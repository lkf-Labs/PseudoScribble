import os
import random
import logging
from copy import deepcopy
from collections import defaultdict
import torch.nn.functional as F

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg, UISM_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 is_multitask=False,
                 fp_loss=None,
                 fn_loss=None,
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.UISM_cfg = UISM_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks
        self.optimizer = optimizer
        self.model_structure = model
        self.optimizer_params = optimizer_params

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob
        self.is_multitask = is_multitask
        self.fp_loss = fp_loss
        self.fn_loss = fn_loss
        # Starting epoch for the second stage
        self.start_epoch_for_stage_two = cfg.next_stage_strat
        # self.with_prev_fp_fn_mask = with_prev_fp_fn_mask

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        # Load the test dataset
        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        # Load the validation dataset
        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        # Set the optimizer
        #    In the first stage of two-stage training, focus on training the shared encoder and segmentation decoder.
        for name, param in self.model_structure.named_parameters():
            if "fn" in name or "fp" in name:
                # Freeze the error prediction decoders
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.optim = get_optimizer(model, optimizer, optimizer_params, fp_lr=0.0, fn_lr=0.0, seg_lr=optimizer_params['lr'],  other_lr=optimizer_params['lr'])

        # load pretrained weight
        model = self._load_weights(model)

        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)
    # 训练阶段
    def training(self, epoch):
        # Unfreeze the error prediction decoders
        # In the first stage of two-stage training, focus on fine-tuning the two error prediction decoders.
        if epoch == self.start_epoch_for_stage_two:
            for name, param in self.model_structure.named_parameters():
                if "fn" in name or "fp" in name:
                    param.requires_grad = True
                else:

                    param.requires_grad = False

            self.optim = get_optimizer(self.model_structure, self.optimizer, self.optimizer_params, fp_lr=0.05, fn_lr=0.05, seg_lr=0.0,  other_lr=0.0)
            # the second phase officially launches multi-tasking
            self.is_multitask = True

        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)\
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            # torch.cuda.empty_cache()
            global_step = epoch * len(self.train_data) + i
            # pass forward
            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, epoch=epoch)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            with torch.no_grad():
                global_step = epoch * len(self.val_data) + i
                loss, batch_losses_logging, splitted_batch_data, outputs = \
                    self.batch_forward(batch_data, validation=True, epoch=epoch)

                batch_losses_logging['overall'] = loss
                reduce_loss_dict(batch_losses_logging)
                for loss_name, loss_value in batch_losses_logging.items():
                    losses_logging[loss_name].append(loss_value.item())

                val_loss += batch_losses_logging['overall'].item()

                if self.is_master:
                    tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                    for metric in self.val_metrics:
                        metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, epoch, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
            orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            prev_fp_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            prev_fn_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            last_click_indx = None

            with torch.no_grad():
                # click n times iteratively
                num_iters = random.randint(0, self.max_num_next_clicks)

                for click_indx in range(num_iters):
                    last_click_indx = click_indx

                    if not validation:
                        self.net.eval()

                    if self.click_models is None or click_indx >= len(self.click_models):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
                    net_input = torch.cat((net_input, prev_fp_output, prev_fn_output), dim=1) if epoch >= self.start_epoch_for_stage_two else net_input

                    model_output = eval_model(net_input, points)
                    prev_output = torch.sigmoid(model_output['instances'])
                    if epoch >= self.start_epoch_for_stage_two:
                        prev_fp_output = torch.sigmoid(model_output['fpmap'])
                        prev_fn_output = torch.sigmoid(model_output['fnmap'])
                    # Iterative sampling strategy to determine the next click location
                    points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)

                    if not validation:
                        self.net.train()

                if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
                    zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            batch_data['points'] = points
            # Concatenate the original input image and the output mask from the previous step along the channel dimension
            net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
            net_input = torch.cat((net_input, prev_fp_output, prev_fn_output), dim=1) if epoch >= self.start_epoch_for_stage_two else net_input
            output = self.net(net_input, points)

            loss = 0.0
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['instances'], batch_data['instances']), epoch) # 输出掩码与真是掩码
            loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                                lambda: (output['instances_aux'], batch_data['instances']), epoch)
            # perform multi-task learning
            if self.is_multitask:
                # train the error prediction decoder using the UISM strategy
                batch_size, c, h, w = output['instances'].size()
                # output normalization
                seg_output = torch.sigmoid(output['instances'])
                seg_output = seg_output.view(seg_output.size(0), -1)
                binary_output = torch.where(seg_output > 0.49, torch.ones_like(seg_output),
                                            torch.zeros_like(seg_output))
                mask_binary = batch_data['instances'].view(batch_data['instances'].size(0), -1)

                # False positive: predicted as gland, but actually the background region
                false_positive_map = torch.logical_and(binary_output == 1, mask_binary == 0)
                false_positive_map = false_positive_map.float()

                false_positive_map = false_positive_map.view(batch_size, c, h, w)
                false_positive_map_ori = false_positive_map
                if self.UISM_cfg is not None:
                    for i in range(batch_size):
                        fp = false_positive_map[i][0].cpu().numpy().astype(np.uint8)
                        seg_output = torch.sigmoid(output['instances'])[i][0].cpu().detach().numpy()
                        fp = (np.logical_and(((seg_output>0.49)&(seg_output<self.UISM_cfg.fp_gamma)).astype(np.uint8),fp)).astype(np.uint8)
                        num_labels, labels = cv2.connectedComponents(fp, connectivity=8)
                        # Count the number of pixels in each connected region
                        label_counts = np.bincount(labels.flatten())
                        # Obtain the indices of regions that need to be modified
                        small_regions = np.where(label_counts < self.UISM_cfg.remove_samll_object_area)[0]
                        # Set the pixel values to 0 within small regions
                        mask = np.isin(labels, small_regions)
                        fp[mask] = 0
                        false_positive_map[i][0] = (torch.from_numpy(fp).to("cuda:" + str(self.cfg.gpu_ids[0]))).float()
                batch_data['fpmap_gt'] = [false_positive_map , false_positive_map_ori] # 8x1x350x741

                # False negative: predicted as background, but actually the target object region
                false_negative_map = torch.logical_and(binary_output == 0, mask_binary == 1)
                false_negative_map = false_negative_map.float()

                false_negative_map = false_negative_map.view(batch_size, c, h, w)
                false_negative_map_ori = false_negative_map
                if self.UISM_cfg is not None:
                    for i in range(batch_size):
                        fn = false_negative_map[i][0].cpu().numpy().astype(np.uint8)
                        seg_output = torch.sigmoid(output['instances'])[i][0].cpu().detach().numpy()
                        fn = (np.logical_and(((seg_output <= 0.49) & (seg_output > self.UISM_cfg.fn_gamma)).astype(np.uint8), fn)).astype(np.uint8)
                        num_labels, labels = cv2.connectedComponents(fn, connectivity=8)
                        # Count the number of pixels in each connected region
                        label_counts = np.bincount(labels.flatten())
                        # Obtain the indices of regions that need to be modified
                        small_regions = np.where(label_counts < self.UISM_cfg.remove_samll_object_area)[0]
                        # Set the pixel values to 0 within small regions
                        mask = np.isin(labels, small_regions)
                        fn[mask] = 0
                        false_negative_map[i][0] = (torch.from_numpy(fn).to("cuda:" + str(self.cfg.gpu_ids[0]))).float()

                batch_data['fnmap_gt'] = [false_negative_map, false_negative_map_ori]  # 8x1x350x741

                loss = self.add_loss('fp_aux_loss', loss, losses_logging, validation,
                                     lambda: (output['fpmap_aux'], batch_data['fpmap_gt']), epoch)
                loss = self.add_loss('fn_aux_loss', loss, losses_logging, validation,
                                     lambda: (output['fnmap_aux'], batch_data['fnmap_gt']), epoch)

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output
    # calculate the loss
    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs, epoch):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if epoch >= self.start_epoch_for_stage_two:
            if  loss_name == 'fp_aux_loss' : loss_weight = 1.0
            elif loss_name == 'fn_aux_loss': loss_weight = 1.0
            elif loss_name == 'instance_aux_loss': loss_weight = 0.0
            else: loss_weight = 0.0
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss
    # save visualization
    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs['instances']).detach().cpu().numpy()
        points = points.detach().cpu().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])
    # load weight
    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return self.cfg.local_rank == 0

# iterative sampling strategy to obtain the next point
def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :] # 去掉channel维度
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5
    # obtain the false positive and false negative regions
    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        # calculate the distance of each pixel in over-segmented and under-segmented regions
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        # Find the maximum distance
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        # Determine if this is a positive click
        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        # Morphological erosion, reducing the area by four times
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        # Obtain the indices of all regions after morphological erosion
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            # Randomly select an index as the click coordinate
            coords = indices[np.random.randint(0, len(indices))]

            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points

# Download pre-trained weights
def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
