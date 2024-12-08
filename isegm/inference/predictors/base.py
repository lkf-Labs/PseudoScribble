import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from time import time
import numpy as np
import cv2


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.prev_fp_prediction = None
        self.prev_fn_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        self.prev_fp_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        self.prev_fn_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
    # model inference
    def get_prediction(self, clicker, generate_pseu, prev_mask=None, pseudo_clicker = None):
        clicks_list = clicker.get_clicks()
        #==============================the coordinates of the pseudo scribble are represented by a series of clicks========================================================
        pseudo_clicks_list = pseudo_clicker.get_clicks()

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            if generate_pseu:
                input_image = torch.cat((input_image, prev_mask, self.prev_fp_prediction, self.prev_fn_prediction), dim=1)
            else:
                input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        _, pseudo_clicks_lists, _ = self.apply_transforms(
            input_image, [pseudo_clicks_list]
        )
        start = time()
        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed, pseudo_clicks_lists=pseudo_clicks_lists)

        execution_time = time() - start
        seg = F.interpolate(pred_logits['instances'], mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        # the generate_pseu parameter is used to control whether the data should be fed into the false positive and false negative decoders
        if generate_pseu:
            fpmap = F.interpolate(pred_logits['fpmap'], mode='bilinear', align_corners=True,
                                       size=image_nd.size()[2:])
            fnmap = F.interpolate(pred_logits['fnmap'], mode='bilinear', align_corners=True,
                                       size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            seg = t.inv_transform(seg)
            if generate_pseu:
                fpmap = t.inv_transform(fpmap)
                fnmap = t.inv_transform(fnmap)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = seg
        # the generate_pseu parameter is used to control whether the data should be fed into the false positive and false negative decoders
        if generate_pseu:
            return seg.cpu().detach().numpy()[0, 0], fpmap.cpu().detach().numpy()[0, 0], fnmap.cpu().detach().numpy()[0, 0], execution_time
        else:
            return seg.cpu().detach().numpy()[0, 0], None, None, execution_time
    # model inference
    def _get_prediction(self, image_nd, clicks_lists, is_image_changed, pseudo_clicks_lists=None):
        points_nd = self.get_points_nd(clicks_lists)
        pseudo_points_nd = self.get_points_nd(pseudo_clicks_lists)

        return self.net(image_nd, points_nd, pseudo_points_nd)

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed


    def apply_pseudo_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.scribble_transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []

        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone(),
            'prev_fp_prediction': self.prev_fp_prediction.clone(),
            'prev_fn_prediction': self.prev_fn_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        self.prev_fp_prediction = states['prev_fp_prediction']
        self.prev_fn_prediction = states['prev_fn_prediction']
