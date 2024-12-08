from isegm.data.datasets.glans import MeibomianGlandsDataset
from isegm.model.is_u2net_model import U2netModel
from isegm.utils.exp_imports.default import *

MODEL_NAME = 'ritm_u2net_pseudo_scribble'



def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (350, 740)
    model_cfg.num_max_points = 24

    model = U2netModel(ocr=False, ocr_width=0, with_aux_output=False, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=True,unet=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    # load pre-trained weights
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.u2netp)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 0.0
    # loss_cfg.instance_aux_loss = MutiBceLoss()
    loss_cfg.instance_aux_loss = MutiNFLLoss()
    loss_cfg.instance_aux_loss_weight = 1.0

    # training data augmentation
    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.8, 1.4)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)
    # validation data augmentation
    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)
    # Set up the sampler
    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = MeibomianGlandsDataset(
        cfg.MeibomianGlands_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=0,
        keep_background_prob=0.01,
        points_sampler=points_sampler
    )

    valset = MeibomianGlandsDataset(
        cfg.MeibomianGlands_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=0,
        points_sampler=points_sampler
    )

    optimizer_params = {
        'lr': 0.025, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[200], gamma=0.5)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=5,
                        image_dump_interval=2000,
                        metrics=[AdaptiveIoU()],
                        # limit the total number of clicks
                        max_interactive_points=cfg.num_max_points,
                        # allow iterative number of clicks
                        max_num_next_clicks=cfg.max_num_next_clicks,
                        )
    trainer.run(num_epochs=220)
