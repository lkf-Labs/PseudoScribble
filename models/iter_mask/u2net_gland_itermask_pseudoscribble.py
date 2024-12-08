from isegm.data.datasets.glans import MeibomianGlandsDataset
from isegm.model.is_u2net_pseudoscribble_model import U2netPseudoScribbleModel
from isegm.utils.exp_imports.default import *
import ml_collections

MODEL_NAME = 'gland_baseline_pseudoscribble'


# cfg是各种参数

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (350, 740)
    # model_cfg.crop_size = (320, 640)
    model_cfg.num_max_points = 24 #40

    model = U2netPseudoScribbleModel(use_leaky_relu=True,use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=True,is_unet=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    # load pre-trained weights
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.u2netp)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size
    # select loss function
    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 0.0
    # loss_cfg.instance_aux_loss = MutiBceLoss()
    loss_cfg.instance_aux_loss = MutiNFLLoss()  # Normalized Focal Loss  for deep supervision
    # loss_cfg.instance_aux_loss = MutiAdaptableNFLLoss()
    loss_cfg.instance_aux_loss_weight = 1.0
    # *** we conducted extensive experiments to select the most suitable loss function ***
    # fpmap loss
    # loss_cfg.fp_aux_loss = MutiNFLLoss(alpha=0.75, gamma=5)
    # loss_cfg.fp_aux_loss = MutilWeightedCrossEntropyLoss()
    # config = ml_collections.ConfigDict()
    # config.data = data = ml_collections.ConfigDict()
    # data.w = 740
    # data.h = 350
    # config.device = (torch.device("cuda:" + cfg.gpus))
    # config.student = student = ml_collections.ConfigDict()
    # student.nu = 1.0
    # student.epsilon = 1e-8
    # loss_cfg.fp_aux_loss = MutiTLoss(config, nu=config.student.nu, epsilon=config.student.epsilon)
    # loss_cfg.fp_aux_loss = MutiAdaptableNFLLoss()
    loss_cfg.fp_aux_loss = MutiSoftIoULoss()  # IoU loss for deep supervision
    # loss_cfg.fp_aux_loss = MutiBoundaryDoULoss(cfg.gpus)  # BDoU loss
    # loss_cfg.fp_aux_loss = MutiDiceBCELoss()  # Dice+bce loss
    # loss_cfg.fp_aux_loss = MutiIoUBCELoss()  # IoU+bce loss
    # loss_cfg.fp_aux_loss = MutiBDLoss()  # bd loss
    # loss_cfg.fp_aux_loss = MutiTverskyLoss()  # Tversky loss
    # loss_cfg.fp_aux_loss = MutiDiceLoss()  # Dice loss
    # loss_cfg.fp_aux_loss = MutiIoUNFLLoss(alpha=0.5, gamma=2)  # iou+nfl loss
    loss_cfg.fp_aux_loss_weight = 0.0
    # fnmap loss
    # loss_cfg.fn_aux_loss = MutiNFLLoss(alpha=0.80, gamma=5)
    # loss_cfg.fn_aux_loss = MutilWeightedCrossEntropyLoss()
    # loss_cfg.fn_aux_loss = MutiAdaptableNFLLoss()
    # loss_cfg.fn_aux_loss = MutiTLoss(config, nu=config.student.nu, epsilon=config.student.epsilon)
    loss_cfg.fn_aux_loss = MutiSoftIoULoss() # IoU loss for deep supervision
    # loss_cfg.fn_aux_loss = MutiBoundaryDoULoss(cfg.gpus)  # BDoU loss
    # loss_cfg.fn_aux_loss = MutiDiceBCELoss()  # Dice+bce loss
    # loss_cfg.fn_aux_loss = MutiIoUBCELoss()  # IoU+bce loss
    # loss_cfg.fn_aux_loss = MutiBDLoss()  # bd loss
    # loss_cfg.fn_aux_loss = MutiTverskyLoss()  # Tversky loss
    # loss_cfg.fn_aux_loss = MutiDiceLoss()  # Dice loss
    # loss_cfg.fn_aux_loss = MutiIoUNFLLoss(alpha=0.5, gamma=2)  # iou+nfl loss
    loss_cfg.fn_aux_loss_weight = 0.0

    # Configure UISM
    # UISM_cfg = None
    UISM_cfg = edict()
    UISM_cfg.remove_samll_object_area = cfg.remove_samll_object_area
    UISM_cfg.fp_gamma = cfg.fp_gamma
    UISM_cfg.fn_gamma = cfg.fn_gamma


    # training data augmentation
    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
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
        'lr': 0.05, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[350], gamma=1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg, UISM_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 100), (250, 1), (300, 100), (450,1)],
                        image_dump_interval=2000,
                        metrics=[AdaptiveIoU()],
                        # limit the total number of clicks
                        max_interactive_points=cfg.num_max_points,
                        # allow iterative number of clicks
                        max_num_next_clicks=cfg.max_num_next_clicks,
                        is_multitask = False,
                        )
    trainer.run(num_epochs=500, validation=False)
