_base_ = 'yolox_l_8xb8-300e_coco.py'
auto_scale_lr = dict(base_batch_size=16, enable=False)
checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='NumClassCheckHook'),
]
dist_params = dict(backend='nccl')
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
load_from = None
loading_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=50)
log_level = 'INFO'
lr_config = dict(
    policy='step',
    step=[
        8,
        11,
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
model = dict(
    backbone=dict(
        act_cfg=dict(type='Swish'),
        deepen_factor=1.0,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type='CSPDarknet',
        use_depthwise=False,
        widen_factor=1.0),
    bbox_head=dict(
        act_cfg=dict(type='Swish'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_classes=4,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        test_cfg=dict(
            nms=dict(iou_threshold=0.65, type='nms'), score_thr=0.01),
        train_cfg=dict(
            assigner=dict(center_radius=2.5, type='SimOTAAssigner')),
        type='YOLOXHead',
        use_depthwise=False),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=10,
                random_size_range=(
                    480,
                    800,
                ),
                size_divisor=32,
                type='BatchSyncRandomResize'),
        ],
        pad_size_divisor=32,
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(type='Swish'),
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=256,
        type='YOLOXPAFPN',
        upsample_cfg=dict(mode='nearest', scale_factor=2),
        use_depthwise=False),
    test_cfg=dict(nms=dict(iou_threshold=0.65, type='nms'), score_thr=0.01),
    train_cfg=dict(assigner=dict(center_radius=2.5, type='SimOTAAssigner')),
    type='YOLOX')
mp_start_method = 'fork'
multi_image_mix_dataset = dict(
    mixup=dict(
        bbox_clip_border=True,
        flip_ratio=0.5,
        img_scale=(
            640,
            640,
        ),
        max_iters=15,
        pad_val=114.0,
        ratio_range=(
            0.8,
            1.6,
        )),
    mosaic=dict(
        bbox_clip_border=True,
        center_ratio_range=(
            0.5,
            1.5,
        ),
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        prob=0.5))
opencv_num_threads = 0
optimizer = dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
resume_from = None
runner = dict(max_epochs=12, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_pipeline = [
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='Pad'),
    dict(keep_empty=False, min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.65, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_to_square=True,
                    pad_val=dict(img=(
                        114.0,
                        114.0,
                        114.0,
                    )),
                    type='Pad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
workflow = [
    (
        'train',
        1,
    ),
]
