
# ============================ data config =============================

point_cloud_range = [-10, -10, -2.0, 10, 10, 6.0]

class_names = [ 'parking_slot', 'wheel_stop']

dataset_type = 'FishDatasetStop'
# data_root = '/vePFS/sample_data/2026_2_5_metrics'
# test_data_root = '/vePFS/sample_data/2026_2_5_metrics'
data_root = '/vePFS/train_data/2026_2_1_full_size'
test_data_root = '/vePFS/val_data/2026_2_1_full_size'
stride_bev = 2

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0
)

file_client_args = dict(backend='disk')

data_config = {
    'cams': [
        'CAM_FISHEYE_FORWARD', 'CAM_FISHEYE_LEFT', 'CAM_FISHEYE_BACKWARD', 'CAM_FISHEYE_RIGHT'
    ],
    'Ncams': 4,
    'input_size': (768, 960),
    'src_size': (1536, 1920),

    'resize': (0, 0),
    'rot': (0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00
}


train_pipeline = [
    dict(
        type='PrepareImageInputsFish_Fullsize',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepthFish',
        bda_aug_conf=bda_aug_conf,
        is_train=True,
        occ_cls=4,
        spe_cls=4),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 
                                'corner_points', 'spe', 'opy', 'wheel_stop_points', 'canvas'])
]

test_pipeline = [
    dict(
        type='PrepareImageInputsFish_Fullsize',
        is_train=False,
        data_config=data_config,
        sequential=False),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs'])
]


data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    persistent_workers=True,
    pin_memory=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_dir= 'jsons',
        pipeline=train_pipeline,
        test_mode=False,
        box_type_3d='LiDAR',
        modality=input_modality,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='mmcv',
        cam_names=data_config['cams']),

    val=dict(
        type=dataset_type,
        data_root=test_data_root,
        ann_dir= 'jsons',
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        modality=input_modality,
        test_mode=False,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='mmcv',
        cam_names=data_config['cams']),
    
    test=dict(
        type=dataset_type,
        data_root=test_data_root,
        ann_dir = 'jsons',
        pipeline=test_pipeline,
        modality=input_modality,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='mmcv',
        cam_names=data_config['cams'])
    )

evaluation = dict(interval=99999)
n_voxels=[400, 400, 6]
voxel_size=[0.05, 0.05, 0.5]
multi_scale_id=[0]

# ========================= model config =========================
model = dict(
    type='FastBEVFish',
    style="v1",
    n_voxels = n_voxels,
    voxel_size = voxel_size,
    multi_scale_id = multi_scale_id,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_cfg=dict(type='BN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[64]),

    neck_3d=dict(
        type='M2BevNeck',
        in_channels=64*6,
        out_channels=256,
        num_layers=6,
        stride=stride_bev,
        is_transpose=False,
        fuse=dict(in_channels=64*6, out_channels=64*6),
        # norm_cfg=dict(type='SyncBN', requires_grad=True)),
        norm_cfg=dict(type='BN', requires_grad=True)),

    psd_head=dict(
        type='ParkingSlotWheelstopHead2D',
        in_channels=256,
        share_conv_channel=64,
        num_heatmap_convs=2,
        common_heads=dict(
            slot_ctr_offset=(2, 2),
            slot_kp0=(2, 2), slot_kp1=(2, 2), slot_kp2=(2, 2), slot_kp3=(2, 2),
            ws_ctr_offset=(2, 2),
            ws_kp0=(2, 2), ws_kp1=(2, 2),
            slot_type=(4, 2),
            slot_occ=(4, 2),
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_reg=dict(type='L1Loss', reduction='mean', loss_weight=1.0),
        bbox_coder=dict(
            type='CenterPointParkingspotBBoxCoder',
            pc_range=[-10, -10, -1.5, 10, 10, 1.5],
            post_center_range=[-15, -15, -5, 15, 15, 5.0],
            max_num=50,
            score_threshold=0.2,
            out_size_factor=stride_bev,
            voxel_size=voxel_size[:2],
            code_size=9,
            nms_kernel_size=15)
        ),

    train_cfg=dict(
        pts=dict(
            max_objs=50,
            dense_reg=1,
            grid_size=n_voxels,
            out_size_factor=stride_bev,
            point_cloud_range=[-10, -10, -1.5, 10, 10, 1.5],
            voxel_size=voxel_size[:2],
            gaussian_overlap=0.1,
            min_radius=2,
        )
    ),
    test_cfg=dict(
        pts=dict(
        )
    )

)


# ============================ optimizer =============================
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=5,
    warmup_by_epoch=True,
    warmup_ratio=0.001,
    min_lr_ratio=0.01)


# ============================ runtime ===============================
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1, max_keep_ckpts=3, save_last=True)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')

log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
# workflow = [('train', 1)]
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'spawn'



# ============================= custom hook ===============================

custom_hooks = [
    dict(
        type='SaveBestValLossHook',
        key='loss_val',
        rule='less',
        ckpt_name='best_loss_val.pth',
        save_optimizer=False,
        strict=False,
        verbose=True,
        priority='LOWEST',
    )
]