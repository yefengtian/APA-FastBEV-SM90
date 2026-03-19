_base_ = ['./fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.py']

# Baseline note:
# This config keeps 3DOD head from FastBEV and uses psd_type-style fish datalink.

class_names = [f'CAT_{i}' for i in range(10)]
dataset_type = 'InternalDataset'
data_root = '/Users/darry/magna/proj/Fastbev_3dod/data/od-demo-c-0701-full/'
ann_file = None  # read raw scene*.json directly
point_cloud_range = [-50, -50, -5, 50, 50, 3]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data_config = dict(
    cams=[
        'CAM_FISHEYE_FORWARD',
        'CAM_FISHEYE_LEFT',
        'CAM_FISHEYE_BACKWARD',
        'CAM_FISHEYE_RIGHT',
    ],
    Ncams=4,
    input_size=(768, 960),
    src_size=(1536, 1920),
    resize=(0, 0),
    rot=(0, 0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

train_pipeline = [
    dict(
        type='PrepareImageInputsFish_Fullsize',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bev_seg=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='PrepareImageInputsFish_Fullsize',
        is_train=False,
        data_config=data_config,
        sequential=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img_inputs']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        with_velocity=False,
        sequential=False,
        load_interval=1,
    ),
    val=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        with_velocity=False,
        sequential=False,
        load_interval=1,
    ),
    test=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        with_velocity=False,
        sequential=False,
        load_interval=1,
    ))

model = dict(
    type='FastBEVFish3DOD',
    n_cams=4,
    fisheye=True,
    n_voxels=[250, 250, 6],
    voxel_size=[0.4, 0.4, 1.0],
    multi_scale_id=[0],
    neck_3d=dict(
        fuse=dict(in_channels=64 * 6, out_channels=64 * 6),
    ),
    bbox_head=dict(num_classes=len(class_names)),
)

load_from = None
resume_from = None
fp16 = None
