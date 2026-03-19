_base_ = ['./fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.py']

# ============================ data config =============================
class_names = [f'CAT_{i}' for i in range(10)]
dataset_type = 'InternalDataset'
data_root = '/Users/darry/magna/proj/Fastbev_3dod/data/od-demo-c-0701-full/'
ann_file = data_root + 'od_fisheye_infos.pkl'
point_cloud_range = [-50, -50, -5, 50, 50, 3]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# Keep fisheye camera settings aligned with teammate PSD config.
data_config = {
    'src_size': (1536, 1920),
    'input_size': (768, 960),
    'resize': (0.0, 0.0),
    'crop': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'test_input_size': (768, 960),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=4,
        n_times=1,
        transforms=[dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bev_seg=False),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=4,
        n_times=1,
        transforms=[dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

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

# ========================= model config =========================
model = dict(
    n_cams=4,
    fisheye=True,
    bbox_head=dict(num_classes=len(class_names)),
)

# Do not load the base nuImages pretrain checkpoint by default.
load_from = None
resume_from = None
