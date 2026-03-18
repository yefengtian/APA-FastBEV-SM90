_base_ = ['./fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py']

class_names = [f'CAT_{i}' for i in range(10)]
dataset_type = 'InternalDataset'
data_root = '/Users/darry/magna/proj/Fastbev_3dod/data/od-demo-c-0701-full/'
ann_file = data_root + 'od_fisheye_infos.pkl'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_config = {
    'src_size': (1280, 1920),
    'input_size': (512, 960),
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    'test_input_size': (512, 960),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=4, n_times=1, transforms=[
        dict(type='LoadImageFromFile')]),
    dict(type='LoadAnnotations3D', with_bbox=True, with_label=True),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='InternalRandomAugImageMultiViewImage',
        data_config=data_config,
        is_train=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=4, n_times=1, transforms=[
        dict(type='LoadImageFromFile')]),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='InternalRandomAugImageMultiViewImage',
        data_config=data_config,
        is_train=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        with_velocity=False,
        sequential=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        with_velocity=False,
        sequential=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        with_velocity=False,
        sequential=False))

model = dict(
    n_cams=4,
    fisheye=True,
    bbox_head=dict(num_classes=len(class_names)))

evaluation = dict(interval=99999)
