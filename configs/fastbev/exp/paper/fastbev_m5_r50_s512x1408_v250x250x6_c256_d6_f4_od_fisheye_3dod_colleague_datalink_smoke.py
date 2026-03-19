_base_ = ['./fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4_od_fisheye_3dod_colleague_datalink.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
)

runner = dict(type='EpochBasedRunner', max_epochs=1)
total_epochs = 1
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

find_unused_parameters = True
opencv_num_threads = 0
mp_start_method = 'spawn'

