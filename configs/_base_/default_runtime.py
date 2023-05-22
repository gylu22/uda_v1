default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10,save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# vis_backends = [dict(type='WandbVisBackend',init_kwargs=dict(project='uda_v1',name ='uda_deformable_detr_mean_teacher_v1'))]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# log_level = 'INFO'
# load_from = None
# resume = False

