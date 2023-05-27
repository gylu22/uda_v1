_base_ = [
    './deformable_detr_coco.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/da_city2foggy_coco.py'
]


detector = _base_.model 
# detector.init_cfg = dict(type='Pretrained', checkpoint='work_dirs/uda_deformable_detr_city2foggy_coco_source/epoch_50.pth')

# detector.data_preprocessor = dict(
#     type='DetDataPreprocessor',
#     mean=[103.530, 116.280, 123.675],
#     std=[1.0, 1.0, 1.0],
#     bgr_to_rgb=True,
#     pad_size_divisor=32)

model=dict(
    _delete_=True,
    type='UDA_DETR',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        pseudo_label_initial_score_thr=0.5,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'),
    ckpt='work_dirs/uda_deformable_detr_city2foggy_coco_source/epoch_50.pth'
    )

"""
# learning policy
max_epochs = 50
train_cfg = dict(
    # type='EpochBasedTrainLoop',
    by_epoch=True,max_epochs=max_epochs, val_begin=1, val_interval=1)

val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')


param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001))


# seed 
randomness=dict(seed=0)

auto_scale_lr = dict(base_batch_size=32,enable=True)

custom_hooks = [dict(type='MeanTeacherHook',skip_buffer=False)]
"""

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=150000, val_interval=3000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=150000,
        by_epoch=False,
        milestones=[120000, 150000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3.2e-5, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=50000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook',skip_buffer=False)]


auto_scale_lr = dict(base_batch_size=32,enable=True)
