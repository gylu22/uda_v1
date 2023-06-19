_base_ = [
    './deformable_detr_coco.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/da_city2foggy_coco_act.py'
]


detector = _base_.model 
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
    ckpt='work_dirs/best_coco_bbox_mAP_epoch_45.pth'
    )

act_hook_cfg = dict(
    interval=500,
    percent = 0.2,
    min_thr = 0.001,
    save_img_interval = 50,
    label_file='data/Devkit/city2foggy/CocoFormatAnnos/cityscapes_train_cocostyle.json',
    classes = ['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus'],
    dataloader = _base_.dataloader_act)
    
    

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=150000, val_interval=500)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
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


custom_hooks = [dict(type='MeanTeacherHook',skip_buffer=False),
                # dict(type='ComputePR',interval=1)
                dict(type='ACTHook',
                     cfg_dict=act_hook_cfg)]
# custom_hooks = [dict(type='MeanTeacherHook',skip_buffer=False)]

auto_scale_lr = dict(base_batch_size=32,enable=True)

randomness=dict(seed=0)