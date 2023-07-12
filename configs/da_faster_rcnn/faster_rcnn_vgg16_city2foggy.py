_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
model=dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.VGG',
        depth=16,
        num_stages=5,
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        init_cfg = dict(type='Pretrained', 
                        checkpoint='work_dirs/vgg/vgg16_weights.pth')),
    neck=dict(
        in_channels=[128, 256, 512, 512]),
    roi_head=dict(bbox_head=dict(num_classes=8)))

data_root = 'data/Devkit/city2foggy/'
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        ann_file='CocoFormatAnnos/cityscapes_train_cocostyle.json',
        data_prefix=dict(img='cityscapes/leftImg8bit/train'),
        ))
val_dataloader = dict(
    dataset=dict(
        ann_file='CocoFormatAnnos/cityscapes_foggy_val_cocostyle.json',
        data_prefix=dict(img='foggy_cityscapes/leftImg8bit_foggy/val')))
val_evaluator = dict(
    ann_file= data_root + 'CocoFormatAnnos/cityscapes_foggy_val_cocostyle.json',
    metric=['bbox'],
    format_only=False)


randomness=dict(seed=0)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

vis_backends = [dict(type='WandbVisBackend',init_kwargs=dict(project='uda_faster_rcnn_vgg',name ='uda_faster_rcnn_bsx4_lr_0.02_city2foggy'))]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.  proposal_fast
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)