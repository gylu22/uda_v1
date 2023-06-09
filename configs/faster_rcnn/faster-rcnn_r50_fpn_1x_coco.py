_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
model =dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.VGG',
        depth=16,
        num_stages=5,
        out_indices=(1, 2, 3, 4),
        init_cfg = dict(type='Pretrained', 
                        checkpoint='https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth'),
                        ),
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



train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)


# vis_backends = [dict(type='WandbVisBackend',init_kwargs=dict(project='uda_single_class',name ='uda_faster_rcnn_bsx4_lr_0.004_all-classes'))]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')



optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.  proposal_fast
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
    
   