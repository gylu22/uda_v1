dataset_type = 'CocoDataset'
data_root = 'data/Devkit/city2foggy/'

backend_args = None
batch_size = 2
num_workers = 6


color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

scale = [(1333, 600), (1333, 1200)]
branch_field = ['sup', 'unsup_teacher', 'unsup_student']
#========================================================dataset pipeline ================================================
# supervised pipline use default deformable_detr train pipeline 
# which will be sent to student model for supervised training 
sup_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]


# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'flip', 'flip_direction',
                   'homography_matrix'))
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(type='LoadEmptyAnnotations'),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor','flip', 'flip_direction',
                   'homography_matrix')),
]

# 'scale_factor'
# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    # dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'scale_factor','ori_shape', 'img_shape'))
]

# ================================================datasets==================================================
# dataset settings 
# use cityscapes as labeled data and foggy cityscapes as unlabeled data

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='CocoFormatAnnos/cityscapes_train_cocostyle.json',
    data_prefix=dict(img='cityscapes/leftImg8bit/train'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline,
    backend_args=backend_args)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='CocoFormatAnnos/cityscapes_foggy_train_cocostyle.json',
    data_prefix=dict(img='foggy_cityscapes/leftImg8bit_foggy/train'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)

# =============================================dataloader================================================

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 1]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='CocoFormatAnnos/cityscapes_foggy_val_cocostyle.json',
        data_prefix=dict(img='foggy_cityscapes/leftImg8bit_foggy/val'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'CocoFormatAnnos/cityscapes_foggy_val_cocostyle.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator


dataloader_act = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='CocoFormatAnnos/cityscapes_foggy_train_cocostyle.json',
        data_prefix=dict(img='foggy_cityscapes/leftImg8bit_foggy/train'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
