import copy
from typing import List, Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.structures.bbox import bbox2roi, bbox_project
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from .semi_base import SemiBaseDetector
from mmengine.runner import load_checkpoint

@MODELS.register_module()
class UDA_DETR(SemiBaseDetector):
   
    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ckpt=None
                ) -> None:
        super().__init__(detector,semi_train_cfg,semi_test_cfg,
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if ckpt is not None:
            load_checkpoint(self.student,ckpt,map_location='cpu')
            load_checkpoint(self.teacher,ckpt,map_location='cpu')
    
    
    # def loss_by_pseudo_instances(self,
    #                              batch_inputs: Tensor,
    #                              batch_data_samples: SampleList,
    #                              batch_info: Optional[dict] = None) -> dict:
    #     """Calculate losses from a batch of inputs and pseudo data samples.

    #     Args:
    #         batch_inputs (Tensor): Input images of shape (N, C, H, W).
    #             These should usually be mean centered and std scaled.
    #         batch_data_samples (List[:obj:`DetDataSample`]): The batch
    #             data samples. It usually includes information such
    #             as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
    #             which are `pseudo_instance` or `pseudo_panoptic_seg`
    #             or `pseudo_sem_seg` in fact.
    #         batch_info (dict): Batch information of teacher model
    #             forward propagation process. Defaults to None.

    #     Returns:
    #         dict: A dictionary of loss components
    #     """
    #     # batch_data_samples = filter_gt_instances(
    #     #     batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
    #     losses = self.student.loss(batch_inputs, batch_data_samples)
    #     pseudo_instances_num = sum([
    #         len(data_samples.gt_instances)
    #         for data_samples in batch_data_samples
    #     ])
    #     unsup_weight = self.semi_train_cfg.get(
    #         'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
    #     return rename_loss_dict('unsup_',
    #                             reweight_loss_dict(losses, unsup_weight))



    @torch.no_grad()
    def get_pseudo_instances(
                            self,batch_inputs: Tensor, batch_data_samples: SampleList
                            ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        # x = self.teacher.extract_feat(batch_inputs)
        self.teacher.eval()
        # get teacher query and teacher's predict
        tea_img_feats = self.teacher.extract_feat(batch_inputs)
        tea_encoder_inputs_dict, tea_decoder_inputs_dict = self.teacher.pre_transformer(
            tea_img_feats, batch_data_samples)
        tea_encoder_outputs_dict = self.teacher.forward_encoder(**tea_encoder_inputs_dict)
        tea_tmp_dec_in, tea_head_inputs_dict = self.teacher.pre_decoder(**tea_encoder_outputs_dict)
        tea_decoder_inputs_dict.update(tea_tmp_dec_in)
        tea_decoder_outputs_dict = self.teacher.forward_decoder(**tea_decoder_inputs_dict)
        tea_head_inputs_dict.update(tea_decoder_outputs_dict)
        
        
        results_list = self.teacher.bbox_head.predict(**tea_head_inputs_dict,
                                                      batch_data_samples=batch_data_samples,rescale=False)
        teacher_query_pos = tea_decoder_inputs_dict["query_pos"]
        
        # results_list = self.teacher.predict(batch_inputs,batch_data_samples,rescale=False)
        
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results
        
        # 去掉低于阈值之外的预测结果
        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=self.semi_train_cfg.pseudo_label_initial_score_thr)
        
        for data_samples in batch_data_samples:
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
       
        batch_info = {
            # 'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': []
        }
        
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)
        return batch_data_samples, batch_info, teacher_query_pos
         


    def init_weights(self):
        pass
      
        
    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 teacher_query_pos,
                                 batch_info: Optional[dict] = None,
                                 ) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        # batch_data_samples = filter_gt_instances(
        #     batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)
        
        stu_img_feats = self.student.extract_feat(batch_inputs)
        stu_head_inputs_dict = self.student.forward_transformer(stu_img_feats,
                                                    batch_data_samples,
                                                    teacher_query_pos)
        losses = self.student.bbox_head.loss(
            **stu_head_inputs_dict, batch_data_samples=batch_data_samples)
        # losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1.) if pseudo_instances_num > 0 else 0.
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))
       
        

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
                multi_batch_data_samples: Dict[str, SampleList]) -> dict:
            """Calculate losses from multi-branch inputs and data samples.

            Args:
                multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                    input images, each value with shape (N, C, H, W).
                    Each value should usually be mean centered and std scaled.
                multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                    The dict of multi-branch data samples.

            Returns:
                dict: A dictionary of loss components
            """
            losses = dict()
            losses.update(**self.loss_by_gt_instances(
                multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

            origin_pseudo_data_samples, batch_info,teacher_query_pos  = self.get_pseudo_instances(
                multi_batch_inputs['unsup_teacher'],
                multi_batch_data_samples['unsup_teacher'])
            multi_batch_data_samples[
                'unsup_student'] = self.project_pseudo_instances(
                    origin_pseudo_data_samples,
                    multi_batch_data_samples['unsup_student'])
            losses.update(**self.loss_by_pseudo_instances(
                multi_batch_inputs['unsup_student'],
                multi_batch_data_samples['unsup_student'],teacher_query_pos,batch_info))
            return losses