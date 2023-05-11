import copy
from typing import List, Optional, Tuple

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


@MODELS.register_module()
class UDA_DETR(SemiBaseDetector):
    """
    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """


    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
    
    
    
    
    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        
        # Calculate losses from a batch of inputs and pseudo data samples
        pass
    
    
    @torch.no_grad()
    def get_pseudo_instances(
                            self,batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        # x = self.teacher.extract_feat(batch_inputs)
        self.teacher.training= False
        results_list = self.teacher.predict(batch_inputs,batch_data_samples,rescale=False)
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
        return batch_data_samples, batch_info
        
    
    
    def cls_loss_by_pseudo_instances(self,):
        pass
    
    
    def reg_loss_by_pseudo_instances(self,):
        pass
    
    