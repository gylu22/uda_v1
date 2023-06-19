import copy
from typing import  Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.structures.bbox import  bbox_project
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBaseDetector
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
import json
import cv2
import numpy as np 


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
        
        # self.save_pr = False
        self.iter = 0
        self.save_img_interval = False
        self.cls_thr = [0.5] * 8
        self.cls_thr_ig = [0.2] * 8
        
        
    @torch.no_grad()
    def get_pseudo_instances(
                            self,batch_inputs: Tensor, batch_data_samples: SampleList
                            ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
         # prepare teacher's gt bboxes 
        tea_gt_instances = batch_data_samples[0].gt_instances
        tea_gt_instances.bboxes = bbox_project(
                tea_gt_instances.bboxes,
                torch.from_numpy(batch_data_samples[0].homography_matrix).inverse().to(
                    self.data_preprocessor.device), batch_data_samples[0].ori_shape)
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
        
            
        for result in results_list:
            result.bboxes = bbox_project(
                result.bboxes,
                torch.from_numpy(batch_data_samples[0].homography_matrix).inverse().to(
                    self.data_preprocessor.device), batch_data_samples[0].ori_shape)
        # filter the pseudo labels with adaptative thr 
        results_list = self.filter_pred_instances(results_list,self.cls_thr)
       
        # with open('work_dirs/visualization/act_0.4/pkl/img_path.txt','a') as f:
        #     f.write(f'{batch_data_samples[0].img_path}\n')
        
    
        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results
        
        
        # for data_samples in batch_data_samples:
        #     data_samples.gt_instances.bboxes = bbox_project(
        #         data_samples.gt_instances.bboxes,
        #         torch.from_numpy(data_samples.homography_matrix).inverse().to(
        #             self.data_preprocessor.device), data_samples.ori_shape)
            
        # if self.save_img_interval:
        #     self.save_vis_img(batch_data_samples=batch_data_samples)     
        #     self.save_img_interval = False
            
        # torch.save(batch_data_samples[0].gt_instances,'work_dirs/gt_pseudo_vis/pkl/pre_1.pkl')
        ############################################################################################
        
        
        # if self.save_pr:
        #     TP,FP,FN,precision, recall = self.compute_pr(tea_gt_instances,batch_data_samples)
        #     # print(f"precision:{precision} recall:{recall}")
        #     data = {'iter':self.iter,'TP':TP,'FP':FP,'FN':FN,'precision':precision,'recall':recall}       
        #     if self.iter ==0:
        #          with open('work_dirs/gt_pseudo_vis/compute_pr.json', 'w') as f:
        #             data = [data]
        #             json.dump(data,f,indent=4)
        #     else:
        #         with open('work_dirs/gt_pseudo_vis/compute_pr.json', 'r') as f:
        #             data_list = json.load(f)   
        #         data_list.append(data)
        #         with open('work_dirs/gt_pseudo_vis/compute_pr.json', 'w') as f:
        #             json.dump(data_list,f,indent=4)
            
        #     self.save_pr=False
            
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
        
        
        
    def project_pseudo_instances(self, batch_pseudo_instances: SampleList,
                                 batch_data_samples: SampleList) -> SampleList:
        """Project pseudo instances."""
        for pseudo_instances, data_samples in zip(batch_pseudo_instances,
                                                  batch_data_samples):
            data_samples.gt_instances = copy.deepcopy(
                pseudo_instances.gt_instances)
            
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.tensor(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
        wh_thr = self.semi_train_cfg.get('min_pseudo_bbox_wh', (1e-2, 1e-2))
        return filter_gt_instances(batch_data_samples, wh_thr=wh_thr)
       
       
    
       
    """
    def compute_pr(self,tea_gt_instances,batch_data_samples,iou_thr=0.5):
        
        gt_bboxes = tea_gt_instances.bboxes.cpu().numpy().tolist()
        pred_bboxes = batch_data_samples[0].gt_instances.bboxes.cpu().numpy().tolist()
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
         # 对于每个检测框，找到 IoU 值最大的 Ground Truth 框
        for pred_bbox in pred_bboxes:
            max_iou = 0
            for gt_bbox in gt_bboxes:
                iou = self.compute_iou(pred_bbox,gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt_bbox

            # 如果最大 IoU 值大于阈值，则将其视为正确检测
            if max_iou > iou_thr:
                true_positives += 1
                gt_bboxes.remove(best_gt)
            else:
                false_positives += 1

        # 计算漏检的目标数量
        false_negatives = len(gt_bboxes)

        # 计算 Precision 和 Recall
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)

        return true_positives,false_positives,false_negatives,precision, recall

    def compute_iou(self,box1,box2):
        
         # 计算两个框的交集的左上角和右下角坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])                                                                                                                                                                                                  

        # 计算交集的面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        # 计算 IoU 值
        iou = intersection / union  if union != 0 else 0
        
        return iou    
    
    """
    
    def filter_pred_instances(self,pred_instances,act_thr):
        outputs_list = []
        pred_instances = pred_instances[0]
        # torch.save(pred_instances,f'work_dirs/visualization/act_0.4/pkl/act_out_{self.iter}.pkl')
        
        if pred_instances.bboxes.shape[0] > 0:
            labels = pred_instances.labels
            for cls in range(len(act_thr)):
                if torch.any(torch.eq(labels,cls)):
                    tmp_instances = pred_instances[pred_instances.labels == cls]
                    tmp_instances = tmp_instances[tmp_instances.scores > act_thr[cls]]
                    outputs_list.append(tmp_instances)
                    
        outputs = InstanceData.cat(outputs_list)
        # torch.save(outputs,f'work_dirs/visualization/act_0.4/pkl/act_out_filter_{self.iter}.pkl')
        return outputs
    
    
    
    
    
    
    # def save_vis_img(self,batch_data_samples):
    #     img_path = batch_data_samples[0].img_path
    #     image = cv2.imread(img_path)
    #     bboxes = batch_data_samples[0].gt_instances.bboxes.cpu().detach().numpy().astype(np.int32)
    #     n = bboxes.shape[0]
    #     for i in range(n):
    #         bbox_x_min = bboxes[i][0]
    #         bbox_y_min = bboxes[i][1]
    #         bbox_x_max = bboxes[i][2]
    #         bbox_y_max = bboxes[i][3]
    #         cv2.rectangle(image, (bbox_x_min,bbox_y_min), (bbox_x_max,bbox_y_max), (0, 0, 255), 1)
            
    #     cv2.imwrite(f'work_dirs/gt_pseudo_vis/act_1/act_{self.iter}.jpg', image)