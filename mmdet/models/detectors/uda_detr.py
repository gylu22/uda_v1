import copy
from typing import  Optional, Tuple, Dict
import torch
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict,multi_apply)
from mmdet.structures.bbox import  bbox_project
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig,reduce_mean
from .semi_base import SemiBaseDetector
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmcv.ops.nms import nms
from mmdet.utils import InstanceList, OptInstanceList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from typing import Dict, List, Tuple
from mmdet.registry import MODELS

@MODELS.register_module()
class UDA_DETR(SemiBaseDetector):
   
    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ckpt=None,
                 loss_reliable_cls:ConfigType = None
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
        self.loss_reliable_cls = MODELS.build(loss_reliable_cls)
        
    @torch.no_grad()
    def get_pseudo_instances(
                            self,batch_inputs: Tensor, batch_data_samples: SampleList
                            ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
         # prepare teacher's gt bboxes 
        # tea_gt_instances = batch_data_samples[0].gt_instances
        # tea_gt_instances.bboxes = bbox_project(
        #         tea_gt_instances.bboxes,
        #         torch.from_numpy(batch_data_samples[0].homography_matrix).inverse().to(
        #             self.data_preprocessor.device), batch_data_samples[0].ori_shape)
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
        
         #change bbox coordinate to original image    
        for result in results_list:   
            result.bboxes = bbox_project(
                result.bboxes,
                torch.from_numpy(batch_data_samples[0].homography_matrix).inverse().to(
                    self.data_preprocessor.device), batch_data_samples[0].ori_shape)
            
        # compute bboxes after nms
        results_list = self.multi_class_nms(results_list)
        # filter the pseudo labels with adaptative thr 
        # results_list = self.filter_pred_instances(results_list,self.cls_thr)
        # pseudo_label_idx_list = self.get_pseudo_label_idx(results_list,self.cls_thr)
        # torch.save(results_list,f'work_dirs/visualization/nms/pkl/instances_nms_{self.iter}.pkl')
        # with open('work_dirs/visualization/nms/pkl/img_path.txt','a') as f:
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
                                 pseudo_label_idx_list,
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
        # torch.save(batch_inputs,'work_dirs/visualization/plus_unliable/pkl/stu_inputs.pkl')
        # torch.save(batch_data_samples[0].gt_instances,'work_dirs/visualization/plus_unliable/pkl/pseudo_instances_1.pkl')
        stu_img_feats = self.student.extract_feat(batch_inputs)
        stu_head_inputs_dict = self.student.forward_transformer(stu_img_feats,
                                                    batch_data_samples,
                                                    teacher_query_pos)
        
        # losses = self.student.bbox_head.loss(
        #     **stu_head_inputs_dict, batch_data_samples=batch_data_samples)
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        stu_head_outs = self.student.bbox_head.forward(stu_head_inputs_dict['hidden_states'],
                                                       stu_head_inputs_dict['references'])
        
        loss_inputs = stu_head_outs + (batch_gt_instances,batch_img_metas,pseudo_label_idx_list)
        losses = self.loss_by_feat(*loss_inputs)
        
        
        pseudo_instances_num = sum([
            len(data_samples.gt_instances)
            for data_samples in batch_data_samples
        ])
        unsup_weight = self.semi_train_cfg.get(
            'unsup_weight', 1) if pseudo_instances_num > 0 else 0.
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

            origin_pseudo_data_samples, batch_info,teacher_query_pos = self.get_pseudo_instances(
                                                                    multi_batch_inputs['unsup_teacher'],
                                                                    multi_batch_data_samples['unsup_teacher'])
            multi_batch_data_samples[
                'unsup_student'],pseudo_label_idx_list= self.project_pseudo_instances(
                    origin_pseudo_data_samples,
                    multi_batch_data_samples['unsup_student'])
                
            losses.update(**self.loss_by_pseudo_instances(
                            multi_batch_inputs['unsup_student'],
                            multi_batch_data_samples['unsup_student'],
                            teacher_query_pos,
                            pseudo_label_idx_list,
                            batch_info,
                            ))
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
        batch_data_samples = filter_gt_instances(batch_data_samples, wh_thr=wh_thr)
        pseudo_label_idx_list = []
        for data_sample in batch_data_samples:
            pseudo_label_idx = self.get_pseudo_label_idx(data_sample.gt_instances,
                                                        self.cls_thr,
                                                        self.cls_thr_ig)
            pseudo_label_idx_list.append(pseudo_label_idx)
            
        # torch.save(batch_data_samples,'work_dirs/visualization/plus_unliable/pkl/pseudo_label.pkl')
        return batch_data_samples,pseudo_label_idx_list
    
       
    
    # def filter_pred_instances(self,pred_instances,act_thr):
    #     outputs_list = []
    #     pred_instances = pred_instances[0]
    #     # torch.save(pred_instances,f'work_dirs/visualization/act_0.4/pkl/act_out_{self.iter}.pkl')
        
    #     if pred_instances.bboxes.shape[0] > 0:
    #         labels = pred_instances.labels
    #         for cls in range(len(act_thr)):
    #             if torch.any(torch.eq(labels,cls)):
    #                 tmp_instances = pred_instances[pred_instances.labels == cls]
    #                 tmp_instances = tmp_instances[tmp_instances.scores > act_thr[cls]]
    #                 outputs_list.append(tmp_instances)
                    
    #     outputs = InstanceData.cat(outputs_list)
    #     # torch.save(outputs,f'work_dirs/visualization/act_0.4/pkl/act_out_filter_{self.iter}.pkl')
    #     return outputs
    
    
    
    def multi_class_nms(self,instances,iou_threshold=0.5):
        results_nms=[]
        for instance in instances:
            labels = instance.labels
            classes = torch.unique(labels)
            idx_list = []
            for i in range(len(classes)):
                tmp_instance = instance[instance.labels == classes[i]]
                # get the index of label class[i] 
                idx_label = torch.nonzero(labels == classes[i]).flatten()
                bboxes = tmp_instance.bboxes
                scores = tmp_instance.scores
                _,idx_nms = nms(bboxes,scores,iou_threshold)
                idx_label = idx_label[idx_nms]
                idx_list.append(idx_label)
            idx_list = torch.cat(idx_list,dim=0)       
            result_nms = instance[idx_list]
            results_nms.append(result_nms)
        return results_nms
    
    
    def get_pseudo_label_idx(self,pred_instance,act_thr,act_thr_ig):
        idx_dict = {}
        idx_reliable =[]
        idx_ig =[]
        scores = pred_instance.scores
        labels = pred_instance.labels
        for cls in range(len(act_thr)): 
            if torch.any(torch.eq(labels,cls)):
                # get cls index in pred_instances
                idx_label = torch.nonzero(labels==cls).flatten()
                tmp_scores = scores[idx_label]
                idx_reliable_single = torch.nonzero(tmp_scores >= act_thr[cls]).flatten() 
                idx_ig_single = torch.nonzero((tmp_scores>act_thr_ig[cls])&(tmp_scores<act_thr[cls])).flatten()
                idx_reliable.append(idx_label[idx_reliable_single])
                idx_ig.append(idx_label[idx_ig_single])
        idx_reliable = torch.cat(idx_reliable,dim=0)        
        idx_ig = torch.cat(idx_ig,dim=0)
        idx_dict['idx_reliable'] = idx_reliable
        idx_dict['idx_ig'] = idx_ig
        return idx_dict
    
    
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas,
        pseudo_label_idx_list,
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou,loss_unliable_cls = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            pseudo_label_idx_list=pseudo_label_idx_list)

        loss_dict = dict()
        loss_dict['loss_cls'] = sum(losses_cls)
        loss_dict['loss_bbox'] = sum(losses_bbox)
        loss_dict['loss_iou'] = sum(losses_iou)
        loss_dict['loss_unliable_cls'] = sum(loss_unliable_cls)
        
        return loss_dict
    
    
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict],
                            pseudo_label_idx_list) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas,
                                           pseudo_label_idx_list)
        
  
        (labels_reliable_list, label_reliable_weights_list, bbox_targets_list,
                bbox_weights_list,labels_unreliable_list,weights_unreliable_list,
                num_total_pos_reliable,num_total_pos_unreliable,num_total_neg)=cls_reg_targets
        
        
        
        labels = torch.cat(labels_reliable_list, 0)
        label_weights = torch.cat(label_reliable_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.student.bbox_head.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos_reliable * 1.0 + \
            num_total_neg * self.student.bbox_head.bg_cls_weight
        if self.student.bbox_head.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.student.bbox_head.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos_reliable = loss_cls.new_tensor([num_total_pos_reliable])
        num_total_pos_reliable = torch.clamp(reduce_mean(num_total_pos_reliable), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        # torch.save(bboxes,'work_dirs/visualization/plus_unliable/pkl/pred_bbox.pkl')
        # torch.save(bboxes_gt,'work_dirs/visualization/plus_unliable/pkl/pseudo_bbox.pkl')
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.student.bbox_head.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos_reliable)

        # regression L1 loss
        loss_bbox = self.student.bbox_head.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos_reliable)
        
        # unliable_pseudo cls loss
        unreliable_lable = torch.cat(labels_unreliable_list,0)
        unreliable_lable_weights = torch.cat(weights_unreliable_list,0)
        loss_unreliable_cls = self.loss_reliable_cls(
                    cls_scores,unreliable_lable,unreliable_lable_weights,
                    avg_factor=num_total_pos_unreliable)
        
        return loss_cls, loss_bbox, loss_iou,loss_unreliable_cls
    
    
 
    
    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    pseudo_label_idx_list) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_reliable_list, label_reliable_weights_list, bbox_targets_list, bbox_weights_list, 
                labels_unreliable_list,weights_unreliable_list,pos_assigned_inds_reliable,
                pos_assigned_inds_unreliable,neg_inds
                ) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                      batch_gt_instances,batch_img_metas,
                                      pseudo_label_idx_list)
         
         
        num_total_pos_reliable = sum((inds.numel() for inds in pos_assigned_inds_reliable))
        num_total_pos_unreliable = sum((inds.numel() for inds in pos_assigned_inds_unreliable))
        num_total_neg = sum((inds.numel() for inds in neg_inds))
        return (labels_reliable_list, label_reliable_weights_list, bbox_targets_list,
                bbox_weights_list,labels_unreliable_list,weights_unreliable_list,
                num_total_pos_reliable,num_total_pos_unreliable,num_total_neg)
    
    
    
    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            pseudo_label_idx) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.student.bbox_head.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        reliable_gt_idx = pseudo_label_idx['idx_reliable']
        unreliable_gt_idx = pseudo_label_idx['idx_ig']

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        teacher_scores = gt_instances.scores
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        
        # 获取预测边界框对应的GTbox在gtbox列表里的索引 b
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        # 获取可靠为标签在最终计算bbox权重（pos_assigned_gt_inds）中的索引
        _,reliable_assign_idx = torch.where(pos_assigned_gt_inds==reliable_gt_idx.unsqueeze(-1))
        pos_assigned_inds_reliable = pos_inds.clone()[reliable_assign_idx]
        
        # 获取不可靠伪标签在最终计算中（pos_assigned_gt_inds）的索引
        _,unreliable_assign_idx = torch.where(pos_assigned_gt_inds==unreliable_gt_idx.unsqueeze(-1))
        pos_assigned_inds_unreliable = pos_inds.clone()[unreliable_assign_idx]
        
        pos_gt_bboxes_reliable_idx = pos_assigned_gt_inds[reliable_assign_idx]
        pos_gt_bboxes_reliable = gt_bboxes[pos_gt_bboxes_reliable_idx.long(), :]

        # reliable label targets
        labels_reliable = gt_bboxes.new_full((num_bboxes, ),
                                    self.student.bbox_head.num_classes,
                                    dtype=torch.long)
        labels_reliable[pos_assigned_inds_reliable] = gt_labels[pos_gt_bboxes_reliable_idx]
        label_reliable_weights = gt_bboxes.new_ones(num_bboxes)
        
        # unreliable_label targets 
        labels_unreliable = gt_bboxes.new_full((num_bboxes, ),
                                    self.student.bbox_head.num_classes,
                                    dtype=torch.long)
    
        labels_unreliable[pos_assigned_inds_unreliable] = gt_labels[pos_assigned_gt_inds[unreliable_assign_idx]]
        weights_unreliable = gt_bboxes.new_zeros(num_bboxes)
        weights_unreliable[pos_assigned_inds_unreliable] = teacher_scores[pos_assigned_gt_inds[unreliable_assign_idx]]
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_assigned_inds_reliable] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes_reliable / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_assigned_inds_reliable] = pos_gt_bboxes_targets
        
        return (labels_reliable, label_reliable_weights, bbox_targets, bbox_weights, 
                labels_unreliable,weights_unreliable,pos_assigned_inds_reliable,
                pos_assigned_inds_unreliable,neg_inds)