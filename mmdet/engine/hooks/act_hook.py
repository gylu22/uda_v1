from mmengine.hooks.hook import DATA_BATCH
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.config import Config, ConfigDict
from mmdet.engine.runner import ACTLoop


from mmdet.registry import HOOKS
from mmengine import load, dump
import numpy as np


@HOOKS.register_module()
class ACTHook(Hook):
    
    def __init__(self,
                 cfg_dict : dict = None
                 ) -> None:
        
        assert isinstance(cfg_dict, dict)
        cfg = Config(cfg_dict)  
        self.classes = cfg.classes
        self.label_file = cfg.label_file 
        self.interval = cfg.interval 
        # self.act_loops = Runner.build_test_loop(cfg.act_loop)
        self.dataloader = Runner.build_dataloader(cfg.dataloader)
        # boxes_per_image_gt, cls_ratio_gt = self.get_data_info(cfg.label_file)
        self.min_thr = cfg.get('min_thr', 0.001)  # min cls score threshold for ignore
        self.percent = cfg.get('percent', 0.2)
        self.save_img_interval = cfg.save_img_interval
        
    def get_data_info(self,json_file):
        """get information from labeled data"""
        info = load(json_file)
        id2cls = {}
        total_image = len(info['images'])
        for value in info['categories']:
            id2cls[value['id']] = self.classes.index(value['name'])
        cls_num = [0] * len(self.classes)
        for value in info['annotations']:
            cls_num[id2cls[value['category_id']]] += 1
        cls_num = [max(c, 1) for c in cls_num]  # for some cls not select, we set it 1 rather than 0
        total_boxes = sum(cls_num)
        cls_ratio_gt = np.array([c / total_boxes for c in cls_num])
        boxes_per_image_gt = total_boxes / total_image
        
        return cls_ratio_gt,boxes_per_image_gt
        
        
        
    def before_train_iter(self, 
                          runner:Runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        
        runner.model.iter = runner.iter
        if self.every_n_train_iters(runner, self.save_img_interval) or runner.iter == 0:
            runner.model.save_img_interval = True
            
            
        if self.every_n_train_iters(runner, self.interval) or runner.iter == 0:
            boxes_per_image_gt, cls_ratio_gt = self.get_data_info(self.label_file)
            potential_positive = len(self.dataloader.dataset) * boxes_per_image_gt * cls_ratio_gt
            act_loops = ACTLoop(runner,self.dataloader,self.classes)
            results = act_loops.run()
            cls_thr, cls_thr_ig=self.eval_score_thr(results,potential_positive,self.percent)
            runner.model.cls_thr = cls_thr
            runner.model.cls_thr_ig = cls_thr_ig
            runner.logger.info(f'boxes per image (label data): {boxes_per_image_gt}')
            runner.logger.info(f'cls ratio (label data): {cls_ratio_gt}')
            runner.logger.info(f'cls_thr (unlabled data):{cls_thr}')
            runner.logger.info(f'cls_thr_ig (unlabled data): {cls_thr_ig}')
         
    

       
    def eval_score_thr(self, results, potential_positive ,percent):
        score_list = [[] for _ in self.classes]
        for result in results:
            for i in range(len(self.classes)):
                score_list[i].append(np.array(result[i]))
                    
        score_list = [np.concatenate(c) for c in score_list]
        score_list = [np.zeros(1) if len(c) == 0 else np.sort(c)[::-1] for c in score_list]
        cls_thr = [0] * len(self.classes)
        cls_thr_ig = [0] * len(self.classes)
        for i, score in enumerate(score_list):
            cls_thr[i] = max(0.05, score[min(int(potential_positive[i] * percent), len(score) - 1)])
            # NOTE:for UDA, we use 0.001
            cls_thr_ig[i] = max(self.min_thr, score[min(int(potential_positive[i]), len(score) - 1)])
        return cls_thr, cls_thr_ig
        
        


