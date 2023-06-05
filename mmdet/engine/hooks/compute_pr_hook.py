from typing import Optional, Union
from mmengine.hooks.hook import DATA_BATCH

import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmengine import load,dump
DATA_BATCH = Optional[Union[dict, tuple, list]]



@HOOKS.register_module()
class ComputePR(Hook):
    
    def __init__(self,
                 interval: int=1,
                 out_dir: str='work_dirs' ) -> None:
        
       self.interval = interval 
      
       
       
    def before_train_iter(self,
                          runner:Runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        
        model = runner.model 
        if self.every_n_train_iters(runner,self.interval):
            model.save_pr = True
            model.iter = runner.iter
    
        # data = model.data_preprocessor(data_batch,False)
        # batch_inputs = data['inputs']['unsup_teacher']
        # batch_data_samples = data['data_samples']['unsup_teacher']
        # tea_predicts,_,_ = model.get_pseudo_instances(batch_inputs,batch_data_samples)
        
        # gt_info = data_batch['data_samples']['unsup_teacher'][1].gt_instances
        
        # print(f"batch_udx:{batch_idx}")
        
        
  
  
  
  
  
 