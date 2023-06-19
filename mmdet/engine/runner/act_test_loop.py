import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import BaseLoop
from mmdet.registry import LOOPS
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from torch.utils.data import DataLoader
import numpy as np

@LOOPS.register_module()
class ACTLoop(BaseLoop):
    
    """
        Loop for act_hook to inference the whole dataset
    
    """
    def __init__(self, 
                 runner, 
                 dataloader: DataLoader,
                 classes) -> None:
        
        assert isinstance(dataloader,DataLoader)
        self._runner = runner
        self.dataloader = dataloader
        self.classes = classes
        
    def run(self)-> None:
        
        assert hasattr(self.runner.model, 'teacher')
        self.runner.model.teacher.eval()
        model = self.runner.model.teacher
        if is_model_wrapper(model):
            model = model.module
        results=[]   
        
        for idx, data_batch in enumerate(self.dataloader):
            results.append(self.run_iter(idx, data_batch,model))
        
        return results    
   
        
    @torch.no_grad()    
    def run_iter(self,idx,data_batch,model):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """    
        scores_cls = scores_cls = [[] for _ in range(len(self.classes))]
        outputs = model.test_step(data_batch)
        labels = outputs[0].pred_instances.labels.cpu().tolist()
        scores = outputs[0].pred_instances.scores.cpu().numpy().tolist()
        
        for idx,cls in enumerate(labels):
            scores_cls[cls].append(scores[idx])
        
        return scores_cls