from .optimizer import Optimizer
from typing import List, Callable

class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        # You need to store base_lrs as to allow closed-form lr computation; you may assume that, optimizer does not add new param groups after initializing the LRSchduler.
        pass

    def get_lr(self) -> List[float]:        
        # Calculate new learning rates for each param_group.
        # To be implemented by subclasses with a closed-form formula based on self.last_epoch.
        raise NotImplementedError

    def step(self, epoch: int = None):
        # Advance to `epoch` (or next epoch if None), then update optimizer.param_groups:
        # 1. Update self.last_epoch 
        # 2. Compute new LRs via self.get_lr()
        # 3. Assign each param_group['lr'] = corresponding new LR
        pass

    def get_last_lr(self) -> List[float]:
        # Return the most recent learning rate for each param_group.
        pass
      
    
class LambdaLR(LRScheduler):
    """
    Lambda learning rate scheduler.
    Applies a user-defined function to the learning rate.
    """
    def __init__(self, optimizer, lr_lambda: Callable[[int], float], last_epoch: int = -1):
        pass

    def get_lr(self) -> List[float]:
        pass
    
class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.
    Multiplies the learning rate by a factor every epoch.
    """
    
    def __init__(self, optimizer, gamma: float = 0.1, last_epoch: int = -1):
        pass

    def get_lr(self) -> List[float]:
        pass
        
class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    Decreases the learning rate by a factor every `step_size` epochs.
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        pass

    def get_lr(self) -> List[float]:
        pass
    
