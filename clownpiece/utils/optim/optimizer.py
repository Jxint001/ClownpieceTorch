from clownpiece.autograd import no_grad
from clownpiece.nn import Parameter
from clownpiece.nn.init import zeros_
from clownpiece.tensor import Tensor

from typing import List, Iterable, Dict, Any, Union

class Optimizer:
    param_groups: List[Dict[str, Any]]       # list of parameter groups
    state: Dict[Parameter, Dict[str, Any]]   # mapping param_id -> optimizer state
    defaults: Dict[str, Any]                 # default hyperparams for each group

    def __init__(self, parameters: Union[Iterable[Parameter], Iterable[Dict]], defaults: Dict[str, Any]):
        """
        - `parameters`: an iterable of `Parameter` objects or a list of dicts defining parameter groups.
            - if iterable of `Parameter`, add it as the first param_group.
        - `defaults`: a dict of default hyperparameters (e.g., learning rate).
        """
        pass

    def add_param_group(self, param_group: Dict[str, Any]):
        """Merge defaults into `param_group` and add to `param_groups`."""
        for k, v in self.defaults.items():
            param_group.setdefault(k, v)
        self.param_groups.append(param_group)

    def step(self):
        """Perform a single optimization step (update all parameters).
        Must be implemented by subclasses."""
        raise NotImplementedError

    def zero_grad(self, set_to_None: bool = True):
        """Reset gradients for all parameters."""
        pass
  
class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0, damping: float = 0.0, weight_decay: float = 0.0):
        pass

    def step(self):
        pass

class Adam(Optimizer):
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    pass

  def step(self):
    pass
