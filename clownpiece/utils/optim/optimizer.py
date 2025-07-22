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
        self.defaults = defaults
        self.param_groups = []
        self.state = {}

        parameters = list(parameters)
        
        if isinstance(parameters[0], dict):
            for param_group in parameters:
                self.add_param_group(param_group)
        else:
            self.add_param_group({'params': list(parameters)})

          # assign state for each param
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = {}

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
        for group in self.param_groups:
            for p in group['params']:
                if set_to_None:
                    p.grad = None
                else:
                    zeros_(p.grad)

class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0, damping: float = 0.0, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            damping = group['damping']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad + weight_decay * p
                state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = Tensor.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.copy_(momentum * buf + (1 - damping) * grad)
                    state['momentum_buffer'] = buf
                    update = buf
                else:
                    update = grad
                p.copy_(p - lr * update)


class Adam(Optimizer):
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # self._step_count = 0
    super().__init__(params, defaults)

  @no_grad()
  def step(self):
    for group in self.param_groups:
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if weight_decay != 0:
                grad = grad + weight_decay * p

            state = self.state[p]
            if 'step' not in state:
                state['step'] = 0
                state['exp_avg'] = Tensor.zeros_like(p)
                state['exp_avg_sq'] = Tensor.zeros_like(p)
            state['step'] += 1
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            exp_avg.copy_(beta1 * exp_avg + (1 - beta1) * grad)
            exp_avg_sq.copy_(beta2 * exp_avg_sq + (1 - beta2) * (grad * grad))
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)) + eps
            step_size = lr / bias_correction1
            p.copy_(p - step_size * (exp_avg / denom))