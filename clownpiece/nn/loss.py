# MSE, CrossEntropy

from clownpiece.nn.module import Module
from clownpiece import Tensor


# loss.py
class MSELoss(Module):
  def __init__(self, reduction: str = 'mean'):
    pass
  def forward(self, input: Tensor, target: Tensor) -> Tensor:
    pass

class CrossEntropyLoss(Module):
  def __init__(self, reduction: str = 'mean'):
    pass
  def forward(self, logits: Tensor, target: Tensor) -> Tensor:
    # logits is of shape (..., num_class)
    # target is of shape (...), and it's value indicate the index of correct label

    # You need to ensure your implement is differentiable under our autograd engine.
    # However, you can assume target has requires_grad=False and ignore the gradient flow towards it.
    pass