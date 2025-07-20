# Sigmoid, ReLU, Tanh, LeakyReLU

from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module

class Sigmoid(Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x: Tensor) -> Tensor:
    neg_x = -x
    exp_neg = neg_x.exp()
    one = Tensor(1.0, requires_grad=False)
    denominator = one + exp_neg
    return one / denominator
  
class ReLU(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: Tensor) -> Tensor:
    zero = Tensor(0.0, requires_grad=False)
    mask = x > zero
    return mask * x + (1 - mask) * zero
    
class Tanh(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: Tensor) -> Tensor:
    return Tensor.tanh(x)
      
class LeakyReLU(Module):
  def __init__(self, negative_slope: float = 0.01):
    super().__init__()
    self.negative_slope = negative_slope

  def forward(self, x: Tensor) -> Tensor:
    alpha = Tensor(self.negative_slope, requires_grad=False)
    alpha_x = x * alpha
    
    mask = x > alpha_x
    return mask * x + (1 - mask) * alpha_x