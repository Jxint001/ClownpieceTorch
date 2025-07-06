# Sigmoid, ReLU, Tanh, LeakyReLU

from clownpiece.tensor import Tensor
from clownpiece.nn.module import Module

class Sigmoid(Module):
  def __init__(self):
    pass
  
  def forward(self, x: Tensor) -> Tensor:
    pass
  
class ReLU(Module):
  def __init__(self):
    pass
  def forward(self, x: Tensor) -> Tensor:
    pass
    
class Tanh(Module):
  def __init__(self):
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass
      
class LeakyReLU(Module):
  def __init__(self, negative_slope: float = 0.01):
    pass

  def forward(self, x: Tensor) -> Tensor:
    pass