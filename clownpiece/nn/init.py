from clownpiece.tensor import Tensor
from clownpiece.autograd.no_grad import no_grad

from typing import Callable
import random
import math

_gain_lookup_table = {
  "linear": 1.0,
  "identity": 1.0,
  "sigmoid": 1.0,
  "tanh": 5/3,
  "relu": math.sqrt(2),
  "leaky_relu": lambda a: math.sqrt(2 / (1 + a * a)),
  "selu": 3/4
}

def calculate_gain(nonlinearity: str, a: float = 0) -> float:
  nonlinearity = nonlinearity.lower()

  if nonlinearity not in _gain_lookup_table:
    raise KeyError(f"Unkown nonlinearity: {nonlinearity}, choices are {list(_gain_lookup_table.keys())}")
  
  value = _gain_lookup_table[nonlinearity]
  if nonlinearity == "leaky_relu":
    return value(a)
  else:
    return value
  
def _no_grad_init(func):
    def wrapper(*args, **kwargs):
        with no_grad():
            return func(*args, **kwargs)
    return wrapper  
  
def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if tensor.dim() == 2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = len(tensor) // (num_output_fmaps * num_input_fmaps)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out

def generate(tensor: Tensor, func):
  vals = [func() for _ in range(tensor.__len__())]
  tensor.copy_(Tensor(vals).reshape(tensor.shape))

@_no_grad_init
def constants_(tensor: Tensor, value: float):
  generate(tensor, lambda: value)

@_no_grad_init
def zeros_(tensor: Tensor):
  generate(tensor, lambda: 0)

@_no_grad_init
def ones_(tensor: Tensor):
  generate(tensor, lambda: 1)

@_no_grad_init
def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
  generate(tensor, lambda: random.gauss(mean, std))

@_no_grad_init
def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0):
  generate(tensor, lambda: random.uniform(low, high))

def xavier_uniform_(tensor: Tensor, gain: float = 1.0):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  a = gain * math.sqrt(6.0 / (fan_in + fan_out))
  
  generate(tensor, lambda: random.uniform(-a, a))
  
  return tensor

def xavier_normal_(tensor: Tensor, gain: float = 1.0):
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  std = gain * math.sqrt(2.0 / (fan_in + fan_out))
  
  generate(tensor, lambda: random.gauss(0, std))
  
  return tensor

def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
  if mode not in ["fan_in", "fan_out"]:
      raise ValueError(f"Invalid mode: {mode}. Choose 'fan_in' or 'fan_out'")
  
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  fan = fan_in if mode == "fan_in" else fan_out
  gain = calculate_gain(nonlinearity, a)
  
  bound = gain * math.sqrt(3.0 / fan)

  generate(tensor, lambda: random.uniform(-bound, bound))
  
  return tensor

def kaiming_normal_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
):
  if mode not in ["fan_in", "fan_out"]:
      raise ValueError(f"Invalid mode: {mode}. Choose 'fan_in' or 'fan_out'")
  
  fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
  fan = fan_in if mode == "fan_in" else fan_out
  gain = calculate_gain(nonlinearity, a)
  
  std = gain / math.sqrt(fan)
  
  generate(tensor, lambda: random.gauss(0.0, std))
  
  return tensor