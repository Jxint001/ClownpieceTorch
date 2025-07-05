"""
    Implement Various Functions
"""

from typing import List, Union
import copy

from clownpiece.tensor import Tensor, zeros, zeros_like
from clownpiece.autograd.autograd import Node, Edge
from clownpiece.autograd.no_grad import no_grad
from clownpiece.utils_ import wrap_tuple


class Context():
    def __init__(self):
        self.saved_tensors = []
        
    def save_for_backward(self, *args) -> None:
        self.saved_tensors.extend(
            [self.repack_tensor(tensor) for tensor in args if isinstance(tensor, Tensor)]
        )
        
    def get_saved_tensors(self) -> List[Tensor]:
        return self.saved_tensors
    
    @staticmethod
    def repack_tensor(tensor: Tensor):
        # avoid cyclic reference
        if isinstance(tensor, Tensor):
            return copy.copy(tensor) # shallow copy
        else:
            return tensor
    

class Function(Node):
    """
    Base class for all functions.
    """
    ctx: Context
    
    def __init__(self):
        super().__init__()
        self.ctx = None
        
    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError("Forward method not implemented")

    @staticmethod
    def backward(ctx: Context, *args):
        raise NotImplementedError("Backward method not implemented")    
    
    # run forward pass
    def apply(self, *args, **kwargs):
      # your implement here

      # step 1. initialize self.ctx and populate self.next_edges

      # step 2. outputs = self.forward(...) with no_grad

      # step 3. set grad_fn for outputs to self

      # step 4. return outputs

      pass
    
    # run backward pass
    def run(self, *args):
      # your implement here

      # step 1. grad_inputs = self.backward(...) with no_grad

      # step 2. return grad_inputs
      pass

class AccumulateGrad(Function):
    """
    Accumulate gradient to .grad field
    
    grad_fn for leaf tensors
    """
    def __init__(self, input: Tensor):
      # your implement here

      pass
    
    # this forward should never be called
    @staticmethod
    def forward(ctx: Context):
        return None
    
    @staticmethod
    def backward(ctx: Context, output_grad: Tensor):
      # your implement here
      
      pass

"""
    Clone Contiguous
"""

class Clone(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.clone()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output

class Contiguous(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
"""
    Subscriptor
"""

class Subscriptor(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, index_or_slice: Union[int, slice, List[int], List[slice]]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
"""
    Element-wise Binary and Unary Operators
"""

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

# backward method for broadcast
def reduce_broadcast(grad_output: Tensor, input_shape: List[int], output_shape: List[int], end_dim: int = 0) -> Tensor:
  # end_dim argument is for matmul, which only broadcasts dim <= dim() - 2
  pass

# binary op forward decorator
def binary_op_forward_wrapper(forward_impl):
  # save input shapes into ctx
  # call forward_impl
  pass

# binary op backward decorator
def binary_op_backward_wrapper(backward_impl):
  # call backward_impl to get grad_inputs_broadcasted
  # call reduce_broadcast to get grad_inputs
  pass

class Add(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        pass
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Sub(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        pass
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Mul(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        pass
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        pass
    
class Div(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        pass
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        pass
    
class Sign(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Abs(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Cos(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Clamp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, min_val: float, max_val: float):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Log(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, exponent: float): 
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
"""
    Matrix Multiplication
"""

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

"""
    Reduction and Normalization Operations
"""

def reduce_forward_wrapper(forward_impl):
    pass

class Sum(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: Union[int, List[int], None], keepdims: bool = False):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Max(Function):
    @staticmethod
    @reduce_forward_wrapper
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor, grad_indices: Tensor = None):
        pass
    
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
"""
    Shape Manipulation
"""

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, perm: List[int]):
        pass

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim0: int, dim1: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class View(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Narrow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, start: int, length: int):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Chunk(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, chunks: int, dim: int = 0):
        pass
        
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass
    
class Split(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, split: Union[int, List[int]], dim: int = 0):
        pass

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        pass
        
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):    
        pass
    
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        pass
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        pass
    
class Broadcast(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor):
        pass
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        pass