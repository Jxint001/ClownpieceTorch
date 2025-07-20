"""
    Implement Various Functions
"""

from typing import List, Union
import copy

from clownpiece import TensorBase, ones_like

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
        # step 1. initialize self.ctx and populate self.next_edges
        if self.ctx is None:
            self.ctx = Context()
        
        self.next_edges = [Edge.gradient_edge(tensor) for tensor in args]
        
        # step 2. outputs = self.forward(...) with no_grad
        with no_grad():
            outputs = self.forward(self.ctx, *args, **kwargs)  # outputs should be tensors
        
        is_list = isinstance(outputs, (list, tuple))

        if not is_list:
            outputs = wrap_tuple(outputs)

        # step 3. set grad_fn for outputs to self (and ouput_nr)
        i = 0
        for out in outputs:
            out.grad_fn = self
            out.requires_grad = True
            out.output_nr = i
            i += 1

        # step 4. return outputs
        return outputs if is_list else outputs[0]
    
    # run backward pass
    def run(self, *args):
        # step 1. grad_inputs = self.backward(...) with no_grad
        with no_grad():
            args = tuple(list(_ for _ in args if _ is not None))
            grad_inputs = self.backward(self.ctx, *args)
        # step 2. return grad_inputs
        return grad_inputs

class AccumulateGrad(Function):
    """
    Accumulate gradient to .grad field
    
    grad_fn for leaf tensors
    """
    def __init__(self, input: Tensor):
        super().__init__()
        self.ctx = Context()
        self.ctx.input = input
    
    # this forward should never be called
    @staticmethod
    def forward(ctx: Context):
        return None
    
    @staticmethod
    def backward(ctx: Context, output_grad: Tensor):
        input = ctx.input
        if input.requires_grad:
            if input.grad is None:
                input.grad = zeros_like(input)
            input.grad += output_grad
        return input.grad

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
        return input.contiguous()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output
    
"""
    Subscriptor
"""

class Subscriptor(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, index_or_slice: Union[int, slice, List[int], List[slice]]):
        ctx.index_or_slice = index_or_slice
        ctx.input_shape = input.shape
        res = input.__getitem__(index_or_slice)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        grad_input = zeros(ctx.input_shape)
        sub_grad_input = grad_input.__getitem__(ctx.index_or_slice)
        sub_grad_input.copy_(grad_output)
        return grad_input
    
"""
    Element-wise Binary and Unary Operators
"""

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.__neg__()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.__neg__()

# backward method for broadcast
def reduce_broadcast(grad_output: Tensor, input_shape: List[int], output_shape: List[int], end_dim: int = 0) -> Tensor:
    # end_dim argument is for matmul, which only broadcasts dim <= dim() - 2

    while grad_output.shape.__len__() > input_shape.__len__():
        grad_output = grad_output.sum(0, keepdims=False)

    n = input_shape.__len__() - end_dim
    for i in range(n):
        if input_shape[i] == 1 and output_shape[i] > 1:
            print(input_shape, output_shape)
            grad_output = grad_output.sum(i, keepdims=True)

    return grad_output

# binary op forward decorator
def binary_op_forward_wrapper(forward_impl):
    def decorator(ctx: Context, arg1: Tensor, arg2: Tensor):
        # save input shapes into ctx
        ctx.save_for_backward(arg1, arg2)
        ctx.shape1 = arg1.shape
        ctx.shape2 = arg2.shape
        # call forward_impl
        return forward_impl(ctx, arg1, arg2)
    return decorator
    

# binary op backward decorator
def binary_op_backward_wrapper(backward_impl):
    def decorator(ctx: Context, arg: Tensor):
        # call backward_impl to get grad_inputs_broadcasted
        grad_inputs_broadcasted = backward_impl(ctx, arg)
        # call reduce_broadcast to get grad_inputs
        if "MatMul" in backward_impl.__name__:
            grad1 = reduce_broadcast(grad_inputs_broadcasted[0], ctx.shape1, arg.shape, 2)
            grad2 = reduce_broadcast(grad_inputs_broadcasted[1], ctx.shape2, arg.shape, 2)
        else:
            grad1 = reduce_broadcast(grad_inputs_broadcasted[0], ctx.shape1, arg.shape, 0)
            grad2 = reduce_broadcast(grad_inputs_broadcasted[1], ctx.shape2, arg.shape, 0)

        return grad1, grad2
    return decorator

class Add(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 + input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output, grad_output
    
class Sub(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 - input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output, -grad_output
    
class Mul(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 * input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx: Context, grad_output):
        input1, input2, = ctx.get_saved_tensors()
        return grad_output * input2, grad_output * input1
    
class Div(Function):
    @staticmethod
    @binary_op_forward_wrapper
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        return input1 / input2
    
    @staticmethod
    @binary_op_backward_wrapper
    def backward(ctx, grad_output):
        input1, input2, = ctx.get_saved_tensors()
        return grad_output * (1 / input2), grad_output * input1 * (-1) * (1 / (input2 * input2))
    
class Sign(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        return input.sign()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output * 0
    
class Abs(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.abs()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * input.sign()
    
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.sin()
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * input.cos()

class Cos(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.cos()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * input.sin() * (-1)

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        res = input.tanh()
        ctx.save_for_backward(res)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        res,  = ctx.get_saved_tensors()
        return grad_output * (1 - res * res)

class Clamp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, min_val: float, max_val: float):
        ctx.min = min_val
        ctx.max = max_val
        ctx.save_for_backward(input)
        return input.clamp(min_val, max_val)
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        mask = (ctx.min < input) * (input < ctx.max)
        return grad_output * mask, None, None

class Log(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        ctx.save_for_backward(input)
        return input.log()
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        return grad_output * (1 / input)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        res = input.exp()
        ctx.save_for_backward(res)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        res,  = ctx.get_saved_tensors()
        return grad_output * res

class Pow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, exponent: float): 
        ctx.exp = exponent
        res = input.pow(exponent)
        ctx.save_for_backward(input, res)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, res, = ctx.get_saved_tensors()
        return grad_output * ctx.exp * (res / input)
    
class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor):
        res = input.sqrt()
        ctx.save_for_backward(res)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        res, = ctx.get_saved_tensors()
        return grad_output * 0.5 * (1 / res)
    
"""
    Matrix Multiplication
"""

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, input1: Tensor, input2: Tensor):
        ctx.save_for_backward(input1, input2)
        return input1.matmul(input2)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input1, input2, = ctx.get_saved_tensors()
        d1, d2 = input1.dim(), input2.dim()

        if d1 == 1 and d2 == 1:
            return grad_output * input2, grad_output * input1
        
        if d1 == 1:
            input1 = input1.unsqueeze(0)
            grad_output = grad_output.unsqueeze(-2)
        if d2 == 1:
            input2 = input2.unsqueeze(1)
            grad_output = grad_output.unsqueeze(-2).transpose(-1, -2)

        grad_input1_broadcasted = grad_output.matmul(input2.transpose(-1, -2))
        grad_input2_broadcasted = input1.transpose(-1, -2).matmul(grad_output)

        grad_input1 = reduce_broadcast(grad_input1_broadcasted, input1.shape, grad_input1_broadcasted.shape, 2)
        grad_input2 = reduce_broadcast(grad_input2_broadcasted, input2.shape, grad_input2_broadcasted.shape, 2)

        if d1 == 1:
            grad_input1 = grad_input1.squeeze(-2)
        if d2 == 1:
            grad_input2 = grad_input2.squeeze(-1)

        return grad_input1, grad_input2

"""
    Reduction and Normalization Operations
"""

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Union[int, List[int], None], keepdims: bool = False):
        ctx.keepdims, ctx.input_shape = keepdims, input.shape
        if dim is None:
            dim = [_ for _ in range(input.dim())]
        elif isinstance(dim, int):
            dim = [dim]
        else:
            dim = list(dim)
        ctx.dim = dim
        ctx.dim.sort()
        res = input.sum(dim, keepdims=keepdims)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        dim, keepdims, input_shape = ctx.dim, ctx.keepdims, ctx.input_shape
        if keepdims == False:
            for d in dim:
                grad_output = grad_output.unsqueeze(d)
        grad_input = grad_output.broadcast_to(input_shape)
        return grad_input, None, None
        
    
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        ctx.dim, ctx.keepdims, ctx.input_shape = dim, keepdims, input.shape
        res = input.max(dim, keepdims=keepdims)
        ctx.save_for_backward(res[1])
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor, grad_indices: Tensor = None):
        dim, keepdims, input_shape = ctx.dim, ctx.keepdims, ctx.input_shape
        argmax, = ctx.get_saved_tensors()
        if keepdims == False:
            grad_output = grad_output.unsqueeze(dim)
            argmax = argmax.unsqueeze(dim)
        grad_input = zeros(input_shape)
        grad_input.scatter_(dim, argmax, grad_output)
        return grad_input, None, None

    
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int):
        ctx.dim = dim
        y = input.softmax(dim)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        y, = ctx.get_saved_tensors()
        return y * (grad_output - (grad_output * y).sum(ctx.dim, keepdims=True))

class Mean(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False):
        ctx.dim, ctx.keepdims, ctx.input_shape = dim, keepdims, input.shape
        res = input.mean(dim, keepdims=keepdims)
        return res

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        dim, keepdims, input_shape = ctx.dim, ctx.keepdims, ctx.input_shape
        if keepdims == False:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.broadcast_to(input_shape) / input_shape[dim], None, None


class Var(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, keepdims: bool = False, unbiased: bool = True):
        ctx.save_for_backward(input)
        ctx.den = input.shape[dim] - 1 if unbiased == True else input.shape[dim]
        ctx.miu = input.mean(dim, keepdims=True)
        ctx.keepdims, ctx.dim, ctx.input_shape = keepdims, dim, input.shape
        return input.var(dim, keepdims=keepdims, unbiased=unbiased)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        input, = ctx.get_saved_tensors()
        keepdims, dim, input_shape = ctx.keepdims, ctx.dim, ctx.input_shape
        if keepdims == False:
            grad_output = grad_output.unsqueeze(dim)
        grad_input = (2 / ctx.den) * grad_output.broadcast_to(input_shape) * (input - ctx.miu)
        return grad_input, None, None, None

"""
    Shape Manipulation
"""

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, perm: List[int]):
        ctx.perm = perm
        return input.permute(perm)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.permute(ctx.perm)
    
class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim0: int, dim1: int):
        ctx.dim0, ctx.dim1 = dim0, dim1
        return input.transpose(dim0, dim1)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.transpose(ctx.dim0, ctx.dim1)

class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.input_shape = input.shape
        res = input.reshape(shape)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.reshape(ctx.input_shape)
    
class View(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.input_shape = input.shape
        res = input.view(shape)
        return res
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.view(ctx.input_shape)
    
class Narrow(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int, start: int, length: int):
        ctx.dim, ctx.start, ctx.length, ctx.input_shape = dim, start, length, input.shape
        return input.narrow(dim, start, length)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        dim, start, length, input_shape = ctx.dim, ctx.start, ctx.length, ctx.input_shape
        grad_input = zeros(input_shape)
        sliced_grad_input = grad_input.narrow(dim, start, length)
        sliced_grad_input.copy_(grad_output)
        return grad_input

    
class Chunk(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, chunks: int, dim: int = 0):
        ctx.dim = dim
        res = input.chunk(chunks, dim)
        return res
        
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        base_list = [TensorBase(_) for _ in grad_outputs]
        grad_input = TensorBase.cat(base_list, dim = ctx.dim)
        return grad_input, None, None

    
class Split(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, split: Union[int, List[int]], dim: int = 0):
        ctx.split, ctx.dim = split, dim
        return input.split(split, dim)

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        base_list = [TensorBase(_) for _ in grad_outputs]
        grad_input = TensorBase.cat(base_list, dim=ctx.dim)
        return grad_input, None, None
    
class Stack(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        ctx.dim = dim
        return Tensor.stack(inputs, dim=dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        dim = ctx.dim
        grad_inputs = grad_output.chunk(grad_output.shape[dim], dim)
        squeezed_grad_inputs = [Tensor.squeeze(gi, dim) for gi in grad_inputs]
        return tuple(squeezed_grad_inputs) + (None, )
        
    
class Cat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor, dim: int = 0):
        ctx.dim = dim
        ctx.split = [input.shape[dim] for input in inputs]
        return Tensor.cat(inputs, dim)  
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_inputs = grad_output.split(ctx.split, ctx.dim)
        return tuple(grad_inputs) + (None, )
        
class Squeeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        ctx.dim = dim
        return input.squeeze(dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):    
        return grad_output.unsqueeze(ctx.dim), None
    
class Unsqueeze(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int = 0):
        ctx.dim = dim
        return input.unsqueeze(dim)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return grad_output.squeeze(ctx.dim), None
    
class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, shape: List[int]):
        ctx.input_shape = input.shape
        return input.broadcast_to(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor):
        return reduce_broadcast(grad_output, ctx.input_shape, grad_output.shape)
    
class Broadcast(Function):
    @staticmethod
    def forward(ctx: Context, *inputs: Tensor):
        ctx.inputs_shape = [input.shape for input in inputs]
        base_list = [TensorBase(_) for _ in inputs]
        return Tensor.broadcast(*base_list)
    
    @staticmethod
    def backward(ctx: Context, *grad_outputs: Tensor):
        grad_inputs = []
        for input_shape, grad_output in zip(ctx.inputs_shape, grad_outputs):
            grad_inputs.append(reduce_broadcast(grad_output, input_shape, grad_output.shape))
        return grad_inputs