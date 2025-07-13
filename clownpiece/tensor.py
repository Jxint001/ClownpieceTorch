from sympy import FunctionClass
from . import tensor_impl as cp
# from cp import TensorBaseImpl

from typing import TYPE_CHECKING, List, Optional, Union
import copy
import importlib

if TYPE_CHECKING:
  from clownpiece.autograd.autograd import Function

"""
  Autograd Utils (avoid circular import only)
"""
def is_grad_enabled():
  from clownpiece.autograd.no_grad import is_grad_enabled
  return is_grad_enabled()

def is_grad_enabled_with_params(*args):
  flatten_args = []
  for arg in args:
    if isinstance(arg, (list, tuple)):
      flatten_args.extend(arg)
    else:
      flatten_args.append(arg)
  
  return is_grad_enabled() and any(tensor.requires_grad for tensor in flatten_args if isinstance(tensor, Tensor))


def scalar_to_tensorbase(function):
    def wrapped_function(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, (int, float)):
                new_args.append(TensorBase(arg, requires_grad=False))
            else:
                new_args.append(arg)
        return function(*new_args, **kwargs)
    return wrapped_function

def tensor_base_op(impl_method_name):
  """
  Decorator for TensorBase operations that follow the pattern:
  return self.__class__(self._impl.method_name(*args))
  """
  def decorator(func):
    def wrapper(self, *args, **kwargs):
      impl_method = getattr(self._impl, impl_method_name)
      result_impl = impl_method(*args, **kwargs)
      return self.__class__(result_impl)
    return wrapper
  return decorator

def tensor_base_binary_op(impl_method_name, reverse=False):
  """
  Decorator for TensorBase binary operations that support both TensorBase and scalar types.
  @param impl_method_name: The name of the method on the _impl object
  @param reverse: If True, this is a reverse operation (e.g., __radd__)
  """
  def decorator(func):
    def wrapper(self, other):
      if isinstance(other, TensorBase):
        if reverse:
          result_impl = getattr(other._impl, impl_method_name)(self._impl)
        else:
          result_impl = getattr(self._impl, impl_method_name)(other._impl)
        return self.__class__(result_impl)
      elif isinstance(other, (int, float)):
        if reverse:
          # For reverse operations with scalars, use the reverse method if available
          reverse_method_name = impl_method_name.replace('__', '__r', 1) if '__' in impl_method_name else f'__r{impl_method_name}__'
          if hasattr(self._impl, reverse_method_name):
            result_impl = getattr(self._impl, reverse_method_name)(other)
          else:
            # Fallback to regular method (some operations are commutative)
            result_impl = getattr(self._impl, impl_method_name)(other)
        else:
          result_impl = getattr(self._impl, impl_method_name)(other)
        return self.__class__(result_impl)
      else:
        raise TypeError(f"Expected TensorBase, int or float, got {type(other).__name__}")
    return wrapper
  return decorator

def tensor_base_comparison_op(impl_method_name):
  """
  Decorator for TensorBase comparison operations.
  """
  def decorator(func):
    def wrapper(self, value):
      if isinstance(value, TensorBase):
        result_impl = getattr(self._impl, impl_method_name)(value._impl)
        return self.__class__(result_impl)
      elif isinstance(value, (int, float)):
        result_impl = getattr(self._impl, impl_method_name)(value)
        return self.__class__(result_impl)
      else:
        raise TypeError(f"Expected TensorBase, int or float, got {type(value).__name__}")
    return wrapper
  return decorator

"""
  Tensor Base Class
"""

class TensorBase:
  def __init__(self, array=None, requires_grad=None):
    if isinstance(array, TensorBase):
      self._impl = array._impl
    elif isinstance(array, cp.TensorBaseImpl):
      self._impl = array
    elif array is not None:
      self._impl = cp.TensorBaseImpl(array)
    else:
      self._impl = cp.TensorBaseImpl()
      
  @classmethod
  def randn(cls, shape, **kwargs):
    impl = cp.TensorBaseImpl.randn(shape)
    return cls(impl, **kwargs)

  @classmethod
  def randn_like(cls, tensor, **kwargs):
    impl = cp.TensorBaseImpl.randn_like(tensor._impl)
    return cls(impl, **kwargs)

  @classmethod
  def ones(cls, shape, **kwargs):
    print("[call] ones in TensorBase", flush=True)
    impl = cp.TensorBaseImpl.ones(shape)
    print("impl in ones-TensorBase get", flush=True)
    return cls(impl, **kwargs)
  
  @classmethod
  def ones_like(cls, tensor, **kwargs):
    print("[call] ones_like in TensorBase", flush=True)
    impl = cp.TensorBaseImpl.ones_like(tensor._impl)
    return cls(impl, **kwargs)
  
  @classmethod
  def zeros(cls, shape, **kwargs):
    impl = cp.TensorBaseImpl.zeros(shape)
    return cls(impl, **kwargs)
  
  @classmethod
  def zeros_like(cls, tensor, **kwargs):
    impl = cp.TensorBaseImpl.zeros_like(tensor._impl)
    return cls(impl, **kwargs)
  
  @classmethod
  def empty(cls, shape, **kwargs):
    impl = cp.TensorBaseImpl.empty(shape)
    return cls(impl, **kwargs)
  
  @classmethod
  def empty_like(cls, tensor, **kwargs):
    impl = cp.TensorBaseImpl.empty_like(tensor._impl)
    return cls(impl, **kwargs)
  
  @classmethod
  def stack(cls, inputs: List["TensorBase"], dim=0, **kwargs):
    print("Base Stack called with inputs:", inputs, "dim:", dim)
    if not isinstance(inputs, (list, tuple)):
      raise TypeError(f"Expected list or tuple, got {type(inputs).__name__}")
    if not all(isinstance(t, TensorBase) for t in inputs):
      raise TypeError("All inputs must be instances of TensorBase")
    impl = cp.TensorBaseImpl.stack([t._impl for t in inputs], dim)
    return cls(impl, **kwargs)
  
  @classmethod
  def cat(cls, inputs: List["TensorBase"], dim=0, **kwargs):
    if not isinstance(inputs, (list, tuple)):
      raise TypeError(f"Expected list or tuple, got {type(inputs).__name__}")
    if not all(isinstance(t, TensorBase) for t in inputs):
      print(type(inputs[0]).__name__)
      raise TypeError("All inputs must be instances of TensorBase")
    print("in TensorBase cat, dim is ", dim)
    print(inputs[0].shape)
    print(inputs[1].shape)
    impl = cp.TensorBaseImpl.cat([t._impl for t in inputs], dim)
    return cls(impl, **kwargs)
  
  @classmethod
  def broadcast(cls, *inputs: "TensorBase"):
    if not all(isinstance(t, TensorBase) for t in inputs):
      print(type(inputs[0]).__name__)
      raise TypeError("All inputs must be instances of TensorBase")
    impls = cp.TensorBaseImpl.broadcast([t._impl for t in inputs])
    return [cls(impl) for impl in impls]
  

  @property
  def shape(self):
    return self._impl.shape
      
  def reshape(self, new_shape):
    print("Base Reshape called with new shape:", new_shape)
    reshaped_impl = self._impl.reshape(new_shape)
    return self.__class__(reshaped_impl)

  def __getitem__(self, idx):
    print("in TensorBase getitem, idx is ", type(idx).__name__)
    if isinstance(idx, (tuple, list)):
      print(type(idx[0]).__name__)
    item = self._impl[idx]
    # print("in TensorBase, item is a ", type(item).__name__)
    if isinstance(item, cp.TensorBaseImpl):
      # print("self.__class__(item) is ", type(self.__class__(item)).__name__)
      return self.__class__(item)
    return item

  def __setitem__(self, idx, value):
    if isinstance(value, TensorBase):
      self._impl[idx] = value._impl
    else:
      self._impl[idx] = value
  
  def item(self):
    return self._impl.item()

  def __len__(self):
      return cp.numel(self._impl)
  def clone(self):
    return type(self)(self._impl.clone())
  
  def __repr__(self):
    return f"{self._impl}"

  def tolist(self):
    return self._impl.tolist()
  

  """
    Part1
  """
  def is_contiguous(self): return self._impl.is_contiguous()

  @tensor_base_op('contiguous')
  def contiguous(self): pass

  @tensor_base_op('transpose')
  def transpose(self, dim0=-1, dim1=-2): pass 
  
  def copy_(self, other):
    if isinstance(other, TensorBase):
      self._impl.copy_(other._impl)
    else:
      raise TypeError(f"Expected TensorBase, got {type(other).__name__}")
    
  """
    Part2
  """
  
  @tensor_base_op('__neg__')
  def __neg__(self): pass
  
  @tensor_base_op('sign')
  def sign(self): pass
  
  @tensor_base_op('abs')
  def abs(self): pass
  
  @tensor_base_op('sin')
  def sin(self): pass
  
  @tensor_base_op('cos')
  def cos(self): pass
  
  @tensor_base_op('tanh')
  def tanh(self): pass
  
  @tensor_base_op('log')
  def log(self): pass
  
  @tensor_base_op('exp')
  def exp(self): pass
  
  @tensor_base_op('sqrt')
  def sqrt(self): pass
  
  @tensor_base_op('clamp')
  def clamp(self, min_val, max_val): pass
  
  def pow(self, exponent):
    if isinstance(exponent, TensorBase):
      exponent = exponent._impl
    elif not isinstance(exponent, (int, float)):
      raise TypeError(f"Expected int, float or TensorBase, got {type(exponent).__name__}")
    powered_impl = self._impl.pow(exponent)
    return self.__class__(powered_impl)



  @tensor_base_binary_op('__add__')
  def __add__(self, other): pass
  
  @tensor_base_binary_op('__add__', reverse=True)
  def __radd__(self, other): pass
  
  @tensor_base_binary_op('__sub__')
  def __sub__(self, other): pass
  
  @tensor_base_binary_op('__sub__', reverse=True)
  def __rsub__(self, other): pass
  
  @tensor_base_binary_op('__mul__')
  def __mul__(self, other): pass
  
  @tensor_base_binary_op('__mul__', reverse=True)
  def __rmul__(self, other): pass
  
  @tensor_base_binary_op('__truediv__')
  def __truediv__(self, other): pass
  
  @tensor_base_binary_op('__truediv__', reverse=True)
  def __rtruediv__(self, other): pass
    
  @tensor_base_comparison_op('__gt__')
  def __gt__(self, value): pass
  
  @tensor_base_comparison_op('__lt__')
  def __lt__(self, value): pass
  
  @tensor_base_comparison_op('__ge__')
  def __ge__(self, value): pass
  
  @tensor_base_comparison_op('__le__')
  def __le__(self, value): pass
  
  @tensor_base_comparison_op('__eq__')
  def __eq__(self, value): pass
  
  @tensor_base_comparison_op('__ne__')
  def __ne__(self, value): pass
  
  """
    Part3
  """
  def sum(self, dim=None, keepdims=False):
    if isinstance(dim, int):
      summed_impl = self._impl.sum(dim, keepdims)
      return self.__class__(summed_impl)
    elif isinstance(dim, (list, tuple)):
      result = self.clone()
      dim = list(dim)
      dim.sort(reverse=True)
      for d in dim:
        result = result.sum(d, keepdims=keepdims)
      return result
    elif dim is None:
      dim = list(range(self.dim()))
      return self.sum(dim, keepdims=keepdims)
    else :
      raise TypeError(f"Expected int or list/tuple of single int, got {type(dim).__name__}")
      
  """
    Part4
  """
  def matmul(self, other):
    if not isinstance(other, TensorBase): raise TypeError(f"Expected TensorBase, got {type(other).__name__}")
    print("in TensorBase, left is ", self, " right is ", other)
    matmul_impl = self._impl.matmul(other._impl)
    temp = self.__class__(matmul_impl)
    print("result is ", temp)
    return temp
  def dim(self): return len(self.shape)
  
  def squeeze(self, dim=0):
    squeezed_impl = cp.squeeze(self._impl, dim)
    return self.__class__(squeezed_impl)
  def unsqueeze(self, dim=0):
    unsqueezed_impl = cp.unsqueeze(self._impl, dim)
    return self.__class__(unsqueezed_impl)
  
  """
    Part5
  """
  def max(self, dim=None, keepdims=False):
    print("max called with dim=", dim, "keepdims=", keepdims)
    if isinstance(dim, int):
      pass
    elif isinstance(dim, (list, tuple)):
      if len(dim) == 1:
        dim = dim[0]
      else:
        raise TypeError(f"Expected int or list/tuple of single int, got {type(dim).__name__}")
    elif dim is None:
      dim = -1
    else :
      raise TypeError(f"Expected int or list/tuple of single int, got {type(dim).__name__}")
      
    max_impl, indices_impl = self._impl.max(dim, keepdims)
    return self.__class__(max_impl), self.__class__(indices_impl)
  
  def softmax(self, dim=-1):
    if isinstance(dim, int):
      pass
    elif isinstance(dim, (list, tuple)):
      if len(dim) == 1:
        dim = dim[0]
      else:
        raise TypeError(f"Expected int or list/tuple of single int, got {type(dim).__name__}")
    elif dim is None:
      dim = -1
    else :
      raise TypeError(f"Expected int or list/tuple of single int, got {type(dim).__name__}")
      
    softmax_impl = self._impl.softmax(dim)
    return self.__class__(softmax_impl)
  
  def scatter_(self, dim, index, src):
    self = self.__class__(self._impl.scatter_(dim, index._impl, src._impl))
    
  def broadcast_to(self, shape):
    if not isinstance(shape, (list, tuple)):
      raise TypeError(f"Expected list or tuple, got {type(shape).__name__}")
    broadcasted_impl = cp.broadcast_to(self._impl, shape)
    return self.__class__(broadcasted_impl)


  """
    Part6
  """
  def permute(self, perm: List[int]):
    if not isinstance(perm, (list, tuple)):
      raise TypeError(f"Expected list or tuple, got {type(perm).__name__}")
    permuted_impl = cp.permute(self._impl, perm)
    return self.__class__(permuted_impl)
  
  def transpose(self, dim0=-1, dim1=-2):
    if not isinstance(dim0, int) or not isinstance(dim1, int):
      raise TypeError(f"Expected int for dim0 and dim1, got {type(dim0).__name__} and {type(dim1).__name__}")
    transposed_impl = self._impl.transpose(dim0, dim1)
    return self.__class__(transposed_impl)
  
  def view(self, shape: List[int]):
    if not isinstance(shape, (list, tuple)):
      raise TypeError(f"Expected list or tuple, got {type(shape).__name__}")
    viewed_impl = self._impl.view(shape)
    return self.__class__(viewed_impl)
  
  def narrow(self, dim: int, start: int, length: int):
    if not isinstance(dim, int) or not isinstance(start, int) or not isinstance(length, int):
      raise TypeError(f"Expected int for dim, start and length, got {type(dim).__name__}, {type(start).__name__} and {type(length).__name__}")
    narrowed_impl = cp.narrow(self._impl, dim, start, length)
    return self.__class__(narrowed_impl)
  
  def chunk(self, chunks: int, dim: int = 0):
    if not isinstance(chunks, int) or not isinstance(dim, int):
      raise TypeError(f"Expected int for chunks and dim, got {type(chunks).__name__} and {type(dim).__name__}")
    chunked_impls = cp.chunk(self._impl, chunks, dim)
    return [self.__class__(impl) for impl in chunked_impls]
  
  def split(self, split: Union[int, List[int]], dim: int = 0):
    print("split called with split=", split, "dim=", dim)
    if not isinstance(dim, int):
      raise TypeError(f"Expected int for dim, got {type(dim).__name__}")
    split_impls = cp.split(self._impl, split, dim)
    return [self.__class__(impl) for impl in split_impls]  

  def mean(self, dim: int, keepdims: bool = False):
    if not isinstance(dim, int):
      raise TypeError(f"Expected int for dim, got {type(dim).__name__}")
    mean_impl = self._impl.mean(dim, keepdims)
    return self.__class__(mean_impl)
  
  def var(self, dim: int, keepdims: bool = False, unbiased: bool = True):
    if not isinstance(dim, int):
      raise TypeError(f"Expected int for dim, got {type(dim).__name__}")
    var_impl = self._impl.var(dim, keepdims, unbiased)
    return self.__class__(var_impl)
  

"""
  Utils for Binding
"""

"""
  Wrap scalar args to singleton Tensor (requires_grad = False)
"""
def scalar_to_tensor(function):
  def wrapped_function(*args, **kwargs):
    new_args = []
    for arg in args:
      if isinstance(arg, (int, float)):
        new_args.append(Tensor(arg, requires_grad=False))
      else:
        new_args.append(arg)
    return function(*new_args, **kwargs)
  return wrapped_function
  
"""
  Wrap around a Tensor operator that traces gradient.
  @arg: op_name: the name of the TensorBase method to call.
  @arg: Function_name: the name of the Function class to use for autograd.
"""
def tensor_op(op_name, Function_name):
  def decorator(function):
    def wrapped_function(*args, **kwargs):
      if not is_grad_enabled_with_params(*args):
        op = getattr(TensorBase, op_name)
        raw_results = op(*args, **kwargs)
        
        def TensorBase2Tensor(x):
          return Tensor(x, requires_grad=False) if isinstance(x, TensorBase) else x
        
        if isinstance(raw_results, (list, tuple)):
          return tuple(TensorBase2Tensor(x) for x in raw_results)
        else:
          return TensorBase2Tensor(raw_results)
      
      module = importlib.import_module("clownpiece.autograd.function")
      FunctionClass = getattr(module, Function_name)
      # print("[in tensor_op decorator:] function is ", str(function))
      return function(*args, **kwargs, FunctionClass=FunctionClass)
    
    return wrapped_function
  return decorator  

"""
  Tensor Class 
"""

class Tensor(TensorBase):
  # if grad tracing should be enabled for this tensor
  requires_grad: bool

  # grad of the Tensor, possibly None
  grad: Optional["Tensor"] 
  
  # function instance that produces the Tensor, possibly None
  grad_fn: Optional["Function"] 
  # this tensor is the i-th output of grad_fn's forward. In the backward graph, it becomes input_nr of the edge.  
  output_nr: int

  def __init__(self, data: TensorBase, requires_grad: bool = None) -> "Tensor":
    super().__init__(data)
    self.requires_grad_(requires_grad)
    self.grad_fn = None
    self.grad = None
    self.output_nr = 0

  # set requires_grad based on is_grad_enabled.
  # ! you must implement this method: grader lib uses this function
  def requires_grad_(self, requires_grad: bool = None):
    if requires_grad is None:
      requires_grad = is_grad_enabled()
    
    self.requires_grad = requires_grad and is_grad_enabled()
  # """
  #   Other
  # """
  def backward(self, grad: Optional["Tensor"]=None):
    print("[call] backward in Tensor")
    from clownpiece.autograd.autograd import backward
    backward(self, grad)
      
  # def requires_grad_(self, requires_grad:bool=None):
  #   if requires_grad is None:
  #     requires_grad = is_grad_enabled()
  #   self.requires_grad = requires_grad 
  
  # def tolist(self):
  #   return super().tolist()

      
  # """
  #   Operator Binding for Autograd
  # """
  
  # """
  #   Part 1
  # """
  @tensor_op('clone', 'Clone')
  def clone(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)

  
  @tensor_op('contiguous', 'Contiguous')
  def contiguous(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('__getitem__', 'Subscriptor')
  def __getitem__(self, index, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, index)
    
  """
    Part 2
  """
  @tensor_op('__neg__', 'Neg')
  def __neg__(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('sign', 'Sign')
  def sign(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('abs', 'Abs')
  def abs(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self) 
  
  @tensor_op('sin', 'Sin')
  def sin(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('cos', 'Cos')
  def cos(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('tanh', 'Tanh')
  def tanh(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('clamp', 'Clamp')
  def clamp(self, min_val, max_val, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, min_val, max_val)


  @tensor_op('log', 'Log')
  def log(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('exp', 'Exp')
  def exp(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  @tensor_op('pow', 'Pow')
  def pow(self, exponent, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, exponent)

  @tensor_op('sqrt', 'Sqrt')
  def sqrt(self, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self)
  
  # """
  #   Part 3
  # """
  
  @tensor_op('__add__', 'Add')
  @scalar_to_tensor
  def __add__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
  @tensor_op('__radd__', 'Add')
  @scalar_to_tensor
  def __radd__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)

  @tensor_op('__sub__', 'Sub')
  @scalar_to_tensor
  def __sub__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
  @tensor_op('__rsub__', 'Sub')
  @scalar_to_tensor
  def __rsub__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)
  
  @tensor_op('__mul__', 'Mul')
  @scalar_to_tensor
  def __mul__(self, other, FunctionClass=None)->"Tensor":
    print("Tensor __mul__ called with other:", other)
    return FunctionClass().apply(self, other)
    
  @tensor_op('__rmul__', 'Mul')
  @scalar_to_tensor
  def __rmul__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)

  @tensor_op('__truediv__', 'Div')
  @scalar_to_tensor
  def __truediv__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, other)
  
  @tensor_op('__rtruediv__', 'Div')
  @scalar_to_tensor
  def __rtruediv__(self, other, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(other, self)
  
  # """
  # Part 4
  # """
  @tensor_op('matmul', 'MatMul')
  def matmul(self, other, FunctionClass=None)->"Tensor":
    print("[call] matmul Tensor")
    if not isinstance(other, Tensor):
      raise TypeError(f"Expected Tensor, got {type(other).__name__}")
    print(type(FunctionClass).__name__)
    return FunctionClass().apply(self, other)
  
  def __matmul__(self, other)->"Tensor":    
    return self.matmul(other)
  
  def __rmatmul__(self, other)->"Tensor":
    if not isinstance(other, Tensor):
      raise TypeError(f"Expected Tensor, got {type(other).__name__}")
    return other.matmul(self)
  
  # """
  # Part 5
  # """
  @tensor_op('sum', 'Sum')
  def sum(self, dim=None, keepdims=False, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, dim, keepdims)
  
  @tensor_op('max', 'Max')
  def max(self, dim=-1, keepdims=False, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, dim, keepdims)
  
  @tensor_op('softmax', 'Softmax')
  def softmax(self, dim=-1, FunctionClass=None)->"Tensor":
    return FunctionClass().apply(self, dim)
  
  # """
  # Part 6
  # """

  @tensor_op('permute', 'Permute')
  def permute(self, perm: List[int], FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, perm)
  
  @tensor_op('transpose', 'Transpose')
  def transpose(self, dim0: int, dim1: int, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, dim0, dim1)
  
  @tensor_op('reshape', 'Reshape')
  def reshape(self, shape: List[int], FunctionClass=None) -> "Tensor":
    if not isinstance(shape, list):
      if isinstance(shape, int):
        shape = [shape]
      else:
        raise TypeError(f"Expected list or int for shape, got {type(shape).__name__}")
    return FunctionClass().apply(self, shape)
  
  @tensor_op('view', 'View')
  def view(self, shape: List[int], FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, shape)
  
  @tensor_op('narrow', 'Narrow')
  def narrow(self, dim: int, start: int, length: int, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, dim, start, length)
  
  @tensor_op('chunk', 'Chunk')
  def chunk(self, chunks: int, dim: int = 0, FunctionClass=None) -> List["Tensor"]:
    return FunctionClass().apply(self, chunks, dim)
  
  @tensor_op('split', 'Split')
  def split(self, split: Union[int, List[int]], dim: int = 0, FunctionClass=None) -> List["Tensor"]:
    return FunctionClass().apply(self, split, dim)
  
  @tensor_op('stack', 'Stack')
  def stack(inputs: List["Tensor"], dim: int = 0, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(*inputs, dim=dim)
  
  @tensor_op('cat', 'Cat')
  def cat(inputs: List["Tensor"], dim: int = 0, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(*inputs, dim=dim)
  
  @tensor_op('squeeze', 'Squeeze')
  def squeeze(self, dim: int = 0, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, dim)
  
  @tensor_op('unsqueeze', 'Unsqueeze')
  def unsqueeze(self, dim: int = 0, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, dim)
  
  @tensor_op('broadcast_to', 'BroadcastTo')
  def broadcast_to(self, shape: List[int], FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, shape)
  
  @tensor_op('broadcast', 'Broadcast')
  def broadcast(inputs: List["Tensor"], FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(*inputs)

  @tensor_op('mean', 'Mean')
  def mean(self, dim: int, keepdims: bool = False, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, dim, keepdims)
  
  @tensor_op('var', 'Var')
  def var(self, dim: int, keepdims: bool = False, unbiased: bool = True, FunctionClass=None) -> "Tensor":
    return FunctionClass().apply(self, dim, keepdims, unbiased)
      
  """
  STR
  """
  
  def __repr__(self):    
    grad_fn_name = self.grad_fn.__class__.__name__ if self.grad_fn else None
    return f"({super().__repr__()}, requires_grad={self.requires_grad}, grad_fn={grad_fn_name})"
  
def stack(inputs: List[Tensor], dim: int = 0) -> Tensor:
    return Tensor.stack(inputs, dim)

def cat(inputs: List[Tensor], dim: int = 0) -> Tensor:
    return Tensor.cat(inputs, dim)

def broadcast(inputs: List[Tensor]) -> Tensor:
    return Tensor.broadcast(inputs)

# """
#   Constructors
# """
  
def empty(shape, requires_grad: bool = False) -> Tensor:
  return Tensor.empty(shape, requires_grad=requires_grad)

def empty_like(tensor: Tensor, requires_grad: bool = False) -> Tensor:
  return Tensor.empty(tensor.shape, requires_grad=requires_grad)

def zeros(shape, requires_grad=False):
  return Tensor.zeros(shape, requires_grad=requires_grad)

def zeros_like(tensor: Tensor, requires_grad=False) -> Tensor:
  return Tensor.zeros(tensor.shape, requires_grad=requires_grad)

def ones(shape, requires_grad=False):
  # print("[call] ones that create a Tensor", flush=True)
  return Tensor.ones(shape, requires_grad=requires_grad)

def ones_like(tensor: Tensor, requires_grad=False) -> Tensor:
  return Tensor.ones(tensor.shape, requires_grad=requires_grad)

def randn(shape, requires_grad=False):
  return Tensor.randn(shape, requires_grad=requires_grad)

def randn_like(tensor, requires_grad=False):
  return Tensor.randn(tensor.shape, requires_grad=requires_grad)