# Core Module System

from typing import Dict, Iterable, Tuple, Union, Optional
from clownpiece import Tensor, zeros_like
from clownpiece.tensor import empty_like


class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)


class Module(object):
  def __init__(self):
    # It's a good practice to add a mechanism to enforce that:
    #   All subclasses of Module must call super().__init__ in their __init__
    #   (User often forgets to do so!)
    # For example:
    #   add a boolean variable _init_called, 
    #   and check at beginning of __setattr__ call.
    #
    # this mechanism is optional and does not account for score.

    pass

  def train(self, flag: bool = True):
    # set module and submodule to training = flag
    pass

  def eval(self):
    # set module and submodule to inferencing mode
    pass

  def __setattr__(self, name, value):
    pass

  def __getattr__(self, name):
    pass
    
  """
    Forward
  """
    
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  """
    Parameter
  """
  def register_parameter(self, name: str, param: Optional[Parameter]):
    # why does this function even exist? 
    # well, sometimes we want to register None as placeholder for disabled optioanl parameters. (e.g., bias in Linear)
    pass

  def parameters(self, recursive: bool=True) -> Iterable[Parameter]:
    # return a generator of all parameters in this module
    # yield immediate parameters first,
    # if recursive, then yield parameters from children.

    # HINT: use `yield` and `yield from` semantic
    pass

  def named_parameters(self, recursive: bool=True) -> Iterable[Tuple[str, Parameter]]:
    # similar to parameters, but return a name along with the parameter
    # the name is obtained by joining the recurisve attr name with '.'
    # for example
    """
      class A(Module):
        a: Parameter
        b: B

      class B(Moudle)
        c: Parameter
      
      Then, A.named_parameters() -> [
        ("a", ...),
        ("b.c", ...)
      ]
    """
    pass

  """
    Buffer
  """

  def register_buffer(self, name: str, buffer: Optional[Buffer]):
    pass

  def buffers(self, recursive: bool=True) -> Iterable[Buffer]:
    pass

  def named_buffers(self, recursive: bool=True) -> Iterable[Tuple[str, Buffer]]:
    pass

  """
    Modules
  """

  def register_modules(self, module: Optional["Module"]):
    pass

  def modules(self, recursive: bool=True) -> Iterable["Module"]:
    pass

  def named_modules(self, recursive: bool=True) -> Iterable["Module"]:
    pass
    
  """
    State Dict
  """

  def state_dict(self) -> Dict:
    pass

  def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True):
    pass
    
  """
    Printing
  """
  def __repr__(self) -> str:
    pass

  def extra_repr(self) -> str:
    return ""