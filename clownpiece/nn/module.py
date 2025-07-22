# Core Module System

from typing import Dict, Iterable, Tuple, Union, Optional
from clownpiece import Tensor, zeros_like
from clownpiece.tensor import empty_like


class Parameter(Tensor):
  _next_id = 0

  def __init__(self, data):
    super().__init__(data, requires_grad=True)
    self.param_id = Parameter._next_id
    Parameter._next_id += 1

  def __hash__(self):
    return hash(self.param_id)
  
  def __eq__(self, other):
    if not isinstance(other, Parameter):
      return False
    return self.param_id == other.param_id
    

class Buffer(Tensor):
  def __init__(self, data):
    super().__init__(data, requires_grad=False)

def _addindent(s_: str, numSpaces: int):
  s = s_.split("\n")
  if len(s) == 1:
    return s_
  first = s.pop(0)
  s = [(numSpaces * " ") + line for line in s]
  s = "\n".join(s)
  s = first + "\n" + s
  return s

class Module(object):

  training: bool

  _parameters: Dict[str, Parameter]
  _buffers: Dict[str, Buffer]
  _modules: Dict[str, "Module"]

  def __init__(self):
    # print("[call] init in Module")
    super().__setattr__("call_super_init_", True)
    super().__setattr__("training", True)
    super().__setattr__("_parameters", {})
    super().__setattr__("_buffers", {})
    super().__setattr__("_modules", {})


  def train(self, flag: bool = True):
    # set module and submodule to training = flag
    if not isinstance(flag, bool):
      raise ValueError("training flag is expected to be boolean")
    
    # self.training = flag
    super().__setattr__("training", flag)

    for module in self.modules(True):
      if module is not self:
        module.train(flag)

    # Actually I do not know what to return
    # return self

  def eval(self):
    # set module and submodule to inferencing mode
    self.train(False)

  def __setattr__(self, name, value):
    if self.call_super_init_ == False:
      raise RuntimeError("super().__init__ should be called")
    
    if not isinstance(value, (Parameter, Buffer, Module)):
      super().__setattr__(name, value)

    elif isinstance(value, Parameter):
      self.register_parameter(name, value)

    elif isinstance(value, Buffer):
      self.register_buffer(name, value)

    else:  # value is Module
      self.register_modules(name, value)

  def __getattr__(self, name):
    if name in self._parameters:
      return self._parameters[name]
    
    elif name in self._buffers:
      return self._buffers[name]
    
    elif name in self._modules:
      return self._modules[name]

    else:
      raise RuntimeError("Cannot get attribute, not in _parameters, _buffers and _modules") 
    
  """
    Forward
  """
    
  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward method not implemented")
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
  
  """"
    Helper function for yielding names + members of modules
  """
  def _named_members(self, get_members_fn, recursive: bool=True):
    modules = (
      self.named_modules(recursive=recursive)
    )

    for module_prefix, module in modules:
      members = get_members_fn(module)
      for k, v in members:
        name = module_prefix + ("." if module_prefix else "") + k
        yield name, v

  """
    Parameter
  """
  def register_parameter(self, name: str, param: Optional[Parameter]):
    if not isinstance(param, Parameter) and param is not None:
      raise TypeError("param should be Parameter")

    self._parameters[name] = param

  def parameters(self, recursive: bool=True) -> Iterable[Parameter]:
    # return a generator of all parameters in this module
    # yield immediate parameters first,
    # if recursive, then yield parameters from children.
    for _, param in self.named_parameters(recursive=recursive):
      yield param

  def named_parameters(self, recursive: bool=True) -> Iterable[Tuple[str, Parameter]]:
    gen = self._named_members(
      lambda module: module._parameters.items(),
      recursive=recursive
    )

    yield from gen

  """
    Buffer
  """

  def register_buffer(self, name: str, buffer: Optional[Buffer]):
    if not isinstance(buffer, Buffer) and buffer is not None:
      raise TypeError("buffer should be Buffer")
    
    self._buffers[name] = buffer

  def buffers(self, recursive: bool=True) -> Iterable[Buffer]:
    for _, buffer in self.named_buffers(recursive=recursive):
      yield buffer

  def named_buffers(self, recursive: bool=True) -> Iterable[Tuple[str, Buffer]]:
    gen = self._named_members(
      lambda module: module._buffers.items(),
      recursive=recursive
    )

    yield from gen

  """
    Modules
  """

  def register_modules(self, name: str, module: Optional["Module"]):
    if not isinstance(module, Module) and module is not None:
      raise TypeError("module should be Module")
    
    self._modules[name] = module

  def modules(self, recursive: bool=True) -> Iterable["Module"]:
    for _, module in self.named_modules(recursive=recursive):
      yield module

  def _named_modules(self, prefix: str = "", recursive: bool = True) -> Iterable["Module"]:
    yield prefix, self

    for name, module in self._modules.items():
      if module is None:
        continue
      submodule_prefix = prefix + ("." if prefix else "") + name
      yield from module._named_modules(
        submodule_prefix,
        recursive=recursive
      )

  def named_modules(self, recursive: bool=True) -> Iterable["Module"]:
    return self._named_modules(recursive=recursive)
    
  """
    State Dict
  """

  def state_dict(self) -> Dict:
    ret = {}
    for name, param in self.named_parameters():
      ret[name] = param
    
    for name, buffer in self.named_buffers():
      ret[name] = buffer

    return ret

  def load_state_dict(self, state: Dict[str, Tensor], strict: bool = True):
    model_params = self.state_dict()
    
    # initialize missing_keys
    missing_keys = set(model_params.keys())
    unused_keys = set(state.keys())
    
    for name, tensor in state.items():
      # 我也不知道这个是干啥的  好像是跳过禁用权重
      if tensor is None:
        if name in model_params:
          unused_keys.discard(name)
          missing_keys.discard(name)
        continue
          
      # if name is not in this model, skip
      if name not in model_params:
        continue
          
      # remove used key
      unused_keys.discard(name)
      missing_keys.discard(name)
      
      # check shape alignment
      model_tensor = model_params[name]
      if model_tensor.shape != tensor.shape:
        if strict:
          raise RuntimeError(
              f"Size mismatch for '{name}': "
              f"expected {model_tensor.shape}, got {tensor.shape}"
          )
        continue
      
      # deep copy
      model_tensor.copy_(tensor)
    
    if strict:
      if missing_keys:
        missing_str = ", ".join(f"'{k}'" for k in sorted(missing_keys))
        raise RuntimeError(f"Missing keys in state_dict: {missing_str}")
      
      if unused_keys:
        unused_str = ", ".join(f"'{k}'" for k in sorted(unused_keys))
        raise RuntimeError(f"Unexpected keys in state_dict: {unused_str}")
    
  """
    Printing
  """
  def __repr__(self) -> str:
    # treat hte extra repr like the submodule, one item per line
    extra_lines = []
    extra_repr = self.extra_repr()
    if extra_repr:
      extra_lines = extra_repr.split("\n")
    child_lines = []
    for key, module in self._modules.items():
      mod_str = repr(module)
      mod_str = _addindent(mod_str, 2)
      child_lines.append("(" + key + "): " + mod_str)
    lines = extra_lines + child_lines

    main_str = self.__class__.__name__ + "("
    if lines:
      if len(extra_lines) == 1 and not child_lines:
        main_str += extra_lines[0]
      else:
        main_str += "\n " + "\n ".join(lines) + "\n"

    main_str += ")"
    return main_str
  
  def extra_repr(self) -> str:
    return ""