# Sequential, ModuleList, ModuleDict

from typing import Iterable, Dict, Tuple
from clownpiece.nn.module import Module
class Sequential(Module):
  
  def __init__(self, *modules: Module):
    super().__init__()
    for idx, module in enumerate(modules):
      self.register_modules(str(idx), module)
    self.mod_seq = list(modules)


  def forward(self, input):
    x = input
    for module in self.mod_seq:
      if module is not self:
        # print(module.__class__.__name__)
        x = module.forward(x)
    return x


class ModuleList(Module):
  
  def __init__(self, modules: Iterable[Module] = None):
    # hint: try to avoid using [] (which is mutable) as default argument. it may lead to unexpected behavor.
    # also be careful to passing dictionary or list around in function, which may be modified inside the function.
    super().__init__()
    self.mod_list: tuple
    if modules is None:
      self.mod_list = tuple()
    else:
      mlist = []
      for mod in modules:
        mlist.append(mod)
      self.mod_list = tuple(mlist)

    for idx, mod in enumerate(self.mod_list):
      self.register_modules(str(idx), mod)

  def __add__(self, other: Iterable[Module]):
    self.mod_list = tuple(list(self.mod_list) + list(other))
    org = self.mod_list.__len__()
    for idx, mod in enumerate(other):
      self.register_modules(str(idx + org), mod)

  def __setitem__(self, index: int, value: Module):
    self.mod_list = tuple(list(self.mod_list).__setitem__(index, value))
    self._modules.__setitem__(str(index), value)

  def __getitem__(self, index: int) -> Module:
    return self.mod_list.__getitem__(index)

  def __delitem__(self, index: int):
    self.mod_list = tuple(list(self.mod_list).__delitem__(index))
    self._modules.__delitem__(str(index))

  def __len__(self):
    return self.mod_list.__len__()

  def __iter__(self) -> Iterable[Module]:
    return iter(self.mod_list[i] for i in range(self.mod_list.__len__()))

  def append(self, module: Module):
    self.register_modules(str(self.__len__()), module)
    self.mod_list = tuple(list(self.mod_list).append(module))

  def extend(self, other: Iterable[Module]):
    org = self.mod_list.__len__()
    for idx, mod in enumerate(other):
      self.register_modules(str(idx + org), mod)
    self.mod_list = tuple(list(self.mod_list) + list(other))

class ModuleDict(Module):
  
  def __init__(self, dict_: Dict[str, Module]):
    super().__init__()
    print(dict_)
    self.mod_dict = {}
    for name, mod in dict_.items():
      self.register_modules(name, mod)
      self.mod_dict[name] = mod

  def __setitem__(self, name: str, value: Module):
    self.mod_dict[name] = value
    self._modules[name] = value

  def __getitem__(self, name: str) -> Module:
    return self.mod_dict[name]

  def __delitem__(self, name: str):
    self.mod_dict.__delitem__(name)
    self._modules.__delitem__(name)

  def __len__(self):
    return self.mod_dict.__len__()

  def __iter__(self) -> Iterable[str]:
    return iter(self.mod_dict.keys())
  
  def keys(self) -> Iterable[str]:
    return iter(self.mod_dict.keys())

  def values(self) -> Iterable[Module]:
    return iter(self.mod_dict.values())

  def items(self) -> Iterable[Tuple[str, Module]]:
    return iter(tuple(self.mod_dict.items()))

  def update(self, dict_: Dict[str, Module]):
    for name, mod in dict_.items():
      self.register_modules(name, mod)
      self.mod_dict[name] = mod