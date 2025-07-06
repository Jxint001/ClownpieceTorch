# Sequential, ModuleList, ModuleDict

from typing import Iterable, Dict, Tuple
from clownpiece.nn.module import Module
class Sequential(Module):
  
  def __init__(self, *modules: Module):
    pass

  def forward(input):
    pass


class ModuleList(Module):
  
  def __init__(self, modules: Iterable[Module] = None):
    # hint: try to avoid using [] (which is mutable) as default argument. it may lead to unexpected behavor.
    # also be careful to passing dictionary or list around in function, which may be modified inside the function.
    pass

  def __add__(self, other: Iterable[Module]):
    pass

  def __setitem__(self, index: int, value: Module):
    pass

  def __getitem__(self, index: int) -> Module:
    pass

  def __delitem__(self, index: int):
    pass

  def __len__(self):
    pass

  def __iter__(self) -> Iterable[Module]:
    pass

  def append(self, module: Module):
    pass

  def extend(self, other: Iterable[Module]):
    pass

class ModuleDict(Module):
  
  def __init__(self, dict_: Dict[str, Module]):
    pass

  def __setitem__(self, name: str, value: Module):
    pass

  def __getitem__(self, name: str) -> Module:
    pass

  def __delitem__(self, name: str):
    pass

  def __len__(self):
    pass

  def __iter__(self) -> Iterable[str]:
    pass
  
  def keys(self) -> Iterable[str]:
    pass

  def values(self) -> Iterable[Module]:
    pass

  def items(self) -> Iterable[Tuple[str, Module]]:
    pass

  def update(self, dict_: Dict[str, Module]):
    pass