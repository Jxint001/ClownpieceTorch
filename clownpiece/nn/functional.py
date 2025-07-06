from typing import Iterable, Dict, Tuple
from clownpiece.nn.module import Module

class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.register_module(str(i), module)

    def forward(self, input):
        x = input
        for module in self._modules.values():
            x = module(x)
        return x

class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = None):
        if modules is None:
            modules = []
        super().__init__()
        for i, module in enumerate(modules):
            self.register_module(str(i), module)

    def __add__(self, other: Iterable[Module]):
        return ModuleList(list(self._modules.values()) + list(*other))

    def __setitem__(self, index: int, value: Module):
        self.register_module(str(index), value)

    def __getitem__(self, index: int) -> Module:
        return self._modules[str(index)]

    def __delitem__(self, index: int):
        del self._modules[str(index)]

    def __len__(self):
        return len(self._modules)
      
    def __iter__(self) -> Iterable[Module]:
        return iter(self._modules.values())

    def append(self, module: Module):
        self.register_module(str(len(self._modules)), module)
        return self
      
    def extend(self, other: Iterable[Module]):
        for module in other:
            self.append(module)
        return self

class ModuleDict(Module):
    def __init__(self, dict_: Dict[str, Module] = None):
        super().__init__()
        if dict_ is not None:
            for k, v in dict_.items():
                self.register_module(k, v)

    def __setitem__(self, name: str, value: Module):
        self.register_module(name, value)

    def __getitem__(self, name: str) -> Module:
        return self._modules[name]

    def __delitem__(self, name: str):
        del self._modules[name]

    def __len__(self):
        return len(self._modules)
      
    def __iter__(self) -> Iterable[str]:
        return iter(self._modules.keys())
    
    def keys(self) -> Iterable[str]:
        return self._modules.keys()
    
    def values(self) -> Iterable[Module]:
        return self._modules.values()
      
    def items(self) -> Iterable[Tuple[str, Module]]:
        return self._modules.items()

    def update(self, dict_: Dict[str, Module]):
        for k, v in dict_.items():
            self.register_module(k, v)
