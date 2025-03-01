from collections import defaultdict
import torch
from typing import Any
from .Data_Hook_Component import DataHookComponent
class Identical_Wrapper(torch.nn.Module,DataHookComponent):
    class BasicHook():
        def __call__(self,x):
            self.x=x
            return x

    ID_Map=defaultdict(lambda : 0)
    def __init__(self,datahook : list | Any = None,name: str = 'idt') -> None:
        '''
        An empty module does nothing, return the original input

        hooks: list[ callable(input)]
        '''
        super().__init__()
        DataHookComponent.__init__(self,datahook)

        self.name=name
        self.id=Identical_Wrapper.ID_Map[name]
        Identical_Wrapper.ID_Map[name]+=1

    def __call__(self, x,*args: Any, **kwds: Any) -> Any:
        for _datahook in self._datahooks:
            x=_datahook(x)
        return x
