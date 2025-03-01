from typing import Any
class DataHookComponent():
    def __init__(self,datahook: list | Any) -> None:
        self._datahooks=[]
        if datahook is not None:
            if isinstance(datahook,list):
                self._datahooks.extend(datahook)
            else:
                self._datahooks.append(datahook)
        pass

    def add_hook(self,hook):
        self._datahooks.append(hook)

    def remove_hooks(self):
        self._datahooks.clear()
