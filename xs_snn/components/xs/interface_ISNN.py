import torch

class ISNN():
    '''
    接口:该对象为SNN神经元模型
    '''
    def __init__(self) -> None:
        pass
    
    def aggregation(self):
        '''
        该神经元输出是否是聚合模式输出
        '''
        return False
    
    def reset(self) ->None:
        '''
        重置神经元和突触状态
        '''
        return
    
    def add_states(self,dict:dict):
        '''
        添加托管的神经元突触变量
        '''
        # self.__neuromorphic_states=dict
        self.__dict__['_neuromorphic_states']=dict
        self.__dict__.update(dict)
        

    def snn_to_cuda(self,device=None):
        '''
        将托管的神经元和突触状态变量转移到cuda设备上,(非auto grad模式)

        变量名应另存在 self._neuromorphic_states:dict 中
        '''
        if hasattr(self,'_neuromorphic_states'):
            for key in [ k for k,v in self.__dict__.items() if k in self._neuromorphic_states]:
                self.__dict__[key]=self.__dict__[key].cuda(device)
                pass
        
        if isinstance(self,torch.nn.Module):
            for module in self.children():
                if isinstance(module,ISNN):
                    module.snn_to_cuda(device)
        return

class SNN_To_CUDA():
    def __init__(self,device=None):
        self.device=device

    def __call__(self,module):
        if isinstance(module, ISNN):
            if hasattr(module,'_neuromorphic_states'):
                for key in [ k for k in module.__dict__.keys() if k in module._neuromorphic_states]:
                    module.__dict__[key]=module.__dict__[key].cuda(self.device)
                    pass

class SNN_Reset:
    def __init__(self,extra_models:tuple=()) -> None:
        """_summary_
            Reset the neuron states
        Args:
            extra_models (tuple): reset instances beyond ISNN. Such an instance should have reset() method. 
        """
        self._models=(ISNN,*extra_models)
        pass
    def __call__(self,module):
        if isinstance(module, self._models):
            module.reset()
        return
