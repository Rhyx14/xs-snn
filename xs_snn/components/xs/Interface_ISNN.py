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
    
    def update_states(self,dict:dict):
        '''
        添加托管的到isnn神经元的突触变量
        '''
        if not hasattr(self,'_neuromorphic_states'):
            self._neuromorphic_states={}
        self._neuromorphic_states.update(dict)

    class SNN_To_CUDA():
        def __init__(self,device='cuda:0'):
            self.device=device
    
        def __call__(self,module):
            if isinstance(module, ISNN):
                for key in module._neuromorphic_states.keys():
                    if isinstance(module._neuromorphic_states[key],torch.Tensor):
                       module._neuromorphic_states[key]=module._neuromorphic_states[key].to(self.device)
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
