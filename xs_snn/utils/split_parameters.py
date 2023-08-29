# from https://zhuanlan.zhihu.com/p/267535838?ivk_sa=1024320u
import torch
from torchvision import models
from ..components.interface_ISNN import ISNN
def split_parameters_for_SNN(module):
    '''
    get params of SNN

    @return : decay, // including CONV, Linear,
            : no_decay, // including BN, bias of each layer
    '''

    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,ISNN)):
            params_no_decay.extend([*m.parameters()])

        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

def split_bn_and_snn_parameters(module):
    '''
    get params of bn and spiking neuron
    '''
    _params = []
    for m in module.modules():
        if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,ISNN)):
            _params.extend([*m.parameters()])
    return _params

def print_parameters_info(parameters):
    for k, param in enumerate(parameters):
        print('[{}/{}] {}'.format(k+1, len(parameters), param.shape))
        
if __name__ == '__main__':
    model = models.resnet18(pretrained=False)
    params_decay, params_no_decay = split_parameters_for_SNN(model)
    print_parameters_info(params_decay)
    print_parameters_info(params_no_decay)