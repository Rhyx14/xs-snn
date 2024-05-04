import torch
import torch.nn as nn
from ...components.xs.net_aggregated import Aggregated_Spiking_Layer as ASL

from ...components.xs.net_RateBN import RateBatchNorm
from ...components.xs.net_BN import BatchNorm
from ..fuse_bn import fuse_conv2d
class Fuse_BN():
    '''
    Fuse BN to Conv2d, only support RateBatchNorm
    '''
    def __call__(self,module):
        
        if isinstance(module,(ASL,)):
            if isinstance(module._layer,(nn.Conv2d)):
                _conv_layer=module._layer
                if isinstance(module._norm,(RateBatchNorm,BatchNorm)):
                    _b2d_norm=module._norm._norm
                    with torch.no_grad():
                        _new_conv = fuse_conv2d(_conv_layer,_b2d_norm)
                    module._layer=_new_conv
                    module._norm=None

def validate(net, input_, cuda=True):
    net.eval()
    if cuda:
        input_ = input_.cuda()
        net.cuda()
    # import time
    # s = time.time()
    a = net(input_)
    if cuda:
        torch.cuda.synchronize()
    # print(time.time() - s)
    fuse_module(net)
    # print(mbnet)
    # s = time.time()
    b = net(input_)
    if cuda:
        torch.cuda.synchronize()
    # print(time.time() - s)
    return (a - b).abs().max().item()


if __name__ == '__main__':
    import torchvision
    mbnet = torchvision.models.mobilenet_v2(True)
    mbnet.eval()
    print(validate(mbnet, torch.randn(32, 3, 224, 224), True))