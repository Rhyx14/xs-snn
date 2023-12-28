
import torch
from ..components.interface_ISNN import ISNN
from ..utils.override import Override

class G_arctan(torch.autograd.Function):
    """ 
    近似梯度 arctan
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0) 
        return output.type_as(input)

    @staticmethod
    def backward(ctx, dy):
        input, = ctx.saved_tensors 
        return _back(input, dy)
        # frac=(3.1416 /2 * alpha)**2
        # hu = alpha / (2 * (1 + input**2 * frac))
        # return dy * hu

@torch.jit.script
def _back(input,dy):
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + input**2 * frac))
    return dy*hu

class LIF(torch.nn.Module,ISNN):
    '''
    LIF

    vth step is according to 
    https://arxiv.org/pdf/2210.06386.pdf

    tau 
    https://github.com/langfengQ/MLF-DSResNet/blob/main/parallel_nets/spike_layer_for_cifar10.py
    '''
    duplicate=1
    g=G_arctan.apply
    id=0

    def __init__(self,delay=None,vth_base=0.6,tau=0.25,small_id=0,**kwargs):
        super(LIF,self).__init__()
        
        # serial number
        self.id=LIF.id+small_id
        LIF.id+=1

        self.vth_base=vth_base  
        self.delay=delay

        self.tau=tau

        self.Vth=vth_base
        self.reset()
        self.state_hooks=[]

    def reset(self):
        self.U=None
        
    
    def forward(self,input):

        rslt=self._update(input)

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,input,rslt)
        return rslt

    def _update(self,input):
        if(self.U is None):
            self.U=input
        else:
            # t=torch.zeros(*input.shape)+self.U
            self.U=self.U+ input
        o=LIF.g(self.U-self.Vth)

        self.U =self.tau * (1-o.detach()) * self.U

        return o

    def extra_repr(self):
        s = (f'delay={self.delay},id={self.id},vth_base={self.vth_base}')
        return s
