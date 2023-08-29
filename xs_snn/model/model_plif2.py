import torch
from ..components.interface_ISNN import ISNN
from ..utils.override import Override

class firing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psp,tau):
        ctx.save_for_backward(psp,tau)
        
        output,U=_forward(psp,tau,P_LIF2.vth)
        
        return output,U

    @staticmethod
    def backward(ctx, dy,dU):
        psp, tau= ctx.saved_tensors 
        return _back(psp, tau,dy,dU,P_LIF2.vth)

@torch.jit.script
def _forward(psp,tau,vth:float):
    output = torch.gt(psp, vth).type_as(psp)
    U =tau * (1-output) * psp
    return output,U

@torch.jit.script
def _back(psp,tau,dy,dU,vth:float):

    _o = torch.gt(psp,vth).type_as(dy)
    # dp = dU * (1-_o) * tau
    # dtau= dU * (1-_o)* psp
    foo = dU * (1-_o)
    dp= foo * tau
    dtau= foo * psp
    
    # 近似梯度 arctan
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (psp-vth)**2 * frac))

    return dy*hu + dp, dtau

class P_LIF2(torch.nn.Module,ISNN):
    '''
    P_LIF
    '''
    duplicate=1
    g=firing.apply
    id=0
    vth=1
    def __init__(self,delay=None,small_id=0,**kwargs):
        super(P_LIF2,self).__init__()
        
        # serial number
        self.id=P_LIF2.id+small_id
        P_LIF2.id+=1

        self.delay=delay

        self.tau=torch.nn.Parameter(torch.Tensor([0]))

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
            PSP=input
        else:
            # t=torch.zeros(*input.shape)+self.U
            PSP=self.U + input
        o,self.U=P_LIF2.g(PSP,torch.sigmoid(self.tau))

        return o

    def extra_repr(self):
        s = (f'delay={self.delay},id={self.id},vth={P_LIF2.vth}')
        return s
