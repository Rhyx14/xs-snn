import torch
from spikingjelly.activation_based.base import MemoryModule

class firing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psp,tau,vth):
        ctx.save_for_backward(psp,tau)
        ctx.vth=vth
        output,U=_forward(psp,tau,vth)
        
        return output,U

    @staticmethod
    def backward(ctx, do,dU):
        psp, tau= ctx.saved_tensors 
        return _back(psp, tau,do,dU,ctx.vth)

@torch.jit.script
def _forward(psp,tau,vth:float):
    _s = torch.gt(psp, vth).type_as(psp)
    U = tau * (1-_s) * psp
    return _s * psp, U

@torch.jit.script
def _back(psp,tau,do,dU,vth:float):

    # dp1
    _s = torch.gt(psp,vth).type_as(do)
    foo = dU * (1-_s)
    dp= foo * tau
    dtau= foo * psp
    
    # 近似梯度 arctan
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (psp-vth)**2 * frac))

    return do*(hu*psp +_s) + dp, dtau,None

class FP32(MemoryModule):
    '''
    FP32, clamp the potential
    '''
    g=firing.apply
    id=0
    vth=1
    def __init__(self,delay=None,small_id=0,**kwargs):
        super(FP32,self).__init__()
        
        # serial number
        self.id=FP32.id+small_id
        FP32.id+=1

        self.delay=delay

        self.tau=torch.nn.Parameter(torch.Tensor([0]))
        self.vth=FP32.vth

        self.reset()
        self.state_hooks=[]

    def reset(self):
        self.U=None
    
    def single_step_forward(self,input):
        # assert input shape: [b,c,h,w]
        rslt=self._update(input)

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks: hooks(self.id,input,rslt)
        return rslt

    def _update(self,input):
        if(self.U is None):
            PSP=input
        else:
            PSP=self.U + input
            
        o,self.U=FP32.g(PSP,
                         torch.sigmoid(self.tau),
                         self.vth)

        return o

    def extra_repr(self):
        s = (f'delay={self.delay},id={self.id},vth={FP32.vth}')
        return s
