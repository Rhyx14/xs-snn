import torch
from ..components.NeuronBase import NeuronBase
class PLIF_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psp,tau,vth):
        ctx.save_for_backward(psp,tau)
        ctx.vth=vth
        output,U=_forward(psp,tau,vth)
        return output,U

    @staticmethod
    def backward(ctx,dy,dU):
        psp, tau= ctx.saved_tensors 
        return _back(psp,tau,ctx.vth,dy,dU)

@torch.jit.script
def _forward(spsp,tau,vth:float):
    output = torch.gt(spsp, vth).type_as(spsp)
    U =tau * (1-output) * spsp
    return output,U

@torch.jit.script
def _back(spsp,tau,vth:float,dy,dU):

    _o = torch.gt(spsp,vth).type_as(dy)
    foo = dU * (1-_o)
    dp= foo * tau
    dtau= foo * spsp
    
    # 近似梯度 arctan
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (spsp-vth)**2 * frac))

    d_sg = dy * hu
    return d_sg + dp, dtau,None

class PLIF(NeuronBase):
    '''
    PLIF, with adaptive vth
    '''
    id=0
    def __init__(self,small_id=0,base_vth=1.,**kwargs):
        super().__init__()

        # serial number
        self.id=PLIF.id+small_id
        PLIF.id+=1

        self.base_vth=base_vth
        self.vth=base_vth

        self.tau=torch.nn.Parameter(torch.Tensor([0]))

        self.reset()

    def reset(self):
        self.U=None
        
    def single_step_forward(self,input):

        rslt=self._update(input)

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks: hooks(self.id,input,rslt)
        return rslt

    def _update(self,input):
        if(self.U is None):
            sPSP=input
        else:
            sPSP=self.U + input
            
        o,self.U=PLIF_dynamic.apply(
            sPSP,
            torch.sigmoid(self.tau),
            self.vth,
        )

        return o

    def extra_repr(self):
        s = f'{super().extra_repr()},id={self.id},vth={self.vth}'
        return s
