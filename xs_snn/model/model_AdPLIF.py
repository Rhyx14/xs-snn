import torch
from xs_snn.components import ISNN

class firing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psp,tau,vth,):
        ctx.save_for_backward(psp,tau,vth)
        output,U=_forward(psp,tau,vth)
        
        return output,U

    @staticmethod
    def backward(ctx,dy,dU):
        psp, tau,vth= ctx.saved_tensors 
        return _back(psp,tau,vth,dy,dU)

@torch.jit.script
def _forward(psp,tau,vth):
    output = torch.gt(psp, vth).type_as(psp)
    U =tau * (1-output) * psp
    return output,U

@torch.jit.script
def _back(psp,tau,vth,dy,dU):

    _o = torch.gt(psp,vth).type_as(dy)
    foo = dU * (1-_o)
    dp= foo * tau
    dtau= foo * psp
    
    # 近似梯度 arctan
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (psp-vth)**2 * frac))

    d_sg = dy * hu
    return d_sg + dp, dtau, -d_sg*0.1

class AdPLIF(torch.nn.Module,ISNN):
    '''
    PLIF, with adaptive vth
    '''
    g=firing.apply
    id=0
    def __init__(self,delay=None,small_id=0,base_vth=0.75,vth_scale=0.5,**kwargs):
        super().__init__()

        # serial number
        self.id=AdPLIF.id+small_id
        AdPLIF.id+=1

        self.delay=delay
        self.base_vth=base_vth
        self.vth_scale=vth_scale

        if ('inplane' in kwargs) and ('channel_wise' in kwargs) and kwargs['channel_wise']:
            self.inplane=kwargs['inplane']
            self.channel_wise=kwargs['channel_wise']
            self.vth=torch.nn.Parameter(torch.zeros(1,self.inplane,1,1)) # assert shape: [b,c,h,w]
        else:
            self.channel_wise=False
            self.vth=torch.nn.Parameter(torch.Tensor([0]))

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
        o,self.U=AdPLIF.g(
            PSP,
            torch.sigmoid(self.tau),
            self.base_vth+ self.vth_scale*torch.sigmoid(self.vth))

        return o

    def extra_repr(self):
        s = (f'delay={self.delay},id={self.id},base_vth={self.base_vth},vth_scale={self.vth_scale},channel_wise={self.channel_wise}')
        return s
