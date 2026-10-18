"""Original Drive with a separately optimized, trainable diagonal vector."""
import torch
from torch import nn
from train_channel_ablation import Drive as OriginalDrive

class Drive(OriginalDrive):
    def __init__(self,state,T,device):
        super().__init__(state,T,device)
        self.diagonal=nn.Parameter(state['bilinear.weight'][0].diagonal().to(device).clone())
    def forward(self,xf):
        tr=self.log_tau_rise.exp()[:,None];td=self.log_tau_decay.exp()[:,None];w=self.omega[:,None]
        k=w*(1-torch.exp(-self.time/tr))*torch.exp(-self.time/td)
        kf=torch.fft.rfft(k,n=self.nfft,dim=-1)
        y=torch.fft.irfft(xf*kf[None],n=self.nfft,dim=-1)[...,:self.T].transpose(1,2).contiguous()
        full=self.weight*self.offdiag+torch.diag(self.diagonal)
        return ((y@full)*y).sum(-1)+y.sum(-1)+self.bias
    def original_state(self,base,reset_params):
        s=super().original_state(base,reset_params)
        s['bilinear.weight']=(self.weight*self.offdiag+torch.diag(self.diagonal)).detach().cpu()[None].clone()
        return s
