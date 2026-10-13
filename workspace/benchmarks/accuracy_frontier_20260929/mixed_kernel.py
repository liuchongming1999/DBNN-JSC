"""Two double-exponential components per site, unchanged quadratic integration."""
import torch
from torch import nn
from train_channel_ablation import Drive

class MixedDrive(Drive):
    def __init__(self,state,T,device):
        super().__init__(state,T,device)
        for name,init in [('tau_rise',state['tau_rise']*.5),('tau_decay',state['tau_decay']*3.)]:
            v=state.get('aux_'+name,init).to(device).clone()
            setattr(self,'log_aux_'+name,nn.Parameter(v.log()))
        self.aux_omega=nn.Parameter(state.get('aux_omega',torch.zeros_like(state['omega'])).to(device).clone())

    def forward(self,xf):
        k=self.omega[:,None]*(1-torch.exp(-self.time/self.log_tau_rise.exp()[:,None]))*torch.exp(-self.time/self.log_tau_decay.exp()[:,None])
        k=k+self.aux_omega[:,None]*(1-torch.exp(-self.time/self.log_aux_tau_rise.exp()[:,None]))*torch.exp(-self.time/self.log_aux_tau_decay.exp()[:,None])
        kf=torch.fft.rfft(k,n=self.nfft,dim=-1)
        y=torch.fft.irfft(xf*kf[None],n=self.nfft,dim=-1)[...,:self.T].transpose(1,2).contiguous()
        return ((y@(self.weight*self.offdiag))*y).sum(-1)+y.sum(-1)+self.bias

    def original_state(self,base,reset_params):
        s=super().original_state(base,reset_params)
        for name in ['tau_rise','tau_decay']:s['aux_'+name]=getattr(self,'log_aux_'+name).detach().exp().cpu().clone()
        s['aux_omega']=self.aux_omega.detach().cpu().clone()
        return s
