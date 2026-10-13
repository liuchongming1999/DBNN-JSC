"""Faithful causal Conv1D -> activation -> BN architecture from Beniaguev et al.

The public Keras HDF5 can be read directly; its (kernel,input,output) weights
are transposed to PyTorch (output,input,kernel). Evaluation BN uses saved
population statistics, including the original epsilon.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class KerasBatchNorm(nn.BatchNorm1d):
    """Match Keras population variance and moving-statistics convention in training."""
    def forward(self,x):
        if not self.training:return super().forward(x)
        mean=x.mean((0,2));var=x.var((0,2),unbiased=False)
        with torch.no_grad():
            self.running_mean.mul_(1-self.momentum).add_(mean.detach(),alpha=self.momentum)
            self.running_var.mul_(1-self.momentum).add_(var.detach(),alpha=self.momentum)
            self.num_batches_tracked.add_(1)
        return (x-mean[None,:,None])/torch.sqrt(var[None,:,None]+self.eps)*self.weight[None,:,None]+self.bias[None,:,None]

class CausalLayer(nn.Module):
    def __init__(self,ni,no,k,eps=.001,activation='relu',dilation=1):
        super().__init__();self.conv=nn.Conv1d(ni,no,k,dilation=dilation)
        self.bn=KerasBatchNorm(no,eps=eps,momentum=.01)
        self.left=(k-1)*dilation;self.activation=activation
    def forward(self,x):
        x=self.conv(F.pad(x,(self.left,0)))
        if self.activation=='relu':x=F.relu(x)
        elif self.activation=='tanh':x=torch.tanh(x)
        elif self.activation=='linear':pass
        else:raise ValueError(self.activation)
        return self.bn(x)

class TCN(nn.Module):
    def __init__(self,inputs,kernels,widths,eps=.001,dendritic_outputs=0):
        super().__init__();layers=[];ni=inputs
        for k,w in zip(kernels,widths):layers.append(CausalLayer(ni,w,k,eps));ni=w
        self.layers=nn.ModuleList(layers);self.spikes=nn.Conv1d(ni,1,1);self.somatic=nn.Conv1d(ni,1,1)
        self.dendritic=nn.Conv1d(ni,dendritic_outputs,1) if dendritic_outputs else None
        self.history=1+sum(k-1 for k in kernels)
    def forward(self,x):
        # Public interface [batch,time,input] -> [batch,time,(spike logit,voltage)]
        z=x.transpose(1,2)
        for layer in self.layers:z=layer(z)
        return torch.cat([self.spikes(z),self.somatic(z)],1).transpose(1,2)
    def forward_all(self,x):
        z=x.transpose(1,2)
        for layer in self.layers:z=layer(z)
        core=torch.cat([self.spikes(z),self.somatic(z)],1).transpose(1,2)
        dend=self.dendritic(z).transpose(1,2) if self.dendritic is not None else None
        return core,dend

def load_public_h5(path):
    import h5py
    with h5py.File(path,'r') as h:
        cfg=h.attrs['model_config'];cfg=json.loads(cfg.decode() if isinstance(cfg,bytes) else cfg)
        config=cfg['config'];layers=config.get('layers',config)
        convs=[r['config'] for r in layers if r['class_name']=='Conv1D' and r['config']['name'].startswith('layer_')]
        norms=[r['config'] for r in layers if r['class_name']=='BatchNormalization']
        assert len(convs)==len(norms)>0
        store=h['model_weights'];weights={}
        def gather(name,obj):
            if isinstance(obj,h5py.Dataset):weights[name]=np.asarray(obj)
        store.visititems(gather)
        def weight(layer,var):
            found=[v for n,v in weights.items() if n.split('/')[-1].split(':')[0]==var and n.split('/')[0]==layer]
            if len(found)!=1:raise ValueError((layer,var,[n for n in weights if layer in n]))
            return torch.as_tensor(found[0])
        first=weight(convs[0]['name'],'kernel');nin=first.shape[1]
        model=TCN(nin,[r['kernel_size'][0] for r in convs],[r['filters'] for r in convs],norms[0]['epsilon'])
        with torch.no_grad():
            for mod,c,b in zip(model.layers,convs,norms):
                assert c['padding']=='causal' and c['strides']==[1]
                assert c.get('dilation_rate',[1])==[1]
                mod.activation=c['activation'];mod.bn.eps=b['epsilon']
                mod.conv.weight.copy_(weight(c['name'],'kernel').permute(2,1,0));mod.conv.bias.copy_(weight(c['name'],'bias'))
                mod.bn.weight.copy_(weight(b['name'],'gamma'));mod.bn.bias.copy_(weight(b['name'],'beta'))
                mod.bn.running_mean.copy_(weight(b['name'],'moving_mean'));mod.bn.running_var.copy_(weight(b['name'],'moving_variance'))
            for name,mod in [('spikes',model.spikes),('somatic',model.somatic)]:
                mod.weight.copy_(weight(name,'kernel').permute(2,1,0));mod.bias.copy_(weight(name,'bias'))
        model.eval()
    return model,cfg

def numpy_forward(model,x):
    # Small float64 independent check for weight layout, causal padding and BN.
    x=np.asarray(x,dtype=np.float64).transpose(0,2,1)
    for layer in model.layers:
        w=layer.conv.weight.detach().numpy().astype(float);bias=layer.conv.bias.detach().numpy().astype(float)
        pad=np.pad(x,((0,0),(0,0),(layer.left,0)));windows=np.lib.stride_tricks.sliding_window_view(pad,w.shape[-1],axis=2)
        x=np.einsum('bctk,ock->bot',windows,w,optimize=True)+bias[None,:,None]
        if layer.activation=='relu':x=np.maximum(x,0)
        elif layer.activation=='tanh':x=np.tanh(x)
        mean=layer.bn.running_mean.detach().numpy()[None,:,None];var=layer.bn.running_var.detach().numpy()[None,:,None]
        x=(x-mean)/np.sqrt(var+layer.bn.eps)*layer.bn.weight.detach().numpy()[None,:,None]+layer.bn.bias.detach().numpy()[None,:,None]
    out=[]
    for head in [model.spikes,model.somatic]:
        out.append(np.einsum('bct,oc->bot',x,head.weight.detach().numpy()[:,:,0])+head.bias.detach().numpy()[None,:,None])
    return np.concatenate(out,axis=1).transpose(0,2,1)
