"""Read-only optimizer/BN/LSTM implementation checks, no model training."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');B=R/'benchmarks/baselines_20260930';W=R/'benchmarks/baseline_fidelity_audit_20261016'
sys.path[:0]=[str(B),str(B/'vendor/elmneuron')]
from tcn import KerasBatchNorm
from train_default import LSTM,config,losses,model_for
from src.neuronio.neuronio_train_utils import NeuronioLoss

def main():
 torch.set_num_threads(1);rng=np.random.default_rng(20261016);torch.manual_seed(20261016);result={}
 # Independent NumPy transcription of Keras2.2.4 NAdam, matching the source
 # equations; gradient stream spans tiny and ordinary magnitudes.
 q=rng.normal(size=11);p=torch.nn.Parameter(torch.tensor(q,dtype=torch.float64));worst=0
 for lr in [1e-4,3e-5,1e-5]:
  opt=torch.optim.NAdam([p],lr=lr,eps=1e-7,momentum_decay=.004);m=np.zeros(11);v=np.zeros(11);prod=1.
  for t in range(1,201):
   g=rng.normal(size=11)*np.logspace(-8,0,11);p.grad=torch.tensor(g,dtype=torch.float64)
   mu=.9*(1-.5*.96**(t*.004));mu_next=.9*(1-.5*.96**((t+1)*.004));newprod=prod*mu
   m=.9*m+.1*g;v=.999*v+.001*g*g
   mbar=(1-mu)*g/(1-newprod)+mu_next*m/(1-newprod*mu_next)
   q-=lr*mbar/(np.sqrt(v/(1-.999**t))+1e-7);prod=newprod;opt.step();worst=max(worst,float(np.max(np.abs(p.detach().numpy()-q))))
 result['nadam_vs_keras224_numpy']=dict(passed=worst<1e-7,max_parameter_abs=worst,steps=600,stage_resets=3,gradient_scale_range=[1e-8,1.],source='keras-team/keras tag2.2.4 optimizers.py Nadam equations; currentPyTorchNAdam actual hyperparameters')
 # Population-variance BN values, moving statistics, and finite-difference
 # input gradients are checked independently of nn.BatchNorm training.
 x=rng.normal(size=(2,3,4));w=rng.normal(size=x.shape);bn=KerasBatchNorm(3,eps=.001,momentum=.01).double().train();bn.weight.data.copy_(torch.tensor([.7,1.2,-.4]));bn.bias.data.copy_(torch.tensor([.2,.1,-.1]));xx=torch.tensor(x,requires_grad=True);y=bn(xx)
 def numpy_bn(z):return (z-z.mean((0,2))[None,:,None])/np.sqrt(z.var((0,2))[None,:,None]+.001)*bn.weight.detach().numpy()[None,:,None]+bn.bias.detach().numpy()[None,:,None]
 forward=float(np.max(np.abs(y.detach().numpy()-numpy_bn(x))));(y*torch.tensor(w)).sum().backward();g=xx.grad.numpy();ng=np.empty_like(x);eps=1e-5
 for ix in np.ndindex(x.shape):
  xp=x.copy();xm=x.copy();xp[ix]+=eps;xm[ix]-=eps;ng[ix]=((numpy_bn(xp)-numpy_bn(xm))*w).sum()/(2*eps)
 grad=float(np.max(np.abs(g-ng)));variance=float(np.max(np.abs(bn.running_var.detach().numpy()-(.99+.01*x.var((0,2))))));mean=float(np.max(np.abs(bn.running_mean.detach().numpy()-.01*x.mean((0,2)))))
 result['training_batchnorm']=dict(passed=max(forward,grad,variance,mean)<1e-7,forward_max=forward,input_gradient_max=grad,moving_variance_max=variance,moving_mean_max=mean,scope='Population-moment formula/gradient fixture; not an end-to-end originalTensorFlow training replay')
 # Current LSTM against its explicit four-gate equations, gate order and bias.
 lstm=LSTM().double();inp=torch.tensor(rng.normal(size=(2,7,500)),dtype=torch.float64);actual=lstm(inp);h=torch.zeros(2,50,dtype=torch.float64);c=h.clone();outs=[]
 for t in range(7):
  gates=inp[:,t]@lstm.recurrent.weight_ih_l0.T+h@lstm.recurrent.weight_hh_l0.T+lstm.recurrent.bias_ih_l0+lstm.recurrent.bias_hh_l0;i,f,g,o=gates.chunk(4,-1);c=f.sigmoid()*c+i.sigmoid()*g.tanh();h=o.sigmoid()*c.tanh();outs.append(lstm.output(h))
 gap=float(torch.max(torch.abs(actual-torch.stack(outs,1))).detach());result['lstm_internal_equations']=dict(passed=gap<1e-10,max_abs=gap,current_trainable_parameters=sum(p.numel() for p in lstm.parameters() if p.requires_grad),published1278input_parameter_count=4*50*(1278+50+1)+2*(50+1),scope='Confirms current standardLSTM equations; does not establish author unpublished JAX initializer/gate implementation equivalence')
 # Author loss is the exact module on same official-ELM model outputs.
 cfg=config('elm',1.);elm=model_for(B,cfg);x=torch.randn(2,13,500);y=torch.rand(2,13,2);out=elm(x);original=NeuronioLoss()(out,(y[...,0],y[...,1]));current,_=losses(elm,x,y,None,cfg,0)
 gap=abs(float(current-original));result['elm_official_full_window_loss']=dict(passed=gap<1e-7,max_abs=gap,scope='AuthorPython code loss, not paper150ms-burn alternative')
 result['all_passed']=all(r.get('passed',False) for r in result.values());result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(W/'numerical_probes.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 assert result['all_passed']
if __name__=='__main__':main()
