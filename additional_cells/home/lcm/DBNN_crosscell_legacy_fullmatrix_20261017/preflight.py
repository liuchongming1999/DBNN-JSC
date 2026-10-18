import sys,json,math
from pathlib import Path
import numpy as np
import torch
root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
sys.path.insert(0,str(root/'benchmarks/channel_ablation_20260929'))
from full_matrix_drive import Drive,OriginalDrive
from common8 import sha
p=Path(__file__).parent;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
a0=Path('/home/lcm/DBNN_noncerebellar_20261002/results/human_l23/ampa/dt1.0ms/best_checkpoint.pt')
s=torch.load(a0,map_location='cpu',weights_only=False)['model_state'];assert s['bilinear.weight'][0].diagonal().count_nonzero()==0
n=500;dev='cuda:0';torch.manual_seed(127)
x=torch.randint(-1,2,(2,500,n),device=dev).float();xf=torch.fft.rfft(x,n=1<<(2*n-2).bit_length())
a=OriginalDrive(s,n,dev);b=Drive(s,n,dev);a.time=a.time.float();b.time=b.time.float()
assert torch.equal(a(xf),b(xf))
# Gradient clipping must be compared to the unrestricted full-matrix class.
# Its diagonal gradients correctly contribute to the global gradient norm.
a.offdiag.fill_(1.)
oa=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=1e-6)
ob=torch.optim.AdamW([{'params':[v for k,v in b.named_parameters() if k!='diagonal'],'lr':.0003},{'params':[b.diagonal],'lr':.00003}],weight_decay=1e-6)
for m,o in [(a,oa),(b,ob)]:
 o.zero_grad();m(xf).square().mean().backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step()
mask=~torch.eye(500,dtype=torch.bool,device=dev)
offdiag_gap=float((a.weight[mask]-b.weight[mask]).abs().max())
assert torch.allclose(a.weight[mask],b.weight[mask],atol=1e-7,rtol=0),offdiag_gap
# Compare against an unrestricted full-matrix Adam update under the same norm.
c=Drive(s,n,dev);c.time=c.time.float();oc=torch.optim.AdamW(c.parameters(),lr=.0003,weight_decay=1e-6);oc.zero_grad();c(xf).square().mean().backward();torch.nn.utils.clip_grad_norm_(c.parameters(),1.);oc.step()
assert torch.allclose(b.diagonal,.1*c.diagonal,atol=1e-9,rtol=1e-5)
for epochs in [1000,400]:
 sch=torch.optim.lr_scheduler.LambdaLR(ob,lambda ep:.02+.98*(1+math.cos(math.pi*ep/epochs))/2)
 for ep in range(epochs):ob.step();sch.step();assert math.isclose(ob.param_groups[1]['lr']/ob.param_groups[0]['lr'],.1,rel_tol=1e-12)
reset={k:float(s[k]) for k in ['v_th','reset_amp','tau_reset']};physical=b.original_state(s,reset);assert physical['bilinear.weight'][0].diagonal().count_nonzero()>0
assert torch.equal(physical['bilinear.weight'][0].diagonal(),b.diagonal.detach().cpu())
# This is a numerical implementation test, never a model-selection candidate.
r={'passed':True,'original_zero_diagonal_source':str(a0),'source_sha256':sha(a0),'initial_forward_exact':True,'non_diagonal_update_vs_full_reference_max_gap':offdiag_gap,'diagonal_update_ratio':.1,'schedule_ratio_verified':[1000,400],'physical_export_diagonal_retained':True}
(p/'preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
