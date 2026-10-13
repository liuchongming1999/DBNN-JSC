"""Frozen new-cell checkpoint versus independent FFT/matmul/CPU-reset oracle."""
import argparse,gc,os,sys,json
from pathlib import Path
import numpy as np
import torch
from common import read,sha,write

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--model',required=True);a=p.parse_args()
    root=a.root;work=a.work
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/channel_ablation_20260929'),str(root/'benchmarks/training_refinement_20260929'),str(root/'benchmarks/resolution_frontier_20260929')]
    from backends import subset_state
    from gpu_backend import ResolutionModel,Graph,options
    from train_channel_ablation import reset
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(61002701);records=[]
    for dt in [1.,.5,.25]:
        f=work/'results'/a.model/'ampa'/f'dt{dt}ms/best_checkpoint.pt';sel=read(f.parent/'final_selection.json');assert sha(f)==sel['checkpoint_sha256']
        state=torch.load(f,map_location='cuda',weights_only=False)['model_state']
        for A in [64,500]:
            ids=np.sort(rng.choice(500,A,False));st=subset_state(state,torch.tensor(ids,device='cuda'))
            raw=rng.poisson(.025,(1,A,1001)).astype(np.int8);raw[...,0]=0;raw=raw*(1-2*(ids[None,:,None]>=250)).astype(np.int8)
            x=torch.from_numpy(raw).cuda();T=1000*round(1/dt)+1;nfft=1<<(2*T-2).bit_length();time=torch.arange(T,device='cuda')*dt
            k=st['omega'][:,None]*(1-torch.exp(-time/st['tau_rise'][:,None]))*torch.exp(-time/st['tau_decay'][:,None])
            full=torch.zeros((1,A,T),device='cuda');full[...,::round(1/dt)]=x.float()
            y=torch.fft.irfft(torch.fft.rfft(full,n=nfft)*torch.fft.rfft(k,n=nfft),n=nfft)[...,:T].transpose(1,2).contiguous()
            u=((y@st['bilinear.weight'][0])*y).sum(-1)+y.sum(-1)+st['bilinear.bias']
            ref,events=reset(u[0].cpu().numpy(),float(st['v_th']),float(st['reset_amp']),float(st['tau_reset'])/dt)
            choices=[]
            for mode,block in options(dt):
                m=ResolutionModel(st,x,dt,mode,block);g=Graph(m);v=g();torch.cuda.synchronize();v=v[0].cpu().numpy()
                err=float(np.max(np.abs(v-ref)));ev=np.flatnonzero(v>=m.th)
                passed=bool(np.isfinite(v).all() and err<.001 and np.array_equal(events,ev))
                choices.append(dict(mode=mode,block=block,max_abs_mv=err,events_equal=bool(np.array_equal(events,ev)),passed=passed))
                del g,m;gc.collect();torch.cuda.empty_cache()
            assert any(z['passed'] for z in choices),(dt,A,choices)
            records.append(dict(dt_ms=dt,A=A,checkpoint_sha256=sha(f),oracle_events=len(events),choices=choices))
            print('INFERENCE_PREFLIGHT',a.model,dt,A,sum(z['passed'] for z in choices),'/',len(choices),flush=True)
            del x,st,full,y,u,k;torch.cuda.empty_cache()
    write(work/'results'/a.model/'inference_preflight.json',dict(passed=True,records=records,purpose='Independent random events and frozen real new-neuron checkpoints. Implementation equivalence only; no main31 accuracy or timing claim.',gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))

if __name__=='__main__':main()
