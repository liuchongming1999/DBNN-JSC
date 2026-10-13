import json,sys,time,hashlib
from pathlib import Path
import numpy as np,torch
from experiment import ROOT,HERE,case_data,accuracy,measure_group,stamp,resource_snapshot
from backends import Prepared,capture,native,subset_state
from DBNN import DBNN_R,DBNN_F

def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    result={'created':stamp(),'resources':resource_snapshot(),'checkpoints':{},'equivalence':{},'ablation':{}}
    d=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    signs=np.where(d['channel_type'].astype(str)=='exc',1.,-1.).astype(np.float32)
    xc=torch.from_numpy(d['test_ip'].astype(np.float32)*signs[None,:,None]);target=torch.from_numpy(d['test_op'].astype(np.float32))
    ckpts={'reference':'source/dbnn_checkpoint.pt',
      'scratch_1ep_cpu':'results/train_cpu_bench/dbnn_best.pt',
      'scratch_1ep_gpu':'results/train_gpu5_b1_bench/dbnn_best.pt',
      'finetune_1ep':'results/train_gpu5_b1_lr2e4/dbnn_best.pt',
      'finetune_4ep_partial':'results/dbnn_repro_finetune_5ep_lr2e4/dbnn_best.pt'}
    with torch.no_grad():
        for name,path in ckpts.items():
            raw=torch.load(ROOT/path,map_location='cpu',weights_only=False);s=raw['model_state']
            # Model construction outside inference_mode preserves reference _version counters.
            m=DBNN_R(500,6001,torch.device('cpu'));m.load_state_dict(s);m.eval()
            output=torch.cat([m(xc[b:b+1]) for b in range(10)])
            errs=output-target;rmse=errs.square().mean(-1).sqrt()
            result['checkpoints'][name]={'path':path,'mean_sample_rmse_mv':float(rmse.mean()),
                'pooled_rmse_mv':float(errs.square().mean().sqrt()),'per_sample_rmse_mv':rmse.tolist(),
                'max_abs_mv':float(errs.abs().max()),'config':raw.get('config')}
            if name=='reference':ref=output.clone();state=s
        np.savez_compressed(HERE/'fit_traces.npz',target=target.numpy(),prediction=ref.numpy(),time_ms=np.arange(6001))
        # Every backend is checked on all 10 held-out traces, including reset events.
        x=xc.cuda();sd={k:v.cuda() for k,v in state.items()}
        for scan,quad in [('cuda','cublas'),('triton','cublas'),('cute','cublas'),('fft','cutlass'),('cuda','cutlass_tensor')]:
            try:
                p=Prepared(sd,x,scan,quad);out=p().clone()
                result['equivalence'][f'{scan}_{quad}']=accuracy(out,ref,p.vth)
            except Exception as e:result['equivalence'][f'{scan}_{quad}']={'error':str(e)}
        # Adversarial driven reset: zero, frequently-spiking, negative and near-threshold.
        g=torch.Generator().manual_seed(919)
        drive=torch.randn((4,1001),generator=g)*5-52
        drive[0,:]=-75.;drive[1,:]=-30.
        drive[2,10:20]=-55.+torch.linspace(-1e-4,1e-4,10)
        from DBNN import _apply_reset_closed_loop
        rref=_apply_reset_closed_loop(drive,torch.tensor(-55.),torch.tensor(10.),torch.tensor(12.),10.)
        dg=drive.cuda();og=torch.empty_like(dg);alpha=float(torch.exp(torch.tensor(-1/12)))
        ap=torch.tensor(alpha,device='cuda').pow(torch.arange(1001,device='cuda',dtype=torch.float32))
        native().call('dbnn_reset_event',dg,og,ap,4,1001,alpha,10.,-55.,torch.cuda.current_stream().cuda_stream)
        result['adversarial_reset']=accuracy(og,rref,-55.)
        # Controlled ablation holds scan/quad/input fixed and changes reset only.
        for N,A,T,B in [(500,500,6001,1),(500,32,6001,1),(500,32,6001,8)]:
            sc,xc2,ids,_=case_data(N,A,T,B);xs=xc2[:,ids].contiguous().cuda()
            sp={k:v.cuda() for k,v in subset_state(sc,ids).items()}
            p=Prepared(sp,xs,'cuda','cublas');event=capture(p)
            def sequential_reset():
                p.stream=torch.cuda.current_stream().cuda_stream
                y=p.synaptic();p.drive_only(y)
                native().call('dbnn_reset',p.drive,p.out,p.B,p.T,p.alpha,p.amp,p.vth,p.stream)
                return p.out
            seq=capture(sequential_reset)
            fns={'sequential_reset':seq,'event_reset':event}
            tensor=Prepared(sp,xs,'cuda','cutlass_tensor');fns['cutlass_tensor_event']=capture(tensor)
            for block in [64,128,256,512]:
                pp=Prepared(sp,xs,'triton','cublas',block=block)
                fns[f'triton_block_{block}']=capture(pp)
            for fn in fns.values():
                for _ in range(3):fn()
            errs={name:accuracy(fn().clone(),event().clone(),p.vth) for name,fn in fns.items()}
            timing=measure_group(fns,50)
            for name in timing:timing[name].update(errs[name])
            result['ablation'][f'N{N}_A{A}_B{B}']=timing
    (HERE/'validation.json').write_text(json.dumps(result,indent=2,default=str))
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','checkpoints')},indent=2),flush=True)

if __name__=='__main__':main()
