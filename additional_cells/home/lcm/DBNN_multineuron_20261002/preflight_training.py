"""One-batch numerical check on real new-neuron references, not a model fit."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from common import mask,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',required=True);a=p.parse_args()
    sys.path.insert(0,str(a.root/'benchmarks/channel_ablation_20260929'))
    from train_channel_ablation import Drive,teacher_reset,reset
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    d=a.data/a.model/'ampa/train_1.0ms'
    rows=json.loads((d/'candidate_manifest.json').read_text())['cases']
    ready=[r for r in rows if Path(r['reference_file']).with_suffix('.json').exists()]
    chosen=[]
    for split in ['fit','validation']:
        subset=[r for r in ready if r['split']==split]
        if len(subset)<2:raise RuntimeError(('Need two ready cases per split',split,len(subset)))
        chosen+=subset[:2]
    op=json.loads((a.work/'results'/a.model/'screen/operating_point.json').read_text())
    base={'tau_rise':torch.full((500,),2.),'tau_decay':torch.full((500,),20.),'omega':torch.full((500,),.001),
          'bilinear.weight':torch.zeros((1,500,500)),'bilinear.bias':torch.tensor([op['rest_mv']]),
          'v_th':torch.tensor(op['rest_mv']+15),'reset_amp':torch.tensor(8.),'tau_reset':torch.tensor(16.)}
    results=[]
    for dt in [1.,.5,.25]:
        stride=round(1/dt);T=6000*stride+1;nfft=1<<(2*T-2).bit_length()
        dense=[];target=[];spikes=[]
        for row in chosen:
            inp=np.load(row['input_file']);ref=np.load(row['reference_file'])
            x=np.zeros((500,T),np.float32);ids=inp['ids'];x[ids,::stride]=inp['x']*(1-2*(ids[:,None]>=250))
            vv=ref['voltage'][::round(dt/.25)];assert len(vv)==T
            dense.append(x);target.append(vv);spikes.append(ref['spike_times'])
        target=np.asarray(target);masks=np.asarray([mask(T,s,dt) for s in spikes]);assert masks.sum(axis=1).min()>1
        var=np.array([np.var(v[m]) for v,m in zip(target,masks)])
        weights=.3+.7/np.maximum(var,.04);weights/=weights[:2].mean()
        xf=torch.fft.rfft(torch.tensor(np.asarray(dense),device='cuda'),n=nfft)
        teacher=np.array([teacher_reset(T,s/dt,8.,16./dt) for s in spikes],dtype=np.float32)
        y=torch.tensor(target-teacher,device='cuda');w=torch.tensor(masks,dtype=torch.float32,device='cuda')
        model=Drive(base,T,torch.device('cuda'));model.time=model.time.float()*dt
        error=model(xf[:2])-y[:2];dw=torch.minimum(w[:2,1:],w[:2,:-1])
        per=(error.square()*w[:2]).sum(1)/w[:2].sum(1)+.1*((error[:,1:]-error[:,:-1]).div(dt).square()*dw).sum(1)/dw.sum(1)
        loss=(per*torch.tensor(weights[:2],device='cuda')).mean();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(z.grad).all() for z in model.parameters())
        with torch.inference_mode():pred=model(xf[2:]).cpu().numpy()
        for row in pred:
            v,ev=reset(row,op['rest_mv']+15,8.,16./dt);assert np.isfinite(v).all()
        results.append(dict(dt_ms=dt,samples=T,loss=float(loss.detach()),finite_gradients=True,
                            reference_mask_retained_fraction=masks.mean(axis=1).tolist()))
        del model,xf,y,w,loss,error,per;torch.cuda.empty_cache()
    write(a.work/'results'/a.model/'training_preflight.json',dict(passed=True,checks=results,
        cases=[r['case_id'] for r in chosen],purpose='Real-data loading, physical resampling, original Drive, balanced loss, backpropagation and closed-loop reset implementation check. Synthetic seed; no fitted accuracy or checkpoint claim.',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print('TRAINING_PREFLIGHT_PASS',a.model,results,flush=True)

if __name__=='__main__':main()
