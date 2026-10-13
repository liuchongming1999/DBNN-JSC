"""Lock validation selection, then evaluate and time the deployed model on main31."""
import argparse, gc, hashlib, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(v):return dict(median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),runs_ms=list(v))
def measure(fn,n):
    for _ in range(3):fn()
    torch.cuda.synchronize();values=[]
    for _ in range(n):
        t=time.perf_counter();fn();torch.cuda.synchronize();values.append((time.perf_counter()-t)*1000)
    return stat(values)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--repeats',type=int,default=30);a=ap.parse_args();root=a.root.resolve();work=a.work.resolve()
    sys.path[:0]=[str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/channel_ablation_20260929')]
    from backends import Prepared,subset_state,capture
    from quadratic_tiled import TiledDrive
    from train_channel_ablation import reset
    out=work/'main31';out.mkdir(exist_ok=True)
    selection=work/'final_selection.json';selected=work/'best_checkpoint.pt'
    if not selection.exists():
        studies=[json.loads((work/'selection.json').read_text()),json.loads((work/'self_terms/selection.json').read_text())]
        candidates=[c for study in studies for c in study['candidates']]
        winner=min(candidates,key=lambda c:c['validation']['score'])
        ck=torch.load(winner['checkpoint'],map_location='cpu',weights_only=False)
        locked=datetime.now().isoformat();ck['final_locked_at']=locked;torch.save(ck,selected)
        info=dict(selected=winner,candidates=candidates,locked_at=locked,selection_rule='Pooled masked validation MSE + 0.5*(1-F1 at +/-5 ms)',checkpoint_sha256=sha(selected),training_only=True,studies=[str(work/'selection.json'),str(work/'self_terms/selection.json')])
        selection.write_text(json.dumps(info,indent=2)+'\n')
    info=json.loads(selection.read_text());assert sha(selected)==info['checkpoint_sha256']
    ck=torch.load(selected,map_location='cpu',weights_only=False);state={k:v.detach().clone() for k,v in ck['model_state'].items()};threshold=float(state['v_th']);amp=abs(float(state['reset_amp']));tau=abs(float(state['tau_reset']))
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    class Deployed(Prepared):
        def __init__(self,st,x):
            super().__init__(st,x,'triton','cublas',512)
            # The generic base masks the diagonal. Restore the locked model here;
            # the tiled kernel already counts arbitrary diagonal entries once.
            self.W=st['bilinear.weight'][0].detach().clone().contiguous()
            self.tile=TiledDrive(self.y.reshape(-1,self.K),self.W,self.bias,backend='triton',config=3,triangle=self.K>32,precision='ieee')
            self.drive=self.tile.out.view(self.B,self.T);del self.z
        def drive_only(self,y):self.tile(y.reshape(-1,self.K));return self.drive
    manifest_path=root/'benchmarks/neuron_single_expanded_20260929/selected_20260929/manifest.json';manifest=json.loads(manifest_path.read_text());rows=[]
    resource=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout
    for r in manifest['cases']:
        cid=r['case_id'];oj=out/f'dbnn_{cid}.json';on=oj.with_suffix('.npz')
        if oj.exists() and on.exists():
            prev=json.loads(oj.read_text());assert prev['checkpoint_sha256']==info['checkpoint_sha256'];rows.append(prev);continue
        ip=manifest_path.parent/r['input_file']
        if not ip.exists():ip=manifest_path.parent/'inputs'/r['input_file']
        inp=np.load(ip);ids=inp['ids'].astype(np.int64);raw=inp['x'].astype(np.float32);assert hashlib.sha256(raw.astype(np.uint8).tobytes()).hexdigest()==r['input_sha256']
        ii=torch.from_numpy(ids);sign=torch.where(ii<250,1.,-1.);packed=(torch.from_numpy(raw)*sign[:,None])[None].contiguous();x=packed.cuda();s=subset_state({k:v.cuda() for k,v in state.items()},ii.cuda());T=x.shape[-1]
        with torch.inference_mode():
            prep=Deployed(s,x);fn=capture(prep);prediction=fn().clone();drive=prep.drive.clone();torch.cuda.synchronize()
            # Independent FFT + dense matmul + CPU event recurrence, same model.
            t=torch.arange(T,device='cuda');nfft=1<<(2*T-2).bit_length();k=s['omega'][:,None]*(1-torch.exp(-t/s['tau_rise'][:,None]))*torch.exp(-t/s['tau_decay'][:,None])
            y=torch.fft.irfft(torch.fft.rfft(x,n=nfft)*torch.fft.rfft(k,n=nfft),n=nfft)[...,:T].transpose(1,2).contiguous()
            u=((y@s['bilinear.weight'][0])*y).sum(-1)+y.sum(-1)+s['bilinear.bias'];ud=u[0].cpu().numpy();oracle,oracle_events=reset(ud,threshold,amp,tau)
            pred=prediction[0].cpu().numpy();pred_events=np.flatnonzero(pred>=threshold);err=pred-oracle
            validation=dict(full_curve_max_abs_mv=float(np.max(abs(err))),full_curve_rmse_mv=float(np.sqrt(np.mean(err**2))),drive_max_abs_mv=float((u-drive).abs().max()),event_indices_equal=bool(np.array_equal(pred_events,oracle_events)),finite=bool(np.isfinite(pred).all()))
            validation['passed']=validation['finite'] and validation['event_indices_equal'] and validation['full_curve_max_abs_mv']<.001
            if not validation['passed']:raise RuntimeError((cid,validation))
            ph=packed.pin_memory();dense=torch.zeros((1,500,T));dense[:,ii]=packed;dh=dense.pin_memory();host=torch.empty((1,T),pin_memory=True)
            def packed_e2e():x.copy_(ph,non_blocking=True);host.copy_(fn(),non_blocking=True)
            def dense_e2e():x.copy_(dh.index_select(1,ii),non_blocking=True);host.copy_(fn(),non_blocking=True)
            timing=dict(packed_e2e=measure(packed_e2e,a.repeats),dense_gather_e2e=measure(dense_e2e,a.repeats),graph_core=measure(fn,a.repeats))
        np.savez_compressed(on,voltage=pred,drive=drive.cpu().numpy()[0],reset_events=pred_events,ids=ids)
        rec=dict(finished=True,created=datetime.now().isoformat(),case_id=cid,checkpoint_sha256=info['checkpoint_sha256'],candidate=info['selected']['name'],input_sha256=r['input_sha256'],A=len(ids),T=T,reset_events_ms=pred_events.tolist(),threshold_mv=threshold,reset_amp_mv=amp,tau_reset_ms=tau,validation=validation,timing=timing,precision='FP32',implementation='active packing + affine prefix scan + complete quadratic/linear/bias fusion + event reset + cached workspace + CUDA Graph',self_terms=bool(ck.get('allow_self_terms',False)),gpu_device=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),resource='Shared server; other users/jobs retained; our training searches finished before this timer')
        oj.write_text(json.dumps(rec,indent=2)+'\n');rows.append(rec);print(cid,'RMSE numerical',validation['full_curve_rmse_mv'],'e2e',timing['dense_gather_e2e']['median_ms'],flush=True)
        del prep,fn,x,s,ph,dh,dense,host,prediction,drive,u,y,k;gc.collect();torch.cuda.empty_cache()
    result=dict(finished=True,selected=info,records=rows,resource_snapshot=resource,repeats=a.repeats,manifest_sha256=sha(manifest_path),timer='Warm prepared model. DBNN dense CPU active-ID gather + H2D + complete GPU computation + D2H + synchronize. Compile, coefficient preparation and graph capture excluded.')
    (work/'main31_complete.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',len(rows),info['selected']['name'],flush=True)

if __name__=='__main__':main()
