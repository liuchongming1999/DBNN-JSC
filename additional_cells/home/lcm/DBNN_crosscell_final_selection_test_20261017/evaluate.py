"""Frozen full-matrix test evaluation, complete outputs and prepared GPU timing."""
from pathlib import Path
import argparse
import gc
import fcntl
import hashlib
import os
import sys
import time
import numpy as np
import torch
from common8 import read,write,sha,metric,aggregate,match_events,event_aggregate,reset_function
def fft_length(n):
    return 1 << (2*n-2).bit_length()


def bootstrap(rows,n=10000):
    rng=np.random.default_rng(20261017)
    ix=rng.integers(0,len(rows),(n,len(rows)))
    sse=np.array([r['voltage']['sse'] for r in rows]);size=np.array([r['voltage']['n'] for r in rows])
    den_v=size[ix].sum(1)
    rmse=np.sqrt(np.divide(sse[ix].sum(1),den_v,out=np.full(n,np.nan),where=den_v>0))
    tp,fp,fn=[np.array([r['events'][k] for r in rows])[ix].sum(1) for k in ['tp','fp','fn']]
    den=2*tp+fp+fn;f1=np.divide(2*tp,den,out=np.full(n,np.nan),where=den>0)
    return dict(replicates=n,unit='complete independent input trajectory',
        rmse_ci95_mv=np.nanquantile(rmse,[.025,.975]).tolist() if np.isfinite(rmse).any() else None,
        f1_ci95=np.nanquantile(f1,[.025,.975]).tolist() if np.isfinite(f1).any() else None)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--job',type=int,required=True);ap.add_argument('--repeats',type=int,default=3)
    a=ap.parse_args();work=a.work.resolve();job=read(work/'locked_protocol.json')['jobs'][a.job]
    base=Path(job['base']);out=base/'evaluation';out.mkdir(parents=True,exist_ok=True)
    guard=open(out/'evaluation.lock','w');fcntl.flock(guard,fcntl.LOCK_EX)
    if (out/'complete.json').exists():return
    assert read(work/'models_locked.json')['all_models_locked']
    locked=read(work/'models_locked.json')['models'][str(a.job)]
    cp=Path(locked['checkpoint']);sel=locked
    assert sha(cp)==sel['checkpoint_sha256']
    assert sha(job['test_manifest'])==job['test_manifest_sha256']
    assert sha(job['reference_completion'])==job['reference_completion_sha256']
    assert read(job['reference_completion'])['all_inputs_retained']
    root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/fair_window_20261005/timing_w500')]
    from gpu_backend import ResolutionModel,Graph,options
    from backends import subset_state
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    state=torch.load(cp,map_location='cpu',weights_only=False)['model_state']
    assert torch.isfinite(state['bilinear.weight']).all()
    assert locked['diagonal_trainable']
    # A zero-valued winner is permitted by the unchanged validation rule.
    reference_meta={r['case_id']:r for r in read(job['reference_completion'])['records']}
    reset=reset_function(work);pp={k:float(state[k]) for k in ['v_th','reset_amp','tau_reset']}
    state_gpu={k:v.cuda() for k,v in state.items()};records=[]
    rows=read(job['test_manifest'])['cases']
    # Same archived synthetic-shape policy used by the primary DBNN. A backend
    # may change only to restore numerical equivalence, never based on test error.
    policy=read(root/'benchmarks/fair_window_20261005/timing_w500/gpu_policy.json')['records']
    for row in rows:
        record_path=out/'records'/f"{row['case_id']}.json"
        if record_path.exists():
            rec=read(record_path);assert rec['checkpoint_sha256']==sha(cp) and rec['test_manifest_sha256']==job['test_manifest_sha256'];records.append(rec);continue
        f=np.load(row['input_file']);ids=f['ids'];raw=f['x']
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        assert hashlib.sha256(ids.tobytes()).hexdigest()==row['ids_sha256']
        assert sha(row['reference_file'])==reference_meta[row['case_id']]['reference_sha256']
        signed=raw.astype(np.int8)*np.where(ids[:,None]<250,1,-1).astype(np.int8)
        st=subset_state(state_gpu,torch.as_tensor(ids.astype(np.int64),device='cuda'))
        assert torch.equal(st['bilinear.weight'][0].diagonal().cpu(),state['bilinear.weight'][0].diagonal()[torch.from_numpy(ids.astype(np.int64))])
        x=torch.from_numpy(signed[None]).cuda().contiguous();n=x.shape[-1]
        # Independent subset FFT implementation retains all diagonal entries.
        age=torch.arange(n,device='cuda')[None]
        k=st['omega'][:,None]*(1-torch.exp(-age/st['tau_rise'][:,None]))*torch.exp(-age/st['tau_decay'][:,None])
        y=torch.fft.irfft(torch.fft.rfft(x.float(),n=fft_length(n))*torch.fft.rfft(k,n=fft_length(n))[None],n=fft_length(n))[...,:n].transpose(1,2).contiguous()
        drive=(((y@st['bilinear.weight'][0])*y).sum(-1)+y.sum(-1)+st['bilinear.bias'])[0].cpu().numpy()
        oracle,events=reset(drive,pp['v_th'],pp['reset_amp'],pp['tau_reset'])
        near=min([r for r in policy if r['dt_ms']==1. and r['A']==row['A']],key=lambda r:abs(np.log(r['input_steps']/n)))
        tags=sorted(near['timing'],key=lambda k:near['timing'][k]['median_ms'])
        validation=None
        for tag in tags:
            mode,block=tag.rsplit('_',1);m=ResolutionModel(st,x,1.,mode,int(block));v=m();torch.cuda.synchronize()
            pred=v[0].cpu().numpy().copy();pred_events=np.flatnonzero(pred>=m.th).astype(float)
            validation=dict(max_abs_mv=float(np.max(abs(pred-oracle))),events_equal=bool(np.array_equal(pred_events,events)))
            if validation['max_abs_mv']<.001 and validation['events_equal']:break
            del m,v
        else:
            from precision_fallback import FFT64ResetModel
            prior_failure=validation
            m=FFT64ResetModel(st,x);v=m();torch.cuda.synchronize()
            pred=v[0].cpu().numpy().copy();pred_events=np.flatnonzero(pred>=m.th).astype(float)
            validation=dict(max_abs_mv=float(np.max(abs(pred-oracle))),events_equal=bool(np.array_equal(pred_events,events)),fast_backend_failure=prior_failure)
            assert validation['max_abs_mv']<.001 and validation['events_equal'],(row['case_id'],validation)
            tag='prepared_fft_fp64_sequential_reset'

        graph=Graph(m);wall=[]
        for _ in range(a.repeats):
            torch.cuda.synchronize();start=time.perf_counter();graph();torch.cuda.synchronize()
            wall.append((time.perf_counter()-start)*1000)
        measured=m.out[0].cpu().numpy().copy()
        assert np.max(abs(measured-oracle))<.001 and np.array_equal(np.flatnonzero(measured>=m.th),events),('Captured graph parity',row['case_id'])
        validation['captured_graph_max_abs_mv']=float(np.max(abs(measured-oracle)))
        validation['captured_graph_events_equal']=bool(np.array_equal(np.flatnonzero(measured>=m.th),events))
        ref=np.load(row['reference_file']);times=np.arange(n,dtype=float)
        target=np.interp(times,ref['time_ms'],ref['voltage'])
        rec=dict(case_id=row['case_id'],duration_ms=row['duration_ms'],A=row['A'],family=row['family'],
            site_mode=row['site_mode'],rate_hz=row['rate_hz'],input_sha256=row['input_sha256'],
            checkpoint_sha256=sha(cp),reference_sha256=sha(row['reference_file']),
            test_manifest_sha256=job['test_manifest_sha256'],input_seed=row['seed'],ids_sha256=row['ids_sha256'],
            reference_events_ms=ref['spike_times'].tolist(),predicted_events_ms=pred_events.tolist(),
            voltage=metric(target,pred,ref['spike_times']),events=match_events(ref['spike_times'],pred_events),
            prepared_wall_ms=float(np.median(wall)),runs_ms=wall,backend=tag,numerical=validation)
        (out/'predictions').mkdir(exist_ok=True)
        np.savez_compressed(out/'predictions'/f"{row['case_id']}.npz",time_ms=times,
            voltage=pred,drive=drive,reference_voltage=target,
            reference_events_ms=ref['spike_times'],predicted_events_ms=pred_events)
        write(record_path,rec);records.append(rec)
        write(out/'status.json',dict(completed=len(records),total=len(rows)))
        print('EVALUATE',job['cell'],job['profile'],len(records),len(rows),flush=True)
        del m,v,graph,st,x,y;gc.collect();torch.cuda.empty_cache()
    assert len(records)==182 and sha(cp)==sel['checkpoint_sha256']
    groups={}
    for key in ['family','duration_ms','A','rate_hz','site_mode']:
        groups[key]={}
        for value in sorted({r[key] for r in records},key=str):
            rr=[r for r in records if r[key]==value]
            groups[key][str(value)]=dict(inputs=len(rr),voltage=aggregate([r['voltage'] for r in rr]),events=event_aggregate([r['events'] for r in rr]))
    write(out/'complete.json',dict(finished=True,all_inputs_retained=True,records=records,
        checkpoint_sha256=sha(cp),test_manifest_sha256=sha(job['test_manifest']),
        selected_candidate=locked['selected_candidate'],diagonal_max_abs=locked['diagonal_max_abs'],
        undefined_voltage_inputs=sum(r['voltage']['n']==0 for r in records),test_used_for_selection=False,
        voltage=aggregate([r['voltage'] for r in records]),events=event_aggregate([r['events'] for r in records]),
        groups=groups,bootstrap=bootstrap(records),prepared_wall_ms=sum(r['prepared_wall_ms'] for r in records),
        timing=dict(repeats=a.repeats,scope='Prepared complete GPU filtering+bilinear+linear+bias+closed-loop reset, synchronized; external packing/upload/download/capture excluded',
                    shared_server=True,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name())))


if __name__=='__main__':main()
