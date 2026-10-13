"""Accuracy-focused fitting and checkpoint selection use fit/validation only."""
import argparse,copy,fcntl,math,os,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha,raster,mask,metric,aggregate

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--dt',type=float,required=True);p.add_argument('--fresh',action='store_true')
    p.add_argument('--scratch',action='store_true');p.add_argument('--epochs',type=int,default=1000);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--refinement-epochs',type=int,default=400)
    p.add_argument('--two-kernels',action='store_true',help='Archived architecture experiment; excluded from this fixed-architecture campaign.')
    p.add_argument('--init-checkpoint',type=Path,help='Learned starting state; time constants are stored in physical milliseconds.')
    p.add_argument('--manifest',type=Path,help='Explicit fit/validation manifest with native-time reference files.')
    p.add_argument('--input-cache',choices=['fft','raster'],default='fft',help='Raster stores exact int8 events on GPU and computes each batch FFT on demand.')
    p.add_argument('--regime',choices=['low-spike','all'],default='low-spike',help='Declared deployment regime: reference-only selection of 0–3 spikes and no ISI below 20ms. All-activity pilot is archived separately.')
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();out=work/a.name;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'training.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'final_selection.json').exists():
        print('ALREADY_COMPLETE',a.name,flush=True);return
    dt=a.dt;stride=round(1/dt);assert dt in (1.,.5,.25)
    sys.path[:0]=[str(root/'benchmarks/channel_ablation_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from train_channel_ablation import Drive,reset,teacher_reset
    from train_refinement import match_events,event_aggregate
    if a.two_kernels:
        from mixed_kernel import MixedDrive
        Drive=MixedDrive
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(291049);rng=np.random.default_rng(291049);device=torch.device('cuda:0')
    if a.scratch and a.init_checkpoint is not None:raise ValueError('Random initialization and a learned starting checkpoint are mutually exclusive')
    original=a.init_checkpoint.resolve() if a.init_checkpoint else root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt'
    ck=torch.load(original,map_location='cpu',weights_only=False);base=ck['model_state'];params={k:float(base[k]) for k in ['v_th','reset_amp','tau_reset']}
    if not a.two_kernels and any(k.startswith('aux_') for k in base):raise ValueError('The fixed architecture requires one temporal kernel per site')
    if ck.get('time_constant_units','ms')!='ms':raise ValueError('Starting checkpoint must store physical millisecond time constants')
    rows=[];old=root/'benchmarks/dt_resolution_20260929/train'
    if a.manifest:
        manifest_data=read(a.manifest);rows=manifest_data['cases']
    else:
        for r in read(old/'manifest.json')['cases']:
            rows.append({**r,'input_interval_ms':1.,'reference_file':str(old/'references'/(r['case_id']+'_reference.npz'))})
    if a.fresh and not a.manifest:
        assert dt==.5
        if not (work/'fresh/complete.json').exists():raise RuntimeError('Fresh references not finished')
        rows+=read(work/'fresh/manifest.json')['cases']
    source_rows=rows;excluded=[]
    if a.regime=='low-spike':
        retained=[]
        for r in rows:
            ref_meta=read(Path(r['reference_file']).with_suffix('.json'))
            # One window per independent source seed. Preserve its split and
            # use the longest admissible prefix. This covers the higher event
            # rates permitted by 1/2/4s tests without introducing recurrent
            # state resets inside a sampled trajectory or crossing splits.
            ss=np.array(ref_meta['spike_times_ms'])
            chosen=None
            for duration in [6000,4000,2000,1000]:
                if duration>r['duration_ms']:continue
                sp=ss[ss<=duration]
                if len(sp)<=3 and (len(sp)<2 or np.diff(sp).min()>=20):chosen=duration;break
            if chosen is not None:
                rr={**r,'source_case_id':r['case_id'],'source_duration_ms':r['duration_ms'],'source_input_sha256':r['input_sha256'],'duration_ms':chosen}
                if chosen<r['duration_ms']:rr['case_id']=r['case_id']+f'__prefix_{chosen}ms'
                retained.append(rr)
            else:excluded.append(dict(case_id=r['case_id'],split=r['split'],spike_count=ref_meta['spike_count'],burst=ref_meta['burst']))
        rows=retained
    fit=np.array([i for i,r in enumerate(rows) if r['split']=='fit']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    if len(fit)<8 or len(val)<4:raise RuntimeError(('Insufficient independently sampled low-spike fit/validation',len(fit),len(val)))
    T=6000*stride+1;nfft=1<<(2*T-2).bit_length()
    if a.input_cache=='fft':xf=torch.empty((len(rows),500,nfft//2+1),dtype=torch.complex64,device=device)
    else:xc=torch.empty((len(rows),500,T),dtype=torch.int8,device=device)
    def inputs(ii):
        if a.input_cache=='fft':return xf[ii]
        return torch.fft.rfft(xc[ii].float(),n=nfft)
    targets=[];spikes=[];lengths=[]
    for i,r in enumerate(rows):
        d=np.load(r['input_file']);ids=d['ids'];raw=d['x'];step=round(r['input_interval_ms']/dt);assert step>=1
        length=round(r['duration_ms']/dt)+1;input_length=round(r['duration_ms']/r['input_interval_ms'])+1;raw=raw[:,:input_length]
        dense=np.zeros((500,T),np.float32);dense[ids,:length:step]=raw*(1-2*(ids[:,None]>=250))
        r['input_sha256']=__import__('hashlib').sha256(raw.tobytes()).hexdigest();r['input_events']=int(raw.sum())
        if a.input_cache=='fft':xf[i]=torch.fft.rfft(torch.from_numpy(dense).to(device),n=nfft)
        else:
            assert np.max(abs(dense))<=127 and np.array_equal(dense,dense.astype(np.int8))
            xc[i]=torch.from_numpy(dense.astype(np.int8)).to(device)
        ref=np.load(r['reference_file']);reference_dt=float(r.get('reference_output_interval_ms',.5));sample_step=round(dt/reference_dt)
        assert sample_step>=1 and abs(sample_step*reference_dt-dt)<1e-8
        vv=ref['voltage'][::sample_step][:length];assert len(vv)==length
        padded=np.full(T,vv[-1],np.float32);padded[:length]=vv
        targets.append(padded);spikes.append(ref['spike_times'][ref['spike_times']<=r['duration_ms']]);lengths.append(length)
    target=np.stack(targets);masks=np.stack([mask(T,s,dt)&(np.arange(T)<length) for s,length in zip(spikes,lengths)]);wm=torch.from_numpy(masks.astype(np.float32)).to(device)
    variances=np.array([np.var(target[i,masks[i]].astype(float)) for i in range(len(rows))])
    case_weights=.3+.7/np.maximum(variances,.04);case_weights/=np.mean(case_weights[fit]);cw=torch.tensor(case_weights,dtype=torch.float32,device=device)
    init={k:v.clone() for k,v in base.items()}
    if a.scratch:
        # Independent random weights, physical initialization, no fitted parameters copied.
        init['tau_rise'].fill_(5.);init['tau_decay'].fill_(20.);init['omega'].fill_(.3)
        init['bilinear.weight'].uniform_(-1/math.sqrt(500),1/math.sqrt(500));init['bilinear.weight'][0].fill_diagonal_(0)
        init['bilinear.bias'].fill_(-70.);params=dict(v_th=-55.,reset_amp=15.,tau_reset=10.)
        for k,v in params.items():init[k].fill_(v)
    model=Drive(init,T,device);model.time=model.time.float()*dt
    def predict():
        with torch.inference_mode():return np.concatenate([model(inputs(val[j:j+4])).cpu().numpy() for j in range(0,len(val),4)])
    def score(drives,p,detail=False):
        mm=[];ee=[];norm=[]
        for j,i in enumerate(val):
            pred,ev=reset(drives[j,:lengths[i]],p['v_th'],p['reset_amp'],p['tau_reset']/dt)
            m=metric(target[i,:lengths[i]:stride],pred[::stride],spikes[i],1.);mm.append(m);ee.append(match_events(spikes[i],ev*dt))
            norm.append((m['sse']/m['n'])/max(m['reference_ss']/m['n'],.04))
        s=aggregate(mm);s['events']=event_aggregate(ee);s['normalized_mse_floor_0p04']=float(np.mean(norm))
        s['score']=s['pooled_rmse_mv']**2+.02*s['normalized_mse_floor_0p04']+.05*(1-s['events']['f1'])
        if detail:s['cases']=[dict(case_id=rows[i]['case_id'],voltage=m,events=e) for i,m,e in zip(val,mm,ee)]
        return s
    protocol=dict(created=datetime.now().isoformat(),dt_ms=dt,epochs=a.epochs,refinement_epochs=a.refinement_epochs,lr=a.lr,
        operating_regime=a.regime,source_cases=len(source_rows),excluded_cases=len(excluded),
        input_selection='NEURON-only 0–3 reference spikes, minimum ISI 20ms; one longest eligible 6/4/2/1s prefix per independent source seed. Original fit/validation group retained. Valid-time mask excludes padded tail. No surrogate errors used.' if a.regime=='low-spike' else 'All sampled activity ranges (diagnostic pilot).',
        duration_histogram_ms={str(z):sum(r['duration_ms']==z for r in rows) for z in [1000,2000,4000,6000]},
        fit_cases=len(fit),validation_cases=len(val),fresh=a.fresh,random_initialization=a.scratch,
        kernels_per_site=2 if a.two_kernels else 1,
        initialization='Independent random weights with tau_rise=5,tau_decay=20,omega=.3,bias=-70' if a.scratch else str(original),
        initial_checkpoint_sha256=None if a.scratch else sha(original),seed=291049,batch_size=8,time_constant_units='ms',
        training_loss='Per-case masked MSE weighted by 0.3+0.7/max(reference variance,0.04), normalized over fit cases; +0.1 similarly weighted squared physical derivative error.',
        validation_selection='Pooled masked MSE + 0.02 mean normalized MSE (variance floor .04) + .05*(1-event F1); common 1ms observation grid; events at +/-5ms.',
        reset_training=f'Teacher reset initially fixed; select calibrated reset on validation; rebuild teacher residual and refine for {a.refinement_epochs} epochs.',
        dataset=manifest_data['dataset_description'] if a.manifest else 'Old 117 fit/32 validation, plus fresh 72 fit/24 validation when enabled. Same original fully active L5PC and synapse locations.',
        input_cache=a.input_cache,manifest_sha256=sha(a.manifest) if a.manifest else None,
        precision='FP32 IEEE, AdamW wd1e-6, cosine to 2%lr, gradient clipping 1, EMA .995',
        test_used=False,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(out/'protocol.json',protocol);write(out/'training_manifest.json',dict(cases=rows));write(out/'reference_regime_selection.json',dict(regime=a.regime,included=[r['case_id'] for r in rows],excluded=excluded,locked_before_fitting=True));candidates=[];start=time.perf_counter()
    def calibrate(name,epoch):
        drives=predict();grid=[]
        for th in list(np.arange(float(base['bilinear.bias'])+4.,float(base['bilinear.bias'])+24.1,1.))+[params['v_th']]:
            for amp in [4.,6.,8.,10.,12.]:
                for tau in [8.,16.,24.,40.]:
                    pp=dict(v_th=float(th),reset_amp=amp,tau_reset=tau);grid.append(dict(params=pp,**score(drives,pp)))
        grid.append(dict(params=params.copy(),**score(drives,params)))
        win=min(grid,key=lambda r:r['score'])
        for delta in [-.375,-.25,-.125,0.,.125,.25,.375]:
            pp={**win['params'],'v_th':win['params']['v_th']+delta};grid.append(dict(params=pp,**score(drives,pp)))
        winner=min(grid,key=lambda r:r['score']);cp=out/'candidates'/name;cp.mkdir(parents=True,exist_ok=True)
        path=cp/'checkpoint.pt';torch.save(dict(model_state=model.original_state(base,winner['params']),sample_interval_ms=dt,time_constant_units='ms',protocol=protocol,candidate=name),path)
        rec=dict(name=name,epoch=epoch,checkpoint=str(path),checkpoint_sha256=sha(path),reset=winner['params'],validation=score(drives,winner['params'],True))
        write(cp/'validation.json',rec);write(cp/'reset_grid.json',grid);candidates.append(rec)
        print('CANDIDATE',name,'RMSE',rec['validation']['pooled_rmse_mv'],'F1',rec['validation']['events']['f1'],flush=True)
        return rec
    for stage,epochs,lr in [('primary',a.epochs,a.lr),('refine',a.refinement_epochs,a.lr*.2)]:
        if stage=='refine':
            best=min(candidates,key=lambda c:c['validation']['score']);params=best['reset'].copy()
            st=torch.load(best['checkpoint'],map_location='cpu',weights_only=False)['model_state'];model=Drive(st,T,device);model.time=model.time.float()*dt
        teacher=np.stack([teacher_reset(T,s/dt,params['reset_amp'],params['tau_reset']/dt) for s in spikes]).astype(np.float32)
        y=torch.from_numpy(target-teacher).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-6);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=lr*.02)
        ema=copy.deepcopy(model.state_dict());hist=[];bestj=bestv=float('inf');saved={}
        for epoch in range(epochs+1):
            losses=[]
            if epoch:
                for ii in np.array_split(rng.permutation(fit),math.ceil(len(fit)/8)):
                    opt.zero_grad(set_to_none=True);error=model(inputs(ii))-y[ii];w=wm[ii];dw=torch.minimum(w[:,1:],w[:,:-1])
                    per=((error.square()*w).sum(1)/w.sum(1))+.1*((((error[:,1:]-error[:,:-1])/dt).square()*dw).sum(1)/dw.sum(1))
                    loss=(per*cw[ii]).mean()
                    if not torch.isfinite(loss):raise RuntimeError(('nonfinite_loss',stage,epoch))
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(norm):raise RuntimeError(('nonfinite_grad',stage,epoch))
                    opt.step()
                    with torch.no_grad():
                        model.log_tau_rise.clamp_(math.log(.25),math.log(50.));model.log_tau_decay.clamp_(math.log(.5),math.log(100.))
                        if a.two_kernels:
                            model.log_aux_tau_rise.clamp_(math.log(.25),math.log(50.));model.log_aux_tau_decay.clamp_(math.log(.5),math.log(200.))
                        for k,v in model.state_dict().items():
                            if k not in ['time','offdiag']:ema[k].mul_(.995).add_(v,alpha=.005)
                    losses.append(float(loss.detach()))
                sched.step()
            if epoch%20==0 or epoch==epochs:
                s=score(predict(),params);rec=dict(stage=stage,epoch=epoch,loss=float(np.mean(losses)) if losses else None,validation=s,elapsed_s=time.perf_counter()-start)
                hist.append(rec)
                if s['score']<bestj:bestj=s['score'];saved['joint']=epoch;torch.save(model.state_dict(),out/(stage+'_joint.pt'))
                if s['pooled_rmse_mv']<bestv:bestv=s['pooled_rmse_mv'];saved['voltage']=epoch;torch.save(model.state_dict(),out/(stage+'_voltage.pt'))
                write(out/(stage+'_history.json'),hist);write(out/'progress.json',rec)
                print('EPOCH',stage,epoch,round(s['pooled_rmse_mv'],5),round(s['mean_case_ve'],5),round(s['events']['f1'],4),'sec',round(rec['elapsed_s'],1),flush=True)
        torch.save(ema,out/(stage+'_ema.pt'));saved['ema']=epochs
        for tag in ['joint','voltage','ema']:
            model.load_state_dict(torch.load(out/(stage+'_'+tag+'.pt'),map_location=device,weights_only=False));calibrate(stage+'_'+tag,saved[tag])
    winner=min(candidates,key=lambda c:c['validation']['score']);locked=datetime.now().isoformat()
    chosen=torch.load(winner['checkpoint'],map_location='cpu',weights_only=False);chosen['locked_before_test_at']=locked
    torch.save(chosen,out/'best_checkpoint.pt')
    write(out/'final_selection.json',dict(finished=True,selected=winner,candidates=candidates,protocol=protocol,locked_at=locked,checkpoint_sha256=sha(out/'best_checkpoint.pt'),elapsed_s=time.perf_counter()-start))
    print('COMPLETE',a.name,winner['name'],flush=True)

if __name__=='__main__':main()
