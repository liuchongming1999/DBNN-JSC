"""Validation-only training search for an unchanged, deployable DBNN.

No main31 or held-out-test voltage enters optimization or model selection.
Selection score, fixed before evaluation: masked validation MSE + .5*(1-F1@5ms).
"""
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

def match_events(ref,pred,tol=5.):
    ref=sorted(map(float,ref));pred=sorted(map(float,pred));i=j=0;dt=[]
    while i<len(ref) and j<len(pred):
        if pred[j]<ref[i]-tol:j+=1
        elif ref[i]<pred[j]-tol:i+=1
        else:dt.append(pred[j]-ref[i]);i+=1;j+=1
    return dict(tp=len(dt),fp=len(pred)-len(dt),fn=len(ref)-len(dt),matched_errors_ms=dt)

def event_aggregate(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    dt=[x for r in rows for x in r['matched_errors_ms']]
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,median_abs_error_ms=float(np.median(np.abs(dt))) if dt else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(exist_ok=True)
    sys.path[:0]=[str(root/'benchmarks/channel_ablation_20260929')]
    from train_channel_ablation import Drive,reset,teacher_reset,mask_for,metrics,aggregate
    data=out/'data'
    if not (data/'complete.json').exists():raise RuntimeError('Sampling must complete before fitting')
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['cases'];N=len(rows);T=6001
    original=torch.load(root/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)
    initial=torch.load(root/'benchmarks/channel_ablation_20260929/fit_logtau_1000/full_active/dbnn_best.pt',map_location='cpu',weights_only=False)
    base=initial['model_state'];amp0=float(base['reset_amp']);tau0=float(base['tau_reset']);th0=float(base['v_th'])
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');sign=np.where(np.array(original['channel_type'])=='exc',1.,-1.).astype(np.float32)
    xf=torch.empty((N,500,8193),device=device,dtype=torch.complex64);target=[];spikes=[]
    for i,r in enumerate(rows):
        inp=np.load(data/r['input_file']);raw=np.zeros((500,T),np.float32);raw[inp['ids']]=inp['x'];raw*=sign[:,None]
        xf[i]=torch.fft.rfft(torch.from_numpy(raw).to(device),n=16384,dim=-1)
        ref=np.load(data/'references'/(r['case_id']+'_reference.npz'));target.append(ref['voltage']);spikes.append(ref['spike_times'])
    target=np.stack(target).astype(np.float32);masks=np.stack([mask_for(T,s) for s in spikes])
    fit=np.array([i for i,r in enumerate(rows) if r['split']=='fit']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);fit_original=np.array([i for i in fit if rows[i]['kind']=='quantized_original'])
    teacher=np.stack([teacher_reset(T,s,amp0,tau0) for s in spikes]).astype(np.float32);unreset=target-teacher
    pos=np.zeros((N,T),np.float32);negative=np.ones((N,T),np.float32)
    for i,sps in enumerate(spikes):
        for sp in sps:
            k=int(round(float(sp)));pos[i,k]=1;negative[i,max(0,k-5):min(T,k+6)]=0
    y=torch.from_numpy(unreset).to(device);wm=torch.from_numpy(masks.astype(np.float32)).to(device);rt=torch.from_numpy(teacher).to(device);pm=torch.from_numpy(pos).to(device);nm=torch.from_numpy(negative).to(device)
    variances=np.array([target[i,masks[i]].var() for i in range(N)],np.float32);scale=torch.from_numpy(np.maximum(variances,.25)).to(device)
    trials=[dict(name='quantized_voltage',epochs=800,lr=.0005,diverse=False,event=0.,balanced=False),
            dict(name='diverse_voltage',epochs=1000,lr=.001,diverse=True,event=0.,balanced=False),
            dict(name='diverse_event',epochs=1000,lr=.001,diverse=True,event=.2,balanced=False),
            dict(name='balanced_event',epochs=1000,lr=.001,diverse=True,event=.5,balanced=True)]
    protocol=dict(created=datetime.now().isoformat(),trials=trials,fit_case_ids=[rows[i]['case_id'] for i in fit],validation_case_ids=[rows[i]['case_id'] for i in val],
        initialization=str(root/'benchmarks/channel_ablation_20260929/fit_logtau_1000/full_active/dbnn_best.pt'),architecture='Original DBNN double-exponential filters, zero-diagonal dense quadratic, linear sum, bias, event reset',
        selection_rule=manifest['selection_rule'],event_tolerance_ms=5.,voltage_mask_ms=[10,10],precision='FP32 no TF32',batch_size=8,seed=20260929,
        loss='Masked per-trace voltage MSE + 0.1 first-difference MSE; optional reference-event threshold hinge + hard-negative threshold penalty. Balanced trial uses 0.5 absolute + 0.5 variance-normalized voltage MSE (variance floor .25 mV^2). EMA decay .995 after each update.',
        fit_target='Exact quantized-event NEURON replay; teacher reset from full-active adapted initialization; ref AP +/-10 ms removed for voltage loss. Event labels retain AP times.',
        training_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def predict(model,ids):
        with torch.inference_mode():return np.concatenate([model(xf[ids[i:i+4]]).cpu().numpy() for i in range(0,len(ids),4)])
    def score(drives,params,ids=val,details=False):
        scores=[];events=[]
        for k,i in enumerate(ids):
            pred,ev=reset(drives[k],params['v_th'],params['reset_amp'],params['tau_reset']);r=metrics(target[i],pred,spikes[i]);ee=match_events(spikes[i],ev);scores.append(r);events.append(ee)
        ss=aggregate(scores);ss['events']=event_aggregate(events);ss['score']=ss['pooled_rmse_mv']**2+.5*(1-ss['events']['f1'])
        if details:ss['cases']=[dict(case_id=rows[i]['case_id'],voltage=r,events=e) for i,r,e in zip(ids,scores,events)]
        return ss
    params0={k:float(base[k]) for k in ['v_th','reset_amp','tau_reset']}
    candidates=[]
    def calibrate(name,model,source):
        path=out/'candidates'/name;path.mkdir(parents=True,exist_ok=True)
        if (path/'validation.json').exists():
            result=json.loads((path/'validation.json').read_text());candidates.append(result);return
        drives=predict(model,val);grid=[]
        for th in list(np.arange(-60.,-53.9,.5))+[th0]:
            for amp in [6.,8.,10.]:
                for tau in [10.,16.,24.]:
                    pp=dict(v_th=float(th),reset_amp=amp,tau_reset=tau);ss=score(drives,pp);grid.append(dict(params=pp,**ss))
        # Include the checkpoint's unchanged reset and refine near the best grid point.
        grid.append(dict(params=params0,**score(drives,params0)));win=min(grid,key=lambda r:r['score'])
        for delta in [-.375,-.25,-.125,0.,.125,.25,.375]:
            pp={**win['params'],'v_th':win['params']['v_th']+delta};grid.append(dict(params=pp,**score(drives,pp)))
        best=min(grid,key=lambda r:r['score']);state=model.original_state(base,best['params'])
        torch.save(dict(model_state=state,channel_type=original['channel_type'],fit_indices=original['fit_indices'],val_indices=original['val_indices'],protocol=protocol,candidate=name,selection_source=source),path/'checkpoint.pt')
        result=dict(name=name,source=source,checkpoint=str(path/'checkpoint.pt'),checkpoint_sha256=hashlib.sha256((path/'checkpoint.pt').read_bytes()).hexdigest(),validation=score(drives,best['params'],details=True),reset=best['params'])
        np.savez_compressed(path/'validation_drive.npz',drive=drives,indices=val)
        (path/'reset_grid.json').write_text(json.dumps(grid,indent=2)+'\n');(path/'validation.json').write_text(json.dumps(result,indent=2)+'\n');candidates.append(result)
        print('CANDIDATE',name,'score',result['validation']['score'],'RMSE',result['validation']['pooled_rmse_mv'],'F1',result['validation']['events']['f1'],flush=True)
    # Both earlier checkpoints remain visible controls, calibrated on the same validation panel.
    for name,state in [('original_checkpoint',original['model_state']),('previous_1000epoch',base)]:
        model=Drive(state,T,device);calibrate(name,model,'Earlier weights; validation-only reset calibration');del model
    for trial in trials:
        path=out/'training'/trial['name'];path.mkdir(parents=True,exist_ok=True)
        if (path/'completed.json').exists():
            for name in ['best_joint','best_voltage','ema']:
                model=Drive(base,T,device);model.load_state_dict(torch.load(path/(name+'.pt'),map_location=device,weights_only=False));calibrate(trial['name']+'_'+name,model,dict(trial=trial,epoch_policy=name));del model
            continue
        torch.manual_seed(20260929);rng=np.random.default_rng(20260929);model=Drive(base,T,device);indices=fit if trial['diverse'] else fit_original
        opt=torch.optim.AdamW(model.parameters(),lr=trial['lr'],weight_decay=1e-6);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,trial['epochs'],eta_min=trial['lr']*.02)
        ema=copy.deepcopy(model.state_dict());bestjoint=float('inf');bestvoltage=float('inf');hist=[];start=time.perf_counter()
        def loss(pred,ix):
            e=pred-y[ix];w=wm[ix];mse=(e.square()*w).sum(1)/w.sum(1);vw=.5+.5/scale[ix] if trial['balanced'] else 1.
            dw=torch.minimum(w[:,1:],w[:,:-1]);dif=((e[:,1:]-e[:,:-1]).square()*dw).sum()/dw.sum();ll=(mse*vw).mean()+.1*dif
            if trial['event']:
                pre=pred+rt[ix];positive=(torch.relu(th0+.15-pre).square()*pm[ix]).sum()/pm[ix].sum().clamp_min(1)
                negative=(torch.relu(pre-(th0-.15)).square()*nm[ix]).sum()/pm[ix].sum().clamp_min(4)
                ll=ll+trial['event']*(positive+.05*negative)
            return ll
        for epoch in range(1,trial['epochs']+1):
            losses=[]
            for ii in np.array_split(rng.permutation(indices),math.ceil(len(indices)/8)):
                opt.zero_grad(set_to_none=True);ll=loss(model(xf[ii]),ii)
                if not torch.isfinite(ll):raise RuntimeError((trial,epoch,'nonfinite'))
                ll.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                with torch.no_grad():
                    model.log_tau_rise.clamp_(math.log(.25),math.log(50.));model.log_tau_decay.clamp_(math.log(.5),math.log(100.))
                    for k,v in model.state_dict().items():
                        if k in ['time','offdiag']:continue
                        ema[k].mul_(.995).add_(v,alpha=.005)
                losses.append(float(ll.detach()))
            sched.step()
            if epoch%50==0 or epoch==trial['epochs']:
                dr=predict(model,val);sc=score(dr,params0);rec=dict(epoch=epoch,fit_loss=float(np.mean(losses)),elapsed_s=time.perf_counter()-start,validation_nominal=sc);hist.append(rec)
                if sc['score']<bestjoint:bestjoint=sc['score'];torch.save(model.state_dict(),path/'best_joint.pt')
                if sc['pooled_rmse_mv']<bestvoltage:bestvoltage=sc['pooled_rmse_mv'];torch.save(model.state_dict(),path/'best_voltage.pt')
                (path/'history.json').write_text(json.dumps(hist,indent=2)+'\n');print(trial['name'],epoch,'val RMSE',round(sc['pooled_rmse_mv'],4),'F1',round(sc['events']['f1'],4),'s',round(rec['elapsed_s'],1),flush=True)
        torch.save(ema,path/'ema.pt');(path/'completed.json').write_text(json.dumps(dict(finished=True,trial=trial,elapsed_s=time.perf_counter()-start))+'\n');del opt,model
        for name in ['best_joint','best_voltage','ema']:
            model=Drive(base,T,device);model.load_state_dict(torch.load(path/(name+'.pt'),map_location=device,weights_only=False));calibrate(trial['name']+'_'+name,model,dict(trial=trial,epoch_policy=name));del model
    winner=min(candidates,key=lambda r:r['validation']['score']);chosen=torch.load(winner['checkpoint'],map_location='cpu',weights_only=False)
    chosen['locked_before_test_at']=datetime.now().isoformat();torch.save(chosen,out/'selected_checkpoint.pt')
    result=dict(finished=True,locked_before_test_at=chosen['locked_before_test_at'],selected=winner,candidates=candidates,protocol=protocol,selected_checkpoint=str(out/'selected_checkpoint.pt'),sha256=hashlib.sha256((out/'selected_checkpoint.pt').read_bytes()).hexdigest())
    (out/'selection.json').write_text(json.dumps(result,indent=2)+'\n');print('LOCKED',winner['name'],winner['validation']['score'],flush=True)

if __name__=='__main__':main()
