#!/usr/bin/env python3
"""Matched double-exponential DBNN adaptation for three L5PC channel profiles.

Fits unreset drive outside reference AP +/-10 ms using teacher reset on fit
data; selects an epoch on validation drive loss, then calibrates closed-loop
reset on validation only. The final 10 test traces are evaluated after locking.
All profiles share input rasters, splits, initialization and optimizer budget.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

POWERS = {}

def mask_for(n, spikes):
    t=np.arange(n,dtype=float);m=np.ones(n,bool)
    for s in spikes:m &= ~((t>=s-10)&(t<=s+10))
    return m


def reset(drive,threshold,amp,tau):
    d=np.asarray(drive,float);pred=np.empty_like(d);alpha=np.exp(-1./tau)
    key=(len(d),tau)
    if key not in POWERS:POWERS[key]=alpha**np.arange(len(d))
    powers=POWERS[key];start=0;r=0.;spikes=[]
    while start<len(d):
        v=d[start:]+r*powers[:len(d)-start];ii=np.flatnonzero(v>=threshold)
        if not len(ii):pred[start:]=v;break
        k=int(ii[0]);end=start+k;pred[start:end+1]=v[:k+1]
        r=alpha*r*powers[k]-amp;spikes.append(end);start=end+1
    return pred,np.asarray(spikes,float)


def teacher_reset(n,spikes,amp,tau):
    bins={int(round(float(s))) for s in spikes};alpha=np.exp(-1./tau);r=0.;out=np.zeros(n)
    for i in range(n):out[i]=r;r=alpha*r-(amp if i in bins else 0.)
    return out


def metrics(ref,pred,spikes):
    keep=mask_for(len(ref),spikes);r=np.asarray(ref,float)[keep];p=np.asarray(pred,float)[keep];e=p-r
    return dict(n=int(keep.sum()),rmse_mv=float(np.sqrt(np.mean(e*e))),ve=float(1-e.var()/r.var()),
                bias_mv=float(e.mean()),sse=float(np.sum(e*e)),error_ss=float(np.sum((e-e.mean())**2)),reference_ss=float(np.sum((r-r.mean())**2)))


def aggregate(rows):
    return dict(cases=len(rows),pooled_rmse_mv=float(np.sqrt(sum(x['sse'] for x in rows)/sum(x['n'] for x in rows))),
                mean_case_ve=float(np.mean([x['ve'] for x in rows])),
                within_trace_weighted_ve=float(1-sum(x['error_ss'] for x in rows)/sum(x['reference_ss'] for x in rows)))


class Drive(nn.Module):
    def __init__(self,state,T,device):
        super().__init__();self.T=T;self.nfft=1<<(2*T-2).bit_length()
        for name in ['tau_rise','tau_decay']:
            setattr(self,'log_'+name,nn.Parameter(state[name].to(device).clone().log()))
        self.omega=nn.Parameter(state['omega'].to(device).clone())
        self.weight=nn.Parameter(state['bilinear.weight'][0].to(device).clone());self.bias=nn.Parameter(state['bilinear.bias'].to(device).clone())
        self.register_buffer('time',torch.arange(T,device=device));self.register_buffer('offdiag',1-torch.eye(500,device=device))

    def forward(self,xf):
        tr=self.log_tau_rise.exp()[:,None];td=self.log_tau_decay.exp()[:,None];w=self.omega[:,None]
        k=w*(1-torch.exp(-self.time/tr))*torch.exp(-self.time/td)
        kf=torch.fft.rfft(k,n=self.nfft,dim=-1)
        y=torch.fft.irfft(xf*kf[None],n=self.nfft,dim=-1)[...,:self.T].transpose(1,2).contiguous()
        return ((y@(self.weight*self.offdiag))*y).sum(-1)+y.sum(-1)+self.bias

    def original_state(self,base,reset_params):
        s={k:v.detach().cpu().clone() for k,v in base.items()}
        for key in ['tau_rise','tau_decay']:s[key]=getattr(self,'log_'+key).detach().exp().cpu().clone()
        s['omega']=self.omega.detach().cpu().clone()
        s['bilinear.weight']=(self.weight*self.offdiag).detach().cpu()[None].clone();s['bilinear.bias']=self.bias.detach().cpu().clone()
        for k,v in reset_params.items():s[k]=torch.tensor(v,dtype=torch.float32)
        return s


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=1000);ap.add_argument('--lr',type=float,default=.002);ap.add_argument('--fit-tag',default='fit_logtau_1000')
    a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');source=np.load(root/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    ck=torch.load(root/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False);base=ck['model_state'];fit=np.array(ck['fit_indices']);val=np.array(ck['val_indices']);test=np.arange(100,110)
    raw=np.concatenate([source['train_ip'],source['test_ip']]);signs=np.where(source['channel_type']=='exc',1.,-1.).astype(np.float32);T=raw.shape[-1];nfft=1<<(2*T-2).bit_length()
    xf=torch.empty((len(raw),500,nfft//2+1),device=device,dtype=torch.complex64)
    for i in range(0,len(raw),4):xf[i:i+4]=torch.fft.rfft(torch.from_numpy(raw[i:i+4].astype(np.float32)*signs[None,:,None]).to(device),n=nfft,dim=-1)
    protocol=dict(profiles=['full_active','passive_dendrites','all_passive'],fit_indices=fit.tolist(),validation_indices=val.tolist(),test_indices=list(range(100,110)),
                  input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((root/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
                  architecture='Original per-channel double-exponential filters + dense zero-diagonal quadratic + sum + bias + event reset',
                  epochs=a.epochs,lr=a.lr,batch_size=8,optimizer='Adam weight_decay=1e-6 + cosine',seed=20260425,
                  scoring='Reference AP +/-10 ms removed from both RMSE and both VE variances; per-case VE averaged',
                  training='Fit unreset voltage with teacher resets (amp/tau from frozen checkpoint), masked MSE + 0.1 masked first-difference MSE. Fit-only mean bias calibration. Select checkpoint by validation drive loss, then reset parameters by closed-loop masked validation MSE. Test evaluated only after locking.',
                  precision='FP32, no TF32',resource='Shared GPU; training times are not speed benchmark results')
    protocol['kernel_time_constant_parameterization']='Log of tau, constrained to original physical range; same double-exponential model'
    fitout=out/a.fit_tag;fitout.mkdir(exist_ok=True)
    (fitout/'training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def predict(model,indices):
        with torch.inference_mode():return np.concatenate([model(xf[indices[i:i+4]]).cpu().numpy() for i in range(0,len(indices),4)])
    for profile in protocol['profiles']:
        path=fitout/profile;path.mkdir(exist_ok=True)
        if (path/'summary.json').exists():print(profile,'already completed',flush=True);continue
        data_path=root/'data/shai_ei_500syn_6s_dataset.npz' if profile=='full_active' else out/profile/'dataset.npz'
        if not data_path.exists():raise FileNotFoundError(f'Sampling not complete: {data_path}')
        d=source if profile=='full_active' else np.load(data_path,allow_pickle=True)
        target=np.concatenate([d['train_op'],d['test_op']]).astype(np.float32);spikes=list(d['train_spike_time'])+list(d['test_spike_time'])
        masks=np.stack([mask_for(T,s) for s in spikes]);amp0=float(base['reset_amp']);tau0=float(base['tau_reset']);th0=float(base['v_th'])
        unreset=np.stack([r-teacher_reset(T,s,amp0,tau0) for r,s in zip(target,spikes)]).astype(np.float32)
        y=torch.from_numpy(unreset).to(device);wm=torch.from_numpy(masks.astype(np.float32)).to(device)
        torch.manual_seed(20260425);rng=np.random.default_rng(20260425);model=Drive(base,T,device)
        frozen_drive=predict(model,np.arange(110));frozen_test=[]
        for i in test:
            p,ev=reset(frozen_drive[i],th0,amp0,tau0);mm=metrics(target[i],p,spikes[i]);mm.update(index=int(i),reference_spikes=len(spikes[i]),predicted_resets=len(ev));frozen_test.append(mm)
        with torch.no_grad():
            bshift=float(np.sum((unreset[fit]-frozen_drive[fit])*masks[fit])/masks[fit].sum());model.bias.add_(bshift)
        opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-6);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*.05)
        def loss(pred,ix):
            w=wm[ix];tar=y[ix];e=pred-tar;dw=torch.minimum(w[:,1:],w[:,:-1]);return (e.square()*w).sum()/w.sum()+.1*((e[:,1:]-e[:,:-1]).square()*dw).sum()/dw.sum()
        def validate():
            with torch.inference_mode():return float(np.mean([float(loss(model(xf[ii]),ii)) for ii in [val[i:i+4] for i in range(0,len(val),4)]]))
        bestloss=validate();best=copy.deepcopy(model.state_dict());bestepoch=0;hist=[];start=time.perf_counter()
        print(profile,'initial validation drive loss',bestloss,'bias shift',bshift,flush=True)
        for epoch in range(1,a.epochs+1):
            order=rng.permutation(fit);vals=[]
            for i in range(0,len(order),8):
                ii=order[i:i+8];opt.zero_grad(set_to_none=True);ll=loss(model(xf[ii]),ii)
                if not torch.isfinite(ll):raise RuntimeError(f'{profile} nonfinite epoch {epoch}')
                ll.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                with torch.no_grad():model.log_tau_rise.clamp_(math.log(.25),math.log(50.));model.log_tau_decay.clamp_(math.log(.5),math.log(100.))
                vals.append(float(ll.detach()))
            sched.step()
            if epoch%5==0 or epoch==a.epochs:
                vl=validate();row=dict(epoch=epoch,fit_loss=float(np.mean(vals)),validation_drive_loss=vl,elapsed_s=time.perf_counter()-start);hist.append(row)
                if vl<bestloss:bestloss=vl;bestepoch=epoch;best=copy.deepcopy(model.state_dict())
                (path/'history.json').write_text(json.dumps(hist,indent=2)+'\n');print(profile,row,flush=True)
        model.load_state_dict(best);drives=predict(model,np.arange(110))
        candidates=[]
        thresholds=list(np.arange(-64.,-47.9,1.))+[-40.,-20.,float(th0)]
        if sum(len(spikes[i]) for i in val)==0:thresholds=[float(np.max(drives[val])+1.)]
        for th in thresholds:
            for amp in [6.,8.,10.,amp0]:
                for tau in [8.,10.,12.,16.,tau0]:
                    ss=0.;nn=0;count_error=0
                    for i in val:
                        p,ev=reset(drives[i],th,amp,tau);e=(p-target[i])[masks[i]];ss+=float(e@e);nn+=len(e);count_error+=abs(len(ev)-len(spikes[i]))
                    candidates.append(dict(v_th=float(th),reset_amp=float(amp),tau_reset=float(tau),masked_val_mse=ss/nn,event_count_mae=count_error/len(val)))
        best_reset=min(candidates,key=lambda r:(r['masked_val_mse'],r['event_count_mae']));params={k:best_reset[k] for k in ['v_th','reset_amp','tau_reset']}
        state=model.original_state(base,params)
        torch.save(dict(model_state=state,channel_type=source['channel_type'].tolist(),fit_indices=fit.tolist(),val_indices=val.tolist(),dataset=str(data_path),config=protocol,profile=profile),path/'dbnn_best.pt')
        test_rows=[];predictions=[];events=[]
        for i in test:
            p,ev=reset(drives[i],params['v_th'],params['reset_amp'],params['tau_reset']);rr=metrics(target[i],p,spikes[i]);rr.update(index=int(i),reference_spikes=len(spikes[i]),predicted_resets=len(ev));test_rows.append(rr);predictions.append(p);events.append(ev.tolist())
        summary=dict(finished=True,profile=profile,created=datetime.now().isoformat(),protocol=protocol,adaptation_seconds=time.perf_counter()-start,best_epoch=bestepoch,best_validation_drive_loss=bestloss,
                     reset_selection=best_reset,bias_shift_on_fit_mv=bshift,final_bias_mv=float(model.bias.detach()),frozen_test=aggregate(frozen_test),adapted_test=aggregate(test_rows),
                     frozen_test_cases=frozen_test,test_cases=test_rows,reference_test_spikes=sum(len(spikes[i]) for i in test),predicted_test_resets=sum(len(x) for x in events))
        np.savez_compressed(path/'test_predictions.npz',reference=target[test],prediction=np.stack(predictions),spikes=np.array([list(spikes[i]) for i in test],object),predicted_events=np.array(events,object))
        (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(profile,'COMPLETE',summary['frozen_test'],summary['adapted_test'],flush=True)
        del model,opt,best


if __name__=='__main__':main()
