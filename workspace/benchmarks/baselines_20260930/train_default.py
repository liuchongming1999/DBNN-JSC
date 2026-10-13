"""Default-author training recipes on the corresponding locked DBNN panels.

No EMA, derivative/variance-balanced losses, early stopping, LR search or model
capacity search. A smoke test is explicitly distinct from a full-budget run.
"""
import argparse,copy,json,math,os,sys,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from panel import rows_for,make_case,sample_batch,save,digest
from tcn import TCN
from job_lock import acquire_lock

class LSTM(nn.Module):
    def __init__(self):
        super().__init__();self.recurrent=nn.LSTM(500,50,batch_first=True);self.output=nn.Linear(50,2)
        # Keras-style classic LSTM: one trainable bias, Glorot input / orthogonal
        # recurrent initialization, unit forget bias. This gives the paper's
        # 265902 trainable parameters with its original 1278 input channels.
        with torch.no_grad():
            nn.init.xavier_uniform_(self.recurrent.weight_ih_l0);nn.init.orthogonal_(self.recurrent.weight_hh_l0)
            self.recurrent.bias_hh_l0.zero_();self.recurrent.bias_ih_l0.zero_();self.recurrent.bias_ih_l0[50:100]=1
            nn.init.xavier_uniform_(self.output.weight);self.output.bias.zero_()
        self.recurrent.bias_hh_l0.requires_grad_(False)
    def forward(self,x):return self.output(self.recurrent(x)[0])

def config(method,dt):
    common=dict(method=method,dt_ms=dt,batch_size=8,precision='FP32 IEEE',input_channels=500,seed=930291,training_process_optimized=False)
    if method=='elm':
        common.update(epochs=35,batches_per_epoch=10000,window_ms=500,burn_in_loss_ms=0,overlap_ms=150,lr=.0005,optimizer='Adam + per-update cosine',loss='0.5 BCE + 0.5 MSE, voltage scaled by 0.1',recipe='Official neuronio_train_script.py non-short-run defaults; 35x10000 updates and full-window loss. Paper Appendix B instead describes 30x11400 with 150ms loss burn-in; code recipe is used here.',model=dict(num_input=500,num_output=2,num_memory=20,num_branch=45,num_synapse_per_branch=100,input_to_synapse_routing='neuronio_routing',memory_tau_min=1.,memory_tau_max=150.,learn_memory_tau=False,delta_t=dt))
    elif method=='lstm':
        common.update(epochs=30,batches_per_epoch=11400,window_ms=500,burn_in_loss_ms=150,overlap_ms=150,lr=.0005,optimizer='Adam + per-update cosine',loss='0.5 BCE + 0.5 MSE, voltage scaled by 0.1',recipe='ICLR-2024 Appendix B / Table S10 LSTM-50. No public LSTM checkpoint provided; paper recipe, not a separately verified checkpoint replay.',model=dict(input=500,hidden_size=50,layers=1,outputs=2))
    else:
        common.update(epochs=2500,stages=250,epochs_per_stage=10,batches_per_epoch=288,validation_batches=144,window_ms=400,burn_in_loss_ms=0,overlap_ms=154,lr=.0001,optimizer='NAdam, epsilon 1e-7, momentum_decay .004; reset each stage',loss='BCE + soma MSE + whitened dendritic-PCA MSE, default five-stage weight schedule + kernel L2 1e-8',recipe='Author fit_CNN.py schedule, generator length 6*int(.2*128*6000/(8*400))=288 batches/epoch. Published seven-layer NMDA TCN architecture; no capacity search.',model=dict(kernels=[1+round((k-1)/dt) for k in [45,19,19,19,19,19,19]],widths=[128]*7,dendritic_outputs=20,context_support_ms=152,initializer_stddev=.002,BN_epsilon=.001,BN_momentum=.99))
    common['adaptations']=['500 fixed E/I input IDs instead of NeuronIO 1278 inputs; exact event multiplicity retained','Physical 400/500ms windows and 150/154ms overlaps retained; ELM delta_t adjusted; TCN kernels discretized to retain 152ms physical context','Corresponding DBNN fit/validation events; no main31 data in training, PCA or selection','Original output moment alignment replaced by one validation-fitted affine map, locked before held-out evaluation; raw results retained']
    return common

def model_for(work,cfg):
    if cfg['method']=='elm':
        sys.path.insert(0,str(work/'vendor/elmneuron'));from src.expressive_leaky_memory_neuron import ELM
        return ELM(**cfg['model'])
    if cfg['method']=='lstm':return LSTM()
    model=TCN(500,cfg['model']['kernels'],cfg['model']['widths'],dendritic_outputs=20)
    with torch.no_grad():
        for layer in model.layers:
            nn.init.trunc_normal_(layer.conv.weight,std=.002,a=-.004,b=.004);nn.init.zeros_(layer.conv.bias)
        for head,std,bias in [(model.spikes,.001,-2.),(model.somatic,.03,0.),(model.dendritic,.05,0.)]:
            nn.init.trunc_normal_(head.weight,std=std,a=-2*std,b=2*std);nn.init.constant_(head.bias,bias)
    return model

def stage_params(stage):
    if stage<40:return 1e-4,[1,.02,5e-6]
    if stage<80:return 3e-5,[2,.01,3e-6]
    if stage<120:return 1e-5,[4,.01,1e-6]
    if stage<160:return 3e-6,[8,.01,1e-8]
    return 1e-6,[9,.003,1e-9]

def losses(model,x,y,aux,cfg,stage):
    if cfg['method']=='tcn':out,dend=model.forward_all(x)
    else:out=model(x);dend=None
    burn=round(cfg['burn_in_loss_ms']/cfg['dt_ms']);out=out[:,burn:];y=y[:,burn:]
    bce=nn.functional.binary_cross_entropy_with_logits(out[...,0],y[...,0]);mse=nn.functional.mse_loss(out[...,1],y[...,1])
    if dend is None:return .5*bce+.5*mse,dict(bce=bce,mse=mse)
    ds=nn.functional.mse_loss(dend,aux);lr,weights=stage_params(stage)
    reg=sum(m.weight.square().sum() for m in model.modules() if isinstance(m,nn.Conv1d))*1e-8
    return weights[0]*bce+weights[1]*mse+weights[2]*ds+reg,dict(bce=bce,mse=mse,dendritic_mse=ds,regularization=reg)

def predict_case(model,x,cfg,device):
    window=round(cfg['window_ms']/cfg['dt_ms']);overlap=round(cfg['overlap_ms']/cfg['dt_ms']);n=len(x);output=[]
    # Author evaluation resets recurrence per overlapping window; include this work in timing.
    for start in range(0,n,window-overlap):
        end=min(n,start+window)
        if start and start+overlap>=n:break
        v=x[start:end][None]
        if v.shape[1]<window:v=nn.functional.pad(v,(0,0,0,window-v.shape[1]))
        y=model(v)[0,:end-start];burn=0 if start==0 else overlap;output.append(y[burn:])
    return torch.cat(output,0)

def evaluate_voltage(model,cases,cfg,device):
    model.eval();pred=[];refs=[]
    with torch.inference_mode():
        for c in cases:
            y=predict_case(model,torch.from_numpy(c['x']).to(device),cfg,device).cpu().numpy()
            p=y[:,1]*(10 if cfg['method']!='tcn' else 1)-67.7
            burn=round(500/cfg['dt_ms']);pred.append(p[burn:]);refs.append(np.minimum(c['raw_voltage'][burn:],-55))
    p=np.concatenate(pred).astype(float);r=np.concatenate(refs).astype(float)
    scale=float(r.std()/max(p.std(),1e-12));bias=float(r.mean()-scale*p.mean())
    return dict(raw_rmse_mv=float(np.sqrt(np.mean((p-r)**2))),aligned_rmse_mv=float(np.sqrt(np.mean((scale*p+bias-r)**2))),calibration=dict(scale=scale,bias_mv=bias,source='Complete validation set, clipped voltage after initial 500ms; never main31'))

def checkpoint(out,name,obj):
    tmp=out/(name+'.tmp');torch.save(obj,tmp);tmp.replace(out/name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--method',choices=['elm','tcn','lstm'],required=True);p.add_argument('--dt',type=float,choices=[1,.5,.25],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');p.add_argument('--elm-cuda-graph',action='store_true');a=p.parse_args()
    assert not a.elm_cuda_graph or (a.method=='elm' and a.device.startswith('cuda'))
    root=a.root.resolve();work=a.work.resolve();data=a.data.resolve();cfg=config(a.method,a.dt);out=work/'aligned'/f'{a.method}_{a.dt:g}ms';out.mkdir(parents=True,exist_ok=True)
    # New idle-GPU workers and the original queues share this lock. A worker
    # keeps its inherited lock through training and evaluation; no duplicate run.
    job_guard=acquire_lock(out/'training.lock',inherited_env='DBNN_TRAIN_LOCK_FD')
    if not a.smoke and (out/'complete.json').exists():return
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    rows,manifest=rows_for(root,a.dt);cfg.update(manifest=str(manifest),manifest_sha256=digest(manifest),fit_cases=sum(r['split']=='fit' for r in rows),validation_cases=sum(r['split']=='validation' for r in rows))
    if not a.smoke:
        gate=work/'original'/a.method/'complete.json'
        if a.method!='lstm':assert gate.exists() and json.loads(gate.read_text())['passed'],'Public checkpoint replay must pass first'
    dend=data/'tcn_aux'/f'pca_{a.dt:g}ms' if a.method=='tcn' else None
    if dend:assert (dend/'complete.json').exists(),'Default TCN dendritic supervision not prepared'
    used=[r for r in rows if r['split']=='fit'][:3] if a.smoke else rows
    cases=[make_case(r,a.dt,dend,signed=a.method!='tcn') for r in used]
    fit=[c for c in cases if c['split']=='fit'];val=[c for c in cases if c['split']=='validation']
    execution_stream=torch.cuda.Stream(device=a.device) if a.elm_cuda_graph else None
    if execution_stream:execution_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(execution_stream) if execution_stream else nullcontext():
        model=model_for(work,cfg).to(a.device)
    if execution_stream:torch.cuda.current_stream().wait_stream(execution_stream)
    cfg.update(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),source_versions=json.loads((work/'source_versions.json').read_text()))
    save(out/('smoke_config.json' if a.smoke else 'resolved_args.json'),cfg)
    total=cfg['epochs']*cfg['batches_per_epoch'];opt=None;scheduler=None;start_epoch=0;best=float('inf');history=[];steps=0;start=time.time()
    if a.method!='tcn':
        opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total)
    last=out/'last.pt'
    if last.exists() and not a.smoke:
        cp=torch.load(last,map_location=a.device,weights_only=False);assert cp['cfg']==cfg
        if execution_stream:execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(execution_stream) if execution_stream else nullcontext():
            model.load_state_dict(cp['model'])
        if execution_stream:torch.cuda.current_stream().wait_stream(execution_stream)
        start_epoch=cp['epoch']+1;best=cp['best'];history=cp['history'];steps=cp['steps'];rng.bit_generator.state=cp['numpy_rng'];torch.set_rng_state(cp['torch_rng'].cpu());torch.cuda.set_rng_state(cp['cuda_rng'].cpu())
        if a.method!='tcn':opt.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        elif start_epoch%10:
            opt=torch.optim.NAdam(model.parameters(),lr=stage_params(start_epoch//10)[0],eps=1e-7,momentum_decay=.004);opt.load_state_dict(cp['optimizer'])
    epochs=1 if a.smoke else cfg['epochs'];nsteps=3 if a.smoke else cfg['batches_per_epoch'];window=round(cfg['window_ms']/a.dt);execution_graph=None
    run_start_step=steps;elapsed_before_resume=sum(r['elapsed_s'] for r in history)
    for epoch in range(start_epoch,epochs):
        stage=epoch//10
        if a.method=='tcn' and (opt is None or epoch%10==0):opt=torch.optim.NAdam(model.parameters(),lr=stage_params(stage)[0],eps=1e-7,momentum_decay=.004)
        model.train();summed={};epoch_start=time.time()
        for k in range(nsteps):
            xx,yy,dd=sample_batch(fit,window,cfg['batch_size'],rng,a.dt);x=torch.from_numpy(xx).to(a.device);y=torch.from_numpy(yy).to(a.device);d=torch.from_numpy(dd).to(a.device) if dd is not None else None
            if a.elm_cuda_graph:
                if execution_graph is None:
                    from elm_cuda_graph import ELMForwardBackwardGraph
                    execution_graph=ELMForwardBackwardGraph(model,x,y,cfg,execution_stream)
                    save(out/('smoke_execution_runtime.json' if a.smoke else 'execution_runtime.json'),dict(backend='CUDA Graph: unchanged author forward/backward; original Adam/cosine outside graph',strategy_changed=False,author_model_changed=False,global_batch_size=cfg['batch_size'],resumed_step=steps,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),capture_s=execution_graph.capture_s,parity_probe=str(work/'execution_probe_20261001'/f'probe_{a.dt:g}ms_v5.json'),activated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
                loss,parts=execution_graph(x,y)
            else:
                opt.zero_grad(set_to_none=True);loss,parts=losses(model,x,y,d,cfg,stage);loss.backward()
            assert torch.isfinite(loss),('Non-finite loss',epoch,k)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Non-finite gradient'
            opt.step()
            if scheduler:scheduler.step()
            steps+=1
            for key,value in dict(loss=loss,**parts).items():summed[key]=summed.get(key,0)+float(value.detach())
            if (k+1)%100==0 or a.smoke:
                save(out/('smoke_progress.json' if a.smoke else 'progress.json'),dict(status='smoke' if a.smoke else 'training',epoch=epoch+1,total_epochs=cfg['epochs'],batch=k+1,batches_per_epoch=cfg['batches_per_epoch'],steps=steps,total_updates=total,loss=summed['loss']/(k+1),lr=opt.param_groups[0]['lr'],elapsed_s=elapsed_before_resume+time.time()-start,run_elapsed_s=time.time()-start,run_updates=steps-run_start_step,run_start_step=run_start_step,execution_backend='CUDA Graph author ELM' if a.elm_cuda_graph else 'Original author execution'))
                print('STEP',a.method,a.dt,epoch+1,k+1,summed['loss']/(k+1),flush=True)
        if a.smoke:
            save(out/'smoke.json',dict(passed=True,steps=3,finite_loss=True,finite_gradients=True,training_complete=False,elapsed_s=time.time()-start));return
        model.eval();metrics={}
        if a.method=='tcn':
            vals={};vrng=np.random.default_rng(cfg['seed']+epoch+1)
            with torch.inference_mode():
                for _ in range(cfg['validation_batches']):
                    xx,yy,dd=sample_batch(val,window,8,vrng,a.dt);loss,parts=losses(model,torch.from_numpy(xx).to(a.device),torch.from_numpy(yy).to(a.device),torch.from_numpy(dd).to(a.device),cfg,stage)
                    for key,value in dict(loss=loss,**parts).items():vals[key]=vals.get(key,0)+float(value)
            metrics={f'validation_{k}':v/cfg['validation_batches'] for k,v in vals.items()}
            # Original stage checkpoint filenames summarize last seven val spike losses.
            scores=[r['validation']['validation_bce'] for r in history[-6:]]+[metrics['validation_bce']];score=float(np.mean(scores));eligible=(epoch+1)%10==0
        else:
            metrics=evaluate_voltage(model,val,cfg,a.device);score=metrics['aligned_rmse_mv'];eligible=True
        history.append(dict(epoch=epoch+1,updates=steps,train={k:v/nsteps for k,v in summed.items()},validation=metrics,selection_score=score,lr=opt.param_groups[0]['lr'],elapsed_s=time.time()-epoch_start));save(out/'history.json',history)
        cp=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},cfg=cfg,epoch=epoch,steps=steps,best=best,history=history,optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        if eligible and score<best:
            best=score;cp['best']=best;checkpoint(out,'best.pt',cp)
        cp['best']=best;checkpoint(out,'last.pt',cp);print('EPOCH',epoch+1,metrics,flush=True)
    bestcp=torch.load(out/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(bestcp['model']);validation=evaluate_voltage(model,val,cfg,a.device)
    lock=dict(finished=True,default_budget_completed=True,epochs=cfg['epochs'],updates=steps,total_updates=total,selected_epoch=bestcp['epoch']+1,checkpoint=str(out/'best.pt'),checkpoint_sha256=digest(out/'best.pt'),validation=validation,config=cfg,locked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),elapsed_s=elapsed_before_resume+time.time()-start)
    save(out/'complete.json',lock);print('COMPLETE',a.method,a.dt,flush=True)

if __name__=='__main__':main()
