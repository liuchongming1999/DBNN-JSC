"""Author-checkpoint replay on the complete public NeuronIO test set.

Retains author clipping, burn-in, and test-moment alignment for this *original*
protocol replay. L5PC transfer uses validation-only calibration instead.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,explained_variance_score
from tcn import load_public_h5,numpy_forward
from download_public import save,digest

def load_elm(work):
    sys.path.insert(0,str(work/'vendor/elmneuron'))
    from src.expressive_leaky_memory_neuron import ELM
    folder=work/'vendor/elmneuron/models/num_memory_20';cfg=json.loads((folder/'model_config.json').read_text());model=ELM(**cfg)
    cp=folder/'neuronio_best_model_state.pt';model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    return model,cfg,cp

def predict(model,x,method,device,window,overlap,batch=16):
    ns,t,channels=x.shape;pred=np.zeros((ns,t,2),np.float32)
    for start in range(0,t,max(1,window-overlap)):
        end=min(t,start+window)
        if start>0 and start+overlap>=t:break
        for j in range(0,ns,batch):
            v=torch.from_numpy(np.ascontiguousarray(x[j:j+batch,start:end])).to(device,dtype=torch.float32)
            if v.shape[1]<window:v=torch.nn.functional.pad(v,(0,0,0,window-v.shape[1]))
            if method=='elm':v[:,:,639:]*=-1
            y=model(v);out=y[:,:end-start].detach().cpu().numpy();out[:,:,0]=1/(1+np.exp(-out[:,:,0]))
            if method=='elm':out[:,:,1]*=10
            burn=0 if start==0 else overlap
            pred[j:j+batch,start+burn:end]=out[:,burn:]
    return pred

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--method',choices=['elm','tcn'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();work=a.work.resolve();out=work/'original'/a.method;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(work/'vendor/elmneuron'))
    from src.neuronio.neuronio_data_utils import parse_sim_experiment_file
    if a.method=='elm':model,cfg,cp=load_elm(work);window=500;overlap=150
    else:
        cp=a.data/'Models/NMDA_TCN__DWT_7_128_153__model.h5';model,cfg=load_public_h5(cp)
        layers=cfg['config']['layers'];inp=next(r['config'] for r in layers if r['class_name']=='InputLayer');shape=inp.get('batch_input_shape',inp.get('batch_shape'));window=int(shape[1]);overlap=min(max(model.history+1,min(150,window-50)),250)
        probe=np.random.default_rng(30091).poisson(.02,(2,165,1278)).astype(np.float32)
        with torch.inference_mode():y=model(torch.from_numpy(probe)).numpy()
        ny=numpy_forward(model,probe);diff=float(np.max(abs(y-ny)));assert diff<.001,(diff,'TCN weight/BN conversion mismatch')
        save(out/'implementation_check.json',dict(passed=True,torch_vs_independent_float64_max_abs=diff,window=window,overlap=overlap,config=cfg))
    model=model.to(a.device).eval();save(out/'config.json',dict(model=cfg,checkpoint_sha256=digest(cp),device=a.device,precision='FP32 IEEE',voltage_clip_mv=-55,ignore_first_ms=500,author_test_moment_alignment=True,transferred_protocol_uses_validation_only=True))
    rows=json.loads((a.data/'listing.json').read_text())['files'];names=sorted(r['name'] for r in rows if r['name'].startswith('Data_test/'))
    if a.limit:names=names[:a.limit]
    allpred=[];allref=[];allsp=[];start=time.time()
    for k,name in enumerate(names,1):
        f=a.data/name
        if not f.with_suffix(f.suffix+'.download.json').exists():raise RuntimeError(('File not completely downloaded',name))
        print('FILE',k,len(names),name,flush=True)
        X,sp,v=parse_sim_experiment_file(str(f));x=X.transpose(2,1,0);ref=np.minimum(v.T,-55)+67.7
        with torch.inference_mode():pred=predict(model,x,a.method,a.device,window,overlap)
        # ELM author helper aligns each file separately; TCN original aligns all files together below.
        if a.method=='elm':
            pred[:,:,1]=(pred[:,:,1]-pred[:,:,1].mean())/pred[:,:,1].std()*ref.std()+ref.mean()
        allpred.append(pred);allref.append(ref);allsp.append(sp.T)
        save(out/'progress.json',dict(files=k,total=len(names),last=name,elapsed_s=time.time()-start))
        del X,x,v
    pred=np.concatenate(allpred);ref=np.concatenate(allref);sp=np.concatenate(allsp)
    if a.method=='tcn':pred[:,:,1]=(pred[:,:,1]-pred[:,:,1].mean())/pred[:,:,1].std()*ref.std()+ref.mean()
    # Both author evaluators filter to 0--24 output spikes per simulation.
    sim_keep=(sp.sum(1)>=0)&(sp.sum(1)<=24)
    yp=pred[sim_keep,500:];r=ref[sim_keep,500:];ss=sp[sim_keep,500:]
    metrics=dict(rmse_mv=float(np.sqrt(np.mean((yp[:,:,1]-r)**2))),ve=float(explained_variance_score(r.ravel(),yp[:,:,1].ravel())),spike_auc=float(roc_auc_score(ss.ravel(),yp[:,:,0].ravel())),trajectories=int(sim_keep.sum()),samples=int(r.size))
    expected={'elm':dict(rmse_mv=.6375671602714604,spike_auc=.9921568089858758),'tcn':dict(rmse_mv=.71,spike_auc=.9911)}[a.method]
    full=not a.limit and len(names)==12
    passed=full and abs(metrics['rmse_mv']-expected['rmse_mv'])<.08 and abs(metrics['spike_auc']-expected['spike_auc'])<.003
    result=dict(finished=True,passed=passed,full_public_test=full,files=names,metrics=metrics,expected=expected,acceptance='All 12 public in-distribution test files; RMSE within 0.08 mV and AUC within 0.003 of declared paper/repository target; original scoring. Failed gate requires investigation before main comparison.',elapsed_s=time.time()-start)
    save(out/'complete.json' if full else out/'partial.json',result);print(json.dumps(result,indent=2),flush=True)
    if full and not passed:raise SystemExit(2)

if __name__=='__main__':main()
