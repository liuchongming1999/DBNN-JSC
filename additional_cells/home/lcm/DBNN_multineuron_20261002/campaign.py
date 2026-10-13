"""Finite per-neuron experiment; no external learning baselines are launched."""
import argparse,fcntl,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from screen import write
from models import REGISTRY

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--model',choices=REGISTRY,required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
    work=a.work.resolve();data=a.data.resolve();root=a.root.resolve();out=work/'results'/a.model;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'campaign.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    state=dict(model=a.model,gpu=a.gpu,pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='started')
    def step(label,script,args,allow_invalid=False):
        state.update(status='running',phase=label,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state)
        cmd=[sys.executable,'-u',str(work/script),*map(str,args)]
        with open(out/(label+'.log'),'a') as f:ret=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if ret and not (allow_invalid and ret==2):raise RuntimeError((label,ret))
        print('STEP_COMPLETE',a.model,label,ret,flush=True)
    def sampling(profile,kind,dt):
        step(f'sample_{profile}_{kind}_{dt}', 'sample_panel.py',['--work',work,'--data',data,'--model',a.model,'--profile',profile,'--kind',kind,'--dt',dt,'--workers',4])
    def training(profile,dt,initial,manifest):
        step(f'train_{profile}_{dt}', 'train_accuracy.py',['--root',root,'--work',out/profile,'--name',f'dt{dt}ms','--dt',dt,'--manifest',manifest,'--input-cache','raster','--epochs',1000,'--lr',.001,'--init-checkpoint',initial])
        return out/profile/f'dt{dt}ms/best_checkpoint.pt'
    def merge(profile,dts):
        rows=[]
        for dt in dts:rows+=json.loads((data/a.model/profile/f'train_{dt}ms/manifest.json').read_text())['cases']
        path=out/profile/f'training_mix_{dts[-1]}ms.json';write(path,dict(cases=rows,dataset_description='Independent same-neuron panels: base 1ms events and added native fine-time Poisson events; original architecture; fit and validation seeds fixed before testing.'));return path
    try:
        for profile in ['ampa','article_expsyn','ampa_nmda']:
            step('screen_'+profile,'screen.py',['--work',work,'--model',a.model,'--profile',profile])
        for profile in ['ampa','ampa_nmda']:
            step('init_'+profile,'init_checkpoint.py',['--work',work,'--model',a.model,'--profile',profile])
            previous=out/profile/'init_checkpoint.pt'
            dts=[1.,.5,.25] if profile=='ampa' else [1.]
            for i,dt in enumerate(dts):
                sampling(profile,'train',dt);previous=training(profile,dt,previous,merge(profile,dts[:i+1]))
            sampling(profile,'test',1.)
            step('gpu_'+profile,'gpu_evaluate.py',['--root',root,'--work',work,'--data',data,'--model',a.model,'--profile',profile,'--repeats',30])
            manifest=data/a.model/profile/'test/manifest.json'
            # Conservative grid for these fast cerebellar channels; coarse
            # steps are recorded only if all traces remain finite/stable.
            for engine in ['multisplit','core']:
                for dt in [.05,.1,.25,.5,1.]:
                    step(f'solver_{profile}_{engine}_{dt}','solver_benchmark.py',['--root',root,'--work',work,'--model',a.model,'--profile',profile,'--manifest',manifest,'--engine',engine,'--dt',dt,'--repeats',5],allow_invalid=True)
        state.update(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state);write(out/'complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state);raise

if __name__=='__main__':main()
