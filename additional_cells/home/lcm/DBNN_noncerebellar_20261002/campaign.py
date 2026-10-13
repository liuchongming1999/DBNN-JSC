"""Finite fixed-architecture extension; reference/GPU/CPU resources serialized."""
import argparse,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from models import REGISTRY
from common import write,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=REGISTRY,required=True);p.add_argument('--gpu',type=int,default=3);a=p.parse_args()
    work=Path(__file__).resolve().parent;root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');data=Path('/data/lcm/DBNN_noncerebellar_20261002')
    out=work/'results'/a.model;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'campaign.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():return
    state=dict(model=a.model,gpu=a.gpu,pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='running')
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=str(a.gpu))
    def step(label,script,args,cpu_pool=False,gpu=False,invalid=False):
        state.update(phase=label,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state)
        resources=[]
        try:
            if cpu_pool:
                state.update(resource_status='waiting for physical CPU pool 64-71');write(out/'state.json',state)
                f=open(work/'cpu_pool64_71.lock','w');fcntl.flock(f,fcntl.LOCK_EX);resources.append(f)
            if gpu:
                state.update(resource_status='waiting for GPU resource lock');write(out/'state.json',state)
                f=open(work/f'gpu{a.gpu}.lock','w');fcntl.flock(f,fcntl.LOCK_EX);resources.append(f)
                for i in range(3):
                    rows={int(r[0]):tuple(map(int,r[1:])) for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines())}
                    if rows[a.gpu][0]>5 or rows[a.gpu][1]>128:raise RuntimeError(('Selected GPU occupied',rows[a.gpu]))
                    if i<2:time.sleep(3)
            state.update(resource_status='executing');write(out/'state.json',state)
            command=[sys.executable,'-u',str(work/script),*map(str,args)]
            if cpu_pool and script!='sample_panel.py':command=['taskset','-c','64-71',*command]
            if gpu:command=['taskset','-c',str(80+a.gpu),*command]
            with open(out/(label+'.log'),'a') as f:rc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
            if rc and not (invalid and rc==2):raise RuntimeError((label,rc))
            print('STEP_COMPLETE',a.model,label,rc,flush=True)
        finally:
            for f in resources:f.close()
    try:
        # The initial screen establishes a physical operating point; its
        # reference step is selected before any independent training panels.
        step('screen_initial','screen.py',['--work',work,'--model',a.model,'--profile','ampa'],cpu_pool=True)
        step('reference_convergence','reference_convergence.py',['--model',a.model],cpu_pool=True)
        # Preserve the initial operating-point pilot if a finer reference is
        # selected, then repeat the same pair pilot at the selected step.
        from common import read
        chosen=read(out/'screen/operating_point.json')['reference_dt_ms']
        pilot=out/'screen/ampa.json'
        if pilot.exists() and read(pilot)['operating_point']['reference_dt_ms']!=chosen:
            pilot.rename(out/'screen/ampa_initial_reference_pilot.json')
            step('screen_selected_reference','screen.py',['--work',work,'--model',a.model,'--profile','ampa'],cpu_pool=True)
        for profile in ('ampa','ampa_nmda'):
            if profile=='ampa_nmda':step('screen_nmda','screen.py',['--work',work,'--model',a.model,'--profile',profile],cpu_pool=True)
            step('init_'+profile,'init_checkpoint.py',['--work',work,'--model',a.model,'--profile',profile],cpu_pool=True)
            previous=out/profile/'init_checkpoint.pt';dts=[1.,.5,.25] if profile=='ampa' else [1.]
            for i,dt in enumerate(dts):
                step(f'sample_{profile}_{dt}','sample_panel.py',['--work',work,'--data',data,'--model',a.model,'--profile',profile,'--kind','train','--dt',dt,'--workers',8],cpu_pool=True)
                rows=[]
                for d in dts[:i+1]:rows+=read(data/a.model/profile/f'train_{d}ms/manifest.json')['cases']
                manifest=out/profile/f'training_mix_{dt}ms.json'
                write(manifest,dict(cases=rows,dataset_description=f'Independent {a.model} {profile} panels: base 1ms events and added native fine-time Poisson events; original architecture; fit and validation seeds fixed before testing.'))
                step(f'train_{profile}_{dt}','train_accuracy.py',['--root',root,'--work',out/profile,'--name',f'dt{dt}ms','--dt',dt,'--manifest',manifest,'--input-cache','raster','--epochs',1000,'--lr',.001,'--init-checkpoint',previous],gpu=True)
                previous=out/profile/f'dt{dt}ms/best_checkpoint.pt'
            step('test_reference_'+profile,'sample_panel.py',['--work',work,'--data',data,'--model',a.model,'--profile',profile,'--kind','test','--workers',8],cpu_pool=True)
            step('gpu_'+profile,'gpu_evaluate.py',['--root',root,'--work',work,'--data',data,'--model',a.model,'--profile',profile,'--repeats',30],gpu=True)
        for profile in ('ampa','ampa_nmda'):
            for engine in ('multisplit','core'):
                for dt in (.25,.5,1.):
                    step(f'solver_{profile}_{engine}_{dt}','solver_benchmark.py',['--root',root,'--work',work,'--model',a.model,'--profile',profile,'--manifest',data/a.model/profile/'test/manifest.json','--engine',engine,'--dt',dt,'--repeats',5],cpu_pool=True,invalid=True)
        state.update(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state);write(out/'complete.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state);raise

if __name__=='__main__':main()
