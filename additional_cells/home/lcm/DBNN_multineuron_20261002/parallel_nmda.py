"""Finite NMDA preparation on a free GPU; main controller retains CPU benchmarks."""
import argparse,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from common import read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--cpu',type=int,default=88);a=ap.parse_args()
    work=Path(__file__).resolve().parent;root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');data=Path('/data/lcm/DBNN_multineuron_20261002')
    base=work/'results'/a.model;profile=base/'ampa_nmda';profile.mkdir(exist_ok=True)
    with open(profile/'parallel_job.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (profile/'gpu/complete.json').exists():return
        os.sched_setaffinity(0,{a.cpu})
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        state={'pid':os.getpid(),'gpu':a.gpu,'model':a.model,'cpu':a.cpu,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'status':'waiting_reference'}
        def step(label,script,args):
            state.update(phase=label,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(base/'parallel_nmda_state.json',state)
            with open(base/('parallel_nmda_'+label+'.log'),'a') as f:
                subprocess.run([sys.executable,'-u',str(work/script),*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        try:
            while not all((data/a.model/'ampa_nmda'/kind/'complete.json').exists() for kind in ['train_1.0ms','test']):
                write(base/'parallel_nmda_state.json',state)
                campaign=read(base/'state.json')
                if campaign['status']=='failed':raise RuntimeError(campaign)
                time.sleep(15)
            state['status']='running'
            assert (base/'screen/ampa_nmda.json').exists(),'NMDA reference pilot not finished'
            step('init','init_checkpoint.py',['--work',work,'--model',a.model,'--profile','ampa_nmda'])
            observations=[]
            for i in range(3):
                text=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
                row={int(r[0]):(int(r[1]),int(r[2])) for r in csv.reader(text.splitlines())}[a.gpu]
                assert row[0]<=2 and row[1]<100,('Requested GPU occupied; no process stopped',row)
                observations.append({'time':time.time(),'gpu':a.gpu,'utilization_memory_mib':row})
                if i<2:time.sleep(3)
            state['gpu_observations']=observations
            manifest=profile/'training_mix_1.0ms.json'
            write(manifest,dict(cases=read(data/a.model/'ampa_nmda/train_1.0ms/manifest.json')['cases'],dataset_description='Independent same-neuron panels: base 1ms events and added native fine-time Poisson events; original architecture; fit and validation seeds fixed before testing.'))
            step('fit','train_accuracy.py',['--root',root,'--work',profile,'--name','dt1.0ms','--dt',1.,'--manifest',manifest,'--input-cache','raster','--epochs',1000,'--lr',.001,'--init-checkpoint',profile/'init_checkpoint.pt'])
            step('gpu','gpu_evaluate.py',['--root',root,'--work',work,'--data',data,'--model',a.model,'--profile','ampa_nmda','--repeats',30])
            state.update(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(base/'parallel_nmda_state.json',state)
        except Exception as exc:
            state.update(status='failed',error=repr(exc));write(base/'parallel_nmda_state.json',state);raise

if __name__=='__main__':main()
