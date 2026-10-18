"""Full frozen-cohort step curves; the source 40-case study remains intact."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
import json, os, subprocess, sys

ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK=ROOT/'benchmarks/solver_main379_20261016'

def write(name,obj):
    p=WORK/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)

def worker(engine,dt,shard,nshards,cpus):
    cmd=[sys.executable,str(WORK/'run_grid.py'),'--worker','--root',str(ROOT),'--work',str(WORK),
         '--engine',engine,'--dt',str(dt),'--shard',str(shard),'--shards',str(nshards),
         '--cpus',','.join(map(str,cpus)),'--repeats','3']
    env={**os.environ,'OMP_NUM_THREADS':'8' if engine=='multisplit8' else '1',
         'OMP_WAIT_POLICY':'PASSIVE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (WORK/'logs'/f'{engine}_{dt:g}_{shard}.log').open('a') as f:
        p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{engine}, {dt}, shard {shard}: {p.returncode}')
    return dict(engine=engine,dt_ms=dt,shard=shard,finished=True)

def learned(method,cpus,gpu=None):
    cp=ROOT/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
    pred=ROOT/'benchmarks/random_init_main379_20261012/predictions'
    cmd=[sys.executable,str(WORK/'benchmark_frozen_main379.py'),'--root',str(ROOT),
         '--work',str(ROOT/'benchmarks/spike_panel_20261009'),'--method',method,'--repeats','3',
         '--checkpoint',str(cp),'--checkpoint-sha256','737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d',
         '--predictions',str(pred),'--output',str(WORK/'learned_gpu5'/method),'--cpus',','.join(map(str,cpus))]
    env={**os.environ,'OMP_WAIT_POLICY':'PASSIVE','OPENBLAS_NUM_THREADS':'1'}
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (WORK/'logs'/f'learned_{method}.log').open('a') as f:
        p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'learned {method}: {p.returncode}')

def gpu_sequence():
    for method in ['gpu','tcn','elm','lstm']:learned(method,[80],gpu=0)


if __name__=="__main__":
    for method in ["gpu","tcn","elm","lstm"]:learned(method,[112],gpu=5)
    write(WORK/"gpu5_sequence_done.json",dict(finished=True))
