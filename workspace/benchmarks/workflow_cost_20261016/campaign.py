"""Bounded timing campaign, no training, original evidence kept untouched."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,os,json,time,datetime
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
PY=R/'.venv/bin/python'
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def job(name,args,gpu=None):
    command=[str(PY),'-u',str(W/'benchmark_cost.py'),*args]
    env={**os.environ,'OMP_WAIT_POLICY':'PASSIVE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else ''}
    with (W/'logs'/f'{name}.log').open('a') as log:
        t=time.perf_counter();p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(name=name,returncode=p.returncode,elapsed_s=time.perf_counter()-t)
    write(W/'jobs'/f'{name}.json',result)
    if p.returncode:raise RuntimeError(result)
    return result
def gpu_and_cold():
    done=[job('dbnn_gpu',['--mode','dbnn','--method','gpu','--cpus','112'],gpu=0)]
    # First-call runs use independent interpreters; runtime/compiler disk caches
    # are retained. Input file deserialization is not part of service-call time.
    protocol=json.loads((W/'protocol.json').read_text())
    for method,cpus,gpu in [('gpu','112',0),('cpu8','104,105,106,107,108,109,110,111',None),('neuron8','104,105,106,107,108,109,110,111',None)]:
        for cid in protocol['cold_case_ids']:
            for rep in range(3):
                output=W/'cold'/method/f'{cid}_rep{rep}.json'
                if output.exists():continue
                args=['--mode','solver' if method=='neuron8' else 'cold','--method',method,'--cpus',cpus,
                      '--case',cid,'--output',str(output)]
                if method=='neuron8':args+=['--first-call-only']
                t=time.perf_counter();done.append(job(f'cold_{method}_{cid}_{rep}',args,gpu))
                whole=time.perf_counter()-t
                d=json.loads(output.read_text());d['fresh_process_total_ms']=whole*1000;write(output,d)
    return done
def main():
    (W/'logs').mkdir(exist_ok=True);done=[]
    write(W/'status.json',dict(phase='running',started=datetime.datetime.now().isoformat()))
    try:
        with ThreadPoolExecutor(max_workers=14) as pool:
            fs=[pool.submit(gpu_and_cold)]
            fs+=[pool.submit(job,'dbnn_'+m,['--mode','dbnn','--method',m,'--cpus',c]) for m,c in [('cpu1','81'),('cpu8','88,89,90,91,92,93,94,95')]]
            fs+=[pool.submit(job,f'neuron8_{s}',['--mode','solver','--method','neuron8','--cpus',','.join(map(str,range(s*8,s*8+8))),
                   '--shard',str(s),'--shards','8','--repeats','10']) for s in range(8)]
            for f in as_completed(fs):
                result=f.result();done.extend(result if isinstance(result,list) else [result])
                write(W/'status.json',dict(phase='running',finished=False,completed_jobs=done))
        write(W/'status.json',dict(phase='complete',finished=True,completed_jobs=done,finished_at=datetime.datetime.now().isoformat()))
    except Exception as e:
        write(W/'status.json',dict(phase='failed',error=repr(e),completed_jobs=done));raise
if __name__=='__main__':main()
