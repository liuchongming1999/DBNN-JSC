"""Eight independent input trials of the same neuron, persistent process models."""
import argparse,json,multiprocessing as mp,os,time
from pathlib import Path
import numpy as np
from neuron_worker import run,stats,stamp
H=Path(__file__).resolve().parent

def worker(i,P,barrier,q):
    os.sched_setaffinity(0,{48+i})
    a=argparse.Namespace(engine='neuron',threads=1,busywait=0,cache=1,state_cache=1,dt=.1,atol=1e-4,gpu=False,mechanisms='core',active=500,steps=1001,
        samples=list(range(i,8,P)),rate='sampled',repeats=3,output=str(H/'results'/f'pool{P}_worker{i}.json'))
    r=run(a,repeat_barrier=barrier);Path(a.output).write_text(json.dumps(r,indent=2));q.put((i,r))

def main():
    ctx=mp.get_context('spawn');result=dict(started=stamp(),cases=[])
    for P in [1,2,4,8]:
        print('POOL',P,flush=True);barrier=ctx.Barrier(P);q=ctx.Queue();ps=[ctx.Process(target=worker,args=(i,P,barrier,q)) for i in range(P)]
        t=time.perf_counter()
        for p in ps:p.start()
        rs=[q.get(timeout=600) for _ in ps]
        for p in ps:p.join(timeout=30);assert p.exitcode==0
        all_wall=(time.perf_counter()-t)*1000
        rs.sort();w=np.array([r['repeat_wall_windows'] for i,r in rs])
        # Same system monotonic clock, start barrier once per batch repeat.
        times=(w[:,:,1].max(0)-w[:,:,0].min(0))*1000
        result['cases'].append(dict(processes=P,batch=8,timing=stats(times.tolist()),launch_setup_warm_repeats_join_ms=all_wall,
            scope='steady batch wall: cached warm-state restore + event scheduling + 1s simulation + output copy; model build and initial500ms preparation excluded',
            worker_files=[f'pool{P}_worker{i}.json' for i in range(P)]))
        (H/'throughput.json').write_text(json.dumps(result,indent=2))
    result['finished']=stamp();(H/'throughput.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
