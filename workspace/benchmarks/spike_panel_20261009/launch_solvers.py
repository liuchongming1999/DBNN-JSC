"""Pinned independent B=1 timing runs for the frozen 379-case panel."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();panel=a.work.resolve();out=panel/'main379';out.mkdir(exist_ok=True)
    old=json.loads((panel/'confirmatory/manifest.json').read_text());score=json.loads((panel/'methods/neuron_dt0.5/complete.json').read_text())
    ids={r['case_id'] for r in score['records']};rows=[r for r in old['cases'] if r['case_id'] in ids];assert len(rows)==379
    manifest=out/'manifest.json';manifest.write_text(json.dumps({**old,'cases':rows,'primary_subset_only':True},indent=2)+'\n')
    (out/'confirmatory').mkdir(exist_ok=True)
    (out/'confirmatory/manifest.json').write_text(manifest.read_text())
    link=out/'confirmatory/references'
    if not link.exists():link.symlink_to(panel/'confirmatory/references',target_is_directory=True)
    jobs=[];py=root/'.venv/bin/python';previous=panel/'methods/neuron_dt0.5/raw'
    for engine in ['neuron8','core1','reduce']:
        for i in range(4):
            folder=out/engine/f'shard{i:02d}';folder.mkdir(parents=True,exist_ok=True)
            cpus=list(range(32+8*i,40+8*i)) if engine=='neuron8' else [96+i] if engine=='core1' else [100+i]
            if engine=='reduce':
                script=out/'reduce_dt.py';s=(panel/'evaluate_reduce_raw.py').read_text()
                s=s.replace("rows=manifest['cases'][a.shard::8]","rows=manifest['cases'][a.shard::4]")
                script.write_text(s)
                cmd=['taskset','-c',','.join(map(str,cpus)),str(py),str(script),'--root',str(root),'--work',str(root/'benchmarks/baselines_20260930'),
                     '--panel-work',str(out),'--dt','.5','--repeats','3','--publication','--shard',str(i)]
            else:
                script=root/('benchmarks/neuron_single_expanded_20260929/run_multisplit_grid.py' if engine=='neuron8' else 'benchmarks/core_threading_20260929/run_core_threads.py')
                cmd=[str(py),str(script),'--root',str(root),'--manifest',str(manifest),'--output-dir',str(folder),
                     '--previous-results',str(previous),'--threads','8' if engine=='neuron8' else '1','--cpus',','.join(map(str,cpus)),
                     '--dt','.5','--repeats','3','--case-ids',*[r['case_id'] for r in rows[i::4]]]
            env={**os.environ,'OMP_NUM_THREADS':'8' if engine=='neuron8' else '1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE',
                 'PYTHONPATH':str(root/'benchmarks/baselines_20260930')+':'+str(root/'benchmarks/baselines_20260930/deps')}
            log=open(out/f'{engine}_shard{i:02d}.log','a');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            jobs.append(dict(engine=engine,shard=i,pid=proc.pid,cpus=cpus,proc=proc,log=log))
    status=dict(started=datetime.now().isoformat(),cases=379,repeats=3,solver_dt_ms=.5,observation_interval_ms=1.,
                scope='Each timing is B=1 latency on declared disjoint physical-core affinity. Independent measurement jobs run concurrently on the shared server; no batching in the measured solve.',
                jobs=[{k:v for k,v in j.items() if k not in ['proc','log']} for j in jobs])
    (out/'solver_launch.json').write_text(json.dumps(status,indent=2)+'\n')
    codes=[]
    for j in jobs:codes.append(j['proc'].wait());j['log'].close()
    (out/'solver_completion.json').write_text(json.dumps(dict(finished=all(c==0 for c in codes),returncodes=codes,completed=datetime.now().isoformat()),indent=2)+'\n')
    assert all(c==0 for c in codes),codes


if __name__=='__main__':main()
