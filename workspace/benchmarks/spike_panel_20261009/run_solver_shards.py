"""Existing validated multisplit/Core runners, split across independent inputs."""
import argparse
import os
from pathlib import Path
import subprocess
from common_panel import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--engine',choices=['neuron8','core1'],required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();rows=read(work/'confirmatory/manifest.json')['cases']
    base=work/'methods'/a.engine;base.mkdir(parents=True,exist_ok=True);py=root/'.venv/bin/python';procs=[]
    for i in range(8):
        out=base/f'shard{i:02d}';out.mkdir(exist_ok=True)
        cpus=','.join(map(str,range(i*8,i*8+8))) if a.engine=='neuron8' else str(80+i)
        script=root/('benchmarks/neuron_single_expanded_20260929/run_multisplit_grid.py' if a.engine=='neuron8' else 'benchmarks/core_threading_20260929/run_core_threads.py')
        cmd=[str(py),str(script),'--root',str(root),'--manifest',str(work/'confirmatory/manifest.json'),
             '--output-dir',str(out),'--previous-results',str(work/'methods/neuron1/raw'),'--threads','8' if a.engine=='neuron8' else '1',
             '--cpus',cpus,'--dt','1','--repeats','1','--case-ids',*[r['case_id'] for r in rows[i::8]]]
        log=open(base/f'shard{i:02d}.log','w')
        env={**os.environ,'OMP_NUM_THREADS':'8' if a.engine=='neuron8' else '1','OMP_WAIT_POLICY':'PASSIVE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        procs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env),log))
    codes=[]
    for proc,log in procs:codes.append(proc.wait());log.close()
    write(base/'raw_complete.json',dict(finished=all(c==0 for c in codes),dt_ms=1.,shards=8,returncodes=codes,
                                       scope='Accuracy evaluation; one run, no publication timing comparison'))
    assert all(c==0 for c in codes),codes


if __name__=='__main__':main()
