"""Run thread configurations sequentially on the same physical core group."""
import json,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
records=[]
def save(finished=False,failed=False):
    tmp=HERE/'campaign_status.tmp'
    tmp.write_text(json.dumps(dict(finished=finished,failed=failed,updated=datetime.now().isoformat(),records=records),indent=2)+'\n')
    tmp.replace(HERE/'campaign_status.json')

for nt in [1,2,4,8]:
    cpus=list(range(72,72+nt));env=dict(os.environ)
    env.update(OMP_NUM_THREADS=str(nt),MKL_NUM_THREADS=str(nt),OPENBLAS_NUM_THREADS='1',OMP_PLACES=','.join('{'+str(c)+'}' for c in cpus),OMP_PROC_BIND='close',OMP_WAIT_POLICY='PASSIVE',CUDA_VISIBLE_DEVICES='')
    output=HERE/f'nt{nt}';cmd=[sys.executable,'-u',str(HERE/'benchmark_cpu.py'),'--root',str(ROOT),'--output',str(output),'--threads',str(nt),'--cpus',','.join(map(str,cpus)),'--repeats','30']
    with (HERE/f'nt{nt}.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        rec=dict(threads=nt,cpus=cpus,pid=proc.pid,started=datetime.now().isoformat(),command=cmd)
        records.append(rec);save();code=proc.wait();rec.update(exit_code=code,completed=datetime.now().isoformat());save()
    if code!=0:
        save(finished=True,failed=True);raise SystemExit(code)
save(finished=True)
