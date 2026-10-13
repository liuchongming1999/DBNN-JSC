"""Continue without colliding with the primary campaign's bound CPU cores."""
import json,os,subprocess,time
from pathlib import Path
from campaign import job
H=Path(__file__).resolve().parent;PY=H.parents[1]/'.venv/bin/python'
while not (H/'campaign_status.json').exists() or not json.loads((H/'campaign_status.json').read_text()).get('complete'):
    time.sleep(10)
(H/'main_campaign_completed.json').write_text((H/'campaign_status.json').read_text())
for n in [1,2,4,8]:job(f'pilot_neuron_n{n}_recheck',threads=n,steps=301,repeats=3)
for n in [2,4,8]:job(f'pilot_ms_n{n}_record_checked',engine='multisplit',threads=n,busywait=0,steps=301,repeats=3)
good=[]
for n in [2,4,8]:
    r=json.loads((H/'results'/f'pilot_ms_n{n}_record_checked.json').read_text())
    if r.get('finished'):good.append(r)
if good:
    r=min(good,key=lambda r:r['timing']['solver_call']['median_ms']);n=r['config']['threads']
    for A in [32,128,500]:
        for T in [1001,6001]:job(f'main_A{A}_T{T}_multisplit_checked',engine='multisplit',threads=n,busywait=0,active=A,steps=T)
    for rate in ['quiet','low','high']:job(f'rate_{rate}_multisplit_checked',engine='multisplit',threads=n,busywait=0,rate=rate)
job('core_single_final',engine='core',threads=1,active=500,steps=6001)
for n in [1,2,4,8]:
    env={**os.environ,'OMP_NUM_THREADS':str(n),'OPENBLAS_NUM_THREADS':str(n),'MKL_NUM_THREADS':str(n),'CUDA_VISIBLE_DEVICES':'6'}
    with (H/f'dbnn_cpu{n}.log').open('w') as f:
        subprocess.run(['taskset','-c',','.join(map(str,range(48,48+n))),str(PY),str(H/'dbnn_worker.py'),'--device','cpu','--threads',str(n),'--repeats','15','--output',f'dbnn_cpu{n}.json'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (H/'throughput.log').open('w') as f:
    subprocess.run([str(PY),str(H/'throughput.py')],env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,check=True)
(H/'extra_complete.json').write_text(json.dumps(dict(finished=time.strftime('%FT%T%z'))))
