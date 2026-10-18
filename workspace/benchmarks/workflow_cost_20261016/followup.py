from pathlib import Path
import time,json,os,subprocess
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
launch=json.loads((W/'launch.json').read_text())
while not (W/'cold/neuron8').exists() or len(list((W/'cold/neuron8').glob('*.json')))<24:
    if (W/'status.json').exists() and json.loads((W/'status.json').read_text()).get('phase')=='failed':raise RuntimeError('Main campaign failed')
    time.sleep(10)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_WAIT_POLICY':'PASSIVE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
for method,cpus in [('gpu','112'),('cpu8','104,105,106,107,108,109,110,111')]:
    with (W/f'stream_{method}.log').open('w') as log:
        subprocess.run([str(R/'.venv/bin/python'),'-u',str(W/'benchmark_stream.py'),'--method',method,'--cpus',cpus],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for i in range(3):
    started=time.perf_counter()
    with (W/f'prep_replay_{i}.log').open('w') as log:
        subprocess.run([str(R/'.venv/bin/python'),str(W/'preparation_replay.py'),f'rep{i}'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    (W/'preparation_replay'/f'rep{i}'/'process_elapsed.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-started)))
(W/'followup_complete.json').write_text(json.dumps(dict(finished=True)))
