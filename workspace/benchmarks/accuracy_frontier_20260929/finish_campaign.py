import os,subprocess,sys,time
from pathlib import Path
root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');work=root/'benchmarks/accuracy_frontier_20260929'
pid=int((work/'campaign.pid').read_text())
while not (work/'fresh_scratch_0p5ms/final_selection.json').exists():
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('Upstream training stopped before completion; inspect campaign.log')
    time.sleep(10)
# Current user scope fixes the model architecture. The archived B5 experiment
# must not be scheduled by future runs of this completion entry point.
cmd=[sys.executable,'-u',str(work/'continue_from_b2.py')]
subprocess.run(cmd,check=True)
print('EXPERIMENT_COMPLETE',flush=True)
