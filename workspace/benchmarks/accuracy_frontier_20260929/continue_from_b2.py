"""Train B2 -> 0.5 ms without changing architecture, then benchmark locked candidates."""
import fcntl,json,os,subprocess,sys
from datetime import datetime
from pathlib import Path

root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
work=root/'benchmarks/accuracy_frontier_20260929'
name='b2_warm_0p5ms'
lock=open(work/'b2_continuation.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=work/name
cmd=[sys.executable,'-u',str(work/'train_accuracy.py'),'--root',str(root),'--work',str(work),
     '--name',name,'--dt','.5','--fresh','--epochs','1000','--lr','.001',
     '--init-checkpoint',str(work/'balanced_1ms/best_checkpoint.pt')]
print('START B6 from B2',datetime.now().isoformat(),flush=True)
with open(work/'b2_warm_0p5ms.log','a') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print('TRAINING_COMPLETE',flush=True)
selection=json.loads((out/'final_selection.json').read_text())
assert selection['finished'] and selection['protocol']['kernels_per_site']==1
cmd=[sys.executable,'-u',str(work/'evaluate_benchmark.py'),'--root',str(root),'--work',str(work)]
with open(work/'b2_continuation_benchmark.log','w') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print('COMPLETE B2 continuation and paired single-kernel benchmark',datetime.now().isoformat(),flush=True)
