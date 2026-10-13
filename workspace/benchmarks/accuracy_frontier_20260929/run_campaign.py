import os,subprocess,sys,time
from pathlib import Path
root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');work=root/'benchmarks/accuracy_frontier_20260929'
configs=[('balanced_1ms',1.,False,False,1000,.001),('fresh_warm_0p5ms',.5,True,False,1000,.001),('fresh_scratch_0p5ms',.5,True,True,2000,.003)]
for name,dt,fresh,scratch,epochs,lr in configs:
    if (work/name/'final_selection.json').exists():continue
    if fresh:
        while not (work/'fresh/complete.json').exists():time.sleep(10)
    cmd=[sys.executable,'-u',str(work/'train_accuracy.py'),'--root',str(root),'--work',str(work),'--name',name,'--dt',str(dt),'--epochs',str(epochs),'--lr',str(lr)]
    if fresh:cmd+=['--fresh']
    if scratch:cmd+=['--scratch']
    print('START',name,flush=True)
    with open(work/(name+'.log'),'w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISH',name,flush=True)
print('TRAINING_COMPLETE',flush=True)
