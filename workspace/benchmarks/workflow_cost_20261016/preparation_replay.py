"""Time only the historical training preamble; exit before any optimizer step.

Original training source/checkpoints are read-only. Scratch initialization and
protocol files go to a new audit directory. No loss backward/update is reached.
"""
from pathlib import Path
import sys,time,hashlib,shutil,json,os
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
trial=sys.argv[1];work=W/'preparation_replay'/trial;work.mkdir(parents=True,exist_ok=False)
source=R/'benchmarks/diagonal_1ms_20261009/train_dbnn_diagonal.py'
shutil.copy2(R/'benchmarks/diagonal_1ms_20261009/seq_reset.so',work/'seq_reset.so')
s=source.read_text();assert s.count('    start_time = time.perf_counter()')==1
s=s.replace('def main():','def main():\n    _audit_preparation_start = time.perf_counter()',1)
stop="""    torch.cuda.synchronize()
    _elapsed = time.perf_counter()-_audit_preparation_start
    write(work/'cost_preparation.json',dict(preparation_elapsed_s=_elapsed,
        reached_training=False,optimizer_steps=0,input_count=len(rows),
        source_sha256=sha(Path(__file__)),scratch_directory=str(work)))
    print('PREPARATION_ONLY',_elapsed,flush=True)
    return
    start_time = time.perf_counter()"""
s=s.replace('    start_time = time.perf_counter()',stop)
sys.argv=[str(source),'--root',str(R),'--work',str(work),'--window','500','--burn','150','--dt','1',
          '--epochs','2000','--refine-epochs','400','--seed','291049','--allow-diagonal']
os.sched_setaffinity(0,{112})
exec(compile(s,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
