"""Install isolated experiment paths; never overwrite historical results."""
from pathlib import Path
import shutil, subprocess, json
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK=ROOT/'benchmarks/fair_window_20261005'
BASE=ROOT/'benchmarks/baselines_20260930'
def link(src,dst):
    if dst.exists() or dst.is_symlink():
        assert dst.resolve()==src.resolve(),(src,dst)
    else: dst.symlink_to(src,target_is_directory=src.is_dir())
def main():
    db=WORK/'dbnn';db.mkdir(exist_ok=True)
    shutil.copy2(WORK/'seq_reset.c',db/'seq_reset.c')
    subprocess.run(['gcc','-O3','-fPIC','-shared',str(db/'seq_reset.c'),'-o',str(db/'seq_reset.so'),'-lm'],check=True)
    old=ROOT/'benchmarks/window_control_20261004/scratch_matched'
    for w in (400,500):
        done=json.loads((old/f'w{w}_burn150/complete.json').read_text())
        assert done['finished'] and done['protocol']['loss_burn_in_ms']==150
        link(old/f'w{w}_burn150',db/f'dbnn_w{w}_dt1ms_burn150')
    boundary=WORK/'boundary_controls';boundary.mkdir(exist_ok=True)
    shutil.copy2(WORK/'train_default.py',boundary/'train_default.py')
    shutil.copy2(WORK/'probe_boundary.py',boundary/'probe_boundary.py')
    for f in ('panel.py','tcn.py','job_lock.py','elm_cuda_graph.py','event_thresholds.py','benchmark_models.py','publication_timing.py'):
        shutil.copy2(BASE/f,boundary/f)
    for d in ('vendor','original','source_versions.json','deps','compat'):
        link(BASE/d,boundary/d)
    (boundary/'aligned').mkdir(exist_ok=True)
    for dt in (1.,.5,.25):
        link(BASE/'aligned'/f'lstm_{dt:g}ms',boundary/'aligned'/f'lstm_{dt:g}ms')
    print('SETUP_COMPLETE',WORK)
if __name__=='__main__':main()
