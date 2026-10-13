"""Adapt validated inference benchmark to the newly locked short-window weights."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK=ROOT/'benchmarks/fair_window_20261005'
def main():
    p=argparse.ArgumentParser();p.add_argument('--window',type=int,choices=[400,500],default=500);a=p.parse_args()
    src=ROOT/'benchmarks/resolution_frontier_20260929';out=WORK/f'timing_w{a.window}';out.mkdir(exist_ok=True)
    for f in ['gpu_benchmark.py','gpu_backend.py','cpu_benchmark.py','common.py']:
        shutil.copy2(src/f,out/f)
    (out/'specs.py').write_text(f'''from pathlib import Path\ndef configurations(root,work):\n    source=root/'benchmarks/fair_window_20261005/dbnn'\n    return {{name:(dt,source/f'dbnn_w{a.window}_dt{{dt:g}}ms_burn150/best_checkpoint.pt') for name,dt in [('B2',1.),('B6',.5),('B7',.25)]}}\n''')
    f=out/'gpu_benchmark.py';s=f.read_text()
    old="b7=read(work/'b6_warm_0p25ms/final_selection.json');assert b7['finished'] and b7['checkpoint_sha256']==sha(specs['B7'][1]);assert b7['protocol']['kernels_per_site']==1"
    new="locks={n:read(path.parent/'final_selection.json') for n,(_,path) in specs.items()};assert all(locks[n]['finished'] and locks[n]['checkpoint_sha256']==sha(path) and locks[n]['protocol']['loss_burn_in_ms']==150 and locks[n]['protocol']['training_window_ms']=="+str(a.window)+" for n,(_,path) in specs.items());b7=locks['B7']"
    assert old in s;s=s.replace(old,new)
    s=s.replace("b7_locked_at=b7['locked_at'],precision=", "checkpoint_locked_at={n:l['locked_at'] for n,l in locks.items()},training_window_ms="+str(a.window)+",loss_burn_ms=150,precision=")
    f.write_text(s)
    c=out/'cpu_benchmark.py';s=c.read_text().replace('Original CPU backend, matched B2/B6/B7 checkpoints, full prepared computation.',f'Validated CPU backend, random-init {a.window}ms-window checkpoints, full prepared computation.')
    c.write_text(s)
    print('PREPARED',out)
if __name__=='__main__':main()
