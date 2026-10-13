"""One finite experiment finalizer; it exits after six complete evaluations."""
import json,time,hashlib
from pathlib import Path
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');WORK=ROOT/'benchmarks/fair_window_20261005'
def read(p):return json.loads(p.read_text())
def main():
    pending=[(m,dt) for m in ['tcn','elm'] for dt in [1.,.5,.25]]
    completed={}
    for method,dt in pending:
        base=WORK/'boundary_controls/aligned'/f'{method}_{dt:g}ms';path=base/'evaluation/complete.json'
        if not path.exists():print(json.dumps(dict(ready=False,waiting=f'{method}_{dt:g}ms')));return
        e=read(path);cp=read(base/'complete.json');assert e['finished'] and cp['finished'] and cp['updates']==cp['total_updates'] and len(e['records'])==31
        assert cp['checkpoint_sha256']==hashlib.sha256((base/'best.pt').read_bytes()).hexdigest()==e['runtime']['checkpoint_sha256']
        assert cp['config']['burn_in_loss_ms']==150 and e['runtime']['repeats']==30
        completed[f'{method}_{dt:g}ms']=dict(resolved_args=read(base/'resolved_args.json'),complete=cp,evaluation=e)
    d=dict(ready=True,finished=True,created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),boundary=completed)
    p=WORK/'boundary_final.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
    print(json.dumps(dict(ready=True,file=str(p),experiments=6)))
if __name__=='__main__':main()
