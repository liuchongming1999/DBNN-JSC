"""Archive completed model outputs and truthful live control-job state."""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');WORK=ROOT/'benchmarks/fair_window_20261005'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    snapshot=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),dbnn={},boundary={},timing={})
    paths=[]
    for window in [400,500]:
        for dt in [1.,.5,.25]:
            name=f'dbnn_w{window}_dt{dt:g}ms_burn150';f=WORK/'dbnn'/name
            d=read(f/'complete.json');assert d['finished'] and len(d['records'])==31
            assert sha(f/'best_checkpoint.pt')==d['checkpoint_sha256']
            for key in ['best_checkpoint.pt','random_initial_checkpoint.pt','complete.json','final_selection.json','protocol.json','primary_history.json','refine_history.json','training_manifest.json','progress.json']:
                p=f/key
                if p.exists():paths.append((p,Path('dbnn')/name/key))
            for p in (f/'test_predictions').glob('*.npz'):paths.append((p,Path('dbnn')/name/'test_predictions'/p.name))
            snapshot['dbnn'][name]=dict(complete=d,selection=read(f/'final_selection.json'))
        for backend in ['gpu','cpu1','cpu8']:
            f=WORK/f'timing_w{window}'/backend
            if (f/'complete.json').exists():
                d=read(f/'complete.json');assert d['finished'] and len(d['records'])==31
                snapshot['timing'][f'{window}_{backend}']=d
                for p in f.rglob('*'):
                    if p.is_file() and p.suffix in ['.json','.npz']:paths.append((p,p.relative_to(WORK)))
            elif (f/'status.json').exists():snapshot['timing'][f'{window}_{backend}']=read(f/'status.json')
    for method in ['tcn','elm','lstm']:
        for dt in [1.,.5,.25]:
            name=f'{method}_{dt:g}ms';f=WORK/'boundary_controls/aligned'/name;item={}
            for key in ['resolved_args.json','progress.json','history.json','complete.json']:
                p=f/key
                if p.exists():item[key[:-5]]=read(p)
            if (f/'evaluation/complete.json').exists():item['evaluation']=read(f/'evaluation/complete.json')
            if method=='lstm':item['reused_author_default']=True
            if (f/'complete.json').exists() and (f/'publication/complete.json').exists():item['publication']=read(f/'publication/complete.json')
            snapshot['boundary'][name]=item
    snapshot['workers']={p.stem:read(p) for p in (WORK/'logs').glob('gpu*_state.json')}
    snapshot['timing_state']=read(WORK/'timing_state.json')
    sp=WORK/'snapshot.json';sp.write_text(json.dumps(snapshot,indent=2));paths.append((sp,Path('snapshot.json')))
    for p in WORK.glob('*.py'):paths.append((p,Path('source')/p.name))
    for p in (WORK/'logs').glob('gpu*_state.json'):paths.append((p,Path('logs')/p.name))
    for key in ['campaign.json','timing_state.json','seq_reset.c']:
        p=WORK/key
        if p.exists():paths.append((p,Path(key)))
    archive=WORK/'current_artifacts.tar.gz';tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',dereference=True) as t:
        for p,name in paths:t.add(p,arcname=str(name),recursive=False)
    tmp.replace(archive);print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,completed_timings=list(snapshot['timing']),dbnn_models=len(snapshot['dbnn']))))
if __name__=='__main__':main()
