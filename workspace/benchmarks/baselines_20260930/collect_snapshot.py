"""Small, auditable evidence bundle; no public dataset/checkpoint duplication."""
import argparse,json,shutil,time
from pathlib import Path
from panel import save,digest
from train_default import config

def read(path):return json.loads(path.read_text()) if path.exists() else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();work=a.work.resolve();data=a.data.resolve();out=work/'snapshot';out.mkdir(exist_ok=True)
    d=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_versions=read(work/'source_versions.json'),original={},training={},reduction={},pca={},campaign=read(work/'campaign.json'),publication=read(work/'publication/state.json'),adapter_validation=read(work/'adapter_validation.json'),default_configs={m:{str(dt):config(m,dt) for dt in [1.,.5,.25]} for m in ['tcn','elm','lstm']},scripts={p.name:digest(p) for p in work.glob('*.py')})
    from collect_latest_results import collect_publication
    d['publication_results']=collect_publication(work)
    for method in ['tcn','elm','reduce_legacy']:d['original'][method]=read(work/'original'/method/'complete.json')
    dispatcher=read(work/'logs/dispatcher_state.json')
    if dispatcher:
        for gpu in dispatcher.get('gpus',[]):
            for proc in gpu.get('processes',[]):
                command=proc.pop('command','')
                proc['category']='baseline' if 'train_default.py' in command or 'benchmark_models.py' in command else ('CIFAR' if 'cifar100_depth_margin_20260927/train.py' in command else 'other')
    d['dispatcher']=dispatcher
    d['elm_execution_handoff']=read(work/'logs/elm_execution_handoff_state.json')
    probe_dir=work/'execution_probe_20261001'
    d['elm_execution_probes']=dict(ddp=read(probe_dir/'ddp.json'),cuda_graph={str(dt):read(probe_dir/f'probe_{dt:g}ms_v5.json') for dt in [.5,.25]})
    for method in ['tcn','elm','lstm']:
        for dt in [1.,.5,.25]:
            folder=work/'aligned'/f'{method}_{dt:g}ms';d['training'][f'{method}_{dt:g}ms']=dict(progress=read(folder/'progress.json'),complete=read(folder/'complete.json'),resolved=read(folder/'resolved_args.json'),history=read(folder/'history.json'),allocation=read(work/'logs'/f'dispatched_{method}_{dt:g}ms_state.json'),execution_runtime=read(folder/'execution_runtime.json'),smoke=read(folder/'smoke.json'),smoke_config=read(folder/'smoke_config.json'),evaluation=read(folder/'evaluation/complete.json'),publication=read(folder/'publication/complete.json'))
    for dt in [1.,.5,.25]:
        folder=work/'aligned'/f'reduce_{dt:g}ms';d['reduction'][str(dt)]=dict(complete=read(folder/'complete.json'),status=read(folder/'status.json'),protocol=read(folder/'protocol.json'))
        d['pca'][str(dt)]=dict(complete=read(data/'tcn_aux'/f'pca_{dt:g}ms'/'complete.json'),status=read(data/'tcn_aux'/f'pca_{dt:g}ms'/'status.json'))
    save(out/'status.json',d)
    for path in work.glob('logs/*_state.json'):shutil.copy2(path,out/path.name)

if __name__=='__main__':main()
