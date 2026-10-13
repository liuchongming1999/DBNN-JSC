"""Finite, owned-job handoff at the next atomically saved epoch checkpoint.

Original workers keep training until that boundary. Preserve the model, Adam,
cosine schedule and all RNG states; use the author model for final evaluation.
Unrelated jobs are never signalled. No scheduled reminder or recurring task.
"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import torch
from panel import save,digest
from job_lock import acquire_lock


def owned_command(pid,required):
    proc=Path('/proc')/str(pid)
    assert proc.stat().st_uid==os.getuid(),('foreign process',pid)
    args=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert all(part in args for part in required),(pid,args,required)
    return args


def gpu_snapshot():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    values={}
    for line in text.splitlines():
        gpu,util,memory=map(int,line.split(','));values[gpu]=dict(util=util,memory_mib=memory)
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    root=a.root.resolve();work=a.work.resolve();out=work/'execution_probe_20261001';plan=json.loads(a.plan.read_text())
    guard=acquire_lock(out/'handoff.lock',blocking=False)
    for process in plan['controllers']:owned_command(process['pid'],process['required'])
    records=[]
    for job in plan['jobs']:
        dt=job['dt'];folder=work/'aligned'/f'elm_{dt:g}ms'
        owned_command(job['old_pid'],['train_default.py','--method elm',f'--dt {dt:g}'])
        probe=json.loads((out/f'probe_{dt:g}ms_v5.json').read_text());assert probe['passed']
        assert all(r['parameter_max_abs_gap']==0 and r['gradient_max_abs_gap']==0 and r['loss_abs_gap']==0 for r in probe['parity'])
        smoke=json.loads((work/'execution_smoke_20261001/aligned'/f'elm_{dt:g}ms/smoke.json').read_text());assert smoke['passed']
        last=folder/'last.pt';cp=torch.load(last,map_location='cpu',weights_only=False)
        assert cp['cfg']==json.loads((folder/'resolved_args.json').read_text())
        record=dict(**job,folder=str(folder),old_checkpoint_mtime_ns=last.stat().st_mtime_ns,
                    old_checkpoint_steps=cp['steps'],status='waiting_next_epoch_checkpoint',
                    probe_speedup=probe['speedup']);records.append(record)
    if a.validate_only:
        print(json.dumps(dict(validated=True,jobs=records),indent=2));return
    # Stop only the now-replaced queue controllers. Their existing children
    # retain their locks and keep training; the TCN queue remains untouched.
    for process in plan['controllers']:
        owned_command(process['pid'],process['required']);os.kill(process['pid'],signal.SIGTERM)
    for f in ['dispatcher_state.json','recurrent_state.json']:
        path=work/'logs'/f
        if path.exists():save(out/('before_handoff_'+f),json.loads(path.read_text()))
    children={};streaks={};finished=False
    def publish(status='running'):
        state=dict(status=status,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),jobs=records,
                   training_strategy_changed=False,checkpoint_boundary_handoff=True,
                   publication_timing='Unchanged author inference; final contemporaneous phase remains queued')
        save(work/'logs/elm_execution_handoff_state.json',state)
        save(work/'logs/recurrent_state.json',dict(status='complete' if status=='complete' else ('failed' if status=='failed' else 'running'),
             job='ELM default training: CUDA Graph execution handoff',updated=state['updated'],jobs=records))
        active={str(r.get('gpu',r['preferred_gpu'])):dict(job=f"elm_{r['dt']:g}ms",pid=r.get('pid',r['old_pid']),resource_mode=r['status']) for r in records if r['status']!='complete'}
        save(work/'logs/dispatcher_state.json',dict(status='complete' if status=='complete' else 'running',updated=state['updated'],running=active,
             active_jobs=[f"elm_{r['dt']:g}ms" for r in records if r['status']!='complete'],failed=[],mode='Finite checkpoint handoff; previous dispatcher retired'))
        for r in records:
            save(work/'logs'/f"dispatched_elm_{r['dt']:g}ms_state.json",dict(status='complete' if r['status']=='complete' else 'running',phase=r['status'],
                 method='elm',dt_ms=r['dt'],gpu=r.get('gpu',r.get('old_gpu')),pid=r.get('pid',r['old_pid']),execution_only=True))
    def launch(record,phase,fallback=False):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(record['gpu']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',PYTHONPATH=str(work/'deps'))
        if phase=='training':
            cmd=[str(root/'.venv/bin/python'),'-u',str(work/'train_default.py'),'--root',str(root),'--work',str(work),'--data',str(a.data),'--method','elm','--dt',str(record['dt'])]
            if not fallback:cmd.append('--elm-cuda-graph')
            cmd=['taskset','-c',str(84+record['gpu']),*cmd]
        else:
            cmd=[str(root/'.venv/bin/python'),'-u',str(work/'benchmark_models.py'),'--root',str(root),'--work',str(work),'--method','elm','--dt',str(record['dt'])]
        log=(out/f"handoff_{record['dt']:g}ms_{phase}{'_fallback' if fallback else ''}.log").open('a')
        child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(pid=child.pid,status=phase,log=str(Path(log.name)));children[record['dt']]=(child,log)
    publish()
    while True:
        gpus=gpu_snapshot()
        for gpu in [2,3,4,5,7]:
            value=gpus[gpu];streaks[gpu]=streaks.get(gpu,0)+1 if value['util']<=5 and value['memory_mib']<128 else 0
        for record in records:
            if record['status']=='waiting_next_epoch_checkpoint':
                folder=Path(record['folder']);last=folder/'last.pt'
                if last.stat().st_mtime_ns==record['old_checkpoint_mtime_ns']:continue
                eligible=[g for g in [record['preferred_gpu'],2,3,4,5,7] if streaks.get(g,0)>=3]
                if not eligible:continue
                gpu=eligible[0];check=gpu_snapshot()[gpu]
                if check['util']>5 or check['memory_mib']>=128:continue
                owned_command(record['old_pid'],['train_default.py','--method elm',f"--dt {record['dt']:g}"])
                os.kill(record['old_pid'],signal.SIGSTOP)
                try:
                    cp=torch.load(last,map_location='cpu',weights_only=False)
                    assert cp['steps']>record['old_checkpoint_steps']
                    assert cp['steps']==(cp['epoch']+1)*cp['cfg']['batches_per_epoch']
                    assert cp['cfg']==json.loads((folder/'resolved_args.json').read_text())
                    record.update(gpu=gpu,resume_step=cp['steps'],resume_epoch=cp['epoch']+2,checkpoint_sha256=digest(last),
                                  handoff_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                except BaseException:
                    os.kill(record['old_pid'],signal.SIGCONT);raise
                os.kill(record['old_pid'],signal.SIGKILL)
                # Killing this confirmed paused child releases its CUDA context
                # and inherited exclusive file lock; its checkpoint is intact.
                for _ in range(100):
                    probe=Path('/proc')/str(record['old_pid'])
                    if not probe.exists() or probe.joinpath('stat').read_text().split()[2]=='Z':break
                    time.sleep(.1)
                launch(record,'training');streaks[gpu]=0;print('HANDOFF',json.dumps(record),flush=True)
            elif record['status'] in ['training','evaluation']:
                child,log=children[record['dt']];rc=child.poll()
                if rc is None:continue
                log.close()
                if rc and record['status']=='training' and not record.get('fallback'):
                    record.update(fallback=True,graph_return_code=rc);launch(record,'training',fallback=True)
                elif rc:
                    record.update(status='failed',return_code=rc);publish('failed');raise SystemExit(rc)
                elif record['status']=='training':launch(record,'evaluation')
                else:record.update(status='complete',completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        if all(r['status']=='complete' for r in records):publish('complete');print('COMPLETE',flush=True);return
        publish();time.sleep(10)


if __name__=='__main__':main()
