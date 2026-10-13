"""Repair the validation score range without retraining or rerunning CPU timers."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from panel import rows_for,make_case,save,digest
from train_default import model_for,predict_case
from job_lock import acquire_lock
from event_thresholds import VERSION,fit_threshold,detected_events


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--dt',type=float,required=True);a=p.parse_args()
    root=a.root.resolve();work=a.work.resolve();src=work/'aligned'/f'{a.method}_{a.dt:g}ms';out=src/'evaluation'
    guard=acquire_lock(out/'evaluation.lock')
    lock=json.loads((src/'complete.json').read_text());assert digest(src/'best.pt')==lock['checkpoint_sha256']
    cfg=lock['config'];torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_for(work,cfg).cuda().eval();model.load_state_dict(torch.load(src/'best.pt',map_location='cpu',weights_only=False)['model'])
    sys.path.insert(0,str(root/'benchmarks/training_refinement_20260929'))
    from train_refinement import match_events,event_aggregate
    rows,_=rows_for(root,a.dt);pred=[];refs=[]
    with torch.inference_mode():
        for row in rows:
            if row['split']!='validation':continue
            c=make_case(row,a.dt,signed=a.method!='tcn');y=predict_case(model,torch.from_numpy(c['x']).cuda(),cfg,'cuda')
            pred.append(torch.sigmoid(y[:,0]).cpu().numpy());refs.append(c['spike_times'])
    selected=fit_threshold(pred,refs,a.dt,match_events,event_aggregate);selected['checkpoint_sha256']=lock['checkpoint_sha256']
    old_threshold=src/'event_threshold.json';backup=src/'event_threshold_grid_v1.json'
    if old_threshold.exists() and not backup.exists():shutil.copy2(old_threshold,backup)
    save(old_threshold,selected)
    complete=out/'complete.json';baseline=json.loads(complete.read_text());backup=out/'complete_threshold_grid_v1.json'
    if not backup.exists():shutil.copy2(complete,backup)
    for rec in baseline['records']:
        path=out/(rec['case_id']+'.npz')
        with np.load(path) as raw:data={k:raw[k] for k in raw.files}
        data['spike_times']=detected_events(data['probability'],a.dt,selected['threshold'])
        row=next(r for r in json.loads((root/'benchmarks/dt_resolution_20260929/test/manifest.json').read_text())['cases'] if r['case_id']==rec['case_id'])
        ref=np.load(root/'benchmarks/dt_resolution_20260929/test/references'/f'{row["case_id"]}_reference.npz')
        rec['events_at_previous_threshold']=rec['events'];rec['events']=match_events(ref['spike_times'],data['spike_times'])
        for name,b in rec['backends'].items():
            if name.startswith('cpu') and 'events' in b:
                b['events_at_previous_threshold']=b.pop('events')
                b['events_requires_final_publication_recompute']=True
        np.savez_compressed(path,**data);save(out/(rec['case_id']+'.json'),rec)
    baseline['summary']['events']=event_aggregate([r['events'] for r in baseline['records']]);baseline['threshold']=selected['threshold']
    baseline['runtime']['event_threshold_protocol']=VERSION
    baseline['runtime']['event_recalibration']=dict(source='Validation only; existing main31 GPU probability outputs rescored. Voltage and forward wall timers unchanged. CPU event counters will be recomputed in the publication phase.',updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    save(complete,baseline)
    print(json.dumps(dict(validation_threshold=selected['threshold'],validation_peak_range=selected['validation_peak_score_range'],best_validation=max(selected['grid'],key=lambda r:r['f1']),main31_events=baseline['summary']['events']),indent=2),flush=True)


if __name__=='__main__':main()
