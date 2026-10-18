"""Score frozen predictions on all 379 inputs against one refined reference.

Run on 5090. No training, event recalibration, filtering, or prediction edits.
Incomplete timing grids are explicitly excluded from frontier/ratio reporting.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent
sys.path.insert(0,str(WORK))
from event_metrics import match_events,aggregate_events

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(ref,pred,events):
    t=np.arange(len(ref));keep=np.ones(len(t),bool)
    for event in events:keep &= np.abs(t-event)>10.
    r=np.asarray(ref,dtype=np.float64)[keep];p=np.asarray(pred,dtype=np.float64)[keep]
    assert len(r)==len(p) and len(r)>0 and np.isfinite(p).all()
    e=p-r;ss=float(np.sum(e**2));rs=float(np.sum((r-r.mean())**2));es=float(np.sum((e-e.mean())**2))
    return dict(n=len(r),sse=ss,error_ss=es,reference_ss=rs,rmse_mv=float(np.sqrt(ss/len(r))),
                ve=1-es/rs if rs>0 else None,bias_mv=float(e.mean()))
def aggregate(rows):
    return dict(cases=len(rows),reference_events=sum(len(r['reference_events_ms']) for r in rows),
      predicted_events=sum(len(r['predicted_events_ms']) for r in rows),
      pooled_rmse_mv=float(np.sqrt(sum(r['voltage']['sse'] for r in rows)/sum(r['voltage']['n'] for r in rows))),
      events=aggregate_events([r['events'] for r in rows]))
def score(meta,ref,pred,events):
    rv,re=ref;return dict(case_id=meta['case_id'],duration_s=meta['duration_s'],family=meta['family'],
      voltage=metric(rv,pred,re),events=match_events(re,events),reference_events_ms=list(map(float,re)),
      predicted_events_ms=list(map(float,events)))

def main():
    old=read(WORK/'frozen_primary.json');manifest=read(WORK/'manifest.json');rows=manifest['cases']
    assert len(rows)==379 and len({r['case_id'] for r in rows})==379
    refs={};methods={};timing={};replays=[]
    finer_complete=all((WORK/'outputs/reference1/dt0.00625'/f'{r["case_id"]}.json').exists() for r in rows)
    # User-approved practical stop: retain the completed common 0.0125 ms
    # reference; finer runs diagnose sensitivity rather than select a new target.
    refdt=.0125
    for dt in ([.00625] if finer_complete else [])+[.0125,.1,.25,.5]:
        folder=WORK/'outputs/reference1'/f'dt{dt:g}';refs[dt]={}
        for row in rows:
            cid=row['case_id'];j=read(folder/f'{cid}.json')
            assert j['finished'] and j['input_sha256']==row['input_sha256']
            assert j['manifest_sha256']==sha(WORK/'manifest.json') and j['script_sha256']==sha(WORK/'run_grid.py')
            with np.load(folder/f'{cid}.npz') as f:
                assert np.array_equal(f['time_ms'],np.arange(row['duration_ms']+1))
                refs[dt][cid]=(f['voltage'].copy(),f['spike_times_ms'].copy())
            if dt==.1:assert j['validation']['frozen_reference_events_equal']
    frozen_pred=ROOT/'benchmarks/random_init_main379_20261012/predictions'
    baseline=ROOT/'benchmarks/spike_panel_20261009/methods'
    for key in ['gpu','tcn','elm','lstm','reduce']:
        lookup={r['case_id']:r for r in old['models'][key]['records']}
        tr={r['case_id']:r for r in old['models'][key]['timing']['records']}
        scored=[];oldscored=[]
        for row in rows:
            cid=row['case_id'];original=lookup[cid]
            if key=='gpu':path=frozen_pred/f'{cid}.npz'
            elif key=='reduce':path=Path(tr[cid]['raw_output']).with_suffix('.npz')
            else:path=baseline/{'tcn':'tcn_author_default','elm':'elm_author_default','lstm':'lstm_common150'}[key]/'predictions'/f'{cid}.npz'
            with np.load(path) as f:pred=f['voltage'].copy()
            events=np.asarray(original['predicted_events_ms'],dtype=float)
            legacy=score(row,refs[.1][cid],pred,events)
            assert abs(legacy['voltage']['rmse_mv']-original['voltage']['rmse_mv'])<1e-7,(key,cid)
            assert all(legacy['events'][k]==original['events'][k] for k in ['tp','fp','fn'])
            oldscored.append(legacy);scored.append(score(row,refs[refdt][cid],pred,events))
        methods[key]=dict(engine=key,dt_ms=.5 if key=='reduce' else 1.,**aggregate(scored),records=scored,
           original_reference_metrics=aggregate(oldscored),prediction_source='unchanged saved primary outputs')
        if key=='reduce':timing[key]=dict(source='frozen primary timing; not remeasured',sum_median_ms=old['models'][key]['timing']['wall_ms'],records=old['models'][key]['timing']['records'])
    for key in ['cpu1','cpu8']:methods[key]={**methods['gpu'],'engine':key}
    for key in ['gpu','cpu1','cpu8','tcn','elm','lstm']:
        p=WORK/('learned' if key.startswith('cpu') else 'learned_gpu5')/key/'complete.json'
        if not p.exists():continue
        t=read(p);assert t['finished'] and len(t['records'])==379
        assert all(r['validation']['passed'] for r in t['records'])
        timing[key]=dict(source='new full-cohort prepared timings',sum_median_ms=t['wall_ms'],records=t['records'])
    for engine in ['multisplit8','core1']:
        for dt in [.5,.25,.1]:
            key=f'{engine}_dt{dt:g}';scored=[];ts=[];folder=WORK/'outputs'/engine/f'dt{dt:g}'
            for row in rows:
                cid=row['case_id'];rv,re=refs[dt][cid];scored.append(score(row,refs[refdt][cid],rv,re))
                path=folder/f'{cid}.json'
                if not path.exists():continue
                j=read(path);assert j['finished'] and j['timing_repeats']==3 and j['discarded_warmup_calls']==1
                assert j['validation']['same_dt_events_equal'] and j['validation']['same_dt_single_thread_max_voltage_gap_mv']<.001
                ts.append(dict(case_id=cid,median_ms=j['median_solver_ms'],runs_ms=[t['solver_ms'] for t in j['timing']]))
            methods[key]=dict(engine=engine,dt_ms=dt,**aggregate(scored),records=scored,
                prediction_source='same-dt single-core trajectories; timed engine parity checked per input')
            if len(ts)==379:timing[key]=dict(source='new full-cohort prepared timings',sum_median_ms=sum(t['median_ms'] for t in ts),records=ts)
    deep_progress={}
    deepwork=WORK.parent/'deepdendrite_fullcell_20261016'
    for dt in [.5,.25,.1]:
        folder=deepwork/'outputs'/f'dt{dt:g}'
        complete=[r for r in rows if (folder/f'{r["case_id"]}.json').exists()]
        deep_progress[str(dt)]=dict(completed=len(complete),total=379)
        if len(complete)!=379:continue
        scored=[];ts=[]
        for row in rows:
            cid=row['case_id'];j=read(folder/f'{cid}.json')
            assert j['finished'] and j['input_sha256']==row['input_sha256'] and j['repeats']==3
            assert all(v['events_equal'] and v['voltage_max_gap_mv']<.001 for v in j['validation'])
            with np.load(folder/f'{cid}.npz') as f:v=f['voltage'].copy();ev=f['spike_times_ms'].copy()
            scored.append(score(row,refs[refdt][cid],v,ev))
            ts.append(dict(case_id=cid,median_ms=j['median_prepared_ms'],runs_ms=j['prepared_runs_ms']))
        key=f'deep_dt{dt:g}';methods[key]=dict(engine='deep',dt_ms=dt,**aggregate(scored),records=scored)
        timing[key]=dict(source='DeepDendrite full-cohort prepared timings, GPU 5, native recording',sum_median_ms=sum(t['median_ms'] for t in ts),records=ts)
    comparisons=[]
    for dt in [.5,.25,.1,.0125]:
        comparisons.append(dict(dt_ms=dt,**aggregate([score(r,refs[refdt][r['case_id']],*refs[dt][r['case_id']]) for r in rows])))
    full_refinement_check=None
    if finer_complete:
        dr=[];count_changes=[];gaps=[]
        for row in rows:
            cid=row['case_id'];cv,ce=refs[.0125][cid];fv,fe=refs[.00625][cid]
            dr.append(score(row,(fv,fe),cv,ce))
            if len(ce)!=len(fe):count_changes.append(dict(case_id=cid,coarse_count=len(ce),fine_count=len(fe)))
            elif len(ce):gaps.extend(np.abs(ce-fe).tolist())
        full_refinement_check=dict(coarse_dt_ms=.0125,fine_dt_ms=.00625,**aggregate(dr),count_change_cases=count_changes,
            largest_corresponding_event_gap_ms=max(gaps,default=0.),
            stable_by_prespecified_strict_rule=not count_changes and max(gaps,default=0.)<=.5 and aggregate(dr)['pooled_rmse_mv']<=.01)
    targets={};dbnn=methods['gpu']
    for criterion in ['rmse','rmse_and_f1']:
        targets[criterion]={}
        for engine in ['multisplit8','core1']:
            eligible=[(k,m) for k,m in methods.items() if m['engine']==engine and k in timing and m['pooled_rmse_mv']<=dbnn['pooled_rmse_mv'] and (criterion=='rmse' or m['events']['f1']>=dbnn['events']['f1'])]
            if not eligible or 'gpu' not in timing:continue
            k,m=min(eligible,key=lambda kv:timing[kv[0]]['sum_median_ms'])
            targets[criterion][engine]=dict(method=k,dt_ms=m['dt_ms'],solver_rmse_mv=m['pooled_rmse_mv'],solver_f1=m['events']['f1'],
                solver_to_dbnn_gpu_time_ratio=timing[k]['sum_median_ms']/timing['gpu']['sum_median_ms'],
                solver_to_same_core_dbnn_time_ratio=timing[k]['sum_median_ms']/timing['cpu8' if engine=='multisplit8' else 'cpu1']['sum_median_ms'])
    out=dict(reference_dt_ms=refdt,observation_interval_ms=1.,cases=379,total_duration_s=sum(r['duration_s'] for r in rows),
       reference_events=sum(len(v[1]) for v in refs[refdt].values()),models=methods,timing=timing,accuracy_targets=targets,
       full_reference_refinement_check=full_refinement_check,gpu_timing_lane='GPU 5, all compared GPU surrogates measured serially',
       deepdendrite_progress=deep_progress,
       reference_stopping_decision=read(WORK/'reference_agreement.json') if (WORK/'reference_agreement.json').exists() else None,
       solver_reference_comparison=comparisons,finished=read(WORK/'status.json').get('finished',False),
       validation=dict(original_reference_replay_and_scoring_match=True,all_379_inputs_retained=True,models_and_events_frozen=True,
           dbnn_checkpoint_sha256=manifest['dbnn_checkpoint_sha256'],primary_source_sha256=sha(WORK/'frozen_primary.json')))
    p=WORK/'analysis.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    print(json.dumps(dict(finished=out['finished'],reference_events=out['reference_events'],
       models={k:dict(rmse=m['pooled_rmse_mv'],f1=m['events']['f1'],seconds=timing[k]['sum_median_ms']/1000 if k in timing else None) for k,m in methods.items()},targets=targets),indent=2))
if __name__=='__main__':main()
