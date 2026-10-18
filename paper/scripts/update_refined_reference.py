"""Import completed solver sweep, preserving input membership and learned outputs.

Source payloads are bundled as compressed JSON; running this script needs no
server, training, calibration or inference. All metrics use one common reference.
"""
from pathlib import Path
import copy,gzip,hashlib,json
import numpy as np
from event_metrics import match_events
from rescore_event_boundary import update_summary

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name):return json.loads(gzip.decompress((E/name).read_bytes()))

def summary(rows):
    n=sum(r['voltage']['n'] for r in rows);sse=sum(r['voltage']['sse'] for r in rows)
    rs=sum(r['voltage']['reference_ss'] for r in rows);es=sum(r['voltage']['error_ss'] for r in rows)
    v=dict(cases=len(rows),pooled_rmse_mv=float(np.sqrt(sse/n)),
           mean_case_ve=float(np.mean([r['voltage']['ve'] for r in rows if r['voltage']['ve'] is not None])),
           within_trace_weighted_ve=1-es/rs if rs>0 else None,
           total_duration_s=sum(r['duration_s'] for r in rows))
    return update_summary(v,rows)

def main():
    old=load('main379_results_before_refinement_20261016.json.gz')
    sweep=load('solver_main379_analysis_20261016.json.gz')
    assert sweep['finished'] and sweep['cases']==379 and sweep['reference_dt_ms']==.0125
    assert sweep['validation']['dbnn_checkpoint_sha256']==old['checkpoint_sha256']
    mapping={k:k for k in ['gpu','cpu1','cpu8','tcn','elm','lstm','reduce']}
    for engine,key in [('multisplit8','neuron8'),('core1','core1')]:
        for dt in [.5,.25,.1]:mapping[key if dt==.5 else f'{key}_dt{dt:g}']=f'{engine}_dt{dt:g}'
    new=copy.deepcopy(old);new['models']={}
    meta={r['case_id']:r for r in old['models']['gpu']['records']}
    for key,source in mapping.items():
        sm=sweep['models'][source];base=key.split('_dt')[0]
        oldrows={r['case_id']:r for r in old['models'][base]['records']}
        rows=[]
        for r in sm['records']:
            cid=r['case_id'];row=copy.deepcopy(oldrows[cid]);qualification=row['reference_spikes']
            if base in ['gpu','cpu1','cpu8','tcn','elm','lstm','reduce']:
                assert row['predicted_events_ms']==r['predicted_events_ms']
            row.update(copy.deepcopy(r));row['qualification_reference_spikes']=qualification
            row['reference_spikes']=len(r['reference_events_ms'])
            row['minimum_isi_ms']=min(np.diff(r['reference_events_ms'])) if len(r['reference_events_ms'])>1 else row['duration_ms']
            assert row['input_sha256']==meta[cid]['input_sha256']
            row['events']=match_events(row['reference_events_ms'],row['predicted_events_ms'])
            rows.append(row)
        acc=summary(rows)
        assert abs(acc['pooled_rmse_mv']-sm['pooled_rmse_mv'])<1e-12
        assert all(acc['events'][k]==sm['events'][k] for k in ['tp','fp','fn','f1'])
        groups={axis:{str(v):summary([r for r in rows if r[axis]==v])
                      for v in sorted({r[axis] for r in rows},key=str)}
                for axis in old['models'][base]['groups']}
        timing=copy.deepcopy(sweep['timing'][source]);times=[]
        for t in timing['records']:
            if 'wall' not in t:
                t['wall']=dict(median_ms=t['median_ms'],q1_ms=float(np.quantile(t['runs_ms'],.25)),
                               q3_ms=float(np.quantile(t['runs_ms'],.75)),runs_ms=t['runs_ms'])
            t.update(duration_s=meta[t['case_id']]['duration_s'],A=meta[t['case_id']]['A'])
            times.append(t['wall']['median_ms'])
        timing.update(wall_ms=float(sum(times)),finished=True)
        expected=timing['sum_median_ms'];assert np.isclose(timing['wall_ms'],expected,rtol=1e-12)
        new['models'][key]=dict(accuracy=acc,groups=groups,records=rows,timing=timing,interval_ms=sm['dt_ms'])
    new.update(reference_events=sweep['reference_events'],reference_dt_ms=.0125,
               solver_steps_ms=[.5,.25,.1],qualification_reference_dt_ms=.1,qualification_reference_events=637)
    new['durations']={}
    for t in old['durations']:
        new['durations'][t]={}
        for key,m in new['models'].items():
            v=[r['wall']['median_ms'] for r in m['timing']['records'] if r['duration_s']==int(t)]
            new['durations'][t][key]=dict(cases=len(v),median_ms=float(np.median(v)),
                q1_ms=float(np.quantile(v,.25)),q3_ms=float(np.quantile(v,.75)),sum_ms=float(sum(v)))
    lookup={r['case_id']:r for r in new['models']['neuron8_dt0.25']['records']};examples=[]
    for count in [0,3]:
        choices=[r for r in new['models']['gpu']['records'] if r['family']!='zero' and r['duration_s']==8 and r['A']==500
                 and len(r['reference_events_ms'])==count and r['events']['fp']==r['events']['fn']==0
                 and lookup[r['case_id']]['events']['fp']==lookup[r['case_id']]['events']['fn']==0]
        row=min(choices,key=lambda r:r['voltage']['rmse_mv'])
        examples.append(dict(case_id=row['case_id'],reference_spikes=count,eligible_candidates=len(choices),
             solver_dt_ms=.25,dbnn_record=row,neuron_record=lookup[row['case_id']]))
    new['examples']=examples
    new['protocol']=dict(cases=379,repeats=3,solver_discarded_warmup_calls=1,reference_step_ms=.0125,
         learned_warmup='unmeasured validation/eager/graph warm-up calls before timing',
         comparator_steps_ms=[.5,.25,.1],observation_interval_ms=1.,gpu_timing_lane='GPU5, learned methods measured serially',
         scope='Prepared single-trajectory computation, B=1; initialization and input preparation excluded',
         source='solver_main379_analysis_20261016.json.gz',reduce_timing='retained original complete-cohort timing')
    new['evaluation_reference_revision']=dict(reference_dt_ms=.0125,reference_events=680,
         qualification_unchanged=True,all_379_inputs_retained=True,learned_event_times_unchanged=True,
         checkpoints_training_reset_thresholds_unchanged=True,voltage_masks_recomputed_from_common_reference=True,
         source_sha256=sha(E/'solver_main379_analysis_20261016.json.gz'),
         source_remote='/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/solver_main379_20261016/analysis.json')
    new['event_scoring_revision']['scope']='Frozen independent predictions scored against common 0.0125 ms reference; historical training decisions unchanged'
    (E/'main379_results.json').write_text(json.dumps(new,indent=2,allow_nan=False)+'\n')
    (E/'selected_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    # Never restore superseded ELM predictions or windowed-TCN timings.
    if (E/'baseline_final_20261017').exists():
        from import_completed_baselines import main as import_baselines
        import_baselines()
        new=json.loads((E/'main379_results.json').read_text())
    print(json.dumps(dict(reference_events=680,examples=[r['case_id'] for r in examples],
        models={k:dict(rmse=m['accuracy']['pooled_rmse_mv'],f1=m['accuracy']['events']['f1'],time_ms=m['timing']['wall_ms']) for k,m in new['models'].items()}),indent=2))

if __name__=='__main__':main()
