"""Recompute event statistics from archived predictions, never run models.

The compressed pre-correction payload is kept as the reproducible raw input.
Only event statistics and bootstrap intervals change. Voltage/time measurements,
event lists, checkpoint selection and model/reset parameters are preserved.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from event_metrics import match_events, aggregate_events, BOUNDARY_EPS_MS

ROOT = Path(__file__).resolve().parents[1]
E = ROOT/'evidence'
NBOOT = 10000
SEED = 20261016
WEIGHTS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def update_summary(old, rows):
    out = copy.deepcopy(old)
    out['events'] = aggregate_events([r['events'] for r in rows])
    out['reference_events'] = sum(len(r['reference_events_ms']) for r in rows)
    out['predicted_events'] = sum(len(r['predicted_events_ms']) for r in rows)
    out['false_positives_per_minute'] = out['events']['fp']/(sum(r['duration_s'] for r in rows)/60)
    stochastic = [r for r in rows if r['family'] != 'zero']
    fixed = np.array([[r['events'][k] for k in ('tp','fp','fn')] for r in rows if r['family']=='zero'], dtype=int).reshape(-1,3).sum(0)
    if stochastic:
        cache_key = tuple(r['case_id'] for r in stochastic)
        if cache_key not in WEIGHTS:
            WEIGHTS[cache_key] = np.random.default_rng(SEED).multinomial(len(stochastic),
                np.full(len(stochastic),1/len(stochastic)), size=NBOOT)
        counts = np.array([[r['events'][k] for k in ('tp','fp','fn')] for r in stochastic])
        totals = WEIGHTS[cache_key] @ counts + fixed
    else:
        totals = np.repeat(fixed[None], NBOOT, axis=0)
    tp,fp,fn = totals.astype(float).T
    arrays = [np.divide(tp,tp+fp,out=np.full(NBOOT,np.nan),where=tp+fp>0),
              np.divide(tp,tp+fn,out=np.full(NBOOT,np.nan),where=tp+fn>0),
              np.divide(2*tp,2*tp+fp+fn,out=np.full(NBOOT,np.nan),where=2*tp+fp+fn>0)]
    out['bootstrap_95ci'] = {k:np.nanquantile(a,[.025,.975]).tolist() if np.isfinite(a).any() else None
                            for k,a in zip(['precision','recall','f1'],arrays)}
    out.update(bootstrap_replicates=NBOOT, bootstrap_seed=SEED,
               bootstrap_unit='independent full input trajectory; paired across methods',
               bootstrap_stochastic_trajectories=len(stochastic),
               deterministic_zero_controls_held_fixed=len(rows)-len(stochastic),
               f1_interpretable=out['reference_events']>0)
    return out


def main():
    if (E/'solver_main379_analysis_20261016.json.gz').exists():
        # Current public entry point: preserve the finer-reference revision.
        from update_refined_reference import main as import_refined
        import_refined()
        return
    src=E/'main379_results_before_boundary_20261016.json.gz'
    before=json.loads(gzip.decompress(src.read_bytes()))
    after=copy.deepcopy(before)
    changes={}
    for key,model in after['models'].items():
        rows=model['records']
        for row in rows:
            row['events']=match_events(row['reference_events_ms'],row['predicted_events_ms'])
        model['accuracy']=update_summary(model['accuracy'],rows)
        for axis,groups in model['groups'].items():
            for name,summary in groups.items():
                selected=[r for r in rows if str(r[axis])==name]
                assert selected,(axis,name)
                groups[name]=update_summary(summary,selected)
        changes[key]=dict(before=before['models'][key]['accuracy']['events'],
                          after=model['accuracy']['events'])
        for old,new in zip(before['models'][key]['records'],rows):
            assert {k:v for k,v in old.items() if k!='events'}=={k:v for k,v in new.items() if k!='events'}
        assert model['timing']==before['models'][key]['timing']
        assert model['accuracy']['pooled_rmse_mv']==before['models'][key]['accuracy']['pooled_rmse_mv']
    # Selected examples remain unchanged; refresh their embedded scored records.
    for example in after.get('examples',[]):
        for field,method in [('dbnn_record','gpu'),('neuron_record','neuron8')]:
            example[field]=next(r for r in after['models'][method]['records'] if r['case_id']==example['case_id'])
    after['event_scoring_revision']=dict(date='2026-10-16',tolerance_ms=5,
        boundary_eps_ms=BOUNDARY_EPS_MS,bootstrap_replicates=NBOOT,bootstrap_seed=SEED,
        source_archive=src.name,source_uncompressed_sha256=hashlib.sha256(gzip.decompress(src.read_bytes())).hexdigest(),
        checkpoints_thresholds_reset_event_lists_unchanged=True,
        scope='Independent evaluation rescored; historical training/checkpoint-selection decisions unchanged')
    target=E/'main379_results.json'
    target.write_text(json.dumps(after,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    manifest=json.loads((E/'source_manifest.json').read_text())
    manifest['main379_results.json']=dict(source=str(src),source_uncompressed_sha256=after['event_scoring_revision']['source_uncompressed_sha256'],
        sha256=sha(target),transformation='scripts/rescore_event_boundary.py',boundary_eps_ms=BOUNDARY_EPS_MS,
        source_campaign='/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/random_init_main379_20261012')
    (E/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (E/'event_boundary_revision_20261016.json').write_text(json.dumps(dict(
        boundary_eps_ms=BOUNDARY_EPS_MS,bootstrap_replicates=NBOOT,bootstrap_seed=SEED,
        event_changes=changes,voltage_and_runtime_unchanged=True,
        frozen_model_and_event_times_unchanged=True),indent=2)+'\n')
    cross_arrays=E/'crosscell_event_arrays_20261016.json'
    if cross_arrays.exists() and not (E/'crosscell_final_20261017/source.json').exists():
        source=E/'crosscell_events_before_boundary_20261016.json.gz'
        if not source.exists():
            source.write_bytes(gzip.compress((E/'crosscell_events.json').read_bytes()))
        cross=json.loads(gzip.decompress(source.read_bytes()))
        arrays=json.loads(cross_arrays.read_text())
        cross_changes=[]
        for condition in arrays['conditions']:
            cell,profile=condition['cell'],condition['profile']
            job=next(j for j in cross['jobs'] if j['cell']==cell and j['profile']==profile)
            assert condition['checkpoint_sha256']==job['selection']['checkpoint_sha256']
            scores=[]
            for row in condition['records']:
                legacy=match_events(row['reference_events_ms'],row['predicted_events_ms'],boundary_eps_ms=0.)
                assert all(legacy[k]==row['previous_events'][k] for k in ['tp','fp','fn'])
                scores.append(match_events(row['reference_events_ms'],row['predicted_events_ms']))
            old=job['evaluation']['1.0']['events']
            current=aggregate_events(scores)
            job['evaluation']['1.0']['events']=current
            cross_changes.append(dict(cell=cell,profile=profile,before=old,after=current))
        cross['event_scoring_revision']=dict(boundary_eps_ms=BOUNDARY_EPS_MS,
            source='crosscell_event_arrays_20261016.json',checkpoints_unchanged=True)
        (E/'crosscell_events.json').write_text(json.dumps(cross,indent=2,ensure_ascii=False)+'\n')
        (E/'crosscell_boundary_revision_20261016.json').write_text(json.dumps(cross_changes,indent=2)+'\n')
        manifest['crosscell_events.json']=dict(source=str(source),event_arrays=str(cross_arrays),
            sha256=sha(E/'crosscell_events.json'),transformation='scripts/rescore_event_boundary.py')
        (E/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(changes,indent=2))


if __name__=='__main__':
    main()
