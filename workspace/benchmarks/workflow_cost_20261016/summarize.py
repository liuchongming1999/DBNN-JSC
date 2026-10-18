"""Validate complete cost evidence and calculate elapsed-time break-even."""
from pathlib import Path
import json,hashlib,math,statistics,sys
import numpy as np
W=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'results'
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
    return dict(n=len(a),median=float(np.median(a)),q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75)),min=min(a),max=max(a))
def summarize():
    protocol=read(W/'protocol.json');rows=read(W/'manifest.json')['cases'];ids={r['case_id'] for r in rows}
    assert len(ids)==379
    solver=[read(p) for p in (W/'results/neuron8_dt0.5').glob('confirmatory_*.json')]
    assert {r['case_id'] for r in solver}==ids
    assert all(r['validation']['events_equal'] and r['validation']['max_abs_mv']<.001 for r in solver)
    solver_map={r['case_id']:r for r in solver}
    solver_total={k:sum(r['timing'][k]['median_ms'] for r in solver)/1000 for k in solver[0]['timing']}
    solver_round_sums=[sum(r['records'][rep]['resident_ms'] for r in solver)/1000 for rep in range(10)]
    solver_stream_s=float(np.median(solver_round_sums))
    solver_s=solver_total['resident_ms'];methods={}
    validations=379
    for method in ['gpu','cpu8','cpu1']:
        assert read(W/'results'/method/'complete.json')['finished']
        rr=[[read(p) for p in (W/'results'/method/f'round{z}').glob('*.json')] for z in range(3)]
        assert all({x['case_id'] for x in rs}==ids for rs in rr)
        assert all(x['validation']['events_equal'] and x['validation']['max_abs_mv']<.001 for rs in rr for x in rs)
        assert all(x['timing']['prepared']['n']==10 and x['timing']['resident_reuse_layout']['n']==10 for rs in rr for x in rs)
        validations+=sum(map(len,rr))
        # Per-input median pools all 30 repetitions across three rounds.
        maps=[{x['case_id']:x for x in rs} for rs in rr]
        total={}
        for boundary in ['prepared','resident_reuse_layout','resident_new_layout']:
            pooled={cid:float(np.median(sum([m[cid]['timing'][boundary]['runs_ms'] for m in maps],[]))) for cid in ids}
            round_sums=[sum(x['timing'][boundary]['median_ms'] for x in rs)/1000 for rs in rr]
            total[boundary]=dict(sum_input_medians_s=sum(pooled.values())/1000,
                                round_sums_s=round_sums,round_sum_range_s=[min(round_sums),max(round_sums)],
                                relative_iqr_median=float(np.median([
                                    (np.quantile(v,.75)-np.quantile(v,.25))/np.median(v)
                                    for cid in ids for v in [sum([m[cid]['timing'][boundary]['runs_ms'] for m in maps],[])] ])))
            total[boundary]['ratio_vs_solver_resident']=solver_s/total[boundary]['sum_input_medians_s']
        methods[method]=dict(boundaries=total)
        for boundary in ['prepared','resident_reuse_layout','resident_new_layout']:
            ds={}
            for t in [1,2,4,8,16]:
                chosen=[x['case_id'] for x in rows if x['duration_s']==t]
                own=[float(np.median(sum([m[cid]['timing'][boundary]['runs_ms'] for m in maps],[]))) for cid in chosen]
                ref=[solver_map[cid]['timing']['resident_ms']['median_ms'] for cid in chosen]
                ds[str(t)]=dict(inputs=len(chosen),dbnn_median_ms=float(np.median(own)),
                               neuron_median_ms=float(np.median(ref)),ratio_sums=sum(ref)/sum(own),
                               dbnn_mean_input_ms=float(np.mean(own)),neuron_mean_input_ms=float(np.mean(ref)))
            methods[method]['boundaries'][boundary]['by_duration']=ds
    streams={}
    for method in ['gpu','cpu8']:
        rr=[read(W/'stream'/method/f'round{z}.json') for z in range(3)]
        assert all(x['finished'] and {y['case_id'] for y in x['records']}==ids for x in rr)
        assert all(y['validation']['events_equal'] and y['validation']['max_abs_mv']<.001 for x in rr for y in x['records'])
        validations+=379*3
        sums=[r['sum_ms']/1000 for r in rr]
        streams[method]=dict(rounds_s=sums,median_panel_s=float(np.median(sums)),
             cache_hits=[sum(y['layout_cache_hit'] for y in r['records']) for r in rr],
             cache_misses=[sum(not y['layout_cache_hit'] for y in r['records']) for r in rr],
             ratio_vs_solver_resident=solver_stream_s/float(np.median(sums)))
    offline=read(W/'offline_audit.json');data=read(W/'data_generation/complete.json');assert data['finished'] and len(data['records'])==149
    prep=[read(W/'preparation_replay'/f'rep{i}'/'process_elapsed.json')['elapsed_s'] for i in range(3)]
    for i in range(3):assert read(W/'preparation_replay'/f'rep{i}'/'cost_preparation.json')['optimizer_steps']==0
    base=offline['training_and_validation_calibration_elapsed_s']+float(np.median(prep))
    costs=dict(training_validation_calibration_historical_s=offline['training_and_validation_calibration_elapsed_s'],
               preparation_new_measured_fresh_process_s=prep,
               reference_generation_16core_batch_s=data['observed_batch_elapsed_s'],
               reference_generation_historical_sum_s=offline['reference_generation_serial_sum_s'],
               offline_parallel_generation_s=data['observed_batch_elapsed_s']+base,
               offline_serial_accounting_s=offline['reference_generation_serial_sum_s']+base,
               missing='Generation of unsigned training event rasters is not historically timed; both scenarios are conditional on those input arrays being available. Research exploration is not included.',
               training_replayed=False,checkpoint_sha256=protocol['checkpoint_sha256'])
    breakeven={}
    for method,s in streams.items():
        per_input_saved=(solver_stream_s-s['median_panel_s'])/379
        assert per_input_saved>0
        breakeven[method]={key:dict(calls=math.ceil(cost/per_input_saved),
                    source_panel_equivalents=cost/(solver_stream_s-s['median_panel_s']),
                    per_input_savings_s=per_input_saved)
                    for key,cost in costs.items() if key in ['offline_parallel_generation_s','offline_serial_accounting_s']}
    by_duration={}
    for method in ['gpu','cpu8']:
        by_duration[method]={}
        for t,d in methods[method]['boundaries']['resident_reuse_layout']['by_duration'].items():
            saved=(d['neuron_mean_input_ms']-d['dbnn_mean_input_ms'])/1000
            assert saved>0
            by_duration[method][t]=dict(offline_parallel_generation_calls=math.ceil(costs['offline_parallel_generation_s']/saved),
                 scenario='Fixed layout; the duration stratum retains its original input mix',mean_savings_s=saved)
    cold={}
    for method in ['gpu','cpu8','neuron8']:
        rr=[read(p) for p in (W/'cold'/method).glob('*.json')];assert len(rr)==24
        cold[method]=dict(fresh_process_total_ms=stats([x['fresh_process_total_ms'] for x in rr]),
             runtime_call_ms=stats([x['cold_runtime_ms'] if method!='neuron8' else x['model_prepare_ms']+x['timing']['resident_ms']['median_ms'] for x in rr]))
    out=dict(finished=True,inputs=379,duration_s=2497,reference_step_ms=.0125,
        frozen_checkpoint_sha256=protocol['checkpoint_sha256'],solver_dt_ms=.5,
        solver_total_s=solver_total,solver_panel_aggregation=dict(round_totals_s=solver_round_sums,
          median_panel_s=solver_stream_s,definition='Sum resident service times across all379 inputs at the same repetition index; first10 repetitions of every input; excludes input-file I/O and scoring, not a claim of one contiguous whole-program run'),
        methods=methods,stream=streams,offline=costs,
        breakeven=breakeven,breakeven_by_duration_fixed_layout=by_duration,cold=cold,validated_output_records=validations,
        measurement_notes=['All calls return CPU voltage on the common1ms grid and actual event times.',
         'No input file I/O or accuracy scoring inside resident service timing.',
         'Three full-panel DBNN rounds; 30 prepared/reuse and3 new-layout repetitions per input. NEURON10 per input,30 on preselected representatives.',
         'Service stream uses frozen input order and a16-entry LRU cache for IDs, length and fixed backend. Empty cache at the start of each panel.',
         'Data-generation batch16single-core workers is not16cores per trajectory; CPU affinity is disjoint from timed inference workers.',
         'Fresh-process timing retains existing compiler caches; total process wall includes imports, disk-input loading and validation. Runtime-call column excludes these diagnostic operations.',
         'One detailed solver setting is selected by voltage RMSE, not by joint spike accuracy. No model/threshold/reset/checkpoint changed.'])
    (W/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['solver_total_s','stream','offline','breakeven']},indent=2))
if __name__=='__main__':summarize()
