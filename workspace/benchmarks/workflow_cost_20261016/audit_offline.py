"""Collect frozen-run cost records and explicitly identify missing boundaries."""
from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=W/'offline_sources';out.mkdir(exist_ok=True)
p=R/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag'
sources={};
for name in ['protocol.json','final_selection.json','primary_history.json','refine_history.json','training_manifest.json']:
    q=p/name;shutil.copy2(q,out/name);sources[name]=dict(path=str(q),sha256=sha(q))
protocol=read(p/'protocol.json');selection=read(p/'final_selection.json');manifest=read(p/'training_manifest.json')['cases']
assert selection['checkpoint_sha256']=='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
training_s=selection['training_elapsed_s']
reference_records=[]
for row in manifest:
    q=Path(row['reference_file']).with_suffix('.json');j=read(q)
    assert row['input_sha256']==j['input_sha256'] and row['ids_sha256']==j['ids_sha256']
    assert j['dt_ms']==.1
    reference_records.append(dict(case_id=row['case_id'],split=row['split'],source=str(q),
        source_sha256=sha(q),wall_ms=j['wall_ms'],schedule_ms=j.get('schedule_ms'),solve_ms=j.get('solve_ms'),
        input_sha256=row['input_sha256'],ids_sha256=row['ids_sha256'],reference_dt_ms=j['dt_ms']))
    shutil.copy2(q,out/(row['case_id']+'_reference.json'))
serial_s=sum(x['wall_ms'] for x in reference_records)/1000
start=datetime.fromisoformat(protocol['created']);locked=datetime.fromisoformat(selection['locked_at'])
report=dict(checkpoint_sha256=selection['checkpoint_sha256'],inputs_fit=protocol['fit_cases'],
  inputs_validation=protocol['validation_cases'],source_inputs=len(manifest),data_physical_duration_s=sum(x['duration_s'] for x in manifest),
  training_and_validation_calibration_elapsed_s=training_s,
  training_timer_boundary='train_dbnn_diagonal.py start_time after data loading, FFT feature caching, random initialization and equivalence checks; includes optimization, periodic validation, six final reset calibrations and checkpoint selection; stops before independent tests',
  protocol_creation_to_selection_wall_s=(locked-start).total_seconds(),
  reference_generation_serial_sum_s=serial_s,reference_generation_records=reference_records,
  combined_documented_serial_budget_s=serial_s+training_s,
  data_generation_accounting='Sum of per-trajectory elapsed intervals, not observed elapsed time of the historical parallel data-generation job. A conservative sequential deployment scenario; cannot interpret as CPU-seconds or reconstruct concurrency by summing wall times.',
  excluded_evaluation_data='All 379 independent test reference trajectories, reference convergence checks, and test scoring',
  missing_historical_costs=['Initial unsigned event-raster generation','Training process imports, data loading and initial FFT caching before timer','Hyperparameter exploration and engineering development (separate research costs)'],
  sources=sources,train_protocol_unchanged=sha(p/'protocol.json'),observed_model_selection_locked_at=selection['locked_at'])
(W/'offline_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['reference_generation_records','sources']},indent=2))
