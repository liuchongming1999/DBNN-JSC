"""Freeze the previously prelocked 182-input panel; never regenerate or filter."""
from pathlib import Path
import json,hashlib,os,time
import numpy as np
from common8 import read,write,sha
p=Path(__file__).resolve().parent
source=Path('/home/lcm/DBNN_adviser8_20261017')
training=Path('/home/lcm/DBNN_crosscell_candidate_completion_20261017')
p.mkdir(exist_ok=True)
if (p/'locked_protocol.json').exists():
 d=read(p/'locked_protocol.json')
 for j in d['jobs']:
  for key in ['test_manifest','reference_completion','training_manifest']:
   assert sha(j[key])==j[key+'_sha256']
 print('ALREADY_LOCKED',len(d['jobs']));raise SystemExit(0)
a=read(source/'locked_protocol.json');b=read(training/'locked_protocol.json')
lookup={(j['cell'],j['profile']):j for j in b['jobs']};jobs=[]
for i,j in enumerate(a['jobs']):
 t=lookup[j['cell'],j['profile']];base=p/'cells'/j['cell']/j['profile'];base.mkdir(parents=True,exist_ok=True)
 manifest=Path(j['test_manifest']);assert sha(manifest)==j['test_manifest_sha256'];rows=read(manifest)['cases'];assert len(rows)==182
 ref_done=Path(j['base'])/'test/complete.json';r=read(ref_done);assert r['all_inputs_retained'] and r['finished'];rr={x['case_id']:x for x in r['records']}
 assert set(rr)=={x['case_id'] for x in rows}
 tr=read(t['manifest'])['cases'];train_hash={x['input_sha256'] for x in tr};train_seeds={x['seed'] for x in tr}
 assert all(x['input_sha256'] not in train_hash and x['seed'] not in train_seeds for x in rows)
 assert len({x['seed'] for x in rows})==182
 for row in rows:
  d=np.load(row['input_file']);assert hashlib.sha256(d['x'].tobytes()).hexdigest()==row['input_sha256'];assert hashlib.sha256(d['ids'].tobytes()).hexdigest()==row['ids_sha256']
  assert sha(row['reference_file'])==rr[row['case_id']]['reference_sha256']
 reference_condition=Path(j['base'])/'reference_condition.json';sites=Path(j['base'])/'sites.json'
 assert sha(reference_condition)==j['reference_condition_sha256'] and sha(sites)==j['sites_sha256']
 jobs.append(dict(cell=j['cell'],profile=j['profile'],base=str(base),test_manifest=str(manifest),test_manifest_sha256=sha(manifest),reference_completion=str(ref_done),reference_completion_sha256=sha(ref_done),reference_condition=str(reference_condition),reference_condition_sha256=sha(reference_condition),sites=str(sites),sites_sha256=sha(sites),training_folder=str(training/'results'/j['cell']/j['profile']/'dt1.0ms'),training_manifest=t['manifest'],training_manifest_sha256=t['manifest_sha256'],expected_updates=t['expected_updates'],test_inputs=182,train_test_inputs_disjoint=True,train_test_seeds_disjoint=True))
# Pairing is a design property, independent of NEURON output.
for cell in sorted({j['cell'] for j in jobs}):
 panels=[read(j['test_manifest'])['cases'] for j in jobs if j['cell']==cell]
 assert len(panels)==2
 assert all((x['seed'],x['input_sha256'],x['ids_sha256'])==(y['seed'],y['input_sha256'],y['ids_sha256']) for x,y in zip(*panels))
write(p/'locked_protocol.json',dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),training_work=str(training),training_protocol_sha256=sha(training/'locked_protocol.json'),source_panel_locked_at=a['locked_at'],source_panel_protocol_sha256=sha(source/'locked_protocol.json'),scope='12 independently fitted cell/synapse DBNNs; same fixed panel previously used for earlier training recipes and the previous cross-cell selection; not an unseen-again panel',panel_previously_used_for_other_recipe=True,all_models_must_be_locked_before_first_test=True,model_selection='Final expanded candidate pool: positive-update snapshots plus original candidates; paired reset-calibrated diagonal ON/OFF chosen by the same original eligibility/J rule. All choices fixed before this replay; no test-based checkpoint, threshold, diagonal or seed selection',output_filtering=False,all_test_inputs_retained=True,inputs_per_condition=182,total_inputs=2184,test_seed_rule=a['test']['seeds'],design='5 durations x4 A x3 rates x3 families +2 zero controls; spatial mode assigned by fixed cycle',durations_ms=[1000,2000,4000,8000,16000],active_sites=[64,128,256,500],rates_hz=[10,25,50],families=['stationary','excitatory_gates','rate_changes','zero'],metrics=dict(sample_interval_ms=1,voltage_mask_ms=10,event_tolerance_ms=5,boundary_epsilon_ms=1e-6,matching='one-to-one',bootstrap=10000,bootstrap_unit='independent complete input trajectory',undefined_voltage='Retain input and event metrics; report masked voltage as undefined if no valid samples'),evaluation=dict(repeats=3,gpus=[2,5],timing='Prepared complete filtering, readout and autonomous reset; synchronized median of3 on shared server',numerical_parity='Max voltage gap <.001mV and identical events to FFT drive + sequential FP64 reset; same captured-graph parity; predeclared FP64-reset fallback only for equivalence, not test accuracy'),main_model_and_figure_unchanged=True,manuscript_update_only_after_verified_results=True,jobs=jobs))
write(p/'execution_plan.json',dict(status='waiting_all_12_training',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),next='wait -> verify all twelve budgets and validation-selected hashes -> snapshot all twelve checkpoints and thresholds -> evaluate all2184 fixed inputs -> verify records/groups/bootstrap -> archive independent report',publication_update=False))
print(json.dumps({'locked':True,'jobs':12,'inputs':2184,'reference_files_verified':2184,'training_test_overlap':0}))
