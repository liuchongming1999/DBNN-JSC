"""Independent reconstruction of all held-out voltage/event/coverage statistics."""
from pathlib import Path
import json,math,sys
import numpy as np
from common8 import read,write,sha,metric,aggregate,match_events,event_aggregate,reset_function
from evaluate import bootstrap
p=Path(__file__).resolve().parent;pr=read(p/'locked_protocol.json');ml=read(p/'models_locked.json');assert ml['all_models_locked'] and len(ml['models'])==12
reset=reset_function(p);checks=[];errors=[]
for i,j in enumerate(pr['jobs']):
 try:
  model=ml['models'][str(i)];assert sha(model['checkpoint'])==model['checkpoint_sha256']
  assert sha(j['test_manifest'])==j['test_manifest_sha256'];assert sha(j['reference_completion'])==j['reference_completion_sha256']
  base=Path(j['base'])/'evaluation';c=read(base/'complete.json');rows=read(j['test_manifest'])['cases'];rr=c['records']
  assert c['finished'] and c['all_inputs_retained'] and not c['test_used_for_selection']
  assert len(rr)==len(rows)==182 and len({r['case_id'] for r in rr})==182
  lookup={r['case_id']:r for r in rows};assert set(lookup)=={r['case_id'] for r in rr}
  ms=[];es=[]
  for r in rr:
   x=lookup[r['case_id']];assert r['checkpoint_sha256']==model['checkpoint_sha256'] and r['input_sha256']==x['input_sha256'] and r['input_seed']==x['seed'] and r['ids_sha256']==x['ids_sha256']
   assert r['test_manifest_sha256']==j['test_manifest_sha256'] and sha(x['reference_file'])==r['reference_sha256']
   a=np.load(base/'predictions'/f"{x['case_id']}.npz");ref=np.load(x['reference_file']);assert np.isfinite(a['voltage']).all() and np.isfinite(a['drive']).all()
   assert np.array_equal(a['reference_events_ms'],ref['spike_times']) and np.array_equal(a['reference_voltage'],np.interp(a['time_ms'],ref['time_ms'],ref['voltage']))
   assert np.array_equal(a['predicted_events_ms'],np.flatnonzero(a['voltage']>=model['reset']['v_th']))
   oracle,ev=reset(a['drive'],model['reset']['v_th'],model['reset']['reset_amp'],model['reset']['tau_reset'])
   assert np.array_equal(ev,a['predicted_events_ms']) and np.max(np.abs(a['voltage']-oracle))<.001
   m=metric(a['reference_voltage'],a['voltage'],a['reference_events_ms']);e=match_events(a['reference_events_ms'],a['predicted_events_ms'])
   assert m['n']==r['voltage']['n'] and math.isclose(m['sse'],r['voltage']['sse'],rel_tol=1e-12,abs_tol=1e-7)
   assert all(e[k]==r['events'][k] for k in ['tp','fp','fn'])
   assert r['numerical']['events_equal'] and r['numerical']['max_abs_mv']<.001 and r['numerical']['captured_graph_events_equal'] and r['numerical']['captured_graph_max_abs_mv']<.001
   assert len(r['runs_ms'])==3 and np.isfinite(r['runs_ms']).all() and min(r['runs_ms'])>0 and math.isclose(float(np.median(r['runs_ms'])),r['prepared_wall_ms'],rel_tol=1e-12)
   ms.append(m);es.append(e)
  v=aggregate(ms);ev=event_aggregate(es);assert math.isclose(v['pooled_rmse_mv'],c['voltage']['pooled_rmse_mv'],rel_tol=1e-12) and all(ev[k]==c['events'][k] for k in ['tp','fp','fn','f1'])
  assert c['undefined_voltage_inputs']==sum(m['n']==0 for m in ms)
  for key,groups in c['groups'].items():
   assert set(groups)=={str(r[key]) for r in rr}
   for label,g in groups.items():
    sub=[r for r in rr if str(r[key])==label];assert len(sub)==g['inputs']
    ag=aggregate([r['voltage'] for r in sub]);eg=event_aggregate([r['events'] for r in sub]);assert ag['n']==g['voltage']['n'] and eg['f1']==g['events']['f1']
  ci=bootstrap(rr,10000)
  for k in ['rmse_ci95_mv','f1_ci95']:
   assert (ci[k] is None and c['bootstrap'][k] is None) or np.allclose(ci[k],c['bootstrap'][k],atol=1e-12,rtol=0)
  checks.append(dict(cell=j['cell'],profile=j['profile'],passed=True,inputs=182,checkpoint_sha256=model['checkpoint_sha256'],test_manifest_sha256=j['test_manifest_sha256'],all_input_ids_retained=True,voltage_and_events_recomputed=True,oracle_and_captured_graph_parity=True,bootstrap_recomputed=True,groups_verified=True,undefined_voltage_inputs=c['undefined_voltage_inputs']))
 except Exception as e:errors.append(dict(job=i,cell=j['cell'],profile=j['profile'],error=repr(e)))
result=dict(complete=len(checks)==12,passed=len(checks)==12 and not errors,completed_conditions=len(checks),total_inputs=sum(x['inputs'] for x in checks),checks=checks,errors=errors,publication_update=False,test_panel_previously_used_for_other_recipe=True)
write(p/'verification.json',result);print(json.dumps({'passed':result['passed'],'conditions':len(checks),'errors':errors}));sys.exit(0 if result['passed'] else 1)
