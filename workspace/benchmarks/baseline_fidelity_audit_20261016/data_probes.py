from pathlib import Path
import json,numpy as np,collections
from scipy.signal import find_peaks
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/baseline_fidelity_audit_20261016'
rows=json.loads((R/'benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json').read_text())['cases'];native_peak_delta=[];label_diff=[];spcnt=collections.Counter();lengths=collections.Counter();rawmax=0
for r in rows:
 f=np.load(r['reference_file']);sp=f['spike_times'].astype(float);spcnt[r['split']]+=len(sp);lengths[str(r['duration_ms'])]+=1
 peaks,_=find_peaks(f['voltage'],height=-25);pk=f['time_ms'][peaks]
 for s in sp:
  if len(pk):
   p=pk[np.argmin(abs(pk-s))]
   if abs(p-s)<=5:
    native_peak_delta.append(float(p-s));label_diff.append(int(np.floor(s+.5))-int(p-.5))
 inp=np.load(r['input_file']);rawmax=max(rawmax,int(inp['x'].max()))
summary=dict(rows=len(rows),duration_ms_counts=dict(lengths),native_events_by_split=dict(spcnt),native_to_local_peak_delta_ms=dict(n=len(native_peak_delta),median=float(np.median(native_peak_delta)),min=float(np.min(native_peak_delta)),max=float(np.max(native_peak_delta))),current_native_nearest_label_minus_author_peak_label_bins={str(k):v for k,v in collections.Counter(label_diff).items()},elm_source_start_range_for_6s=[500,5500],current_elm_start_range_for_6s=[0,5500],early_start_probability_6s=500/5501,max_training_input_count_per_bin=rawmax,notes='Local maxima reconstructed from saved reference grid, not original paper high-res data. Timing-label diagnostic only; no targets or weights changed.')
(W/'sampling_and_label_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
validation={}
for engine in ['core1','multisplit8']:
 for dt in [.5,.25]:
  paths=list((R/'benchmarks/solver_main379_20261016/outputs'/engine/f'dt{dt:g}').glob('*.json'));vals=[json.loads(p.read_text())['validation'] for p in paths];validation[f'{engine}_{dt}']=dict(cases=len(vals),max_voltage_gap_mv=max(v['same_dt_single_thread_max_voltage_gap_mv'] for v in vals),event_equal_cases=sum(v['same_dt_events_equal'] for v in vals))
(W/'solver_equivalence_audit.json').write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2))
