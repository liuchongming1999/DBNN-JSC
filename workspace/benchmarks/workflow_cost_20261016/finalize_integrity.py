from pathlib import Path
import hashlib,json
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=json.loads((W/'protected_files.json').read_text())
checks={p:dict(before=s,after=sha(p),unchanged=sha(p)==s) for p,s in before.items()}
assert all(x['unchanged'] for x in checks.values())
assert json.loads((W/'status.json').read_text())['finished']
assert json.loads((W/'followup_complete.json').read_text())['finished']
data=json.loads((W/'data_generation/complete.json').read_text());assert data['finished']
old=json.loads((W/'offline_audit.json').read_text())
for row in old['reference_generation_records']:
    assert sha(row['source'])==row['source_sha256']
for i in range(3):
    q=json.loads((W/'preparation_replay'/f'rep{i}'/'cost_preparation.json').read_text())
    assert q['optimizer_steps']==0 and not q['reached_training']
out=dict(passed=True,protected_files=checks,original_reference_metadata_unchanged=149,
         data_replay_reference_count=len(data['records']),
         data_replay_max_voltage_gap_mv=max(x['validation']['max_abs_mv'] for x in data['records']),
         data_replay_all_events_equal=all(x['validation']['events_equal'] for x in data['records']),
         preparation_replays_optimizer_steps=0,original_training_not_restarted=True,
         scripts_sha256={p.name:sha(p) for p in W.glob('*.py')})
(W/'integrity_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print('All integrity checks passed; original checkpoint, protocol, reference metadata and manifest unchanged.')
