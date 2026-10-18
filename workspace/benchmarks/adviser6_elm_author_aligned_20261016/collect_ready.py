from pathlib import Path
import json,hashlib,time
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 train=read(W/'aligned/elm_1ms/complete.json');ev=read(W/'elm_evaluation/complete.json');gpu=read(W/'final_gpu_timing/complete.json')
 assert train['finished'] and train['default_budget_completed'] and train['updates']==train['total_updates']==350000
 assert sha(W/'aligned/elm_1ms/best.pt')==train['checkpoint_sha256']
 assert ev['finished'] and len(ev['records'])==379 and ev['corrected_checkpoint_sha256']==train['checkpoint_sha256']
 assert gpu['finished'] and gpu['full_results'] and gpu['checkpoints_unchanged'] and len(gpu['records'])==379
 assert all(all(v['passed'] for v in rec['parity'].values()) for rec in gpu['records'])
 paths={'training':W/'aligned/elm_1ms/complete.json','elm_accuracy':W/'elm_evaluation/complete.json','paired_gpu_timing':W/'final_gpu_timing/complete.json','engineering_cumulative':R/'benchmarks/adviser6_mapping_engineering_20261016/engineering/complete.json','engineering_rollbacks':R/'benchmarks/adviser6_mapping_engineering_20261016/engineering_rollback/complete.json','tcn_whole_validation':R/'benchmarks/baseline_fidelity_audit_20261016/tcn_whole_main379/complete.json'}
 rec=dict(status='results_ready_manuscript_not_yet_updated',finished=True,ready_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),summary=gpu['summary'],elm_checkpoint_sha256=train['checkpoint_sha256'],paths={k:str(p) for k,p in paths.items()},file_hashes={k:sha(p) for k,p in paths.items()},constraints='Keep frozenDBNN/LSTM, TCNwhole1ms, correctedELM1ms; original379 inputs and0.0125ms reference; no tuning or new outcome screening. Local manuscript update is separate and must validate compilation, figures, references and claims.')
 (W/'comparison_ready.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
