"""Verify the actual recorded baseline manifests against all short-window arms."""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');WORK=ROOT/'benchmarks/fair_window_20261005'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for dt in [1.,.5,.25]:
    db=[read(WORK/'dbnn'/f'dbnn_w{w}_dt{dt:g}ms_burn150/protocol.json') for w in [400,500]]
    assert db[0]['manifest_sha256']==db[1]['manifest_sha256']
    assert db[0]['ordered_input_digest']==db[1]['ordered_input_digest']
    assert db[0]['fit_case_ids']==db[1]['fit_case_ids'] and db[0]['validation_case_ids']==db[1]['validation_case_ids']
    sources=[]
    for method in ['tcn','elm','lstm']:
        c=read(ROOT/'benchmarks/baselines_20260930/aligned'/f'{method}_{dt:g}ms/resolved_args.json')
        assert c['manifest_sha256']==db[0]['manifest_sha256']==sha(Path(c['manifest']))
        rows=read(Path(c['manifest']))['cases']
        assert [r['case_id'] for r in rows if r['split']=='fit']==db[0]['fit_case_ids']
        assert [r['case_id'] for r in rows if r['split']=='validation']==db[0]['validation_case_ids']
        sources.append(dict(method=method,manifest_sha256=c['manifest_sha256'],window_ms=c['window_ms'],burn_in_loss_ms=c['burn_in_loss_ms'],updates=c['epochs']*c['batches_per_epoch']))
    records.append(dict(dt_ms=dt,passed=True,manifest_sha256=db[0]['manifest_sha256'],ordered_input_digest=db[0]['ordered_input_digest'],fit_cases=db[0]['fit_cases'],validation_cases=db[0]['validation_cases'],baselines=sources))
output=dict(passed=True,records=records,checks='Actual baseline resolved_args/manifest hashes; fit and validation ID order; paired DBNN raw input digests. Source input/reference paths and numerical labels match exactly within each resolution.')
(WORK/'shared_data_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
