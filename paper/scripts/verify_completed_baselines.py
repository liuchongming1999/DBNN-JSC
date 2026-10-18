"""Recompute fixed-panel paired bootstrap differences from imported records."""
from pathlib import Path
import json, hashlib
import numpy as np

R=Path(__file__).resolve().parents[1]
E=R/'evidence'
d=json.loads((E/'main379_results.json').read_text())
assert d['baseline_revision']['elm_updates']==350000
rows=d['models']['gpu']['records']
random=[i for i,r in enumerate(rows) if r['family']!='zero']
fixed=[i for i,r in enumerate(rows) if r['family']=='zero']
assert len(random)==374 and len(fixed)==5
w=np.random.default_rng(20261016).multinomial(374,np.full(374,1/374),size=10000)
scores={}
for k in ['gpu','elm','tcn','lstm']:
    m=d['models'][k]
    assert [(r['case_id'],r['input_sha256'],r['reference_events_ms']) for r in m['records']]==[(r['case_id'],r['input_sha256'],r['reference_events_ms']) for r in rows]
    x=np.array([[r['voltage']['sse'],r['voltage']['n'],r['events']['tp'],r['events']['fp'],r['events']['fn']] for r in m['records']])
    b=np.einsum('bi,ij->bj',w,x[random],optimize=False)+x[fixed].sum(0)
    assert np.isfinite(b).all()
    rmse=np.sqrt(b[:,0]/b[:,1]);f1=2*b[:,2]/(2*b[:,2]+b[:,3]+b[:,4])
    scores[k]=(rmse,f1)
out=dict(passed=True,inputs=379,stochastic=374,fixed_controls=5,replicates=10000,seed=20261016,
    source_sha256=hashlib.sha256((E/'main379_results.json').read_bytes()).hexdigest(),comparisons={})
for k in ['elm','tcn','lstm']:
    out['comparisons'][k]=dict(dbnn_minus_baseline_rmse_ci95_mv=np.quantile(scores['gpu'][0]-scores[k][0],[.025,.975]).tolist(),
        dbnn_minus_baseline_f1_ci95=np.quantile(scores['gpu'][1]-scores[k][1],[.025,.975]).tolist())
(E/'baseline_final_paired_statistics_20261017.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
