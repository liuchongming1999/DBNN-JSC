"""Package only completed formal diagnostic outputs, retaining raw traces."""
from pathlib import Path
import hashlib
import json
import platform
import tarfile
import time
import numpy
import scipy
import torch
import neuron

R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
W=R/'benchmarks/adviser7_theory_bridge_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['passive_v2','active_error_budget_v2','active_frozen']:
    d=json.loads((W/name/'complete.json').read_text());assert d['finished'],name
cp=R/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
assert sha(cp)=='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
sources=[cp,R/'scripts/sample_dbnn_data_repro.py',
         R/'source/ShaiEtAl2015/models/L5PCbiophys5.hoc',
         R/'source/ShaiEtAl2015/models/L5PCtemplate.hoc',
         R/'source/ShaiEtAl2015/morphologies/cell1.asc',
         R/'source/ShaiEtAl2015/x86_64/libnrnmech.so']
inputs={}
for name in ['passive_v2','active_error_budget_v2']:
    p=json.loads((W/name/'protocol.json').read_text())
    fit={r['input_sha256'] for r in p['profiles'] if r['split']=='fit'}
    test={r['input_sha256'] for r in p['profiles'] if r['split']=='test'}
    assert len(fit)==64 and len(test)==32 and not fit&test
    files=sorted((W/name/'raw').rglob('*.npz'))
    inputs[name]=dict(unique_fit=64,unique_test=32,raw_trajectory_files=len(files),
                      raw_bytes=sum(x.stat().st_size for x in files),
                      hashes={str(x.relative_to(W)):sha(x) for x in files})
manifest=dict(created=time.strftime('%Y-%m-%d %H:%M:%S %z'),host=platform.node(),
              versions=dict(numpy=numpy.__version__,scipy=scipy.__version__,torch=torch.__version__,neuron=neuron.__version__),
              protected_source_hashes={str(x):sha(x) for x in sources},
              final_sources={str(x.relative_to(W)):sha(x) for x in sorted((W/'v2').glob('*.py'))},
              inputs=inputs,main_checkpoint_unchanged=True,manuscript_modified=False,
              active_frozen_reference_sources='active_frozen/complete.json')
(W/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
members=['v2','passive_v2','active_error_budget_v2','active_frozen',
         'active_frozen_diagnostic.py','build_report.py','README.txt',
         'input_deduplication_v2.json','provenance.json',
         'passive_diagnostic.py','active_error_budget.py','prepare_v2.py','archive_evidence.py']
archive=W/'adviser7_diagnostics_evidence_20261016.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
    for name in members:
        tf.add(W/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.lock') else info)
record=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
            includes_raw_traces_and_frozen_readout_arrays=True,
            primary_checkpoint_not_duplicated=True)
(W/'archive_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({**record,'raw_counts':{k:v['raw_trajectory_files'] for k,v in inputs.items()}},indent=2))
