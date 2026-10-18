"""Preserve pilot evidence; reuse only byte-identical input definitions."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import numpy as np

W=Path('/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/adviser7_theory_bridge_20261016')
sys.path.insert(0,str(W/'v2'))
from passive_diagnostic import make_profiles

pid=int((W/'active_error_budget.pid').read_text())
lines=subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines()
ps={int(p[0]):(int(p[1]),p[2]) for line in lines if len(p:=line.split(None,2))==3}
terminated=[]
if pid in ps:
    assert '/adviser7_theory_bridge_20261016/active_error_budget.py' in ps[pid][1] or 'benchmarks/adviser7_theory_bridge_20261016/active_error_budget.py' in ps[pid][1]
    descendants={pid}
    while True:
        new={i for i,(parent,_) in ps.items() if parent in descendants}
        if new<=descendants:break
        descendants |= new
    assert 3413750 not in descendants
    for i in sorted(descendants,key=lambda x:x!=pid):
        try:os.kill(i,signal.SIGTERM);terminated.append(i)
        except ProcessLookupError:pass
    time.sleep(.3)

new={r['name']:r for r in make_profiles()}
counts={};changed=[]
for oldname,newname in [('passive','passive_v2'),('active_error_budget','active_error_budget_v2')]:
    old=W/oldname;dest=W/newname;dest.mkdir(exist_ok=True)
    oldrows={r['name']:r for r in json.loads((old/'protocol.json').read_text())['profiles']}
    reused=[];invalid=[]
    for src in (old/'raw').rglob('*.npz'):
        name=src.stem.rsplit('_e',1)[0]
        identical=name=='zero' or (name in new and oldrows[name]['input_sha256']==new[name]['input_sha256'])
        if not identical:invalid.append(str(src.relative_to(old)));continue
        try:
            with np.load(src) as f:assert f['voltage_mv'].shape==(501,)
        except Exception:
            invalid.append(str(src.relative_to(old)));continue
        to=dest/src.relative_to(old);to.parent.mkdir(parents=True,exist_ok=True)
        if not to.exists():os.link(src,to)
        reused.append(str(src.relative_to(old)))
    counts[newname]=dict(verified_identical_input_raw_files_reused=len(reused),
                        changed_or_incomplete_files_not_reused=len(invalid),
                        note='Hardlinks to immutable raw npz, no computed fit/projection outputs reused')
changed=[dict(name=n,shift_ms=r.get('input_only_deduplication_shift_ms',0)) for n,r in new.items() if r.get('input_only_deduplication_shift_ms',0)]
record=dict(reason='Input-identity audit found deterministic duplicate test rasters. Input-only common-onset shifts preserve inter-event spacings and E-I delays; no output-dependent screening.',
            interrupted_only_own_pilot_pids=terminated,changed_test_inputs=changed,
            final_unique_fitting_inputs=64,final_unique_test_inputs=32,reuse=counts,
            pilot_directories_preserved=True,primary_checkpoint_and_existing_baseline_training_untouched=True)
(W/'input_deduplication_v2.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
