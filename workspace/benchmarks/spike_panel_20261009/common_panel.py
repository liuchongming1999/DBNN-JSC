"""Frozen L5PC/AMPA configuration and auditable event-panel helpers."""
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');q.replace(p)


def config(root):
    return dict(root=str(Path(root)/'source/ShaiEtAl2015'),biophys_file='L5PCbiophys5.hoc',
                morphology_file='cell1.asc',num_synapses=500,duration_ms=16000.,warmup_ms=500.,dt=.1,
                sample_interval_ms=1.,input_rate_hz=50.,exc_weight=1.88e-4,inh_weight=8e-5,
                exc_tau1_ms=.5,exc_tau2_ms=5.,inh_tau1_ms=1.,inh_tau2_ms=10.,inh_reversal_mv=-80.,
                v_init=-75.,spike_threshold=-20.,seed=20260425)
