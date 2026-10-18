"""Resolve preserved provenance paths inside this repository."""
from pathlib import Path
import json,hashlib
REPO=Path(__file__).resolve().parents[1]
ORIGINAL='/home/lcm/DBNN_L5PC_surrogate_20260928'
def resolve(value):
    value=str(value)
    if value==ORIGINAL or value.startswith(ORIGINAL+'/'):
        p=REPO/'workspace'/value[len(ORIGINAL):].lstrip('/')
    elif value.startswith('/home/lcm/') or value.startswith('/data/lcm/'):
        p=REPO/'additional_cells'/value.lstrip('/')
    elif not Path(value).is_absolute():p=REPO/value
    else:raise ValueError('Unreleased path: '+value)
    if not p.exists():raise FileNotFoundError(p)
    return p

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
