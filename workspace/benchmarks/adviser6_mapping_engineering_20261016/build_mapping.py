"""Single preregistered geometry-only adaptation. No simulation/output selection."""
import json, hashlib, sys, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928'); W=R/'benchmarks/adviser6_mapping_engineering_20261016'
W.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'benchmarks/neuron_single_expanded_20260929'))
from run_multisplit_grid import build
cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
h,pc,state,*_=build(SimpleNamespace(root=R,threads=1,dt=.5))
meta=R/'benchmarks/spike_panel_20261009/site_metadata.json'; sites=json.loads(meta.read_text())['sites']; secs={s.name():s for s in h.allsec()}
segs=[secs[s['section']](s['section_position']) for s in sites]
D=np.array([[h.distance(a,b) for b in segs] for a in segs],dtype=float)
assert np.all(np.isfinite(D)) and np.max(D)<1e10 and np.allclose(D,D.T) and np.allclose(D.diagonal(),0)
ids=np.arange(500); soma=np.array([h.distance(state['cell'].soma[0](.5),s) for s in segs])
all_routes=np.array([np.lexsort((ids,D[a]))[:100] for a in ids]); anchors=[int(np.lexsort((ids,soma))[0])]; nearest=D[anchors[0]].copy(); covered=np.zeros(500,dtype=bool); covered[all_routes[anchors[0]]]=True
for _ in range(44):
 gains=(~covered)[all_routes].sum(axis=1); gains[anchors]=-1
 available=nearest.copy();available[anchors]=-1
 a=int(np.lexsort((ids,-available,-gains))[0]) if not covered.all() else int(np.argmax(available))
 anchors.append(a);nearest=np.minimum(nearest,D[a]);covered[all_routes[a]]=True
routes=np.array([np.lexsort((ids,D[a]))[:100] for a in anchors],dtype=np.int64)
counts=np.bincount(routes.ravel(),minlength=500)
assert routes.shape==(45,100) and counts.min()>0 and all(len(set(r))==100 for r in routes)
rec=dict(protocol='geometry_geodesic_cover45_nearest100_v1',precommitted=True,uses_outputs=False,uses_training_or_evaluation_voltage=False,site_metadata=str(meta),site_metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest(),description='First anchor nearest soma, then greedily maximize newly covered sites, using farthest cable-path distance to break ties; after full coverage, use farthest-point anchors; each branch receives its 100 nearest unique sites. Ties use site ID. E/I signs remain attached to physical site IDs.',anchors=anchors,routing=routes.tolist(),valid_indices_mask=[True]*4500,coverage_counts=counts.tolist(),minimum_site_coverage=int(counts.min()),max_site_coverage=int(counts.max()),branch_radius_um=[float(D[a,r].max()) for a,r in zip(anchors,routes)],old_valid_slots=3800,new_valid_slots=4500,old_empty_branches=3,new_empty_branches=0,capacity_note='Declared dimensions and trainable parameter count unchanged. Correcting empty branches increases effective valid slots from 3800 to 4500; spatial ordering and restored branch utilization are not separately identified.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=W/'morphology_mapping.json'; assert not p.exists();p.write_text(json.dumps(rec,indent=2)+'\n');np.savez_compressed(W/'morphology_distances.npz',distance_um=D,route=routes,anchors=anchors)
print(json.dumps({k:v for k,v in rec.items() if k not in ['routing','valid_indices_mask','coverage_counts']},indent=2))
