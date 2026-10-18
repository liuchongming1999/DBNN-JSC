"""Shared-data ELM, official Python defaults + geometry + exact batch rule.
No capacity/optimizer/loss/seed search. This is a fresh, separate experiment.
"""
from pathlib import Path
import sys,json,hashlib,torch,numpy as np
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928'); B=R/'benchmarks/baselines_20260930'; W=Path(__file__).resolve().parent
sys.path.insert(0,str(B));import train_default as trainer
orig_config=trainer.config;orig_model=trainer.model_for
p=W/'morphology_mapping.json'; mapping=json.loads(p.read_text()); digest=hashlib.sha256(p.read_bytes()).hexdigest()
def config(method,dt):
 assert method=='elm' and dt==1
 c=orig_config(method,dt)
 c['morphology_routing_override']=dict(protocol=mapping['protocol'],file=str(p),sha256=digest,selection='Geometry only; fixed before any corrected training or held-out inference',valid_slots=4500,old_valid_slots=3800)
 c['sampler']=dict(source='Official neuronio_data_loader.generate_batch',batch_trajectories_replace=False,window_start_min_ms=500,window_size_ms=500,window_start_replace=True,shared_fit_validation_manifest_unchanged=True)
 c['adaptations'].append('Geodesic morphology neighborhoods on the500 physical sites; all45 virtual branches populated. Same fixed route as mapping-only control.')
 c['adaptations'].append('Restore official ELM batch rule:8 distinct fit trajectories per batch,500ms crop starts>=500ms. No input regeneration, added trajectories, outcome screening, hyperparameter search or shortened update budget.')
 c['specification_source']='Author public PyTorch code commit52e68a6d39523ac6613a586699b116e8e606dda3; paper/JAX differences are disclosed rather than selected by held-out performance.'
 return c
def model_for(work,cfg):
 model=orig_model(B,cfg)
 with torch.no_grad():
  model.input_to_synapse_indices.copy_(torch.tensor(mapping['routing'],dtype=model.input_to_synapse_indices.dtype).reshape_as(model.input_to_synapse_indices))
  model.valid_indices_mask.fill_(True)
 return model
def sample_batch(cases,window,batch,rng,dt):
 assert dt==1. and window==500 and batch==8
 assert len(cases)>=batch
 inds=rng.choice(len(cases),size=batch,replace=False)
 n=len(cases[int(inds[0])]['x']);assert all(len(cases[int(k)]['x'])==n for k in inds)
 ignore=round(500/dt);assert n-window-ignore>0
 starts=rng.choice(n-window-ignore,size=batch,replace=True)+ignore
 xs=[];ys=[]
 for index,start in zip(inds,starts):
  case=cases[int(index)];assert case.get('aux') is None
  xs.append(case['x'][start:start+window]);ys.append(case['y'][start:start+window])
 return np.stack(xs),np.stack(ys),None
trainer.config=config;trainer.model_for=model_for;trainer.sample_batch=sample_batch
if __name__=='__main__':trainer.main()
