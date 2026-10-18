"""Geometry-only adaptation around the unchanged author-default trainer."""
from pathlib import Path
import sys,json,hashlib,torch
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928'); B=R/'benchmarks/baselines_20260930'; W=R/'benchmarks/adviser6_mapping_engineering_20261016'
sys.path.insert(0,str(B));import train_default as trainer
orig_config=trainer.config;orig_model=trainer.model_for
p=W/'morphology_mapping.json'; mapping=json.loads(p.read_text()); digest=hashlib.sha256(p.read_bytes()).hexdigest()
def config(method,dt):
 assert method=='elm' and dt==1
 c=orig_config(method,dt);c['morphology_routing_override']=dict(protocol=mapping['protocol'],file=str(p),sha256=digest,selection='Single geometry-only mapping fixed before training; no output-based selection',valid_slots=4500,old_valid_slots=3800)
 c['adaptations'].append('Replace index-window routing by morphology geodesic neighborhoods; all45 branches nonempty; preserve author architecture, seed, loss, optimizer and complete350000-update budget. No hyperparameter search.')
 return c
def model_for(work,cfg):
 model=orig_model(B,cfg)
 with torch.no_grad():
  model.input_to_synapse_indices.copy_(torch.tensor(mapping['routing'],dtype=model.input_to_synapse_indices.dtype).reshape_as(model.input_to_synapse_indices))
  model.valid_indices_mask.fill_(True)
 assert model.input_to_synapse_indices.numel()==4500
 return model
trainer.config=config;trainer.model_for=model_for
if __name__=='__main__':trainer.main()
