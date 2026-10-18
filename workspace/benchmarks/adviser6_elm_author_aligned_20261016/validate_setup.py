"""Preflight: exact author sampler equivalence plus shared dataset provenance."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch
import train_author_aligned_elm as adapter
R=adapter.R;B=adapter.B;W=adapter.W
sys.path.insert(0,str(B/'vendor/elmneuron'))
from src.neuronio.neuronio_data_loader import generate_batch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ncase,n,k=12,1601,500
 times=np.arange(n,dtype=np.float32)
 raw=np.broadcast_to(times[None,:,None],(ncase,n,k)).copy()
 raw+=np.arange(ncase,dtype=np.float32)[:,None,None]*10000
 signs=np.where(np.arange(k)<250,1.,-1.).astype(np.float32)
 spike=np.broadcast_to((np.arange(n)%13==0)[None,:,None],(ncase,n,1)).copy().astype(np.float32)
 soma=np.broadcast_to(times[None,:,None],(ncase,n,1)).copy()*.001
 cases=[dict(x=raw[i]*signs,y=np.concatenate((spike[i],soma[i]),axis=-1),aux=None) for i in range(ncase)]
 for seed in [0,930291,20261016]:
  xx,yy,dd=adapter.sample_batch(cases,500,8,np.random.default_rng(seed),1.)
  expected,targets=generate_batch(torch.from_numpy(raw),torch.from_numpy(spike),torch.from_numpy(soma),torch.from_numpy(signs),np.random.default_rng(seed),batch_size=8,input_window_size=500,ignore_time_from_start=500,neuronio_sim_per_file=ncase,neuronio_sim_len=n,neuronio_label_dim=2,neuronio_data_dim=500)
  np.testing.assert_array_equal(xx,expected.numpy());np.testing.assert_array_equal(yy[...,0],targets[0].numpy());np.testing.assert_array_equal(yy[...,1],targets[1].numpy())
  indices=(xx[:,0,0]//10000).astype(int);starts=(xx[:,0,0]%10000).astype(int)
  assert len(np.unique(indices))==8 and starts.min()>=500 and starts.max()+500<n
 old=json.loads((B/'aligned/elm_1ms/resolved_args.json').read_text());new=adapter.config('elm',1.)
 for key in ['model','seed','batch_size','epochs','batches_per_epoch','window_ms','burn_in_loss_ms','overlap_ms','lr','optimizer','loss','precision']:
  assert old[key]==new[key],(key,old[key],new[key])
 manifest=R/'benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json';rows=json.loads(manifest.read_text())['cases'];fit=[r for r in rows if r['split']=='fit'];val=[r for r in rows if r['split']=='validation'];held=json.loads((R/'benchmarks/solver_main379_20261016/manifest.json').read_text())['cases'];assert len(fit)==117 and len(val)==32
 assert not ({r['input_sha256'] for r in rows}&{r['input_sha256'] for r in held})
 for r in rows:
  with np.load(r['input_file']) as f:assert hashlib.sha256(f['x'].tobytes()).hexdigest()==r['input_sha256']
 for method in ['tcn','lstm']:
  locked=json.loads((B/'aligned'/f'{method}_1ms/complete.json').read_text())['config'];assert locked['manifest_sha256']==sha(manifest) and locked['fit_cases']==117 and locked['validation_cases']==32
 mapping_hash=sha(W/'morphology_mapping.json');old_hash=sha(R/'benchmarks/adviser6_mapping_engineering_20261016/morphology_mapping.json');assert mapping_hash==old_hash
 rec=dict(passed=True,sampler_exact_against_official_generate_batch=True,sampler_fixture_seeds=[0,930291,20261016],seed_tests_are_index_fixtures_not_training_runs=True,shared_fit_cases=117,shared_validation_cases=32,manifest=str(manifest),manifest_sha256=sha(manifest),heldout_inputs=379,input_hash_overlap=0,all_source_input_hashes_verified=True,shared_manifest_matches_tcn_and_lstm=True,architecture_optimizer_loss_budget_unchanged=True,mapping_sha256=mapping_hash,mapping_unchanged=True,total_updates=350000,specification='PublicPython defaults; nativeNEURON event convention and validation-only calibration retained for common evaluation')
 (W/'preflight.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
