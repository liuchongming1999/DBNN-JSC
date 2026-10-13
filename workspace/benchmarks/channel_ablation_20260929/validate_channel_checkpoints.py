"""Validate adapted checkpoints through the unmodified DBNN_R implementation."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
base=root/'benchmarks/channel_ablation_20260929'
sys.path.insert(0,str(root/'source'))
from DBNN import DBNN_R,DBNN_F,encode_inputs
sys.path.insert(0,str(root/'benchmarks/sparse_20260928'))
from backends import capture
from quad_optimized_backend import TiledPrepared

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
data=np.load(root/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
x=encode_inputs(data['test_ip'],data['channel_type'])
rows=[]
for name in ['full_active','passive_dendrites','all_passive']:
    path=base/'fit_logtau_1000'/name
    ck=torch.load(path/'dbnn_best.pt',map_location='cpu',weights_only=False)
    model=DBNN_F(500,6001,torch.device('cpu'))
    model.load_state_dict(ck['model_state']);model.eval()
    with torch.inference_mode():
        predictions=np.concatenate([model(torch.from_numpy(z[None])).numpy() for z in x])
    saved=np.load(path/'test_predictions.npz',allow_pickle=True)
    e=predictions-saved['prediction']
    state=ck['model_state']
    row=dict(profile=name,max_abs_mv=float(abs(e).max()),rmse_mv=float(np.sqrt(np.mean(e*e))),
             finite=bool(np.isfinite(predictions).all()),
             tau_rise_ms=[float(state['tau_rise'].min()),float(state['tau_rise'].median()),float(state['tau_rise'].max())],
             tau_decay_ms=[float(state['tau_decay'].min()),float(state['tau_decay'].median()),float(state['tau_decay'].max())])
    assert row['finite'] and row['max_abs_mv']<2e-4,row
    rec=DBNN_R(500,6001,torch.device('cpu'),scan_chunk_size=32)
    rec.load_state_dict(state);rec.eval()
    with torch.inference_mode():
        pr=np.concatenate([rec(torch.from_numpy(z[None])).numpy() for z in x])
    row['original_R_chunk32_max_abs_mv']=float(np.max(abs(pr-predictions)))
    assert np.isfinite(pr).all() and row['original_R_chunk32_max_abs_mv']<2e-4,row
    xp=torch.from_numpy(x[:1]).cuda()
    s={k:v.cuda() for k,v in state.items()}
    prep=TiledPrepared(s,xp,dict(backend='triton',config=3,triangle=True,precision='ieee'))
    prep.block=512
    fn=capture(prep);gpu=[]
    with torch.inference_mode():
        for z in x:
            xp.copy_(torch.from_numpy(z[None]).cuda());gpu.append(fn().clone().cpu().numpy())
    pg=np.concatenate(gpu);row['optimized_gpu_max_abs_mv']=float(abs(pg-predictions).max())
    row['optimized_gpu_rmse_mv']=float(np.sqrt(np.mean((pg-predictions)**2)))
    assert np.isfinite(pg).all() and row['optimized_gpu_max_abs_mv']<2e-4,row
    rows.append(row);print(row,flush=True)
(base/'checkpoint_validation.json').write_text(json.dumps(dict(implementation='Unmodified DBNN_F and DBNN_R(chunk32), CPU FP32; optimized GPU affine scan+complete drive fusion+event reset+CUDA Graph',legacy_R_chunk256_note='Default inverse-power scan overflows for the small time constants in passive_dendrites; chunk32 and the stable affine GPU scan are verified instead.',rows=rows),indent=2)+'\n')
