"""Portable FFT + sequential-reset replay of one locked cross-cell model.

This reference implementation is not the optimized timed implementation.
No model, threshold, kernel, or readout coefficient is fitted here.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from release_paths import REPO,resolve,read,digest
p=argparse.ArgumentParser();p.add_argument('--cell',required=True,choices=['granule','basket','golgi','hay_l5','human_l23','ca1']);p.add_argument('--profile',default='ampa',choices=['ampa','ampa_nmda']);p.add_argument('--case',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,default=REPO/'build/crosscell_replay.npz');a=p.parse_args()
w=resolve('/home/lcm/DBNN_crosscell_final_selection_test_20261017');protocol=read(w/'locked_protocol.json');locked=read(w/'models_locked.json')
i=next(i for i,j in enumerate(protocol['jobs']) if j['cell']==a.cell and j['profile']==a.profile);job=protocol['jobs'][i];model=locked['models'][str(i)]
cp=resolve(model['checkpoint']);assert digest(cp)==model['checkpoint_sha256']
row=next(r for r in read(resolve(job['test_manifest']))['cases'] if r['case_id']==a.case)
f=np.load(resolve(row['input_file']));ids=f['ids'].astype(np.int64);raw=f['x'];signed=raw.astype(np.int8)*np.where(ids[:,None]<250,1,-1).astype(np.int8)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
state=torch.load(cp,map_location=a.device,weights_only=False)['model_state'];index=torch.tensor(ids,device=a.device);n=raw.shape[-1];age=torch.arange(n,device=a.device)[None]
with torch.inference_mode():
    omega=state['omega'][index];rise=state['tau_rise'][index];decay=state['tau_decay'][index]
    k=omega[:,None]*(1-torch.exp(-age/rise[:,None]))*torch.exp(-age/decay[:,None]);x=torch.from_numpy(signed[None]).to(a.device)
    nf=1 << (2*n-2).bit_length();y=torch.fft.irfft(torch.fft.rfft(x.float(),n=nf)*torch.fft.rfft(k,n=nf)[None],n=nf)[...,:n].transpose(1,2).contiguous()
    W=state['bilinear.weight'][0][index][:,index];drive=(((y@W)*y).sum(-1)+y.sum(-1)+state['bilinear.bias'])[0].cpu().numpy()
    threshold,amplitude,tau=[float(state[k]) for k in ['v_th','reset_amp','tau_reset']]
alpha=np.exp(-1./tau);reset=0.;voltage=np.empty(n,np.float64);events=[]
for i in range(n):
    voltage[i]=float(drive[i])+reset;fire=voltage[i]>=threshold
    if fire:events.append(float(i))
    reset=alpha*reset-amplitude*int(fire)
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,voltage=voltage,drive=drive,time_ms=np.arange(n),predicted_events_ms=events)
frozen=np.load(resolve(job['base'])/'evaluation/predictions'/(a.case+'.npz'));gap=float(np.max(np.abs(voltage-frozen['voltage'])))
same=np.array_equal(np.asarray(events),frozen['predicted_events_ms'])
print(json.dumps({'cell':a.cell,'profile':a.profile,'case':a.case,'checkpoint_sha256':digest(cp),'max_voltage_gap_mv':gap,'events_identical':same,'reference_backend':'FFT + sequential FP64 reset; not benchmarked timing','output':str(a.output)},indent=2))
assert gap<.001 and same,'Reference replay differs from the frozen numerical-equivalence tolerance on this hardware/library.'
