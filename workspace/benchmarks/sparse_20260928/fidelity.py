"""Keep approximation fidelity separate from backend equivalence."""
import json,sys
from pathlib import Path
import numpy as np,torch
from experiment import ROOT,HERE,case_data
from DBNN import DBNN_R

def match_spikes(true,pred,tolerance):
    candidates=sorted((abs(float(t)-float(p)),i,j) for i,t in enumerate(true) for j,p in enumerate(pred) if abs(float(t)-float(p))<=tolerance)
    ti=set();pj=set()
    for _,i,j in candidates:
        if i not in ti and j not in pj:ti.add(i);pj.add(j)
    return len(ti)

def main():
    torch.set_num_threads(8)
    data=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    traces=np.load(HERE/'fit_traces.npz');pred=traces['prediction'];target=traces['target']
    state=torch.load(ROOT/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
    threshold=float(state['v_th']);gt=data['test_spike_time']
    out={'model_reset_threshold_mv':threshold,'neuron_event_threshold_mv':-20.,'test_samples':[], 'same_raster_neuron':[]}
    for i in range(10):
        ps=np.flatnonzero(pred[i]>=threshold).tolist();ts=list(gt[i])
        err=pred[i]-target[i]
        out['test_samples'].append({'index':i,'rmse_mv':float(np.sqrt(np.mean(err**2))),
          'max_abs_mv':float(np.max(abs(err))),'target_peak_mv':float(target[i].max()),
          'predicted_peak_mv':float(pred[i].max()),'neuron_spikes_ms':[float(t) for t in ts],
          'dbnn_reset_spikes_ms':ps,'matches':{str(tol):match_spikes(ts,ps,tol) for tol in (2,5,10)}})
    for f in sorted(HERE.glob('neuron_case*_voltage.npz')):
        ci=int(f.name.split('case')[1].split('_')[0]);meta=json.loads((HERE/f'neuron_case{ci}.json').read_text())
        if meta['N']!=500:continue
        state,x,ids,_=case_data(*[meta[k] for k in ['N','A','T','B']])
        m=DBNN_R(500,meta['T'],torch.device('cpu'));m.load_state_dict(state);m.eval()
        with torch.no_grad():y=m(x).numpy()
        yn=np.load(f)['voltage'];err=y-yn
        out['same_raster_neuron'].append({'case':ci,'A':meta['A'],'T':meta['T'],'B':meta['B'],
             'rmse_mv':float(np.sqrt((err**2).mean())),'max_abs_mv':float(np.abs(err).max())})
    (HERE/'fidelity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
