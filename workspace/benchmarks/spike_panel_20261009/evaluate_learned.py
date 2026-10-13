"""Existing locked 1-ms TCN, Branch-ELM and LSTM; no new training/calibration."""
import argparse
import hashlib
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import torch
from scipy.signal import find_peaks
from common_panel import read,write,sha
from score_records import grouped


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--method',choices=['tcn','elm','lstm'],required=True);p.add_argument('--recipe',choices=['common150','author_default'],default='common150')
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();method=a.method
    source=root/'benchmarks/baselines_20260930'
    base=source if method=='lstm' or a.recipe=='author_default' else root/'benchmarks/fair_window_20261005/boundary_controls'
    src=base/'aligned'/f'{method}_1ms';lock=read(src/'complete.json');threshold_record=read(src/'event_threshold.json')
    cp=src/'best.pt';assert sha(cp)==lock['checkpoint_sha256']==threshold_record['checkpoint_sha256']
    assert lock['default_budget_completed'] and lock['updates']==lock['total_updates']
    cfg=lock['config'];assert cfg['dt_ms']==1.
    out=work/'methods'/f'{method}_{a.recipe}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('ALREADY_COMPLETE');return
    sys.path[:0]=[str(source),str(source/'deps'),str(root/'benchmarks/accuracy_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from train_default import model_for,predict_case
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_for(source,cfg).cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
    cal=lock['validation']['calibration'];threshold=threshold_record['threshold']

    def predict(row):
        inp=np.load(row['input_file']);ids,raw=inp['ids'],inp['x']
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        x=np.zeros((row['duration_ms']+1,500),np.float32);x[:,ids]=raw.T
        if method!='tcn':x[:,250:]*=-1
        with torch.inference_mode():
            y=predict_case(model,torch.from_numpy(x).cuda(),cfg,'cuda')
            v=y[:,1]*(10 if method!='tcn' else 1)-67.7
            v=v*cal['scale']+cal['bias_mv'];prob=torch.sigmoid(y[:,0])
        v,prob=v.cpu().numpy(),prob.cpu().numpy()
        peaks,_=find_peaks(prob,height=threshold,distance=5)
        return v,prob,peaks.astype(float)

    # Reuse a known held-out raster solely to check the inference adapter.
    old_manifest=read(root/'benchmarks/dt_resolution_20260929/test/manifest.json')
    oldrow=next(r for r in old_manifest['cases'] if r['case_id']=='case_0062')
    pv,pp,pe=predict(oldrow);old=np.load(src/'evaluation/case_0062.npz')
    gap=float(np.max(abs(pv-old['voltage'])))
    assert gap<1e-4,(method,gap,'voltage adapter')
    assert np.array_equal(pe,old['spike_times']),(method,'event adapter')
    protocol=dict(method=method,recipe=a.recipe,checkpoint=str(cp),checkpoint_sha256=sha(cp),sample_interval_ms=1.,
                  training_config=cfg,voltage_calibration=cal,threshold=threshold,threshold_protocol=threshold_record['version'],
                  threshold_calibration_data='Original validation set; reused without new selection',test_used_for_selection=False,
                  adapter_validation=dict(passed=True,case_id='case_0062',voltage_max_gap_mv=gap,event_times_exact=True))
    write(out/'protocol.json',protocol)
    manifest=read(work/'confirmatory/manifest.json');records=[]
    for row in manifest['cases']:
        cid=row['case_id'];v,prob,events=predict(row)
        ref=np.load(work/'confirmatory/references'/f'{cid}_reference.npz');rr=read(work/'confirmatory/references'/f'{cid}_reference.json')
        mm=metric(ref['voltage'],v,ref['spike_times'],1.);ee=match_events(ref['spike_times'],events)
        records.append(dict(**row,reference_spikes=rr['spike_count'],minimum_isi_ms=rr['minimum_isi_ms'],low_density=rr['low_density'],
                            voltage=mm,events=ee,reference_events_ms=ref['spike_times'].tolist(),predicted_events_ms=events.tolist()))
        (out/'predictions').mkdir(exist_ok=True)
        np.savez_compressed(out/'predictions'/f'{cid}.npz',voltage=v,probability=prob,spike_times=events,
                            reference_voltage=ref['voltage'],reference_events_ms=ref['spike_times'],time_ms=ref['time_ms'])
        if len(records)%20==0:
            write(out/'status.json',dict(completed=len(records),total=len(manifest['cases'])));print('EVAL',method,len(records),flush=True)
    assert sha(cp)==lock['checkpoint_sha256']
    result=dict(finished=True,created=datetime.now().isoformat(),protocol=protocol,records=records,
                manifest_sha256=sha(work/'confirmatory/manifest.json'),**grouped(records,aggregate,event_aggregate))
    write(out/'complete.json',result);print('RESULT',method,result['low_density']['events'],flush=True)


if __name__=='__main__':main()
