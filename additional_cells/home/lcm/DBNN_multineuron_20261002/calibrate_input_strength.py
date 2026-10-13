"""Matched no-input/input weak-probe calibration; no surrogate or main test."""
import argparse,json,time
from pathlib import Path
import numpy as np
from models import build,run,register_core_sources
from screen import window,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',required=True);a=p.parse_args()
    w=a.work.resolve();out=w/'results'/a.model/'screen';op=json.loads((out/'operating_point.json').read_text())
    env=build(w,a.model,'ampa');register_core_sources(env)
    kw=dict(holding=op['holding_current_na'],dt=.025,engine='core')
    duration=500.;base=window(run(env,duration,**kw),duration);rng=np.random.default_rng(61002011)
    ids=np.arange(250);x=rng.poisson(.01,(250,501)).astype(np.uint8);x[:,0]=0
    records=[];weight=1e-6;best=None
    for k in range(5):
        r=run(env,duration,ids,x,weight=weight,**kw);v=window(r,duration)
        delta=float(np.mean((v-base)[-801:]));n=len(r['spike_times']);rec=dict(weight_us=weight,mean_depolarization_mv=delta,spike_count=n)
        records.append(rec);print('CALIBRATION_PROBE',a.model,rec,flush=True)
        if n==0 and delta>0:best=rec
        if n==0 and 2.25<=delta<=2.75:break
        if n:weight=weight*.5
        else:weight=float(np.clip(weight*2.5/max(delta,1e-5),1e-7,1e-3))
    if best is None:raise RuntimeError(('No valid subthreshold weak-probe calibration',records))
    candidate={**op,'weight_us':best['weight_us'],'weight_selection':'Independent 250E/10Hz 500ms weak probe, same warm state and matched no-input baseline; scale toward 2.5mV depolarization while retaining zero spikes. No fit/validation/test outcomes used.',
        'input_calibration':dict(reference_dt_ms=.025,engine='core',source='Numerically verified CoreNEURON implementation of the same neuron',duration_ms=duration,records=records,selected=best),
        'calibrated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write(out/'calibrated_operating_point_candidate.json',candidate)

if __name__=='__main__':main()
