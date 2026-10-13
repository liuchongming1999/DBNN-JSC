"""Numerical engine-transfer check, isolated from training/test timing."""
import argparse,json,os
from pathlib import Path
import numpy as np
from models import build,run,REGISTRY,register_core_sources
from screen import write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=REGISTRY,required=True);ap.add_argument('--engine',choices=['neuron','core','multisplit'],required=True);a=ap.parse_args()
    work=Path(__file__).resolve().parent;out=work/'results'/a.model/'adapter_checks';out.mkdir(parents=True,exist_ok=True)
    env=build(work,a.model,'ampa_nmda');h=env['h'];pc=env['pc'];pc.nthread(8 if a.engine=='multisplit' else 1)
    if a.engine=='core':register_core_sources(env)
    if a.engine=='multisplit':
        h.load_file('parcom.hoc');par=h.ParallelComputeTool();par.multisplit(1)
        env['vv'].record(env['clamp'],env['soma'](.5)._ref_v);env['tt'].record(env['clamp'],h._ref_t)
    ids=np.array([5,50,105,155,220,255,300,355,405,470]);x=np.zeros((10,101),np.uint8)
    x[:,[10,40,80]]=1
    for repeat in range(2):r=run(env,100.,ids,x,weight=1e-6,holding=-.005,dt=.025,warmup=200.,engine=a.engine)
    t=np.arange(401)*.25;v=np.interp(t,r['time_ms'],r['voltage']);np.savez_compressed(out/(a.engine+'.npz'),voltage=v,spike_times=r['spike_times'])
    write(out/(a.engine+'.json'),dict(finished=True,engine=a.engine,threads=8 if a.engine=='multisplit' else 1,repetitions=2,dt_ms=.025,profile='ampa_nmda',purpose='Numerical compatibility and cached warm-state/event replay check; not a benchmark or final evaluation.'))
    if (out/'neuron.npz').exists():
        ref=np.load(out/'neuron.npz');err=v-ref['voltage'];checks=dict(max_abs_mv=float(np.max(abs(err))),rmse_mv=float(np.sqrt(np.mean(err**2))),spike_count_equal=len(r['spike_times'])==len(ref['spike_times']))
        checks['passed']=checks['max_abs_mv']<.005 and checks['spike_count_equal'];write(out/(a.engine+'_parity.json'),checks)
        if not checks['passed']:raise RuntimeError(checks)
        print('ADAPTER_PASS',a.model,a.engine,checks,flush=True)

if __name__=='__main__':main()
