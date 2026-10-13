# -*- coding: utf-8 -*-
"""Half-step verification at representative subthreshold working points."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from run_l5pc import init,simulate,key


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);a=ap.parse_args()
    d=json.loads((a.data/'l5pc_finite_strength.json').read_text());bias=d['fit_profiles'][0]['holding_current_na']
    tasks=[];levels={'proximal':1.5,'distributed':3.,'distal':1.5}
    for name,eps in levels.items():tasks.extend((name,e,i) for e,i in [(0.,0.),(eps,0.),(0.,eps),(eps,eps)])
    results={};started=time.perf_counter()
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(a.root),-70.,.05,bias)) as pool:
        for r in pool.map(simulate,tasks):results[key(r['profile'],r['strength_e'],r['strength_i'])]=r
    checks=[];raw={}
    for name,eps in levels.items():
        traces=[];errors=[]
        for e,i in [(0.,0.),(eps,0.),(0.,eps),(eps,eps)]:
            k=key(name,e,i);ref=np.load(a.data/(k+'.npz'));new=results[k]
            gap=new['voltage_mv']-ref['voltage_mv'];assert np.array_equal(new['time_ms'],ref['time_ms'])
            assert np.array_equal(new['spikes_ms'],ref['spikes_ms'])
            errors.append(dict(strength_e=e,strength_i=i,max_absolute_gap_mv=float(max(abs(gap))),rms_gap_mv=float(np.sqrt(np.mean(gap**2)))))
            traces.append(new['voltage_mv']);raw[k+'_dt005']=new['voltage_mv']
        base,ve,vi,full=traces;ve=ve-base;vi=vi-base;full=full-base;c=next(r['fixed_c_per_mv'] for r in d['fit_profiles'] if r['profile']==name)
        w=np.ones(len(base));w[[0,-1]]=.5;w/=w.sum();den=np.sqrt(w@full**2)
        linear=float(np.sqrt(w@(ve+vi-full)**2)/den);bilinear=float(np.sqrt(w@(ve+vi+c*ve*vi-full)**2)/den)
        old=next(r for r in d['rows'] if r['profile']==name and r['strength']==eps)
        checks.append(dict(profile=name,strength=eps,voltage_checks=errors,
            coarse_additive_relative_rms=old['additive_relative_rms'],fine_additive_relative_rms=linear,
            coarse_bilinear_relative_rms=old['fixed_bilinear_relative_rms'],fine_bilinear_relative_rms=bilinear,
            same_event_counts_and_times=True))
    result=dict(finished=True,reference_dt_ms=.1,control_dt_ms=.05,fixed_holding_current_na=bias,
        same_current_and_synaptic_parameters=True,readout_refitted=False,checks=checks,wall_s=time.perf_counter()-started)
    (a.data/'l5pc_half_step_validation.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.data/'l5pc_half_step_validation.npz',**raw)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
