"""Continue the frozen solver campaign; no training or manuscript changes."""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK=ROOT/'benchmarks/solver_convergence_20261016'


def write(name,value):
    p=WORK/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)


def compare(coarse,fine):
    sse=n=0;all_equal=True;maxgap=0.;nref=0;differences=[]
    for row in json.loads((WORK/'manifest.json').read_text())['cases']:
        cid=row['case_id']
        with np.load(WORK/'outputs/reference1'/f'dt{coarse:g}'/f'{cid}.npz') as a, np.load(WORK/'outputs/reference1'/f'dt{fine:g}'/f'{cid}.npz') as b:
            mask=np.ones(len(b['voltage']),bool)
            for spike in b['spike_times_ms']:mask&=np.abs(b['time_ms']-spike)>10
            err=a['voltage'][mask].astype(float)-b['voltage'][mask].astype(float)
            sse+=float(err@err);n+=len(err)
            ra=a['spike_times_ms'];rb=b['spike_times_ms'];nref+=len(rb)
            eq=len(ra)==len(rb);all_equal&=eq
            gap=float(np.max(np.abs(ra-rb))) if eq and len(ra) else 0. if eq else None
            if gap is not None:maxgap=max(maxgap,gap)
            if not eq or (gap is not None and gap>.5):differences.append(dict(case_id=cid,coarse_events=ra.tolist(),fine_events=rb.tolist(),max_time_gap_ms=gap))
    rmse=float(np.sqrt(sse/n))
    return dict(coarse_dt_ms=coarse,fine_dt_ms=fine,pooled_masked_rmse_mv=rmse,
        identical_counts_per_input=bool(all_equal),max_event_time_difference_ms=maxgap if all_equal else None,
        finer_reference_events=nref,material_event_differences=differences,
        passed=bool(rmse<=.01 and all_equal and maxgap<=.5))


def main():
    write('orchestration_status.json',dict(phase='waiting_for_reference_sweep',finished=False))
    while True:
        p=WORK/'reference_status.json'
        if p.exists() and json.loads(p.read_text()).get('finished'):break
        time.sleep(5)
    log=(WORK/'logs/timing_controller.log').open('a')
    cmd=[sys.executable,str(WORK/'run_grid.py'),'--root',str(ROOT),'--work',str(WORK),'--phase','timing']
    timing=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    write('timing_launch.json',dict(pid=timing.pid,command=cmd))
    history=[];coarse=.05;fine=.025
    for candidate in [.025,.0125,.00625,.003125]:
        if candidate!=.025:
            with (WORK/'logs'/f'adaptive_dt{candidate:g}.log').open('a') as f:
                subprocess.run([sys.executable,str(WORK/'run_grid.py'),'--root',str(ROOT),'--work',str(WORK),
                    '--phase','reference','--steps',str(candidate)],stdout=f,stderr=subprocess.STDOUT,check=True)
        fine=candidate;result=compare(coarse,fine);history.append(result)
        write('reference_convergence.json',dict(criterion=dict(rmse_mv=.01,max_event_gap_ms=.5,counts_equal=True),
            history=history,finest_dt_ms=fine,passed=result['passed']))
        write('orchestration_status.json',dict(phase='reference_refinement' if not result['passed'] else 'timing',
            finished=False,reference_history=history,timing_pid=timing.pid))
        if result['passed']:break
        coarse=fine
    rc=timing.wait();log.close()
    write('orchestration_status.json',dict(finished=rc==0,phase='complete' if rc==0 else 'timing_failed',
        timing_returncode=rc,reference_history=history,finest_dt_ms=fine,
        reference_stability_criterion_met=history[-1]['passed'],updated=datetime.now().isoformat()))
    assert rc==0


if __name__=='__main__':main()
