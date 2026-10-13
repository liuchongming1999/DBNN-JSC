"""Continue after the NEURON-only pilot; freeze new test inputs then evaluate."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
from common_panel import read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();py=root/'.venv/bin/python'
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'0'}
    state=dict(started=datetime.now().isoformat(),phase='waiting_for_pilot')
    write(work/'campaign_status.json',state)
    deadline=time.monotonic()+600
    while not (work/'pilot/reference_complete.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Pilot timeout; inspect pilot log')
        time.sleep(5)
    assert read(work/'pilot/reference_complete.json')['finished']
    stages=[('generate_confirmatory','generate_panel.py',['--phase','confirmatory']),
            ('reference_confirmatory','run_references.py',['--phase','confirmatory','--workers','16']),
            ('evaluate_frozen_full_matrix','evaluate_panel.py',[])]
    for phase,script,args in stages:
        state.update(phase=phase,updated=datetime.now().isoformat());write(work/'campaign_status.json',state)
        complete=work/('confirmatory/manifest.json' if phase=='generate_confirmatory' else 'confirmatory/reference_complete.json' if phase=='reference_confirmatory' else 'evaluation.json')
        if complete.exists():continue
        with open(work/f'{phase}.log','w') as log:
            command=[str(py),str(work/script),'--root',str(root),'--work',str(work),*args]
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    state.update(phase='complete',finished=datetime.now().isoformat(),result='evaluation.json')
    write(work/'campaign_status.json',state)


if __name__=='__main__':main()
