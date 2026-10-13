#!/usr/bin/env python3
"""Extend the fixed 31-case panel to 16/32 cores with matched 8-core controls.

Two disjoint, single-socket CPU groups; one active configuration per group.
Every CPU id denotes a distinct physical core, not an SMT sibling.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'multisplit_highcore_results'
PANELS = [dict(cpus=list(range(16,48)), queue=[(.25,n) for n in (8,16,32)]),
          dict(cpus=list(range(80,112)), queue=[(.5,n) for n in (32,8,16)] +
                [(1.,n) for n in (16,32,8)])]


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def save(records, finished=False):
    path = OUT/'campaign_status.json'
    value = dict(updated=stamp(), pid=os.getpid(), finished=finished,
                 expected_cases_per_config=31, expected_configurations=9,
                 panels=PANELS, records=records)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    queues = [list(p['queue']) for p in PANELS]
    running,records = {},[]
    failed = False
    while any(queues) or running:
        for i,panel in enumerate(PANELS):
            if i in running:
                p,rec,log = running[i]
                result = p.poll()
                if result is None:
                    continue
                log.close()
                rec.update(finished=stamp(),exit_code=result)
                if result:
                    failed = True
                    queues[i].clear()
                print('FINISH',rec['name'],'exit',result,flush=True)
                del running[i]
            if queues[i] and i not in running:
                dt,nt = queues[i].pop(0)
                cpus = panel['cpus'][:nt]
                name = f'nt{nt}_dt{dt:g}'
                cmd = [sys.executable,'-u',str(HERE/'run_multisplit_grid.py'),
                       '--root',str(ROOT),'--manifest',str(HERE/'selected_20260929/manifest.json'),
                       '--output-dir',str(OUT),'--previous-results',str(HERE/'solver_results'),
                       '--threads',str(nt),'--cpus',','.join(map(str,cpus)),
                       '--dt',str(dt),'--repeats','2']
                log = open(OUT/(name+'.log'),'a',buffering=1)
                rec = dict(name=name,threads=nt,dt_ms=dt,cpus=cpus,started=stamp(),command=cmd)
                p = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                rec['pid'] = p.pid
                records.append(rec)
                running[i] = (p,rec,log)
                print('START',name,'pid',p.pid,'cpus',cpus,flush=True)
        save(records)
        if running:
            time.sleep(2)
    save(records,finished=not failed)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
