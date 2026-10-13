#!/usr/bin/env python3
"""Run three dt panels on disjoint core sets, sequential thread configs per set."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'multisplit_results'
OUT.mkdir(exist_ok=True)
PANELS = [(.25, list(range(16,24)), [1,2,4,8]),
          (.5, list(range(24,32)), [4,1,8,2]),
          (1., list(range(88,96)), [8,4,2,1])]


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def save(records, finished=False):
    path = OUT/'campaign_status.json'
    value = dict(updated=stamp(), pid=os.getpid(), finished=finished,
                 expected_cases_per_config=31, expected_configurations=12,
                 panels=[dict(dt_ms=d, physical_cpus=c, thread_order=n) for d,c,n in PANELS],
                 records=records)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    queues = {dt:list(order) for dt,cpus,order in PANELS}
    running, records = {}, []
    failure = False
    while any(queues.values()) or running:
        for dt, cpus, order in PANELS:
            if dt in running:
                p, rec, logfile = running[dt]
                exit_code = p.poll()
                if exit_code is None:
                    continue
                logfile.close()
                rec.update(finished=stamp(), exit_code=exit_code)
                if exit_code:
                    failure = True
                    queues[dt].clear()
                print('FINISH', rec['name'], 'exit', exit_code, flush=True)
                del running[dt]
            if queues[dt] and dt not in running:
                nt = queues[dt].pop(0)
                name = f'nt{nt}_dt{dt:g}'
                cmd = [sys.executable, '-u', str(HERE/'run_multisplit_grid.py'),
                       '--root', str(ROOT), '--manifest', str(HERE/'selected_20260929/manifest.json'),
                       '--output-dir', str(OUT), '--previous-results', str(HERE/'solver_results'),
                       '--threads', str(nt), '--cpus', ','.join(map(str,cpus[:nt])),
                       '--dt', str(dt), '--repeats', '2']
                log = open(OUT/(name+'.log'), 'a', buffering=1)
                rec = dict(name=name, threads=nt, dt_ms=dt, cpus=cpus[:nt],
                           started=stamp(), command=cmd)
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                rec['pid'] = p.pid
                records.append(rec)
                running[dt] = (p,rec,log)
                print('START', name, 'pid', p.pid, 'cpus', cpus[:nt], flush=True)
        save(records)
        if running:
            time.sleep(2)
    save(records, finished=not failure)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
