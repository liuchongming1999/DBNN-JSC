"""Accuracy followed by three prepared inference backends, with fixed resources."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK = ROOT / 'benchmarks/random_init_main379_20261012'
PANEL = ROOT / 'benchmarks/spike_panel_20261009'
CHECKPOINT = ROOT / 'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
SHA256 = '737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'


def env(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    logs = WORK / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / 'evaluation.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(WORK / 'evaluate_direct.py'),
                        '--root', str(ROOT), '--panel', str(PANEL), '--work', str(WORK),
                        '--checkpoint', str(CHECKPOINT), '--checkpoint-sha256', SHA256],
                       cwd=ROOT, env=env(0), stdout=log, stderr=subprocess.STDOUT, check=True)
    jobs = []
    for method, gpu, cpus in [('gpu', 0, '72'), ('cpu1', 1, '80'), ('cpu8', 2, '88,89,90,91,92,93,94,95')]:
        log = (logs / f'{method}.log').open('a')
        p = subprocess.Popen([sys.executable, '-u', str(WORK / 'benchmark_direct.py'),
                              '--root', str(ROOT), '--work', str(PANEL), '--method', method,
                              '--checkpoint', str(CHECKPOINT), '--checkpoint-sha256', SHA256,
                              '--predictions', str(WORK / 'predictions'),
                              '--output', str(WORK / method), '--cpus', cpus, '--repeats', '3'],
                             cwd=ROOT, env=env(gpu), stdout=log, stderr=subprocess.STDOUT)
        jobs.append((method, p, log))
    (WORK / 'state.json').write_text(json.dumps(dict(status='timing', started=datetime.now().isoformat(),
                                                  jobs=[dict(method=m, pid=p.pid) for m,p,_ in jobs]), indent=2))
    failures = []
    for method, p, log in jobs:
        rc = p.wait()
        log.close()
        if rc:
            failures.append(dict(method=method, returncode=rc))
    if failures:
        (WORK / 'state.json').write_text(json.dumps(dict(status='failed', failures=failures), indent=2))
        raise RuntimeError(failures)
    (WORK / 'state.json').write_text(json.dumps(dict(status='complete', finished=datetime.now().isoformat()), indent=2))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
