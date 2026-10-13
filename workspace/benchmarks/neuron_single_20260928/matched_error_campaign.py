"""Fixed-step error/latency frontier on the three frozen six-second inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
R = H / 'matched_error_20260929'
R.mkdir(exist_ok=True)
PY = ROOT / '.venv/bin/python'


def snapshot():
    return dict(time=time.strftime('%FT%T%z'), loadavg=os.getloadavg(), gpu=subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu',
         '--format=csv,noheader'], text=True))


def run_job(mode, dt, repeats=3):
    tag = f'{mode}_dt{dt:g}'
    out = R / f'{tag}.json'
    if out.exists() and json.loads(out.read_text()).get('finished'):
        return
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'CUDA_VISIBLE_DEVICES': '6', 'PYTHONFAULTHANDLER': '1'}
    if mode == 'core_gpu':
        env['CORENEURONLIB'] = '/data/lcm/core_gpu/l5pc_model253/x86_64/libcorenrnmech.so'
        env['DBNN_NEURON_MODEL_ROOT'] = '/data/lcm/core_gpu/l5pc_model253'
    cmd = ['taskset', '-c', '74', str(PY), str(H / 'matched_error_worker.py'),
           '--engine', 'neuron' if mode == 'neuron' else 'core', '--dt', str(dt),
           '--steps', '6001', '--samples', '0', '1', '2', '--repeats', str(repeats),
           '--threads', '1', '--cell-permute', '2' if mode == 'core_gpu' else '0',
           '--warp-balance', '4', '--output', str(out)]
    if mode == 'core_gpu':
        cmd.append('--gpu')
    before = snapshot()
    print('START', tag, flush=True)
    (R / 'status.json').write_text(json.dumps(dict(current=tag, phase='running', **before), indent=2))
    with out.with_suffix('.log').open('w') as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=2400)
    after = snapshot()
    record = json.loads(out.read_text()) if out.exists() else dict(finished=False)
    if proc.returncode != 0:
        record = dict(finished=False, created=time.strftime('%FT%T%z'),
                      error=f'Worker exited with return code {proc.returncode}; see {tag}.log')
    record.update(resource_before=before, resource_after=after, command=cmd, returncode=proc.returncode,
                  comparison_mode=mode, shared_resources=True)
    if mode == 'core_gpu':
        record['gpu_library'] = env['CORENEURONLIB']
        record['gpu_library_sha256'] = hashlib.sha256(Path(env['CORENEURONLIB']).read_bytes()).hexdigest()
        record['gpu_build'] = 'NVHPC 25.3 / CUDA 12.8 / sm_120; cell_permute=2 / warp_balance=4'
    out.write_text(json.dumps(record, indent=2))
    print('DONE', tag, record.get('finished'), record.get('timing', {}).get('solver_call', {}), flush=True)
    if not record.get('finished'):
        print(out.with_suffix('.log').read_text()[-2000:], flush=True)


def dbnn():
    import numpy as np
    import torch
    from dbnn_worker import case
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for s in [0, 1, 2]:
        out = R / f'dbnn_sample{s}.json'
        if out.exists() and json.loads(out.read_text()).get('finished'):
            continue
        before = snapshot()
        row, voltage = case(500, 6001, [s], 'sampled', 'cuda', 1, 40)
        row.update(finished=True, resource_before=before, resource_after=snapshot(),
                   shared_resources=True, voltage_file=out.with_suffix('.npz').name)
        np.savez_compressed(out.with_suffix('.npz'), voltage=voltage)
        out.write_text(json.dumps(row, indent=2))
        print('DBNN', s, row['transfer']['dense_gather_h2d_forward_d2h'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['cpu', 'gpu', 'dbnn'], required=True)
    ap.add_argument('--dts', type=float, nargs='+')
    a = ap.parse_args()
    if a.phase == 'dbnn':
        dbnn()
    else:
        dts = a.dts or [.1, .25, .5, .8, 1., 1.25, 2., 4.]
        for dt in dts:
            for mode in (['neuron', 'core_cpu'] if a.phase == 'cpu' else ['core_gpu']):
                run_job(mode, dt)
    required = ([R / f'dbnn_sample{s}.json' for s in [0, 1, 2]] if a.phase == 'dbnn' else
                [R / f'{mode}_dt{dt:g}.json' for dt in dts for mode in
                 (['neuron', 'core_cpu'] if a.phase == 'cpu' else ['core_gpu'])])
    success = all(p.exists() and json.loads(p.read_text()).get('finished') for p in required)
    (R / f'complete_{a.phase}.json').write_text(json.dumps(dict(finished=success,
        requested=[p.name for p in required], **snapshot()), indent=2))
