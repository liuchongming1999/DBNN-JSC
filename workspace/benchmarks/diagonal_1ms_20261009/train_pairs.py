"""Run a paired 1ms diagonal comparison without changing mainline artifacts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK = ROOT / 'benchmarks/diagonal_1ms_20261009'
TRAIN = WORK / 'train_dbnn_diagonal.py'
BASE = ['--root', str(ROOT), '--work', str(WORK), '--dt', '1',
        '--window', '500', '--burn', '150', '--seed', '291049']


def save(name, obj):
    path = WORK / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def run_pair(smoke):
    running = []
    for label, gpu, flags in [('zero_diag', 0, []), ('full_diag', 1, ['--allow-diagonal'])]:
        command = [sys.executable, '-u', str(TRAIN)] + BASE + flags
        if smoke:
            command += ['--smoke', '--epochs', '1', '--refine-epochs', '1']
        else:
            command += ['--epochs', '2000', '--refine-epochs', '400', '--lr', '0.003']
        log = (WORK / 'logs' / (label + ('_smoke' if smoke else '') + '.log')).open('a')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                cwd=ROOT, env=environment(gpu))
        running.append((label, proc, log, gpu))
    save('state.json', dict(status='smoke' if smoke else 'training',
                           started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                           jobs=[dict(arm=a, pid=p.pid, gpu=g) for a, p, _, g in running]))
    failed = []
    for label, proc, log, _ in running:
        rc = proc.wait()
        log.close()
        if rc:
            failed.append(dict(arm=label, returncode=rc))
    if failed:
        save('state.json', dict(status='failed', failures=failed))
        raise RuntimeError(failed)


def main():
    WORK.mkdir(exist_ok=True)
    (WORK / 'logs').mkdir(exist_ok=True)
    lock = (WORK / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (WORK / 'comparison.json').exists():
        print('ALREADY_COMPLETE', flush=True)
        return
    with (WORK / 'logs/diagonal_validation.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(WORK / 'check_diagonal.py')],
                       cwd=WORK, env=environment(0), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    run_pair(True)
    run_pair(False)
    arms = {}
    for label in ['zero_diag', 'full_diag']:
        folder = WORK / ('dbnn_w500_dt1ms_burn150_' + label)
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete['finished'] and not complete['smoke']
        assert complete['updates'] == 36000
        checkpoint = Path(complete['checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == complete['checkpoint_sha256']
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
        diag = state['bilinear.weight'][0].diag()
        if label == 'zero_diag':
            assert diag.abs().max() == 0
        else:
            assert diag.abs().max() > 0
        arms[label] = dict(summary=complete['summary'], selected=complete['selected'],
                          checkpoint=str(checkpoint), checkpoint_sha256=complete['checkpoint_sha256'],
                          diagonal_max_abs=float(diag.abs().max()),
                          diagonal_l2=float(diag.norm()), protocol=complete['protocol'],
                          records=complete['records'])
    z, f = arms['zero_diag']['protocol'], arms['full_diag']['protocol']
    for key in ['initial_state_digest', 'manifest_sha256', 'ordered_input_digest',
                'fit_case_ids', 'validation_case_ids', 'seed', 'primary_epochs',
                'refinement_epochs', 'learning_rate', 'optimizer', 'loss', 'selection']:
        assert z[key] == f[key], key
    assert [r['input_sha256'] for r in arms['zero_diag']['records']] == [r['input_sha256'] for r in arms['full_diag']['records']]
    save('comparison.json', dict(finished=True, completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                scope='Paired from-scratch 1ms training; same initial physical state, source, crops, loss, budget and validation; only diagonal trainability differs. Same31 evaluation after locks.',
                                pairing_verified=True, arms=arms))
    save('state.json', dict(status='complete', finished=True,
                           completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
