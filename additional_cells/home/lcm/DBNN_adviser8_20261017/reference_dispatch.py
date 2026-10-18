"""Use idle physical cores for the fixed references; shared per-job locks avoid duplicates."""
from pathlib import Path
import concurrent.futures as cf
import fcntl
import os
import subprocess
import sys
from common8 import write

work=Path(__file__).resolve().parent
guard=open(work/'reference_dispatch.lock','w')
fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)


def run(index):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
    with (work/'logs'/f'reference_extra_{index:02}.log').open('a') as out:
        rc=subprocess.run([sys.executable,'-u',str(work/'sample_references.py'),
            '--work',str(work),'--job',str(index),'--workers','8'],env=env,stdout=out,stderr=subprocess.STDOUT).returncode
    return dict(job=index,returncode=rc)


with cf.ThreadPoolExecutor(max_workers=10) as pool:
    results=list(pool.map(run,range(2,12)))
write(work/'reference_dispatch_complete.json',dict(passed=all(r['returncode']==0 for r in results),results=results))
