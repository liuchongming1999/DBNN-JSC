"""Build the recorded CPU/GPU implementations; generated binaries stay untracked."""
import argparse,subprocess,shutil,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--cutlass',type=Path);p.add_argument('--cpu-only',action='store_true');p.add_argument('--cuda-arch',default='sm_120');a=p.parse_args()
w=a.workspace.resolve();s=w/'benchmarks/sparse_20260928'
def run(cmd):subprocess.run([str(x) for x in cmd],check=True)
run(['g++','-O3','-std=c++17','-fPIC','-shared','-fopenmp','-mavx512f','-mavx512dq','-mfma','-ffp-contract=off',w/'benchmarks/cpu_20260929/cpu_inference.cpp','-o',w/'benchmarks/cpu_20260929/cpu_inference.so'])
run(['g++','-O3','-std=c++17','-fPIC','-shared','-fopenmp',s/'native_cpu.cpp','-o',s/'native_cpu.so'])
reset=w/'benchmarks/fair_window_20261005/seq_reset.c'
for d in ['diagonal_1ms_20261009','spike_panel_20261009','fair_window_20261005','fair_window_20261005/dbnn']:
 dest=w/'benchmarks'/d;dest.mkdir(parents=True,exist_ok=True)
 run(['gcc','-O3','-shared','-fPIC',reset,'-lm','-o',dest/'seq_reset.so'])
if not a.cpu_only:
 if a.cutlass is None:raise SystemExit('Provide --cutlass /path/to/cutlass-3.9.2 (download instructions in docs/REPRODUCING.md).')
 nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
 run([nvcc,'-O3','-std=c++17','-shared','-Xcompiler','-fPIC','-arch='+a.cuda_arch,'-I'+str(a.cutlass.resolve()/'include'),s/'native_kernels.cu','-o',s/'native_kernels_v2.so'])
 run([nvcc,'-O3','-std=c++17','-shared','-Xcompiler','-fPIC','-arch='+a.cuda_arch,s/'quadratic_tiled.cu','-o',s/'quadratic_tiled_native.so'])
print('Native builds completed.')
