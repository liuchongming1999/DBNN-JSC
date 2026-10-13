"""Refresh evidence and the canonical report, preserving other mainline sections."""
import os,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE='/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/baselines_20260930'
cmd=shlex.join(['env','LD_LIBRARY_PATH=/home/lcm/miniconda3/envs/torch/lib','/home/lcm/DBNN_L5PC_surrogate_20260928/.venv/bin/python',REMOTE+'/collect_snapshot.py','--work',REMOTE,'--data','/data/lcm/DBNN_baselines_20260930'])
subprocess.run(['ssh','5090',cmd],check=True)
subprocess.run(['rsync','-az','5090:'+REMOTE+'/snapshot/',str(HERE/'snapshot')+'/'],check=True)
subprocess.run([sys.executable,str(HERE/'update_report.py')],check=True)
