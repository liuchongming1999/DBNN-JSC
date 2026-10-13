"""Dependent experiment postprocessing, serialized across the three models."""
import argparse,fcntl,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    with open(a.work/'report.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run([sys.executable,str(a.work/'collect_snapshot.py'),'--work',str(a.work),'--data',str(a.data)],check=True)
        subprocess.run([sys.executable,str(a.work/'build_report.py'),'--work',str(a.work)],check=True)

if __name__=='__main__':main()
