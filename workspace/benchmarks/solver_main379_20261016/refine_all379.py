from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from campaign import worker,write,WORK
with ThreadPoolExecutor(max_workers=16) as pool:
 done=[]
 for f in as_completed([pool.submit(worker,"reference1",.00625,i,16,[80+i]) for i in range(16)]):
  done.append(f.result());write("fine_reference_status.json",dict(finished=len(done)==16,completed=len(done),total=16))
