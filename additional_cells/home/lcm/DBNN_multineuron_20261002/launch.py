"""Launch three finite campaigns only after stable unoccupied-GPU checks."""
import csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from models import REGISTRY
from screen import write

def main():
    work=Path(__file__).resolve().parent;root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');data=Path('/data/lcm/DBNN_multineuron_20261002');data.mkdir(parents=True,exist_ok=True)
    lock=open(work/'launch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (work/'launch.json').exists():
        old=json.loads((work/'launch.json').read_text());active=[]
        for r in old.get('workers',[]):
            p=Path(f'/proc/{r["pid"]}/cmdline')
            if p.exists() and 'DBNN_multineuron_20261002/campaign.py' in p.read_bytes().replace(b'\0',b' ').decode():active.append(r)
        if active:print('ALREADY_RUNNING',active);return
    mapping=dict(granule=4,golgi=5,basket=6);observations=[]
    for i in range(3):
        result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        rows={int(r[0]):(int(r[1]),int(r[2])) for r in csv.reader(result.stdout.splitlines())};observations.append(dict(time=time.time(),gpus=rows))
        if any(rows[g][0]>2 or rows[g][1]>100 for g in mapping.values()):
            write(work/'launch_blocked.json',dict(reason='A selected GPU is occupied; no process was stopped.',observations=observations));raise RuntimeError('Selected GPUs not idle')
        if i<2:time.sleep(4)
    workers=[]
    for name,gpu in mapping.items():
        out=work/'results'/name;out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',str(work/'campaign.py'),'--work',str(work),'--root',str(root),'--data',str(data),'--model',name,'--gpu',str(gpu)]
        with open(out/'campaign.log','a') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        workers.append(dict(model=name,gpu=gpu,pid=p.pid,command=cmd))
    write(work/'launch.json',dict(started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),workers=workers,observations=observations,
        resource_note='GPU0 reserved for existing unified benchmark, GPU2 existing ELM; none stopped. CPU solver group48-55, sampling96-111; shared-server timing.'))
    print('LAUNCHED',workers,flush=True)

if __name__=='__main__':main()
