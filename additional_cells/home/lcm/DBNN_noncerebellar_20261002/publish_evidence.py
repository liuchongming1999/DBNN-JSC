"""Compact raw-artifact status for the new non-cerebellar campaign."""
import argparse,json,time,fcntl
from pathlib import Path
from models import REGISTRY
from common import write

def read(path):return json.loads(path.read_text()) if path.exists() else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    with open(a.work/'report.lock','w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);models={}
        for name,spec in REGISTRY.items():
            base=a.work/'results'/name
            r=dict(spec=spec,state=read(base/'state.json'),reference_convergence=read(base/'screen/reference_convergence.json'),
                   operating_point=read(base/'screen/operating_point.json'),gpu={},sampling={})
            for profile in ('ampa','ampa_nmda'):
                r['gpu'][profile]=read(base/profile/'gpu/complete.json')
                r['sampling'][profile]={}
                for kind in ('train_1.0ms','train_0.5ms','train_0.25ms','test'):
                    d=a.data/name/profile/kind;r['sampling'][profile][kind]=dict(complete=(d/'complete.json').exists(),progress=read(d/'status.json'))
            models[name]=r
        write(a.work/'snapshot.json',dict(collected=time.strftime('%Y-%m-%dT%H:%M:%S%z'),models=models))

if __name__=='__main__':main()
