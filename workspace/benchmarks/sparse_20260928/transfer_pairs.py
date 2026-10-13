"""Randomized paired core/transfer timing; separate from baseline comparison.

Sequential transfer-only runs can encounter a different level of contention
than an interleaved PyTorch comparison. Keep the paired core in the record.
"""
import json,gc,time
import torch
from experiment import HERE,case_data,measure_group,resource_snapshot,stamp,accuracy
from backends import Prepared,capture,subset_state

def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    data=json.loads((HERE/'confirmation.json').read_text())
    out={'started':stamp(),'resources_before':resource_snapshot(),'cases':[]}
    for c in data['cases']:
        sc,x,ids,_=case_data(*[c[k] for k in ['N','A','T','B']])
        host=x[:,ids].contiguous().pin_memory();xs=host.cuda()
        sp={k:v.cuda() for k,v in subset_state(sc,ids).items()}
        _,scan,quad,mode=c['backend'].split('_')
        p=Prepared(sp,xs,scan,quad);fn=capture(p) if mode=='graph' else p
        yhost=torch.empty((c['B'],c['T']),pin_memory=True)
        def e2e():
            xs.copy_(host,non_blocking=True)
            y=fn();yhost.copy_(y,non_blocking=True)
            return yhost
        with torch.no_grad():
            ref=fn().clone()
            for _ in range(8):fn();e2e()
            torch.cuda.synchronize()
            t=measure_group({'core':fn,'packed_h2d_forward_d2h':e2e},100)
            e2e();torch.cuda.synchronize()
            err=accuracy(yhost,ref,p.vth)
        out['cases'].append({'index':c['index'],'timings':t,'transfer_accuracy':err})
        (HERE/'transfer_pairs.json').write_text(json.dumps(out,indent=2))
        print(c['index'],round(t['core']['median_ms'],4),round(t['packed_h2d_forward_d2h']['median_ms'],4),flush=True)
        del p,fn,sp,xs,host,x,sc,ref,yhost;gc.collect();torch.cuda.empty_cache()
    out['finished']=stamp();out['resources_after']=resource_snapshot()
    (HERE/'transfer_pairs.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
