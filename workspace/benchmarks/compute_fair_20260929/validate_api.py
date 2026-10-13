"""Check graph replay after input mutation, including zeroing and fallback A."""
import argparse,sys
from pathlib import Path
import torch

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();work=root/'benchmarks/compute_fair_20260929'
    sys.path[:0]=[str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/dt_resolution_20260929')]
    from common import read,write
    from backend import make_prepared,Model,numerical
    from backends import subset_state
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(1)
    state=torch.load(root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt',map_location='cuda',weights_only=False)['model_state']
    policy=read(work/'deployment_policy.json');rows=[]
    for K,T in [(64,1001),(256,4001),(500,16001),(96,1001)]:
        ids=torch.arange(K,device='cuda');s=subset_state(state,ids)
        for dt in [1.,.5]:
            x=torch.zeros((1,K,T),device='cuda',dtype=torch.int8);m,g,selection=make_prepared(s,x,dt,policy)
            for rep in range(3):
                if rep==1:x.zero_()
                else:
                    gen=torch.Generator(device='cuda').manual_seed(6123+K+rep)
                    x.copy_((torch.rand(x.shape,device='cuda',generator=gen)<.025).to(torch.int8)*(1-2*(ids>=250).to(torch.int8))[None,:,None])
                got=g();oracle=Model(s,x,dt,'baseline',512);expected=oracle();torch.cuda.synchronize()
                q=numerical(got,expected,m.th);assert q['passed'],(K,T,dt,rep,q)
                rows.append(dict(A=K,input_steps=T,dt_ms=dt,input_variant=rep,selection=selection,validation=q));del oracle,expected
            del m,g,x
    write(work/'api_validation.json',dict(finished=True,checks=rows));print('API PASS',len(rows),max(q['validation']['max_abs_mv'] for q in rows))

if __name__=='__main__':main()
