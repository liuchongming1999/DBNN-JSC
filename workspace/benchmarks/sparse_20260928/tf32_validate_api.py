"""Validate dispatch, tail dimensions, input updates and explicit precision modes."""
import json
import numpy as np
import torch
from experiment import HERE,stamp,accuracy
from quad_tune import make_data
from quadratic_benchmark import capture,error,save
from tf32_pipeline import make_precision_prepared,PrecisionPipeline
from tf32_kernels import TensorDrive,TensorFilter


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    policy=json.loads((HERE/'tf32_backend_policy.json').read_text())
    r=dict(started=stamp(),dispatch=[],arbitrary_quad=[],arbitrary_filter=[])
    with torch.no_grad():
        for profile,v in policy['profiles'].items():
            for key,entry in v['shapes'].items():
                A,T,B=map(int,key.split(':'));p,_,_=make_data(A,T,B)
                obj=make_precision_prepared(p.source_state,p.x,policy,profile)
                spec=entry['spec'];expected=PrecisionPipeline(p.source_state,p.x,spec['drive_spec'],spec.get('filter_spec'),spec.get('scan_block',128))
                fn=capture(obj,1);out=fn().clone();ref=expected().clone()
                assert torch.equal(out,ref),(profile,key,'dispatch differs')
                saved=p.x.clone();p.x.zero_();assert float((fn().clone()-p.bias).abs().max())<1e-5
                p.x.copy_(saved);assert torch.equal(fn().clone(),out)
                r['dispatch'].append(dict(profile=profile,shape=key,selected=entry['name'],exact_match=True,graph_update_passed=True))
        # Unknown shapes retain the previous IEEE/cuBLAS fallback.
        p,_,_=make_data(17,301,1);obj=make_precision_prepared(p.source_state,p.x,policy)
        assert type(obj).__name__=='Prepared' and obj.quad=='cublas'
        r['unknown_shape_fallback']=True
        torch.manual_seed(99213)
        for A,M in [(7,3),(31,129),(63,17),(127,257),(513,33)]:
            y=torch.randn(M,A,device='cuda')*.1;w=torch.randn(A,A,device='cuda')*.01
            ref=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)-70
            for kind,i in [('tile',3),('pair',3)]:
                for mode in ['tf32','tf32x3']:
                    obj=TensorDrive(y,w,-70,backend=kind,config=i,triangle=True,precision=mode,rounding=mode=='tf32')
                    er=error(obj().clone(),ref)
                    assert er['finite'] and er['max_abs']<(1e-3 if mode=='tf32' else 1e-4)
                    r['arbitrary_quad'].append(dict(A=A,M=M,backend=kind,precision=mode,error=er))
        # Noninteger input amplitudes and short tails also obey the filter
        # formula; TF32 remains an explicitly approximate matrix product.
        for A,T,B in [(7,17,1),(31,65,2),(127,301,1)]:
            p,_,_=make_data(A,T,B);p.x.copy_(torch.randn_like(p.x)*.1)
            ref=p.synaptic().clone().double()
            for mode in ['tf32','tf32x3']:
                obj=TensorFilter(p,block=32,rows=32,precision=mode,rounding=mode=='tf32')
                er=error(obj().clone(),ref)
                assert er['finite'] and er['max_abs']<(1e-3 if mode=='tf32' else 1e-5)
                r['arbitrary_filter'].append(dict(A=A,T=T,B=B,precision=mode,error=er))
        r['passed']=True;r['finished']=stamp();save(HERE/'tf32_api_validation.json',r)
        print('PASS',len(r['dispatch']),'dispatch',len(r['arbitrary_quad']),'quadratic tails',len(r['arbitrary_filter']),'filter tails')


if __name__=='__main__':main()
