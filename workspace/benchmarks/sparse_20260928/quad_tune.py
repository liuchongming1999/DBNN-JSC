"""Explore explicit fixed kernel configurations, then freeze winners for retest."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
from experiment import HERE,ROOT,case_data,stamp,resource_snapshot
from backends import Prepared,subset_state
from quadratic_benchmark import Modules,capture,measure_device,measure_wall,error,digest,save
from quadratic_tiled import TiledDrive,TRITON_TILES,PAIR_TILES


def make_data(A,T,B):
    # Above 500 sites this is a declared synthetic compute-scale extension.
    N=max(500,A)
    sc,xc,ids,_=case_data(N,A,T,B)
    sp={k:v.cuda() for k,v in subset_state(sc,ids).items()}
    x=xc[:,ids].contiguous().cuda()
    p=Prepared(sp,x,'triton','cublas')
    p.source_state=sp
    y=p.synaptic().reshape(B*T,A).contiguous().clone()
    return p,y,p.W.clone()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--A',type=int,default=500)
    ap.add_argument('--B',type=int,default=1);ap.add_argument('--T',type=int,default=6001)
    ap.add_argument('--repeats',type=int,default=12);ap.add_argument('--count',type=int,default=8)
    ap.add_argument('--output',type=Path,default=HERE/'quad_tune.json')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--pairs-only',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    with torch.no_grad():
        p,y,w=make_data(args.A,args.T,args.B)
        module=Modules(y,w,p.bias)
        ref=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)+p.bias
        methods={'cublas':lambda:module.drive('cublas'), 'R_expression':lambda:module.drive('original_R'),
                 'old_direct':lambda:module.drive('direct')}
        specs={};owners=[]
        for backend in ([] if args.pairs_only else ['cuda','triton']):
            configs=range(2) if args.smoke else range(8 if backend=='cuda' else len(TRITON_TILES))
            for config in configs:
                for tri in [False,True]:
                    for prec in (['ieee'] if backend=='cuda' else ['ieee','tf32x3']):
                        key=f'{backend}_{config}_{"tri" if tri else "full"}_{prec}'
                        obj=TiledDrive(y,w,p.bias,backend,config,tri,prec)
                        methods[key]=obj;owners.append(obj)
                        specs[key]=dict(backend=backend,config=config,triangle=tri,precision=prec)
        for config in range(len(PAIR_TILES)):
            for prec in ['ieee','tf32x3']:
                key=f'pair_{config}_tri_{prec}'
                obj=TiledDrive(y,w,p.bias,'pair',config,True,prec)
                methods[key]=obj;owners.append(obj)
                specs[key]=dict(backend='pair',config=config,triangle=True,precision=prec)
        for name in ([] if args.pairs_only else ['thread','direct_fma']):
            if name=='thread' and args.A>500:continue
            obj=TiledDrive(y,w,p.bias,name)
            methods[name]=obj;owners.append(obj)
            specs[name]=dict(backend=name,config=0,triangle=name=='thread',precision='ieee')
        result=dict(started=stamp(),physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
            A=args.A,B=args.B,T=args.T,scope='complete drive including linear and bias; prepared Y/W; two-stage tiles include final reduction',
            resources_before=resource_snapshot(),Y_sha256=digest(y),W_sha256=digest(w),specs=specs,
            errors={},compile_errors={},timings={},prepared_upper_triangle_outside_timer=True)
        graphs={};preparation={}
        for name,fn in methods.items():
            try:
                t=time.perf_counter();out=fn().clone();torch.cuda.synchronize()
                er=error(out,ref);result['errors'][name]=er
                if not er['finite'] or er['max_abs']>1e-3:
                    raise RuntimeError(f'accuracy {er}')
                graphs[name]=capture(fn,args.count)
                preparation[name]=time.perf_counter()-t
                print('READY',name,er['max_abs'],flush=True)
            except Exception as ex:
                result['compile_errors'][name]=str(ex)
                print('ERROR',name,str(ex)[:500],flush=True)
            save(args.output,result)
        result['first_call_compile_and_capture_seconds']=preparation
        result['count']=args.count;result['repeats']=args.repeats
        result['timings']=measure_device(graphs,args.count,args.repeats,1928+args.A+args.B)
        # Check replay uses the current input, including the thread layout path.
        saved=y.clone();y.zero_()
        for name,fn in graphs.items():
            out=fn().clone();assert float((out-p.bias).abs().max())<1e-5,name
        y.copy_(saved)
        for name,fn in graphs.items():
            er=error(fn().clone(),ref)
            assert er['finite'] and er['max_abs']<1e-3,(name,er)
        result['changed_input_graph_passed']=True
        result['finished']=stamp();result['resources_after']=resource_snapshot()
        save(args.output,result)
        print('RANKING us',flush=True)
        for name,v in sorted(result['timings'].items(),key=lambda kv:kv[1]['median_ms']):
            print(name,round(v['median_ms']*1000,3),flush=True)
        print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
