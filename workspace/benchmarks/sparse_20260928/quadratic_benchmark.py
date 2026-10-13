"""Isolate GEMM, pure quadratic response, and drive from synaptic scan/reset.

All implementations receive identical row-contiguous Y and diagonal-zero W.
Original F/R *expressions* are measured with one full-window call. Original
per-window preparation, R's time chunk loop, scan, reset and transfers are out
of scope. No baseline or production kernel file is modified.
"""
import argparse, ctypes as C, gc, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import HERE, ROOT, case_data, stamp, stats, resource_snapshot
from backends import Prepared, native, subset_state

CASES=[(a,t,b) for t in [1001,6001] for b in [1,8] for a in [8,32,128,500]]
NAMES={
 'original_F':'Original F.bilinear expression',
 'original_R':'Original mm + multiply + sum expression, full window',
 'cublas':'cuBLAS GEMM + CUDA fused reduction',
 'cutlass':'CUTLASS SIMT GEMM + CUDA fused reduction',
 'fastf32':'CUTLASS FastF32 GEMM + CUDA fused reduction (separate precision)',
 'direct':'CUDA direct fused quadratic',
}

def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def save(path,result):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(path)

def error(output,reference):
    d=output.to(torch.float64)-reference
    return {'finite':bool(torch.isfinite(output).all()),
            'max_abs':float(d.abs().max()),'rmse':float(d.square().mean().sqrt()),
            'relative_l2':float(d.norm()/reference.norm().clamp_min(1e-30))}

class Modules:
    def __init__(self,y,w,bias):
        self.y=y;self.w=w;self.R,self.A=y.shape;self.bias=bias
        self.bias_tensor=torch.tensor([bias],device=y.device,dtype=y.dtype)
        self.w3=w.unsqueeze(0)
        self.z=torch.empty_like(y);self.out=torch.empty(self.R,device=y.device)
        self.lib=C.CDLL(str(HERE/'quadratic_only.so'))
        self.lib.dbnn_q_reduce.argtypes=[C.c_void_p]*3+[C.c_int]*2+[C.c_void_p]
        self.lib.dbnn_q_direct.argtypes=[C.c_void_p]*3+[C.c_int]*2+[C.c_void_p]
        self.tensor=C.CDLL(str(HERE/'cutlass_tensor.so'))
        self.tensor.dbnn_cutlass_tensor.argtypes=[C.c_void_p]*3+[C.c_int]*2+[C.c_void_p]
        native()

    def gemm(self,backend):
        if backend=='cublas':
            torch.mm(self.y,self.w,out=self.z)
        elif backend=='cutlass':
            native().call('dbnn_cutlass',self.y,self.w,self.z,self.R,self.A,
                          torch.cuda.current_stream().cuda_stream)
        else:
            status=self.tensor.dbnn_cutlass_tensor(self.y.data_ptr(),self.w.data_ptr(),self.z.data_ptr(),
                self.R,self.A,torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError(f'FastF32 GEMM status {status}')
        return self.z

    def quadratic(self,backend):
        if backend=='original_F':return F.bilinear(self.y,self.y,self.w3,None).squeeze(-1)
        if backend=='original_R':return ((self.y@self.w)*self.y).sum(-1)
        stream=torch.cuda.current_stream().cuda_stream
        if backend=='direct':
            status=self.lib.dbnn_q_direct(self.y.data_ptr(),self.w.data_ptr(),self.out.data_ptr(),self.R,self.A,stream)
        else:
            self.gemm(backend)
            status=self.lib.dbnn_q_reduce(self.y.data_ptr(),self.z.data_ptr(),self.out.data_ptr(),self.R,self.A,stream)
        if status:raise RuntimeError(f'Quadratic kernel status {status}')
        return self.out

    def drive(self,backend):
        if backend=='original_F':
            return F.bilinear(self.y,self.y,self.w3,self.bias_tensor).squeeze(-1)+self.y.sum(-1)
        if backend=='original_R':
            return self.y.sum(-1)+((self.y@self.w)*self.y).sum(-1)+self.bias
        stream=torch.cuda.current_stream().cuda_stream
        if backend=='direct':
            native().call('dbnn_direct',self.y,self.w,self.out,self.R,self.A,self.bias,stream)
        else:
            self.gemm(backend)
            native().call('dbnn_reduce',self.y,self.z,self.out,self.R,self.A,self.bias,stream)
        return self.out

def capture(fn,count):
    # Use an explicit side stream for CUDA Graph warm-up and capture.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(count):out=fn()
    torch.cuda.synchronize()
    def replay():
        graph.replay();return out
    replay.graph=graph;replay.out=out
    return replay

def measure_wall(fns,repeats,seed):
    values={name:[] for name in fns};rng=random.Random(seed)
    for _ in range(repeats):
        names=list(fns);rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize();start=time.perf_counter()
            out=fns[name]()
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            values[name].append(elapsed)
    return {name:stats(v) for name,v in values.items()}

def measure_device(fns,count,repeats,seed):
    values={name:[] for name in fns};rng=random.Random(seed)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        names=list(fns);rng.shuffle(names)
        for name in names:
            torch.cuda.synchronize()
            start.record();fns[name]();end.record();end.synchronize()
            values[name].append(start.elapsed_time(end)/count)
    return {name:stats(v) for name,v in values.items()}

def one_case(index,args):
    A,T,B=CASES[index];start=time.perf_counter()
    sc,xc,ids,_=case_data(500,A,T,B)
    sp={k:v.cuda() for k,v in subset_state(sc,ids).items()};x=xc[:,ids].contiguous().cuda()
    p=Prepared(sp,x,'cuda','cublas')
    y=p.synaptic().reshape(B*T,A).contiguous().clone();w=p.W.clone();bias=p.bias
    module=Modules(y,w,bias);yd=y.double();wd=w.double()
    z64=yd@wd;q64=(z64*yd).sum(-1);d64=q64+yd.sum(-1)+bias
    refs={'gemm':z64,'quadratic':q64,'drive':d64}
    r={'index':index,'N':500,'A':A,'T':T,'B':B,'rows':B*T,'Y_sha256':digest(y),'W_sha256':digest(w),
       'Y_shape':list(y.shape),'Y_stride':list(y.stride()),'dtype':'float32','scopes':{},
       'resources_before':resource_snapshot()}
    count=16 if B*T*A>=20_000_000 else 32
    r['device_graph_unroll']=count
    for scope in ['gemm','quadratic','drive']:
        backends=['cublas','cutlass','fastf32'] if scope=='gemm' else list(NAMES)
        fns={name:(lambda name=name,scope=scope:getattr(module,scope)(name)) for name in backends}
        errs={name:error(fn().clone(),refs[scope]) for name,fn in fns.items()}
        for name,e in errs.items():
            if not e['finite']:raise RuntimeError(f'Nonfinite {scope}/{name}')
            if scope!='gemm' and e['max_abs']>1e-3:raise RuntimeError(f'Accuracy gate {scope}/{name}: {e}')
        graphs={name:capture(fn,1) for name,fn in fns.items()}
        unrolled={name:capture(fn,count) for name,fn in fns.items()}
        # A replay must consume changed input at the original memory address.
        saved=y.clone();y.zero_();torch.cuda.synchronize()
        expected=bias if scope=='drive' else 0.
        for variant in [graphs,unrolled]:
            for name,fn in variant.items():
                out=fn().clone();max_err=float((out-expected).abs().max())
                if max_err>1e-6:raise RuntimeError(f'Stale Graph {scope}/{name}: {max_err}')
        y.copy_(saved);torch.cuda.synchronize()
        graph_err={name:error(fn().clone(),refs[scope]) for name,fn in graphs.items()}
        for name,e in graph_err.items():
            if not e['finite'] or (scope!='gemm' and e['max_abs']>1e-3):raise RuntimeError(f'Graph accuracy {scope}/{name}')
        all_wall={f'{name}/eager':fn for name,fn in fns.items()}
        all_wall.update({f'{name}/graph':fn for name,fn in graphs.items()})
        for fn in all_wall.values():
            for _ in range(5):fn()
        wall=measure_wall(all_wall,args.repeats,args.seed+index*31+len(scope))
        device=measure_device(unrolled,count,args.device_repeats,args.seed+index*31+len(scope)+5000)
        r['scopes'][scope]={'errors_vs_float64':errs,'graph_errors_vs_float64':graph_err,
                            'changed_input_graph_passed':True,'wall':wall,'device_per_call':device}
        if scope=='quadratic':
            print(f'  q GPU us A={A} B={B} T={T}: '+', '.join(f'{k}={v["median_ms"]*1000:.2f}' for k,v in device.items()),flush=True)
        del all_wall,graphs,unrolled,fns;gc.collect();torch.cuda.empty_cache()
    r['resources_after']=resource_snapshot();r['elapsed_seconds']=time.perf_counter()-start
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'quadratic_isolated.json')
    ap.add_argument('--repeats',type=int,default=80);ap.add_argument('--device-repeats',type=int,default=40)
    ap.add_argument('--seed',type=int,default=929);ap.add_argument('--indices',type=int,nargs='*',default=list(range(len(CASES))))
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    result={'started':stamp(),'scope':'Prepared, row-contiguous fixed Y,W; no packing, scan, reset, H2D/D2H or training.',
      'formulas':{'gemm':'Z=YW','quadratic':'q_i=sum_j Y_ij (YW)_ij','drive':'d=q+sum(Y,-1)+bias'},
      'baseline_scope':'Original F and R algebraic expressions with one full-window call; W diagonal removal is preparation. R temporal chunk loop is excluded.',
      'wall_protocol':'Randomized interleaved eager and one-replay Graph calls; synchronize, perf_counter, call, synchronize.',
      'device_protocol':'Randomized interleaved CUDA events around one unrolled Graph of 16/32 repeated calls, divided by count. Fixed-input hot-cache steady-state throughput.',
      'repeats':args.repeats,'device_repeats':args.device_repeats,'seed':args.seed,
      'precision':'FP32, TF32 disabled; CUTLASS FastF32 shown separately','torch':torch.__version__,
      'cuda':torch.version.cuda,'blas_preference':str(torch.backends.cuda.preferred_blas_library()),
      'physical_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'resources_before':resource_snapshot(),
      'baseline_sha256':hashlib.sha256((ROOT/'source/DBNN.py').read_bytes()).hexdigest(),
      'checkpoint_sha256':hashlib.sha256((ROOT/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
      'method_names':NAMES,'cases':[]}
    save(args.output,result)
    with torch.no_grad():
        for index in args.indices:
            print('CASE',index,CASES[index],flush=True)
            rec=one_case(index,args);result['cases'].append(rec);save(args.output,result)
            gc.collect();torch.cuda.empty_cache()
    result['finished']=stamp();result['resources_after']=resource_snapshot();save(args.output,result)
    print('COMPLETE',args.output,flush=True)

if __name__=='__main__':main()
