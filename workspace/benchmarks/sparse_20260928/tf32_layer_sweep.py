"""Probe packing/reset and systematically screen matrix-form temporal filters."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
from experiment import HERE,ROOT,stamp,resource_snapshot
from quad_tune import make_data
from quadratic_benchmark import capture,measure_device,error,save
from backends import native
from triton_scan import run_scan
from tf32_kernels import TensorFilter
from tf32_sweep import CASES


def flag_call(fn,flag):
    def call():
        old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=flag
        try:return fn()
        finally:torch.backends.cuda.matmul.allow_tf32=old
    return call


def no_matmul_controls(p):
    B,A,T=p.x.shape;N=8000 if A in (32,128) else max(A,500)
    ids=torch.arange(A,device=p.x.device,dtype=torch.long)
    x=torch.zeros((B,N,T),device=p.x.device);x[:,ids]=p.x
    packed=torch.empty_like(p.x)
    def gather():return torch.index_select(x,1,ids,out=packed)
    # Fixed u for both reset calls; no TF32 arithmetic is added to comparisons.
    p();u=p.drive.clone();out=torch.empty_like(u)
    def reset():
        native().call('dbnn_reset_event',u,out,p.alpha_pows,B,T,p.alpha,p.amp,p.vth,torch.cuda.current_stream().cuda_stream)
        return out
    result={}
    for name,fn in [('pack_input',gather),('reset_same_drive',reset)]:
        fns={label:flag_call(fn,flag) for label,flag in [('tf32_off',False),('tf32_on',True)]}
        ref=fns['tf32_off']().clone();out2=fns['tf32_on']().clone()
        assert torch.equal(ref,out2)
        graphs={key:capture(call,16) for key,call in fns.items()}
        result[name]=dict(device=measure_device(graphs,16,30,70200+A+B),bitwise_identical=True,
                          uses_matrix_multiply=False)
    result['N']=N
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--indices',type=int,nargs='*',default=list(range(len(CASES))))
    ap.add_argument('--output',type=Path,default=HERE/'tf32_layer_sweep.json');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    result=dict(started=stamp(),physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),exclusive_resources=False,
        scope='Filter candidates include local Toeplitz GEMM, FP32 cross-block affine scan and FP32 carry/output assembly.',
        resources_before=resource_snapshot(),cases=[])
    if args.resume and args.output.exists():result=json.loads(args.output.read_text())
    with torch.no_grad():
        for index in args.indices:
            if any(c['index']==index for c in result['cases']):continue
            A,T,B=CASES[index];print('CASE',index,A,T,B,flush=True)
            p,_,_=make_data(A,T,B);ref=p.synaptic().clone().double()
            record=dict(index=index,A=A,T=T,B=B,specs={},errors={},failures={},compile_seconds={})
            owners={};fns={}
            for block in [64,128,256,512]:
                def scan(block=block):
                    run_scan(p.x,p.y,p.da,p.dc,p.om,block);return p.y
                key=f'scan_{block}';fns[key]=scan;record['specs'][key]=dict(kind='scan',block=block)
            for L in [16,32,64,128]:
                for rows in [16,32]:
                    for prec in ['tf32','tf32x3']:
                        key=f'tc_{L}_{rows}_{prec}';spec=dict(block=L,rows=rows,precision=prec,rounding=prec=='tf32')
                        obj=TensorFilter(p,**spec);fns[key]=obj;owners[key]=obj;record['specs'][key]=dict(kind='tensor',**spec)
            obj=TensorFilter(p,block=32,rows=32,precision='ieee',rounding=False)
            fns['matrix_ieee_control']=obj;owners['matrix_ieee_control']=obj
            record['specs']['matrix_ieee_control']=dict(kind='tensor',block=32,rows=32,precision='ieee',rounding=False)
            graphs={}
            for key,fn in fns.items():
                try:
                    start=time.perf_counter();out=fn().clone();torch.cuda.synchronize()
                    record['errors'][key]=error(out,ref);assert record['errors'][key]['finite']
                    graphs[key]=capture(fn,8);record['compile_seconds'][key]=time.perf_counter()-start
                except Exception as ex:
                    record['failures'][key]=str(ex);print('FAIL',key,str(ex)[:180],flush=True)
            record['screen_device']=measure_device(graphs,8,15,40033+index)
            def pick(keys):return min(keys,key=lambda k:record['screen_device'][k]['median_ms'])
            selected=dict(scan=pick([k for k in graphs if k.startswith('scan_')]),
                          tf32=pick([k for k in graphs if k.endswith('_tf32')]),
                          tf32x3=pick([k for k in graphs if k.endswith('_tf32x3')]))
            record['selected']={k:dict(key=v,spec=record['specs'][v]) for k,v in selected.items()}
            record['ptx_evidence']={}
            for name in ['tf32','tf32x3']:
                obj=owners[selected[name]];ptx=obj.compiled.asm['ptx']
                record['ptx_evidence'][name]=dict(tf32_mma='mma.sync' in ptx and '.tf32' in ptx,registers=obj.compiled.n_regs,spills=obj.compiled.n_spills)
            saved=p.x.clone();p.x.zero_()
            for key in selected.values():assert float(graphs[key]().abs().max())==0
            p.x.copy_(saved);record['changed_input_check']=True
            record['non_matmul_controls']=no_matmul_controls(p)
            result['cases'].append(record);save(args.output,result)
            print('SELECTED',[(k,v['key'],round(record['screen_device'][v['key']]['median_ms']*1000,3),
                  record['errors'][v['key']]['max_abs']) for k,v in record['selected'].items()],flush=True)
            del graphs,owners,fns,p,ref,obj,out,saved
            gc.collect();torch.cuda.empty_cache()
    result['finished']=stamp();result['resources_after']=resource_snapshot();save(args.output,result)
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
