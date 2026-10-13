"""Screen TF32 matrix layouts before fixed-candidate whole-pipeline retests."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
from experiment import HERE,ROOT,stamp,resource_snapshot
from quad_tune import make_data
from quadratic_tiled import TiledDrive
from quadratic_benchmark import Modules,capture,measure_device,error,save,digest
from tf32_kernels import TensorDrive,TC_TILES,TC_PAIR_TILES

CASES=[(32,6001,1),(128,6001,1),(500,6001,1),(32,6001,8),(128,6001,8),
       (500,6001,8),(1024,1001,1),(2048,1001,1),(2048,6001,1),
       (500,6001,2),(500,6001,4),(8,1001,1),(128,301,1)]


class BlasDrive:
    def __init__(self,module,tf32):self.module=module;self.tf32=tf32
    def __call__(self):
        old=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=self.tf32
        try:return self.module.drive('cublas')
        finally:torch.backends.cuda.matmul.allow_tf32=old


def make_drive(y,w,bias,spec,module=None):
    spec=dict(spec);kind=spec.pop('kind')
    if kind=='cublas':return BlasDrive(module or Modules(y,w,bias),spec['tf32'])
    if kind=='ieee':return TiledDrive(y,w,bias,**spec)
    if kind=='tensor':return TensorDrive(y,w,bias,**spec)
    raise ValueError(kind)


def candidates(A,T,B,policy):
    specs={'cublas_ieee':dict(kind='cublas',tf32=False),'cublas_tf32':dict(kind='cublas',tf32=True)}
    current=policy['shapes'].get(f'{A}:{T}:{B}')
    if current:specs['current_ieee']=dict(kind='ieee',**current['spec'])
    else:
        for backend,configs in [('triton',[1,3,6,9]),('pair',[0,1,3])]:
            for i in configs:specs[f'ieee_{backend}_{i}']=dict(kind='ieee',backend=backend,config=i,triangle=True,precision='ieee')
    for i in range(len(TC_TILES)):
        for tri in [False,True]:
            specs[f'tf32_tile_{i}_{int(tri)}']=dict(kind='tensor',backend='tile',config=i,triangle=tri,precision='tf32',rounding=True)
    for i in range(len(TC_PAIR_TILES)):
        specs[f'tf32_pair_{i}']=dict(kind='tensor',backend='pair',config=i,triangle=True,precision='tf32',rounding=True)
    for i in [1,2,3,5,6,8,9,10,11,13]:
        specs[f'tf32x3_tile_{i}']=dict(kind='tensor',backend='tile',config=i,triangle=True,precision='tf32x3',rounding=False)
    for i in [0,1,3,4,5]:
        specs[f'tf32x3_pair_{i}']=dict(kind='tensor',backend='pair',config=i,triangle=True,precision='tf32x3',rounding=False)
    # A few scheduling/truncation controls; never silently mix their precision.
    for group in [1,4,16]:
        specs[f'tf32_group_{group}']=dict(kind='tensor',backend='tile',config=3,triangle=True,precision='tf32',rounding=True,group=group)
    specs['tf32_truncate_control']=dict(kind='tensor',backend='tile',config=3,triangle=True,precision='tf32',rounding=False)
    return specs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--indices',type=int,nargs='*',default=list(range(len(CASES))))
    ap.add_argument('--output',type=Path,default=HERE/'tf32_quad_sweep.json');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--repeats',type=int,default=12);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    policy=json.loads((HERE/'quad_backend_policy.json').read_text())
    result=dict(started=stamp(),physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),exclusive_resources=False,
        baseline_sha256=hashlib.sha256((ROOT/'source/DBNN.py').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((ROOT/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['tf32_kernels.py','tf32_sweep.py']},
        scope='Full drive including quadratic, linear and bias. Prepared Y/W, FP32 storage; explicit precision modes.',
        resources_before=resource_snapshot(),cases=[])
    if args.resume and args.output.exists():result=json.loads(args.output.read_text())
    with torch.no_grad():
        for index in args.indices:
            if any(c['index']==index for c in result['cases']):continue
            A,T,B=CASES[index];print('CASE',index,A,T,B,flush=True)
            p,y,w=make_data(A,T,B);module=Modules(y,w,p.bias)
            ref=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)+p.bias
            specs=candidates(A,T,B,policy);graphs={};owners={}
            record=dict(index=index,A=A,T=T,B=B,kind='trained_500_site_subset' if A<=500 else 'synthetic_compute_only',
                specs=specs,errors={},failures={},compile_seconds={},Y_sha256=digest(y),W_sha256=digest(w))
            for i,(key,spec) in enumerate(specs.items()):
                try:
                    start=time.perf_counter();obj=make_drive(y,w,p.bias,spec,module)
                    out=obj().clone();torch.cuda.synchronize();record['errors'][key]=error(out,ref)
                    assert record['errors'][key]['finite']
                    owners[key]=obj;graphs[key]=capture(obj,8)
                    record['compile_seconds'][key]=time.perf_counter()-start
                except Exception as ex:
                    record['failures'][key]=str(ex);print('FAIL',key,str(ex)[:180],flush=True)
                if i%12==0:print('READY',i,'/',len(specs),key,flush=True)
            record['screen_device']=measure_device(graphs,8,args.repeats,43120+index)
            def pick(keys):
                return min(keys,key=lambda k:record['screen_device'][k]['median_ms']) if keys else None
            valid=list(graphs)
            ieee=[k for k in valid if specs[k]['kind']=='ieee' or k=='cublas_ieee']
            plain=[k for k in valid if specs[k].get('precision')=='tf32' and specs[k].get('rounding')]
            x3=[k for k in valid if specs[k].get('precision')=='tf32x3']
            strict=[k for k in plain if record['errors'][k]['max_abs']<1e-3]
            selected=dict(ieee=pick(ieee),tf32=pick(plain),tf32_strict=pick(strict),tf32x3=pick(x3),cublas_tf32='cublas_tf32')
            record['selected']={k:dict(key=v,spec=specs[v]) for k,v in selected.items() if v}
            record['ptx_evidence']={}
            for name,v in record['selected'].items():
                obj=owners[v['key']]
                if hasattr(obj,'compiled'):
                    ptx=obj.compiled.asm['ptx']
                    record['ptx_evidence'][name]=dict(tf32_mma='mma.sync' in ptx and '.tf32' in ptx,
                        rounding_instruction='cvt.rn.tf32.f32' in ptx,registers=obj.compiled.n_regs,spills=obj.compiled.n_spills)
            saved=y.clone();y.zero_()
            for v in record['selected'].values():assert float((graphs[v['key']]()-p.bias).abs().max())<1e-5
            y.copy_(saved);record['graph_changed_input_passed']=True
            result['cases'].append(record);save(args.output,result)
            print('SELECTED',[(k,v['key'],round(record['screen_device'][v['key']]['median_ms']*1000,3),
                              record['errors'][v['key']]['max_abs']) for k,v in record['selected'].items()],flush=True)
            del graphs,owners,p,y,w,module,ref,obj,out,saved
            gc.collect();torch.cuda.empty_cache()
    result['finished']=stamp();result['resources_after']=resource_snapshot();save(args.output,result)
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
