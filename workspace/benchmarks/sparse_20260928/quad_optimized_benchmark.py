"""Screen shapes, freeze candidates, then paired retest complete drive/forward.

The actual unchanged original F/R forwards remain end-to-end controls. Internal
R expressions are separately labeled. All timing groups randomize method order.
"""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import torch
from experiment import HERE,ROOT,DBNN_F,DBNN_R,accuracy,measure_group,stamp,resource_snapshot
from backends import Prepared
from quadratic_benchmark import Modules,capture,measure_device,measure_wall,error,digest,save
from quad_tune import make_data
from quadratic_tiled import TiledDrive
from quad_optimized_backend import TiledPrepared

CASES=[(32,6001,1),(128,6001,1),(500,6001,1),(32,6001,8),
       (128,6001,8),(500,6001,8),(1024,1001,1),(2048,1001,1),(2048,6001,1)]


def candidates(y,w,bias):
    methods={};specs={}
    combos=[('triton',i,True,'ieee') for i in [0,1,2,3,4,5,6,9]]
    combos += [('cuda',i,True,'ieee') for i in [1,3]]
    combos += [('pair',i,True,'ieee') for i in [0,1,5]]
    combos += [('triton',i,True,'tf32x3') for i in [0,1,3]]
    combos += [('pair',i,True,'tf32x3') for i in [1,2]]
    combos += [('triton',3,False,'ieee')]
    if y.shape[1]<=500:
        combos += [('warp',i,False,'ieee') for i in [0,1,2]]
    for backend,config,tri,prec in combos:
        key=f'{backend}_{config}_{"tri" if tri else "full"}_{prec}'
        spec=dict(backend=backend,config=config,triangle=tri,precision=prec)
        methods[key]=TiledDrive(y,w,bias,**spec);specs[key]=spec
    return methods,specs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'quad_optimized.json')
    ap.add_argument('--repeats',type=int,default=50);ap.add_argument('--screen-repeats',type=int,default=12)
    ap.add_argument('--original-repeats',type=int,default=20);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--indices',nargs='*',type=int,default=list(range(len(CASES))))
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    result=dict(started=stamp(),physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),exclusive_resources=False,
        scope='complete drive and complete forward; module screening followed by fixed-candidate independently ordered retests',
        baseline_sha256=hashlib.sha256((ROOT/'source/DBNN.py').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((ROOT/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
            ['quadratic_tiled.cu','quadratic_tiled.py','quad_optimized_backend.py','quad_optimized_benchmark.py']},
        resources_before=resource_snapshot(),cases=[])
    if args.resume and args.output.exists():result=json.loads(args.output.read_text())
    with torch.no_grad():
        for ci in args.indices:
            if any(c['index']==ci for c in result['cases']):continue
            A,T,B=CASES[ci];print('CASE',ci,A,T,B,flush=True)
            p,y,w=make_data(A,T,B)
            record=dict(index=ci,A=A,T=T,B=B,kind='trained_500_site_subset' if A<=500 else 'synthetic_parameter_expansion_compute_only',
                resources_before=resource_snapshot(),Y_sha256=digest(y),W_sha256=digest(w),
                precision='FP32 IEEE main comparison; tf32x3 separately labeled',failures={})
            ref=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)+p.bias
            modules=Modules(y,w,p.bias)
            trial,specs=candidates(y,w,p.bias)
            good={};errs={};compile_s={}
            for key,fn in trial.items():
                try:
                    start=time.perf_counter();out=fn().clone();torch.cuda.synchronize()
                    compile_s[key]=time.perf_counter()-start;er=error(out,ref)
                    if not er['finite'] or er['max_abs']>1e-3:raise RuntimeError(str(er))
                    good[key]=capture(fn,8);errs[key]=er
                except Exception as ex:
                    record['failures'][key]=str(ex);print('CANDIDATE_FAIL',key,str(ex)[:200],flush=True)
            good['cublas']=capture(lambda:modules.drive('cublas'),8)
            screen=measure_device(good,8,args.screen_repeats,383+ci)
            def pick(keys):return min(keys,key=lambda k:screen[k]['median_ms'])
            selected={
              'new_ieee':pick([k for k in trial if k in good and specs[k]['precision']=='ieee']),
              'cuda_ieee':pick([k for k in trial if k in good and specs[k]['backend']=='cuda']),
              'pair_ieee':pick([k for k in trial if k in good and specs[k]['backend']=='pair' and specs[k]['precision']=='ieee']),
              'tf32x3':pick([k for k in trial if k in good and specs[k]['precision']=='tf32x3']),
              'full_ieee':'triton_3_full_ieee'}
            record['screen_device']=screen;record['screen_errors']=errs;record['screen_specs']=specs
            record['first_call_compile_seconds']=compile_s
            record['selected']={name:dict(key=k,spec=specs[k]) for name,k in selected.items()}
            print('FROZEN',selected,flush=True)
            fns={'cublas':lambda:modules.drive('cublas'),
                 'R_expression':lambda:modules.drive('original_R'),
                 'old_direct':lambda:modules.drive('direct')}
            fns.update({name:trial[key] for name,key in selected.items()})
            single={name:capture(fn,1) for name,fn in fns.items()}
            count=16 if A<=500 else 8
            many={name:capture(fn,count) for name,fn in fns.items()}
            saved=y.clone();y.zero_()
            for variants in [single,many]:
                for name,fn in variants.items():
                    assert float((fn().clone()-p.bias).abs().max())<1e-5,('changed input',name)
            y.copy_(saved)
            record['module_errors']={name:error(fn().clone(),ref) for name,fn in fns.items()}
            for name,er in record['module_errors'].items():
                assert er['finite'] and er['max_abs']<1e-3,(name,er)
            record['module_device']=measure_device(many,count,args.repeats,8071+ci)
            record['module_graph_wall']=measure_wall(single,args.repeats,1747+ci)
            record['module_eager_wall']=measure_wall(fns,20,7489+ci)
            record['module_graph_unroll']=count
            record['intermediate_bytes']={'cublas_Z':y.numel()*y.element_size()}
            for name,key in selected.items():
                obj=trial[key]
                storage=obj.partial_cols if obj.backend=='pair' else obj.partial
                record['intermediate_bytes'][name]=storage.numel()*storage.element_size()

            # Fixed scan/reset; replace only complete drive in the optimized path.
            forward={'cublas':p};new_owners=[]
            for name in ['new_ieee','tf32x3','full_ieee','cuda_ieee','pair_ieee']:
                obj=TiledPrepared(p.source_state,p.x,record['selected'][name]['spec'])
                forward[name]=obj;new_owners.append(obj)
            direct=Prepared(p.source_state,p.x,'triton','direct');forward['old_direct']=direct
            original={};models=[]
            for label,cls in [('F',DBNN_F),('R',DBNN_R)]:
                model=cls(A,T,torch.device('cuda'));model.load_state_dict(p.source_state);model.eval();models.append(model)
                original[f'original_{label}']=lambda model=model:model(p.x)
                if B>1:original[f'original_{label}_trace_loop']=lambda model=model:torch.cat([model(p.x[b:b+1]) for b in range(B)])
            actual_reference=original['original_R']().clone()
            record['reset_events_per_trace']=(actual_reference>=p.vth).sum(1).tolist()
            record['forward_errors']={name:accuracy(fn().clone(),actual_reference,p.vth) for name,fn in forward.items()}
            for name,er in record['forward_errors'].items():
                assert er['finite'] and er['max_abs_mv']<1e-3 and er['spike_mismatch_bins']==0,(ci,name,er)
            graphs={name:capture(fn,1) for name,fn in forward.items()}
            unrolled={name:capture(fn,8) for name,fn in forward.items()}
            saved_x=p.x.clone();p.x.zero_()
            for name,fn in graphs.items():assert float((fn().clone()-p.bias).abs().max())<1e-4,(name,'stale graph')
            p.x.copy_(saved_x)
            for name,fn in graphs.items():
                er=accuracy(fn().clone(),actual_reference,p.vth)
                assert er['max_abs_mv']<1e-3 and er['spike_mismatch_bins']==0,(name,er)
            record['forward_graph_device']=measure_device(unrolled,8,args.repeats,3813+ci)
            record['forward_graph_wall']=measure_wall(graphs,args.repeats,4911+ci)
            # Match original forward and modern Graph on wall time in one group.
            paired=dict(original)
            paired['optimized_cublas']=graphs['cublas'];paired['optimized_new_ieee']=graphs['new_ieee']
            for fn in paired.values():
                for _ in range(2):fn()
            record['original_paired_wall']=measure_wall(paired,args.original_repeats,8207+ci)
            record['input_change_checks_passed']=True
            record['resources_after']=resource_snapshot()
            result['cases'].append(record);save(args.output,result)
            print('DEVICE us', {k:round(v['median_ms']*1000,3) for k,v in record['module_device'].items()},flush=True)
            print('FORWARD graph ms',{k:round(v['median_ms'],4) for k,v in record['forward_graph_wall'].items()},flush=True)
            del single,many,good,trial,fns,graphs,unrolled,forward,original,models,new_owners,paired,p,y,w,modules,ref
            gc.collect();torch.cuda.empty_cache()
    result['finished']=stamp();result['resources_after']=resource_snapshot();save(args.output,result)
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
