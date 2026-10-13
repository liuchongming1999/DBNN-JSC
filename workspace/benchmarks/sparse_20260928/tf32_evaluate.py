"""Fixed configurations, five-layer timings, original-code and precision controls."""
import argparse,gc,hashlib,json,os,math
from pathlib import Path
import numpy as np
import torch
from experiment import HERE,ROOT,DBNN_F,DBNN_R,accuracy,case_data,stamp,resource_snapshot
from quad_tune import make_data
from quadratic_benchmark import capture,measure_device,measure_wall,error,save
from tf32_pipeline import PrecisionPipeline
from tf32_sweep import make_drive,CASES


def variants(q,f):
    base=q['selected']['ieee']['spec'];block=f['selected']['scan']['spec']['block']
    filt={name:{k:v for k,v in f['selected'][name]['spec'].items() if k!='kind'} for name in ['tf32','tf32x3']}
    d={name:v['spec'] for name,v in q['selected'].items()}
    specs={'previous_ieee':dict(drive_spec=base,scan_block=128),
           'tuned_ieee':dict(drive_spec=base,scan_block=block)}
    for name in ['tf32','tf32_strict','tf32x3','cublas_tf32']:
        if name in d:specs['quad_'+name]=dict(drive_spec=d[name],scan_block=block)
    for name in ['tf32','tf32x3']:
        specs['filter_'+name]=dict(drive_spec=base,filter_spec=filt[name])
    specs['combined_tf32']=dict(drive_spec=d['tf32'],filter_spec=filt['tf32'])
    specs['quad_tf32_filter_x3']=dict(drive_spec=d['tf32'],filter_spec=filt['tf32x3'])
    specs['combined_x3']=dict(drive_spec=d['tf32x3'],filter_spec=filt['tf32x3'])
    return specs


def all_inputs_accuracy(A,T,B,pipelines):
    s,x,ids,_=case_data(max(500,A),A,T,10)
    sx={k:v.cuda() for k,v in __import__('backends').subset_state(s,ids).items()}
    x=x[:,ids].contiguous().cuda()
    original=DBNN_R(A,T,torch.device('cuda'));original.load_state_dict(sx);original.eval()
    ref=original(x).clone();vth=float(sx['v_th'])
    data={};target=None
    if A==500:
        raw=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
        target=torch.from_numpy(raw['test_op'][:10,:T].astype(np.float32)).cuda()
    for name,p in pipelines.items():
        saved=p.x.clone();outs=[]
        # Each replay keeps the selected B. Pad the last group and evaluate only
        # genuine traces, so B-dependent scheduling and reset are exercised.
        for start in range(0,10,B):
            take=min(B,10-start);inds=torch.arange(start,start+B,device=x.device)%10
            p.x.copy_(x[inds]);outs.append(p().clone()[:take])
        out=torch.cat(outs);er=accuracy(out,ref,vth)
        er['samples']=10;er['per_sample_max_mv']=(out-ref).abs().amax(1).tolist()
        er['event_counts']=(out>=vth).sum(1).tolist()
        er['reference_event_counts']=(ref>=vth).sum(1).tolist()
        er['strict_gate_passed']=er['finite'] and er['max_abs_mv']<1e-3 and er['spike_mismatch_bins']==0
        if target is not None:
            er['neuron_pooled_rmse_mv']=float((out-target).square().mean().sqrt())
            er['reference_neuron_pooled_rmse_mv']=float((ref-target).square().mean().sqrt())
        data[name]=er;p.x.copy_(saved)
    del s,x,original,ref
    return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'tf32_evaluation.json')
    ap.add_argument('--repeats',type=int,default=50);ap.add_argument('--seed',type=int,default=6291)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--indices',type=int,nargs='*',default=list(range(len(CASES))))
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    quad=json.loads((HERE/'tf32_quad_sweep.json').read_text());fil=json.loads((HERE/'tf32_layer_sweep.json').read_text())
    assert quad.get('finished') and fil.get('finished')
    qs={c['index']:c for c in quad['cases']};fs={c['index']:c for c in fil['cases']}
    result=dict(started=stamp(),physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),exclusive_resources=False,
        scope='Frozen configurations from independent sweeps; module CUDA events, single-replay/eager wall time, ten-input voltage/reset checks.',
        baseline_sha256=quad['baseline_sha256'],checkpoint_sha256=quad['checkpoint_sha256'],
        source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['tf32_kernels.py','tf32_pipeline.py','tf32_evaluate.py']},
        precision_gate=dict(max_abs_mv=1e-3,reset_mismatch_bins=0),repeats=args.repeats,seed=args.seed,
        resources_before=resource_snapshot(),cases=[])
    if args.resume and args.output.exists():result=json.loads(args.output.read_text())
    with torch.no_grad():
        for index in args.indices:
            if any(c['index']==index for c in result['cases']):continue
            A,T,B=CASES[index];print('CASE',index,A,T,B,flush=True)
            p,y,w=make_data(A,T,B);q=qs[index];f=fs[index]
            specs=variants(q,f);pipelines={k:PrecisionPipeline(p.source_state,p.x,v['drive_spec'],v.get('filter_spec'),v.get('scan_block',128)) for k,v in specs.items()}
            rec=dict(index=index,A=A,T=T,B=B,kind=q['kind'],specs=specs,resources_before=resource_snapshot())
            # Fixed Y isolates precision/tiling from differences in the filter.
            drives={name:make_drive(y,w,p.bias,v['spec']) for name,v in q['selected'].items()}
            ref64=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)+p.bias
            rec['drive_errors_vs_fp64']={k:error(fn().clone(),ref64) for k,fn in drives.items()}
            rec['drive_device']=measure_device({k:capture(fn,8) for k,fn in drives.items()},8,args.repeats,args.seed+index)
            filters={k:pipelines[k].synaptic for k in ['previous_ieee','tuned_ieee','filter_tf32','filter_tf32x3']}
            rec['filter_device']=measure_device({k:capture(fn,8) for k,fn in filters.items()},8,args.repeats,args.seed+100+index)
            rec['heldout_accuracy']=all_inputs_accuracy(A,T,B,pipelines)
            graphs={k:capture(fn,1) for k,fn in pipelines.items()}
            many={k:capture(fn,8) for k,fn in pipelines.items()}
            # Graph must consume changing inputs, including new filter workspace.
            saved=p.x.clone();p.x.zero_()
            for k,fn in graphs.items():assert float((fn().clone()-p.bias).abs().max())<1e-5,(index,k,'zero input')
            p.x.copy_(saved)
            rec['graph_changed_input_passed']=True
            rec['forward_device']=measure_device(many,8,args.repeats,args.seed+200+index)
            rec['forward_graph_wall']=measure_wall(graphs,args.repeats,args.seed+300+index)
            rec['forward_eager_wall']=measure_wall(pipelines,20,args.seed+400+index)
            acceptable=[k for k,er in rec['heldout_accuracy'].items() if er['strict_gate_passed']]
            chosen=min(acceptable,key=lambda k:rec['forward_device'][k]['median_ms'])
            rec['selected_strict']=chosen
            rec['selected_fastest']=min(pipelines,key=lambda k:rec['forward_device'][k]['median_ms'])
            # Re-evaluate the actual unchanged F/R call modes, never relabel a
            # whole-window algebraic expression as the original model.
            models=[];original={}
            for kind,cls in [('F',DBNN_F),('R',DBNN_R)]:
                m=cls(A,T,torch.device('cuda'));m.load_state_dict(p.source_state);m.eval();models.append(m)
                original['original_'+kind]=lambda m=m:m(p.x)
                if B>1:original['original_'+kind+'_trace_loop']=lambda m=m:torch.cat([m(p.x[b:b+1]) for b in range(B)])
            for fn in original.values():fn()
            rec['original_screen']=measure_wall(original,3,args.seed+500+index)
            origkey=min(original,key=lambda k:rec['original_screen'][k]['median_ms'])
            rec['original_method']=origkey
            rec['original_paired_wall']=measure_wall({origkey:original[origkey],
                'previous_ieee':graphs['previous_ieee'],'tuned_ieee':graphs['tuned_ieee'],
                'selected_strict':graphs[chosen]},30,args.seed+600+index)
            rec['resources_after']=resource_snapshot()
            result['cases'].append(rec);save(args.output,result)
            print('RESULT',chosen,'device us',round(rec['forward_device'][chosen]['median_ms']*1000,3),
                  'base',round(rec['forward_device']['tuned_ieee']['median_ms']*1000,3),
                  'maxmv',rec['heldout_accuracy'][chosen]['max_abs_mv'],flush=True)
            print('REJECTED',[(k,round(v['max_abs_mv'],6),v['spike_mismatch_bins']) for k,v in rec['heldout_accuracy'].items() if not v['strict_gate_passed']],flush=True)
            del p,y,w,pipelines,graphs,many,drives,filters,models,original,ref64,saved,m
            gc.collect();torch.cuda.empty_cache()
    result['finished']=stamp();result['resources_after']=resource_snapshot();save(args.output,result)
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
