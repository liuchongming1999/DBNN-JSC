"""Second-session fixed-candidate retest, general-shape checks and opt-in policy."""
import gc,hashlib,json,os,time,sys
from pathlib import Path
import numpy as np
import torch
from experiment import HERE,ROOT,DBNN_F,DBNN_R,accuracy,stamp,resource_snapshot
from quad_tune import make_data
from quadratic_tiled import TiledDrive
from quad_optimized_backend import TiledPrepared,make_optimized_prepared
from quadratic_benchmark import Modules,capture,measure_device,measure_wall,error,save


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('cublas')
    prior=json.loads((HERE/'quad_optimized.json').read_text())
    assert prior.get('finished') and len(prior['cases'])==9
    output=HERE/'quad_confirm.json'
    r=dict(started=stamp(),scope='fixed IEEE configurations from quad_optimized.json; second timing session',
        baseline_sha256=prior['baseline_sha256'],checkpoint_sha256=prior['checkpoint_sha256'],
        source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
            ['quadratic_tiled.cu','quadratic_tiled.py','quad_optimized_backend.py','quad_confirm.py']},
        physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),exclusive_resources=False,resources_before=resource_snapshot(),cases=[])
    policy=dict(version=1,precision='ieee',mode='frozen-checkpoint inference',hardware='RTX 5090',
        unknown_shape='cuBLAS',selection='screen candidates, freeze config, independently retest twice; conservative speed gates',shapes={})
    validate_only='--validate-only' in sys.argv
    if validate_only:
        r=json.loads(output.read_text())
        assert len(r['cases'])==9 and not r.get('finished')
        policy['shapes']={f'{c["A"]}:{c["T"]}:{c["B"]}':c['policy'] for c in r['cases']}
        r['validation_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with torch.no_grad():
        for old in ([] if validate_only else prior['cases']):
            A,T,B=old['A'],old['T'],old['B'];spec=old['selected']['new_ieee']['spec']
            print('RETEST',A,T,B,spec,flush=True)
            p,y,w=make_data(A,T,B);modules=Modules(y,w,p.bias)
            q=TiledDrive(y,w,p.bias,**spec)
            fns={'cublas':lambda:modules.drive('cublas'),'new_ieee':q,'old_direct':lambda:modules.drive('direct')}
            many={k:capture(fn,16 if A<=500 else 8) for k,fn in fns.items()}
            count=16 if A<=500 else 8
            c=dict(index=old['index'],A=A,T=T,B=B,kind=old['kind'],spec=spec,
                module_device=measure_device(many,count,60,9900+old['index']))
            new=TiledPrepared(p.source_state,p.x,spec)
            graph={'cublas':capture(p,1),'new_ieee':capture(new,1)}
            fw_many={'cublas':capture(p,8),'new_ieee':capture(new,8)}
            c['forward_device']=measure_device(fw_many,8,60,9930+old['index'])
            c['forward_graph_wall']=measure_wall(graph,60,9970+old['index'])
            base_keys=[k for k in old['original_paired_wall'] if k.startswith('original_')]
            best=min(base_keys,key=lambda k:old['original_paired_wall'][k]['median_ms'])
            cls=DBNN_R if best.startswith('original_R') else DBNN_F
            model=cls(A,T,torch.device('cuda'));model.load_state_dict(p.source_state);model.eval()
            if best.endswith('trace_loop'):
                base=lambda:torch.cat([model(p.x[b:b+1]) for b in range(B)])
            else:base=lambda:model(p.x)
            paired={best:base,'optimized_cublas':graph['cublas'],'optimized_new_ieee':graph['new_ieee']}
            for fn in paired.values():
                for _ in range(3):fn()
            c['original_paired_wall']=measure_wall(paired,30,9978+old['index']);c['original_method']=best
            reference=base().clone()
            c['error_vs_original']=accuracy(graph['new_ieee']().clone(),reference,p.vth)
            assert c['error_vs_original']['max_abs_mv']<1e-3 and c['error_vs_original']['spike_mismatch_bins']==0
            dev_ratio=c['module_device']['cublas']['median_ms']/c['module_device']['new_ieee']['median_ms']
            fw_ratio=c['forward_device']['cublas']['median_ms']/c['forward_device']['new_ieee']['median_ms']
            wall_ratio=c['forward_graph_wall']['cublas']['median_ms']/c['forward_graph_wall']['new_ieee']['median_ms']
            first_dev_ratio=old['module_device']['cublas']['median_ms']/old['module_device']['new_ieee']['median_ms']
            first_fw_ratio=old['forward_graph_device']['cublas']['median_ms']/old['forward_graph_device']['new_ieee']['median_ms']
            enabled=min(dev_ratio,first_dev_ratio)>1.05 and min(fw_ratio,first_fw_ratio)>1.02 and wall_ratio>.98
            entry=dict(enabled=enabled,spec=spec,module_speed_ratio=dev_ratio,
                       forward_device_ratio=fw_ratio,forward_wall_ratio=wall_ratio)
            policy['shapes'][f'{A}:{T}:{B}']=entry
            c['policy']=entry;c['unused_gemm_workspace_removed']=not hasattr(new,'z')
            r['cases'].append(c);save(output,r)
            print('RATIOS',entry,flush=True)
            del graph,fw_many,many,fns,paired,p,y,w,q,new,model,base,modules,reference
            gc.collect();torch.cuda.empty_cache()

        # Check arbitrary asymmetric W, nonzero diagonal, tiny/tail dimensions.
        specs={json.dumps(c['spec'],sort_keys=True):c['spec'] for c in r['cases']}
        specs['cuda_check']=dict(backend='cuda',config=3,triangle=True,precision='ieee')
        specs['warp_check']=dict(backend='warp',config=0,triangle=False,precision='ieee')
        tests=[];torch.manual_seed(83920)
        for A,M in [(7,3),(31,129),(63,17),(127,257),(500,129),(513,17),(1024,33)]:
            y=torch.randn(M,A,device='cuda')*.1;w=torch.randn(A,A,device='cuda')*.01
            ref=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)-70
            for spec in specs.values():
                fn=TiledDrive(y,w,-70.,**spec);out=fn().clone();er=error(out,ref)
                assert er['finite'] and er['max_abs']<1e-4,(A,M,spec,er)
                y_saved=y.clone();y.zero_();assert float((fn()+70).abs().max())==0
                y.copy_(y_saved)
                tests.append(dict(A=A,M=M,spec=spec,error=er))
        r['arbitrary_shape_checks']=tests

        # Every held-out biological input, with the source model as reference.
        p,_,_=make_data(500,6001,1)
        raw=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
        x=torch.from_numpy(raw['test_ip'].astype(np.float32)).cuda()
        x *= torch.cat([torch.ones(250,device='cuda'),-torch.ones(250,device='cuda')])[None,:,None]
        model=DBNN_R(500,6001,torch.device('cuda'));model.load_state_dict(p.source_state);model.eval()
        reference=model(x).clone();checks=[]
        for spec in {json.dumps(c['spec'],sort_keys=True):c['spec'] for c in r['cases'] if c['A']==500}.values():
            new=TiledPrepared(p.source_state,x,spec);fn=capture(new,1)
            er=accuracy(fn().clone(),reference,p.vth)
            assert er['finite'] and er['max_abs_mv']<1e-3 and er['spike_mismatch_bins']==0,(spec,er)
            x_saved=x.clone();x.zero_();assert float((fn()-p.bias).abs().max())<1e-5
            x.copy_(x_saved)
            restored=accuracy(fn().clone(),reference,p.vth)
            assert restored['max_abs_mv']<1e-3 and restored['spike_mismatch_bins']==0
            checks.append(dict(spec=spec,samples=len(x),error=er,changed_input_check=True))
        r['heldout_forward_checks']=checks
        r['checkpoint_bias_mv']=p.bias
        r['heldout_reset_counts']=(reference>=p.vth).sum(1).tolist()
        # Verify policy dispatch on every supported shape.
        for c in r['cases']:
            pp,_,_=make_data(c['A'],c['T'],c['B'])
            fn=make_optimized_prepared(pp.source_state,pp.x,policy)
            er=accuracy(fn().clone(),pp().clone(),pp.vth)
            assert er['max_abs_mv']<1e-3 and er['spike_mismatch_bins']==0
            assert isinstance(fn,TiledPrepared)==c['policy']['enabled']
        r['dispatch_checks_passed']=True
    r['profiling']='Nsight attempted; ERR_NVGPUCTRPERM. No measured occupancy or bandwidth claims.'
    r['finished']=stamp();r['resources_after']=resource_snapshot()
    save(output,r);save(HERE/'quad_backend_policy.json',policy)
    print('COMPLETE',output,'policy enabled',sum(e['enabled'] for e in policy['shapes'].values()),flush=True)


if __name__=='__main__':main()
