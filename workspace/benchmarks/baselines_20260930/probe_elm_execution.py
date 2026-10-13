"""Execution-only ELM probe; never writes aligned training/checkpoint files.

Tests the untouched author's ScriptModule, fixed batch 8, same Adam/cosine
updates, against CUDA replay of forward/backward only. Timing is training
execution, not a replacement for the author's inference benchmark.
"""
import argparse, copy, json, os, time
from pathlib import Path
import numpy as np
import torch
from panel import rows_for, make_case, sample_batch, save
from train_default import config, model_for, losses
from elm_execution import plain_forward


def sync_time(fn, repeats=6):
    samples=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter(); fn()
        torch.cuda.synchronize(); samples.append(time.perf_counter()-start)
    return dict(median_s=float(np.median(samples)), samples_s=samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--dt',type=float,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--skip-batch-probe',action='store_true')
    p.add_argument('--jit-profiling-off',action='store_true');p.add_argument('--plain-capture',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.manual_seed(930291)
    if a.jit_profiling_off:
        torch._C._jit_set_profiling_executor(False);torch._C._jit_set_profiling_mode(False)
    cfg=config('elm',a.dt); device='cuda:0'
    cp=torch.load(a.work/'aligned'/f'elm_{a.dt:g}ms'/'last.pt',map_location='cpu',weights_only=False)
    rows,_=rows_for(a.root,a.dt)
    fit=[make_case(r,a.dt,signed=True) for r in rows if r['split']=='fit'][:12]
    rng=np.random.default_rng(930291);window=round(500/a.dt)
    batches=[sample_batch(fit,window,8,rng,a.dt)[:2] for _ in range(10)]
    x=torch.from_numpy(batches[0][0]).to(device);y=torch.from_numpy(batches[0][1]).to(device)
    report=dict(dt_ms=a.dt,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                author_model_unchanged=True,global_batch_size=8,checkpoint_step=cp['steps'],
                torch_version=torch.__version__,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    for batch in ([] if a.skip_batch_probe else [1,2,4,8]):
        model=model_for(a.work,cfg).to(device);model.load_state_dict(cp['model']);model.train()
        def run():
            model.zero_grad(set_to_none=True)
            value,_=losses(model,x[:batch],y[:batch],None,cfg,0);value.backward()
        for _ in range(3):run()
        report[f'batch_{batch}_forward_backward']=sync_time(run)
        del model;torch.cuda.empty_cache()
        save(a.out,report);print('BATCH',batch,report[f'batch_{batch}_forward_backward'],flush=True)
    baseline=model_for(a.work,cfg).to(device);baseline.load_state_dict(cp['model']);baseline.train()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        replay=model_for(a.work,cfg).to(device);replay.load_state_dict(cp['model']);replay.train()
        static_x=x.clone();static_y=y.clone()
    torch.cuda.current_stream().wait_stream(stream)
    replay_call=(lambda z:plain_forward(replay,z)) if a.plain_capture else replay
    for _ in range(3):
        baseline.zero_grad(set_to_none=True);value,_=losses(baseline,x,y,None,cfg,0);value.backward()
    capture_start=time.perf_counter()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            replay.zero_grad(set_to_none=True);value,_=losses(replay_call,static_x,static_y,None,cfg,0);value.backward()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph();replay.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph,stream=stream):
        graph_loss,graph_parts=losses(replay_call,static_x,static_y,None,cfg,0);graph_loss.backward()
    torch.cuda.synchronize();report['capture_s']=time.perf_counter()-capture_start
    print('CAPTURED',report['capture_s'],flush=True)
    opts=[torch.optim.Adam(m.parameters(),lr=cfg['lr']) for m in [baseline,replay]]
    schedules=[torch.optim.lr_scheduler.CosineAnnealingLR(o,350000) for o in opts]
    for opt,scheduler in zip(opts,schedules):
        opt.load_state_dict(copy.deepcopy(cp['optimizer']));scheduler.load_state_dict(cp['scheduler'])
    parity=[];baseline_times=[];graph_times=[]
    for i,(xx,yy) in enumerate(batches):
        x.copy_(torch.from_numpy(xx));y.copy_(torch.from_numpy(yy))
        static_x.copy_(x);static_y.copy_(y);torch.cuda.synchronize()
        start=time.perf_counter();opts[0].zero_grad(set_to_none=True)
        eager_loss,eager_parts=losses(baseline,x,y,None,cfg,0);eager_loss.backward();torch.cuda.synchronize()
        baseline_times.append(time.perf_counter()-start)
        start=time.perf_counter();graph.replay();torch.cuda.synchronize()
        graph_times.append(time.perf_counter()-start)
        grads=[]
        for (_,p0),(_,p1) in zip(baseline.named_parameters(),replay.named_parameters()):
            if p0.grad is not None:
                grads.append(float((p0.grad-p1.grad).abs().max()))
                assert torch.isfinite(p0.grad).all() and torch.isfinite(p1.grad).all()
        loss_gap=float((eager_loss-graph_loss).abs())
        opts[0].step();opts[1].step();schedules[0].step();schedules[1].step()
        parameters=[float((p0-p1).abs().max()) for p0,p1 in zip(baseline.parameters(),replay.parameters()) if p0.requires_grad]
        parity.append(dict(update=i+1,loss_abs_gap=loss_gap,gradient_max_abs_gap=max(grads),
                           parameter_max_abs_gap=max(parameters),lr_equal=opts[0].param_groups[0]['lr']==opts[1].param_groups[0]['lr']))
        print('PARITY',parity[-1],flush=True)
    report.update(parity=parity,eager_median_s=float(np.median(baseline_times)),
                  graph_median_s=float(np.median(graph_times)),speedup=float(np.median(baseline_times)/np.median(graph_times)),
                  peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                  passed=all(r['loss_abs_gap']<1e-7 and r['gradient_max_abs_gap']<1e-6 and r['parameter_max_abs_gap']<1e-6 and r['lr_equal'] for r in parity),
                  finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    save(a.out,report);print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
