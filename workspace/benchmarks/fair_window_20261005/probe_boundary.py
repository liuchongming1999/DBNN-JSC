"""Verify ELM CUDA replay against eager for the prespecified cropped loss."""
import argparse, copy, os, time
from pathlib import Path
import numpy as np
import torch
from train_default import config,model_for,losses
from panel import rows_for,make_case,sample_batch,save
from elm_cuda_graph import ELMForwardBackwardGraph
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--work',required=True,type=Path);p.add_argument('--dt',required=True,type=float);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(930291);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=config('elm',a.dt);cfg['burn_in_loss_ms']=150
    eager=model_for(a.work,cfg).cuda().train()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        replay=model_for(a.work,cfg).cuda().train();replay.load_state_dict(eager.state_dict())
    torch.cuda.current_stream().wait_stream(stream)
    rows,_=rows_for(a.root,a.dt);fit=[make_case(r,a.dt,signed=True) for r in rows if r['split']=='fit'][:12]
    rng=np.random.default_rng(930291);batches=[sample_batch(fit,round(500/a.dt),8,rng,a.dt)[:2] for _ in range(5)]
    x=torch.from_numpy(batches[0][0]).cuda();y=torch.from_numpy(batches[0][1]).cuda()
    graph=ELMForwardBackwardGraph(replay,x,y,cfg,stream)
    opts=[torch.optim.Adam(m.parameters(),lr=cfg['lr']) for m in (eager,replay)]
    schedules=[torch.optim.lr_scheduler.CosineAnnealingLR(o,350000) for o in opts]
    checks=[]
    for step,(xx,yy) in enumerate(batches,1):
        x.copy_(torch.from_numpy(xx));y.copy_(torch.from_numpy(yy))
        opts[0].zero_grad(set_to_none=True);el,_=losses(eager,x,y,None,cfg,0);el.backward()
        gl,_=graph(x,y);torch.cuda.synchronize()
        gg=max(float((p.grad-q.grad).abs().max()) for p,q in zip(eager.parameters(),replay.parameters()) if p.grad is not None)
        lg=float(abs(el-gl));opts[0].step();opts[1].step();schedules[0].step();schedules[1].step()
        pg=max(float((p-q).abs().max()) for p,q in zip(eager.parameters(),replay.parameters()) if p.requires_grad)
        checks.append(dict(update=step,loss_gap=lg,gradient_gap=gg,parameter_gap=pg,lr_equal=opts[0].param_groups[0]['lr']==opts[1].param_groups[0]['lr']))
    passed=all(r['loss_gap']<1e-7 and r['gradient_gap']<1e-6 and r['parameter_gap']<1e-6 and r['lr_equal'] for r in checks)
    save(a.work/'execution_probe'/f'probe_{a.dt:g}ms.json',dict(passed=passed,checks=checks,burn_ms=150,model_unchanged=True,optimizer_unchanged=True,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),capture_s=graph.capture_s))
    print('PROBE',a.dt,passed,checks,flush=True);assert passed
if __name__=='__main__':main()
