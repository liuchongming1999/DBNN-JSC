"""Two idle GPUs, global author batch 8 unchanged; side-effect-free probe."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from panel import rows_for, make_case, sample_batch, save
from train_default import config, model_for, losses


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    os.sched_setaffinity(0,{88+rank});torch.set_num_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl');device=f'cuda:{rank}';results=[]
    for dt in [.5,.25]:
        cfg=config('elm',dt);torch.manual_seed(cfg['seed'])
        cp=torch.load(a.work/'aligned'/f'elm_{dt:g}ms'/'last.pt',map_location='cpu',weights_only=False)
        model=model_for(a.work,cfg).to(device);model.load_state_dict(cp['model']);model.train()
        distributed=DDP(model,device_ids=[rank],broadcast_buffers=False)
        rows,_=rows_for(a.root,dt);fit=[make_case(r,dt,signed=True) for r in rows if r['split']=='fit'][:8]
        xx,yy,_=sample_batch(fit,round(500/dt),8,np.random.default_rng(cfg['seed']),dt)
        x=torch.from_numpy(xx).to(device);y=torch.from_numpy(yy).to(device)
        count=8//world;start=rank*count
        def forward_backward():
            distributed.zero_grad(set_to_none=True)
            value,_=losses(distributed,x[start:start+count],y[start:start+count],None,cfg,0);value.backward()
            return value
        for _ in range(3):forward_backward()
        times=[]
        for _ in range(8):
            dist.barrier();torch.cuda.synchronize();begin=time.perf_counter()
            value=forward_backward();torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            maximum=torch.tensor(elapsed,device=device);dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
            times.append(float(maximum))
        avg_loss=value.detach().clone();dist.all_reduce(avg_loss);avg_loss/=world
        if rank==0:
            original=model_for(a.work,cfg).to(device);original.load_state_dict(cp['model']);original.train()
            single=[]
            for i in range(11):
                original.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
                loss,_=losses(original,x,y,None,cfg,0);loss.backward();torch.cuda.synchronize()
                if i>=3:single.append(time.perf_counter()-begin)
            gaps=[float((p0.grad-p1.grad).abs().max()) for p0,p1 in zip(original.parameters(),model.parameters()) if p0.grad is not None]
            record=dict(dt_ms=dt,world_size=world,global_batch=8,per_gpu_batch=count,
                        single_median_s=float(np.median(single)),ddp_median_s=float(np.median(times)),
                        speedup=float(np.median(single)/np.median(times)),loss_abs_gap=float((loss-avg_loss).abs()),
                        gradient_max_abs_gap=max(gaps),checkpoint_steps=cp['steps'])
            results.append(record);save(a.out,dict(results=results,physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            print('RESULT',json.dumps(record),flush=True);del original
        dist.barrier();del distributed,model,fit;torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__=='__main__':main()
