"""Small, separate Nsight / arbitrary-shape correctness entry point."""
import argparse
import torch
from quadratic_tiled import TiledDrive

ap=argparse.ArgumentParser();ap.add_argument('--backend',default='triton')
ap.add_argument('--config',type=int,default=0);ap.add_argument('--triangle',action='store_true')
ap.add_argument('--precision',default='ieee');ap.add_argument('--A',type=int,default=500)
ap.add_argument('--M',type=int,default=6001);args=ap.parse_args()
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(7289)
with torch.no_grad():
    y=torch.randn(args.M,args.A,device='cuda')*.1
    w=torch.randn(args.A,args.A,device='cuda')*.01
    m=TiledDrive(y,w,-70.,args.backend,args.config,args.triangle,args.precision)
    for _ in range(3):out=m()
    torch.cuda.synchronize()
    ref=(y.double()@w.double()*y.double()).sum(-1)+y.double().sum(-1)-70
    print('finite',bool(torch.isfinite(out).all()),'max_abs',float((out.double()-ref).abs().max()))
    assert torch.allclose(out.double(),ref,rtol=2e-6,atol=1e-4)
