"""CuTe DSL warp-parallel affine prefix scan; no finite-tail approximation."""
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def scan_kernel(x:cute.Tensor,y:cute.Tensor,da:cute.Tensor,dc:cute.Tensor,om:cute.Tensor):
    tid,_,_=cute.arch.thread_idx()
    bid,b,_=cute.arch.block_idx()
    lane=tid%32
    n=bid*4+tid//32
    if n<x.shape[1]:
        a=da[n]; c=dc[n]; omega=om[n]
        sa=Float32(0.);sc=Float32(0.)
        for chunk in range(cute.ceil_div(x.shape[2],32)):
            t=chunk*32+lane
            val=Float32(0.)
            if t<x.shape[2]: val=x[b,n,t]
            pa=a;pc=c;ua=val;uc=val
            for k in cutlass.range_constexpr(5):
                dist=1<<k
                aa=cute.arch.shuffle_sync_up(pa,dist,mask_and_clamp=0)
                cc=cute.arch.shuffle_sync_up(pc,dist,mask_and_clamp=0)
                va=cute.arch.shuffle_sync_up(ua,dist,mask_and_clamp=0)
                vc=cute.arch.shuffle_sync_up(uc,dist,mask_and_clamp=0)
                if lane>=dist:
                    ua=pa*va+ua;uc=pc*vc+uc
                    pa=pa*aa;pc=pc*cc
            ya=ua+pa*sa;yc=uc+pc*sc
            if t<x.shape[2]:y[b,t,n]=omega*(ya-yc)
            sa=cute.arch.shuffle_sync(ya,31)
            sc=cute.arch.shuffle_sync(yc,31)

@cute.jit
def launch(x:cute.Tensor,y:cute.Tensor,da:cute.Tensor,dc:cute.Tensor,om:cute.Tensor,stream:cuda.CUstream):
    scan_kernel(x,y,da,dc,om).launch(grid=(cute.ceil_div(x.shape[1],4),x.shape[0],1),block=(128,1,1),stream=stream)

def make_scan(x,y,da,dc,om):
    tensors=tuple(from_dlpack(v) for v in (x,y,da,dc,om))
    stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled=cute.compile(launch,*tensors,stream)
    def run():
        compiled(*tensors,cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return run
