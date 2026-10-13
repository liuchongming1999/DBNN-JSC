"""Compiler resource metadata only; these are not measured occupancy counters."""
import json
from pathlib import Path
import torch,triton
from quadratic_tiled import TiledDrive,TRITON_TILES,_contract

torch.set_num_threads(1)
p=torch.cuda.get_device_properties(0)
result=dict(device=p.name,compute_capability=list(torch.cuda.get_device_capability()),
            multiprocessors=p.multi_processor_count,scope='compiler/static launch metadata; Nsight counters unavailable',kernels=[])
with torch.no_grad():
    for A,M,config,triangle in [(500,6001,3,True),(500,48008,9,True),(2048,6001,6,True),(500,6001,6,False)]:
        y=torch.zeros(M,A,device='cuda');w=torch.zeros(A,A,device='cuda')
        fn=TiledDrive(y,w,-80.84,'triton',config,triangle,'ieee')
        bm,bn,bk,nw,st=TRITON_TILES[config];nt=triton.cdiv(A,bn)
        kernel=_contract[(triton.cdiv(M,bm)*nt,)](y,fn.weight,fn.partial,fn.out,M,A,nt,fn.bias,
            bm,bn,bk,8,triangle,'ieee',num_warps=nw,num_stages=st)
        result['kernels'].append(dict(A=A,M=M,config=config,triangle=triangle,tile=[bm,bn,bk],
            warps_per_block=nw,blocks=triton.cdiv(M,bm)*nt,
            registers_per_thread=getattr(kernel,'n_regs',None),spill_count=getattr(kernel,'n_spills',None),
            shared_bytes=getattr(kernel.metadata,'shared',None)))
result['one_thread_per_timepoint_example']=dict(M=6001,threads_per_block=128,blocks=triton.cdiv(6001,128))
out=Path(__file__).with_name('quad_kernel_metadata.json');out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
