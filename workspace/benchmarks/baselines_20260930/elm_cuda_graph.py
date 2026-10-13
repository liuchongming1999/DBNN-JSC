"""Replay the unmodified author ELM forward/backward; keep Adam outside.

Model construction, loading and first warmup must use the same side stream.
Creating leaf parameters on the legacy stream caused capture/backward stream
dependencies on this PyTorch/CUDA runtime; the probe records the correction.
"""
import time
import torch


class ELMForwardBackwardGraph:
    def __init__(self,model,x,y,cfg,stream):
        from train_default import losses
        self.model=model;self.cfg=cfg;self.stream=stream
        began=time.perf_counter();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.x=x.clone();self.y=y.clone()
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                value,_=losses(model,self.x,self.y,None,cfg,0);value.backward()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph();model.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph,stream=stream):
            self.loss,self.parts=losses(model,self.x,self.y,None,cfg,0)
            self.loss.backward()
        torch.cuda.synchronize();self.capture_s=time.perf_counter()-began

    def __call__(self,x,y):
        self.x.copy_(x);self.y.copy_(y);self.graph.replay()
        return self.loss,self.parts
