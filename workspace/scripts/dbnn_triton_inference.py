"""Inference-only DBNN wrapper using Triton state and reset kernels."""
from __future__ import annotations
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from DBNN import DBNN_R
from triton_backend import dual_exp_scan, reset_closed_loop


class DBNNTritonInference:
    """Drop-in voltage inference for a DBNN checkpoint.

    The learned dense quadratic interaction remains a cuBLAS matmul. Triton
    replaces the reference float64 power/cumsum scan and Python reset loop.
    This class is inference-only and intentionally has no autograd path.
    """
    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model_state"]
        n = len(ckpt["channel_type"])
        # DBNN_R supplies the parameter containers and checkpoint-compatible state.
        self.model = DBNN_R(n, 6001, self.device, scan_chunk_size=256).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.float32:
            raise ValueError("DBNNTritonInference expects a CUDA float32 input")
        m = self.model
        yy = dual_exp_scan(x, m.tau_rise, m.tau_decay, m.omega)
        yb = yy.permute(0, 2, 1)
        W = m.bilinear._effective_weight().squeeze(0).to(dtype=x.dtype, device=x.device)
        drive = ((yb @ W) * yb).sum(dim=-1) + yb.sum(dim=-1) + m.bilinear.bias
        return reset_closed_loop(drive, m.v_th, m.reset_amp, m.tau_reset)
