"""Triton recurrence backend for DBNN inference.

The kernel computes the two stable first-order states directly, avoiding the
float64 power/inverse-power construction used by the reference DBNN_R. It is
inference-only in this first version; training still uses the reference model.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _dual_exp_scan_kernel(
        x_ptr, y_ptr, decay_a_ptr, decay_c_ptr, omega_ptr,
        B, N, T,
        stride_xb, stride_xn, stride_xt,
        stride_yb, stride_yn, stride_yt,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = (N + BLOCK - 1) // BLOCK
        b = pid // grid_n
        n0 = (pid - b * grid_n) * BLOCK
        offs = n0 + tl.arange(0, BLOCK)
        mask_n = offs < N
        da = tl.load(decay_a_ptr + offs, mask=mask_n, other=0.0)
        dc = tl.load(decay_c_ptr + offs, mask=mask_n, other=0.0)
        om = tl.load(omega_ptr + offs, mask=mask_n, other=0.0)
        sa = tl.zeros((BLOCK,), dtype=tl.float32)
        sc = tl.zeros((BLOCK,), dtype=tl.float32)
        for t in range(0, T):
            x = tl.load(x_ptr + b * stride_xb + offs * stride_xn + t * stride_xt, mask=mask_n, other=0.0)
            sa = da * sa + x
            sc = dc * sc + x
            y = om * (sa - sc)
            tl.store(y_ptr + b * stride_yb + offs * stride_yn + t * stride_yt, y, mask=mask_n)


def dual_exp_scan(x: torch.Tensor, tau_rise: torch.Tensor, tau_decay: torch.Tensor, omega: torch.Tensor):
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise ValueError("dual_exp_scan currently requires CUDA")
    if x.ndim != 3 or x.dtype != torch.float32:
        raise ValueError("expected contiguous float32 x with shape (B,N,T)")
    x = x.contiguous()
    B, N, T = x.shape
    da = torch.exp(-1.0 / tau_decay).to(device=x.device, dtype=torch.float32).contiguous()
    dc = torch.exp(-(1.0 / tau_rise + 1.0 / tau_decay)).to(device=x.device, dtype=torch.float32).contiguous()
    om = omega.to(device=x.device, dtype=torch.float32).contiguous()
    y = torch.empty_like(x)
    block = 1
    while block < N and block < 1024:
        block *= 2
    grid_n = (N + block - 1) // block
    _dual_exp_scan_kernel[(B * grid_n,)](
        x, y, da, dc, om, B, N, T,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK=block,
        num_warps=4,
    )
    return y


if triton is not None:
    @triton.jit
    def _reset_kernel(
        drive_ptr, out_ptr, B, T,
        stride_db, stride_dt, stride_ob, stride_ot,
        alpha_ptr, amp_ptr, vth_ptr,
    ):
        b = tl.program_id(0)
        alpha = tl.load(alpha_ptr)
        amp = tl.load(amp_ptr)
        vth = tl.load(vth_ptr)
        r = tl.zeros((), dtype=tl.float32)
        for t in range(0, T):
            drive = tl.load(drive_ptr + b * stride_db + t * stride_dt)
            v = drive + r
            tl.store(out_ptr + b * stride_ob + t * stride_ot, v)
            spike = v >= vth
            r = alpha * r - amp * spike.to(tl.float32)


def reset_closed_loop(drive: torch.Tensor, v_th: torch.Tensor, reset_amp: torch.Tensor, tau_reset: torch.Tensor):
    """Inference-only CUDA reset recurrence, with no Python loop over time."""
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not drive.is_cuda or drive.dtype != torch.float32 or drive.ndim != 2:
        raise ValueError("expected CUDA float32 drive with shape (B,T)")
    drive = drive.contiguous()
    B, T = drive.shape
    out = torch.empty_like(drive)
    alpha = torch.exp(-torch.ones((), device=drive.device, dtype=torch.float32) / torch.clamp(torch.abs(tau_reset).to(torch.float32), min=1e-4))
    amp = torch.abs(reset_amp).to(torch.float32)
    vth = v_th.to(torch.float32)
    _reset_kernel[(B,)](drive, out, B, T, drive.stride(0), drive.stride(1), out.stride(0), out.stride(1), alpha, amp, vth, num_warps=1)
    return out
