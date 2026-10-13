#!/usr/bin/env python3
"""Screen symmetric low-rank approximations of the DBNN quadratic term."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from DBNN import DBNN_R, _apply_reset_closed_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "shai_ei_500syn_6s_dataset.npz")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "source" / "dbnn_checkpoint.pt")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "low_rank_screen_20260928.json")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 192, 256, 320, 384])
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    d = np.load(args.dataset, allow_pickle=True)
    signs = np.where(d["channel_type"].astype(str) == "exc", 1.0, -1.0).astype(np.float32)
    x = torch.from_numpy(d["test_ip"].astype(np.float32) * signs[None, :, None])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DBNN_R(500, 6001, torch.device("cpu"), scan_chunk_size=256)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    with torch.inference_mode():
        dense = model(x)
        decay_a, decay_c, omega = model._compute_coefficients()
        state_a = torch.zeros(x.shape[0], 500)
        state_c = torch.zeros_like(state_a)
        chunks = []
        for start in range(0, 6001, 256):
            end = min(start + 256, 6001)
            x_chunk = x[:, :, start:end]
            powers_a, inv_a = model._get_scan_terms(decay_a, end - start, x.dtype, x.device, "a")
            powers_c, inv_c = model._get_scan_terms(decay_c, end - start, x.dtype, x.device, "c")
            a = model._state_scan_chunk(x_chunk, powers_a, inv_a, state_a)
            c = model._state_scan_chunk(x_chunk, powers_c, inv_c, state_c)
            chunks.append(omega[None, :, None] * (a - c))
            state_a, state_c = a[:, :, -1], c[:, :, -1]
        y = torch.cat(chunks, dim=2).permute(0, 2, 1).contiguous()
        weight = model.bilinear._effective_weight().squeeze(0).double()
        eigvals, eigvecs = torch.linalg.eigh((weight + weight.T) / 2)
        order = torch.argsort(eigvals.abs(), descending=True)
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        result = {
            "dataset": str(args.dataset), "checkpoint": str(args.checkpoint),
            "samples": int(x.shape[0]), "channels": 500, "time_steps": 6001,
            "threads": args.threads, "approximation": "symmetric eigendecomposition, largest absolute eigenvalues",
            "ranks": {},
        }
        full_energy = float((eigvals * eigvals).sum())
        for rank in args.ranks:
            U, lam = eigvecs[:, :rank], eigvals[:rank]
            z = torch.matmul(y.double(), U)
            quadratic = (z * z * lam).sum(dim=-1)
            drive = quadratic + y.sum(dim=-1) + float(model.bilinear.bias.item())
            approx = _apply_reset_closed_loop(drive.float(), model.v_th, model.reset_amp, model.tau_reset, model.surr_beta)
            error = approx - dense
            result["ranks"][str(rank)] = {
                "energy_fraction": float((lam * lam).sum() / full_energy),
                "voltage_rmse_mv": float(torch.sqrt((error * error).mean())),
                "max_abs_mv": float(error.abs().max()),
                "mean_abs_mv": float(error.abs().mean()),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
