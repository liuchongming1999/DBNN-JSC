#!/usr/bin/env python3
"""Train a DBNN surrogate on the reproduced NEURON dataset.

This is a self-contained reproduction of the original training protocol:
E/I sign coding, an 80/20 fit/validation split of the 100 training samples,
Adam, voltage MSE plus a first-difference penalty, gradient clipping and
parameter range constraints. Checkpoints contain the dataset/config/splits.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "source") not in sys.path:
    sys.path.insert(0, str(ROOT / "source"))
from DBNN import DBNN_F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data" / "shai_ei_500syn_6s_dataset.npz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "dbnn_training_repro")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260425)
    p.add_argument("--diff-loss-weight", type=float, default=0.1)
    p.add_argument("--init-exc-omega", type=float, default=0.08)
    p.add_argument("--init-inh-omega", type=float, default=-0.05)
    p.add_argument("--init-exc-rise", type=float, default=1.0)
    p.add_argument("--init-inh-rise", type=float, default=1.5)
    p.add_argument("--init-exc-decay", type=float, default=10.0)
    p.add_argument("--init-inh-decay", type=float, default=16.0)
    p.add_argument("--init-bilinear-scale", type=float, default=0.0)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--init-omega-scale", type=float, default=1.0)
    p.add_argument("--init-weight-scale", type=float, default=1.0)
    p.add_argument("--fixed-baseline-mv", type=float, default=-80.84)
    p.add_argument("--fixed-v-th", type=float, default=-40.0)
    p.add_argument("--fixed-reset-amp", type=float, default=10.0)
    p.add_argument("--fixed-tau-reset", type=float, default=12.0)
    p.add_argument("--freeze-v-th", action="store_true")
    p.add_argument("--freeze-reset-amp", action="store_true")
    p.add_argument("--freeze-tau-reset", action="store_true")
    p.add_argument("--scheduler", choices=["none", "cosine", "step"], default="none")
    p.add_argument("--scheduler-step-size", type=int, default=75)
    p.add_argument("--scheduler-gamma", type=float, default=0.3)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def choose_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_inputs(ip, channel_type):
    signs = np.where(np.asarray(channel_type).astype(str) == "exc", 1.0, -1.0).astype(np.float32)
    return ip.astype(np.float32) * signs[None, :, None]


def split_indices(n, ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_size = max(1, int(round(n * ratio)))
    val_size = min(val_size, n - 1)
    return indices[val_size:], indices[:val_size]


def loss_parts(pred, target, diff_weight):
    mse = F.mse_loss(pred, target)
    if pred.shape[-1] > 1:
        diff = F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    else:
        diff = torch.zeros((), device=pred.device, dtype=pred.dtype)
    return mse + diff_weight * diff, mse


def init_model(model, channel_type, targets, args):
    baseline = float(args.fixed_baseline_mv)
    early = float(targets[:, :50].mean())
    source = "manual"
    exc = torch.as_tensor(np.asarray(channel_type).astype(str) == "exc", device=model.device)
    inh = ~exc
    with torch.no_grad():
        if args.init_checkpoint is not None:
            ckpt = torch.load(args.init_checkpoint, map_location=model.device, weights_only=False)
            state = ckpt["model_state"]
            for key, target in [("tau_rise", model.tau_rise), ("tau_decay", model.tau_decay),
                                ("omega", model.omega), ("bilinear.weight", model.bilinear.weight),
                                ("v_th", model.v_th), ("reset_amp", model.reset_amp),
                                ("tau_reset", model.tau_reset)]:
                target.copy_(state[key].to(model.device))
            model.omega.mul_(args.init_omega_scale)
            model.bilinear.weight.mul_(args.init_weight_scale)
            source = str(args.init_checkpoint)
        else:
            model.tau_rise[exc] = args.init_exc_rise
            model.tau_rise[inh] = args.init_inh_rise
            model.tau_decay[exc] = args.init_exc_decay
            model.tau_decay[inh] = args.init_inh_decay
            model.omega[exc] = args.init_exc_omega
            model.omega[inh] = args.init_inh_omega
            model.bilinear.weight.zero_()
            if args.init_bilinear_scale > 0:
                torch.nn.init.normal_(model.bilinear.weight, mean=0.0, std=args.init_bilinear_scale)
                n = model.num_dimensions
                model.bilinear.weight[:, torch.arange(n), torch.arange(n)] = 0.0
            model.v_th.fill_(args.fixed_v_th)
            model.reset_amp.fill_(args.fixed_reset_amp)
            model.tau_reset.fill_(args.fixed_tau_reset)
        model.bilinear.bias.fill_(baseline)
    return {"baseline_mv": baseline, "early_mv": early, "source": source}


def freeze_params(model, args):
    if args.freeze_v_th:
        model.v_th.requires_grad_(False)
    if args.freeze_reset_amp:
        model.reset_amp.requires_grad_(False)
    if args.freeze_tau_reset:
        model.tau_reset.requires_grad_(False)


def evaluate(model, loader, device, diff_weight):
    model.eval()
    total_loss = total_mse = total_items = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss, mse = loss_parts(pred, y, diff_weight)
            b = x.shape[0]
            total_loss += float(loss.item()) * b
            total_mse += float(mse.item()) * b
            total_items += b
    return total_loss / total_items, float(np.sqrt(total_mse / total_items))


def save_checkpoint(model, args, fit_idx, val_idx, channel_type, path):
    payload = {
        "model_state": model.state_dict(), "dataset": str(args.dataset),
        "fit_indices": fit_idx.tolist(), "val_indices": val_idx.tolist(),
        "config": vars(args), "channel_type": np.asarray(channel_type).astype(str).tolist(),
    }
    # pathlib objects in vars(args) are not JSON/pickle problematic, but storing strings is easier to inspect.
    payload["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in payload["config"].items()}
    torch.save(payload, path)


def main():
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    channel_type = data["channel_type"].astype(str)
    train_ip = encode_inputs(data["train_ip"], channel_type)
    train_op = data["train_op"].astype(np.float32)
    test_ip = encode_inputs(data["test_ip"], channel_type)
    test_op = data["test_op"].astype(np.float32)
    fit_idx, val_idx = split_indices(len(train_ip), args.val_ratio, args.seed)
    x_fit = torch.from_numpy(train_ip[fit_idx])
    y_fit = torch.from_numpy(train_op[fit_idx])
    x_val = torch.from_numpy(train_ip[val_idx])
    y_val = torch.from_numpy(train_op[val_idx])
    x_test = torch.from_numpy(test_ip)
    y_test = torch.from_numpy(test_op)
    train_loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=min(args.batch_size, len(x_fit)), shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=min(args.batch_size, len(x_val)), shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=min(args.batch_size, len(x_test)), shuffle=False)
    device = choose_device(args.device)

    model = DBNN_F(num_dimensions=train_ip.shape[1], time_dur=train_ip.shape[2], device=device).to(device)
    init_stats = init_model(model, channel_type, train_op[fit_idx], args)
    freeze_params(model, args)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    init_val_loss, init_val_rmse = evaluate(model, val_loader, device, args.diff_loss_weight)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history = []
    log_file = (args.output_dir / "training.log").open("w", encoding="utf-8")
    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()
    log(f"device={device}; fit={len(fit_idx)} val={len(val_idx)} test={len(test_ip)} channels={train_ip.shape[1]} T={train_ip.shape[2]}")
    log(f"initialization={init_stats}; initial_val_rmse={init_val_rmse:.4f} mV")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = running_mse = seen = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss, mse = loss_parts(pred, batch_y, args.diff_loss_weight)
            if not torch.isfinite(loss):
                log(f"skip non-finite loss at epoch={epoch} batch={seen}")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
            if bad_grad:
                log(f"skip non-finite gradient at epoch={epoch} batch={seen}")
                optimizer.zero_grad(set_to_none=True)
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                log(f"skip non-finite clipped gradient at epoch={epoch} batch={seen}")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                log(f"non-finite parameter update at epoch={epoch} batch={seen}; stopping")
                raise FloatingPointError("non-finite parameter update")
            with torch.no_grad():
                model.tau_rise.clamp_(min=0.25, max=50.0)
                model.tau_decay.clamp_(min=0.5, max=100.0)
                model.tau_reset.clamp_(min=1.0, max=50.0)
                model.reset_amp.clamp_(min=1.0, max=30.0)
                model.v_th.clamp_(min=-70.0, max=-10.0)
            b = batch_x.shape[0]
            running_loss += float(loss.item()) * b
            running_mse += float(mse.item()) * b
            seen += b
        train_loss = running_loss / seen
        train_rmse = float(np.sqrt(running_mse / seen))
        val_loss, val_rmse = evaluate(model, val_loader, device, args.diff_loss_weight)
        elapsed = time.perf_counter() - start
        epoch_seconds = time.perf_counter() - epoch_start
        rec = {"epoch": epoch, "train_loss": train_loss, "train_rmse_mv": train_rmse,
               "val_loss": val_loss, "val_rmse_mv": val_rmse, "epoch_seconds": epoch_seconds,
               "elapsed_seconds": elapsed, "lr": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, args, fit_idx, val_idx, channel_type, args.output_dir / "dbnn_best.pt")
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % max(1, args.save_every) == 0 or epoch == args.epochs:
            save_checkpoint(model, args, fit_idx, val_idx, channel_type, args.output_dir / f"dbnn_epoch_{epoch:04d}.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            log(f"epoch {epoch:04d}/{args.epochs} train_loss={train_loss:.5f} train_rmse={train_rmse:.3f}mV val_loss={val_loss:.5f} val_rmse={val_rmse:.3f}mV epoch_s={epoch_seconds:.2f}")
    save_checkpoint(model, args, fit_idx, val_idx, channel_type, args.output_dir / "dbnn_last.pt")
    model.load_state_dict(best_state)
    final_fit = evaluate(model, train_loader, device, args.diff_loss_weight)
    final_val = evaluate(model, val_loader, device, args.diff_loss_weight)
    final_test = evaluate(model, test_loader, device, args.diff_loss_weight)
    save_checkpoint(model, args, fit_idx, val_idx, channel_type, args.output_dir / "dbnn_best.pt")
    summary = {"dataset": str(args.dataset), "checkpoint": str(args.output_dir / "dbnn_best.pt"),
               "fit_samples": int(len(fit_idx)), "val_samples": int(len(val_idx)), "test_samples": int(len(test_ip)),
               "channels": int(train_ip.shape[1]), "time_dur": int(train_ip.shape[2]), "device": str(device),
               "initial_val_rmse_mv": init_val_rmse, "best_val_loss": best_val_loss,
               "total_train_seconds": time.perf_counter() - start, "final_fit_loss": final_fit[0], "final_fit_rmse_mv": final_fit[1],
               "final_val_loss": final_val[0], "final_val_rmse_mv": final_val[1], "final_test_loss": final_test[0], "final_test_rmse_mv": final_test[1]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    log_file.close()


if __name__ == "__main__":
    main()
