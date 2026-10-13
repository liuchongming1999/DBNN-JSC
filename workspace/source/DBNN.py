import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent


def choose_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def encode_inputs(ip, channel_type):
    signs = np.where(np.asarray(channel_type) == "exc", 1.0, -1.0).astype(np.float32)
    return ip.astype(np.float32) * signs[None, :, None]


def channel_signs(channel_type):
    return np.where(np.asarray(channel_type) == "exc", 1.0, -1.0).astype(np.float32)


def _spike_ste(v, v_th, surr_beta):
    s_hard = (v >= v_th).to(v.dtype)
    if not torch.is_grad_enabled():
        return s_hard
    s_soft = torch.sigmoid(surr_beta * (v - v_th))
    return s_hard + (s_soft - s_soft.detach())


def _apply_reset_closed_loop(v_drive, v_th, reset_amp, tau_reset, surr_beta):
    bsz, steps = v_drive.shape
    dtype = v_drive.dtype
    device = v_drive.device

    tau = torch.clamp(torch.abs(tau_reset), min=1e-4).to(dtype=dtype)
    alpha = torch.exp(-torch.ones((), device=device, dtype=dtype) / tau)
    amp = torch.abs(reset_amp).to(dtype=dtype)
    v_th = v_th.to(dtype=dtype)

    if (not torch.is_grad_enabled()) and bsz == 1:
        v = torch.empty_like(v_drive)
        alpha_pows = alpha.pow(torch.arange(steps, device=device, dtype=dtype))
        t = 0
        r_t = torch.zeros((), device=device, dtype=dtype)

        while t < steps:
            remain = steps - t
            decay = alpha_pows[:remain]
            v_seg = v_drive[0, t:] + r_t * decay
            spike_mask = v_seg >= v_th
            spike_idx = torch.nonzero(spike_mask, as_tuple=False)

            if spike_idx.numel() == 0:
                v[0, t:] = v_seg
                break

            k = int(spike_idx[0, 0].item())
            end = t + k
            v[0, t : end + 1] = v_seg[: k + 1]
            r_at_spike = r_t * decay[k]
            r_t = alpha * r_at_spike - amp
            t = end + 1

        return v

    v = torch.empty_like(v_drive)
    r_t = torch.zeros((bsz,), device=device, dtype=dtype)

    for t in range(steps):
        v_t = v_drive[:, t] + r_t
        s_t = _spike_ste(v_t, v_th, surr_beta)
        v[:, t] = v_t
        r_t = alpha * r_t - amp * s_t

    return v


class BilinearModified(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in1_features, in2_features, device=device))
        if bias:
            self.register_buffer("bias", torch.tensor([-70.0], device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        self._zero_diagonal_()

    def _zero_diagonal_(self):
        with torch.no_grad():
            diagonal_size = min(self.in1_features, self.in2_features)
            diagonal_index = torch.arange(diagonal_size, device=self.weight.device)
            self.weight[:, diagonal_index, diagonal_index] = 0

    def _effective_weight(self):
        self._zero_diagonal_()
        return self.weight

    def forward(self, x1, x2):
        return F.bilinear(x1, x2, self._effective_weight(), self.bias)


class DBNN_F(nn.Module):
    def __init__(self, num_dimensions, time_dur, device):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.time_dur = time_dur
        self.device = device

        self.tau_rise = nn.Parameter(torch.ones(num_dimensions, device=device) * 5)
        self.tau_decay = nn.Parameter(torch.ones(num_dimensions, device=device) * 20)
        self.omega = nn.Parameter(torch.ones(num_dimensions, device=device) * 0.3)

        self.bilinear = BilinearModified(num_dimensions, num_dimensions, 1, device=device).to(device)
        self.v_th = nn.Parameter(torch.tensor(-55.0, device=device))
        self.reset_amp = nn.Parameter(torch.tensor(15.0, device=device))
        self.tau_reset = nn.Parameter(torch.tensor(10.0, device=device))
        self.surr_beta = 10.0

    def create_kernels(self):
        time_index = torch.arange(self.time_dur, device=self.tau_rise.device)
        tau_rise = self.tau_rise.unsqueeze(1)
        tau_decay = self.tau_decay.unsqueeze(1)
        omega = self.omega.unsqueeze(1)
        kernels = omega * (1 - torch.exp(-time_index / tau_rise)) * torch.exp(-time_index / tau_decay)
        return kernels.unsqueeze(1)

    def forward(self, x):
        bsz, nd, steps = x.shape
        assert nd == self.num_dimensions
        assert steps == self.time_dur

        kernels = self.create_kernels()
        kernel_time = kernels.squeeze(1)
        full_length = 2 * steps - 1
        fft_size = 1 << (full_length - 1).bit_length()

        x_fft = torch.fft.rfft(x, n=fft_size, dim=2)
        k_fft = torch.fft.rfft(kernel_time, n=fft_size, dim=1)
        y_fft = x_fft * k_fft.unsqueeze(0)
        y_full = torch.fft.irfft(y_fft, n=fft_size, dim=2)[..., :full_length]
        y = y_full[..., :steps]

        y_perm = y.permute(0, 2, 1)
        bilinear_term = self.bilinear(y_perm, y_perm)
        linear_term = torch.sum(y_perm, dim=2).unsqueeze(-1)
        v_drive = (bilinear_term + linear_term).squeeze(-1)
        return _apply_reset_closed_loop(v_drive, self.v_th, self.reset_amp, self.tau_reset, self.surr_beta)


class DBNN_R(nn.Module):
    def __init__(self, num_dimensions, time_dur, device, scan_chunk_size=256, enable_inference_cache=True):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.time_dur = time_dur
        self.device = device
        self.scan_chunk_size = int(scan_chunk_size)
        self.enable_inference_cache = enable_inference_cache

        self.tau_rise = nn.Parameter(torch.ones(num_dimensions, device=device) * 5)
        self.tau_decay = nn.Parameter(torch.ones(num_dimensions, device=device) * 20)
        self.omega = nn.Parameter(torch.ones(num_dimensions, device=device) * 0.3)

        self.bilinear = BilinearModified(num_dimensions, num_dimensions, 1, device=device).to(device)
        self.v_th = nn.Parameter(torch.tensor(-55.0, device=device))
        self.reset_amp = nn.Parameter(torch.tensor(15.0, device=device))
        self.tau_reset = nn.Parameter(torch.tensor(10.0, device=device))
        self.surr_beta = 10.0

        self._cached_coefficients = None
        self._cache_signature = None
        self._cached_scan_terms = {}
        self._cached_quadratic_weight = {}

    def clear_inference_cache(self):
        self._cached_coefficients = None
        self._cache_signature = None
        self._cached_scan_terms = {}
        self._cached_quadratic_weight = {}

    def _cache_enabled_for_this_forward(self):
        return self.enable_inference_cache and (not self.training) and (not torch.is_grad_enabled())

    def _current_cache_signature(self):
        return (
            int(self.tau_rise._version),
            int(self.tau_decay._version),
            int(self.omega._version),
        )

    def _compute_coefficients(self):
        decay_a = torch.exp(-1.0 / self.tau_decay)
        decay_c = torch.exp(-(1.0 / self.tau_rise + 1.0 / self.tau_decay))
        return decay_a, decay_c, self.omega

    def _get_coefficients(self):
        if not self._cache_enabled_for_this_forward():
            return self._compute_coefficients()

        signature = self._current_cache_signature()
        if self._cache_signature != signature or self._cached_coefficients is None:
            decay_a, decay_c, omega = self._compute_coefficients()
            self._cached_coefficients = (decay_a.detach(), decay_c.detach(), omega.detach())
            self._cache_signature = signature
        return self._cached_coefficients

    def _scan_dtype(self, dtype):
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.float64
        return dtype

    def _get_scan_terms(self, decay, steps, dtype, device, cache_key):
        scan_dtype = self._scan_dtype(dtype)
        if not self._cache_enabled_for_this_forward():
            step_index = torch.arange(1, steps + 1, device=device, dtype=scan_dtype)
            powers = decay.to(dtype=scan_dtype, device=device).unsqueeze(1).pow(step_index)
            inv_powers = torch.reciprocal(powers)
            return powers, inv_powers

        signature = (self._current_cache_signature(), cache_key, steps, str(device), str(scan_dtype))
        cached = self._cached_scan_terms.get(signature)
        if cached is None:
            step_index = torch.arange(1, steps + 1, device=device, dtype=scan_dtype)
            powers = decay.to(dtype=scan_dtype, device=device).unsqueeze(1).pow(step_index).detach()
            inv_powers = torch.reciprocal(powers).detach()
            cached = (powers, inv_powers)
            self._cached_scan_terms[signature] = cached
        return cached

    def _get_quadratic_weight(self, dtype, device):
        if not self._cache_enabled_for_this_forward():
            return self.bilinear._effective_weight().squeeze(0).to(dtype=dtype, device=device).contiguous()

        signature = (int(self.bilinear.weight._version), str(device), str(dtype))
        cached = self._cached_quadratic_weight.get(signature)
        if cached is None:
            cached = self.bilinear._effective_weight().squeeze(0).to(dtype=dtype, device=device).contiguous().detach()
            self._cached_quadratic_weight[signature] = cached
        return cached

    def _state_scan_chunk(self, x_chunk, powers, inv_powers, state_prev):
        scan_dtype = powers.dtype
        x_scaled = x_chunk.to(dtype=scan_dtype) * inv_powers.unsqueeze(0)
        csum = x_scaled.cumsum(dim=2)
        state = powers.unsqueeze(0) * (state_prev.to(dtype=scan_dtype).unsqueeze(-1) + csum)
        return state.to(dtype=x_chunk.dtype)

    def forward(self, x):
        bsz, nd, steps = x.shape
        assert nd == self.num_dimensions
        assert steps == self.time_dur

        decay_a, decay_c, omega = self._get_coefficients()
        quadratic_weight = self._get_quadratic_weight(x.dtype, x.device)
        bias = self.bilinear.bias

        state_a = torch.zeros(bsz, nd, device=x.device, dtype=x.dtype)
        state_c = torch.zeros(bsz, nd, device=x.device, dtype=x.dtype)
        output = torch.empty(bsz, steps, device=x.device, dtype=x.dtype)

        omega_b = omega.unsqueeze(0).unsqueeze(-1)
        chunk = max(1, self.scan_chunk_size)

        for start in range(0, steps, chunk):
            end = min(start + chunk, steps)
            chunk_steps = end - start
            x_chunk = x[:, :, start:end]
            powers_a, inv_powers_a = self._get_scan_terms(decay_a, chunk_steps, x.dtype, x.device, "a")
            powers_c, inv_powers_c = self._get_scan_terms(decay_c, chunk_steps, x.dtype, x.device, "c")

            state_a_chunk = self._state_scan_chunk(x_chunk, powers_a, inv_powers_a, state_a)
            state_c_chunk = self._state_scan_chunk(x_chunk, powers_c, inv_powers_c, state_c)
            y_chunk = omega_b * (state_a_chunk - state_c_chunk)

            if (not torch.is_grad_enabled()) and bsz == 1:
                y_ln = y_chunk[0].transpose(0, 1).contiguous()
                linear = y_ln.sum(dim=1, keepdim=True).transpose(0, 1)
                quadratic = torch.sum(torch.mm(y_ln, quadratic_weight) * y_ln, dim=1, keepdim=True).transpose(0, 1)
            else:
                y_btn = y_chunk.permute(0, 2, 1).contiguous()
                linear = y_btn.sum(dim=2)
                quadratic = ((y_btn @ quadratic_weight) * y_btn).sum(dim=2)

            out_chunk = linear + quadratic
            if bias is not None:
                out_chunk = out_chunk + bias
            output[:, start:end] = out_chunk

            state_a = state_a_chunk[:, :, -1]
            state_c = state_c_chunk[:, :, -1]

        return _apply_reset_closed_loop(output, self.v_th, self.reset_amp, self.tau_reset, self.surr_beta)


DBNN = DBNN_F


def build_model(model_kind, num_dimensions, time_dur, device):
    if model_kind == "f":
        return DBNN_F(num_dimensions=num_dimensions, time_dur=time_dur, device=device).to(device)
    if model_kind == "r":
        return DBNN_R(num_dimensions=num_dimensions, time_dur=time_dur, device=device).to(device)
    raise ValueError(f"Unsupported model kind: {model_kind}")


def load_model_from_checkpoint(checkpoint_path, model_kind, time_dur, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    channel_type = np.asarray(checkpoint["channel_type"]).astype(str)
    model = build_model(model_kind, len(channel_type), time_dur, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, channel_type


def make_poisson_input(num_dimensions, time_dur, freq_hz, channel_type, seed):
    rng = np.random.default_rng(int(seed))
    dt_ms = 1.0
    p = (float(freq_hz) / 1000.0) * dt_ms
    spikes = (rng.random((num_dimensions, time_dur), dtype=np.float32) < p).astype(np.float32)
    signs = channel_signs(channel_type).reshape(num_dimensions, 1)
    return spikes * signs


def detect_spike_times(voltage_mv, threshold_mv):
    above = voltage_mv >= float(threshold_mv)
    rising = np.flatnonzero(np.logical_and(above[1:], ~above[:-1])) + 1
    if above[0]:
        rising = np.concatenate(([0], rising))
    return rising.astype(np.int32).tolist()


def simulate_poisson_response(
    checkpoint_path=REPO_ROOT / "dbnn_checkpoint.pt",
    model_kind="f",
    freq_hz=50.0,
    duration_ms=6000,
    seed=20260426,
    device_arg="auto",
):
    device = choose_device(device_arg)
    time_dur = int(duration_ms) + 1
    model, channel_type = load_model_from_checkpoint(checkpoint_path, model_kind, time_dur, device)

    input_array = make_poisson_input(
        num_dimensions=len(channel_type),
        time_dur=time_dur,
        freq_hz=freq_hz,
        channel_type=channel_type,
        seed=seed,
    )
    batch_x = torch.from_numpy(input_array[None, :, :]).to(device)

    with torch.inference_mode():
        voltage_mv = model(batch_x).detach().cpu().numpy()[0]

    spike_times_ms = detect_spike_times(voltage_mv, float(model.v_th.detach().cpu()))
    return {
        "model_kind": model_kind,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "freq_hz": float(freq_hz),
        "duration_ms": int(duration_ms),
        "time_dur": int(time_dur),
        "seed": int(seed),
        "threshold_mv": float(model.v_th.detach().cpu()),
        "reset_amp": float(model.reset_amp.detach().cpu()),
        "tau_reset": float(model.tau_reset.detach().cpu()),
        "spike_times_ms": spike_times_ms,
        "voltage_mv": voltage_mv.astype(np.float32),
        "input_array": input_array.astype(np.float32),
        "channel_type": channel_type,
    }


def save_simulation_outputs(result, output_prefix=None):
    if output_prefix is None:
        output_prefix = (
            REPO_ROOT
            / "figures"
            / f"dbnn_{result['model_kind']}_poisson_{int(result['freq_hz'])}hz_{int(result['duration_ms'])}ms_seed{result['seed']}"
        )
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    time_ms = np.arange(result["time_dur"], dtype=np.float32)
    fig_path = output_prefix.with_suffix(".png")

    plt.figure(figsize=(12, 4.5))
    plt.plot(time_ms, result["voltage_mv"], color="tab:red", linewidth=1.2, label=f"DBNN_{result['model_kind'].upper()}")
    for t in result["spike_times_ms"]:
        plt.axvline(t, color="tab:red", alpha=0.18, linestyle="--", linewidth=1.0)
    plt.xlabel("Time (ms)")
    plt.ylabel("Somatic Voltage (mV)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    return fig_path


def save_simulation_npz(result, npz_output=None):
    if npz_output is None:
        npz_output = (
            REPO_ROOT
            / "results"
            / f"dbnn_{result['model_kind']}_poisson_{int(result['freq_hz'])}hz_{int(result['duration_ms'])}ms_seed{result['seed']}.npz"
        )
    npz_output = Path(npz_output)
    npz_output.parent.mkdir(parents=True, exist_ok=True)

    time_ms = np.arange(result["time_dur"], dtype=np.float32)
    np.savez(
        npz_output,
        time_ms=time_ms,
        voltage_mv=result["voltage_mv"].astype(np.float32),
        spike_times_ms=np.asarray(result["spike_times_ms"], dtype=np.int32),
    )
    return npz_output


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate DBNN_F or DBNN_R with Poisson input and save voltage/spike-time outputs.")
    parser.add_argument("--backend", choices=["f", "r"], default="f")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "dbnn_checkpoint.pt")
    parser.add_argument("--freq-hz", type=float, default=50.0)
    parser.add_argument("--duration-ms", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--npz-output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = simulate_poisson_response(
        checkpoint_path=args.checkpoint,
        model_kind=args.backend,
        freq_hz=args.freq_hz,
        duration_ms=args.duration_ms,
        seed=args.seed,
        device_arg=args.device,
    )
    fig_path = save_simulation_outputs(result, output_prefix=args.output_prefix)
    npz_path = None
    if args.save_npz:
        npz_path = save_simulation_npz(result, npz_output=args.npz_output)
    print(
        f"Saved DBNN_{result['model_kind'].upper()} output to {fig_path} "
        f"(freq={result['freq_hz']} Hz, spikes={len(result['spike_times_ms'])}, device={result['device']})"
    )
    if npz_path is not None:
        print(f"Saved NPZ output to {npz_path}")
    print(f"spike_times_ms={result['spike_times_ms']}")


if __name__ == "__main__":
    main()
