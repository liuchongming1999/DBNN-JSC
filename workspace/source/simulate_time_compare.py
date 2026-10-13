import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
SHAI_ROOT = REPO_ROOT / "ShaiEtAl2015"
FIGURES_DIR = REPO_ROOT / "figures"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


NUM_SYNAPSES = 500
NUM_EXC = 250
NUM_INH = 250
DT_MS = 0.1
V_INIT_MV = -75.0
SPIKE_THRESHOLD_MV = -20.0
INPUT_RATE_HZ = 50.0
EXC_WEIGHT = 1.88e-4
INH_WEIGHT = 8.0e-5
EXC_TAU1_MS = 0.5
EXC_TAU2_MS = 5.0
INH_TAU1_MS = 1.0
INH_TAU2_MS = 10.0
INH_REVERSAL_MV = -80.0
SITE_SEED = 20260425
SAMPLE_INDEX = 0
BIOPHYS_FILE = "L5PCbiophys5.hoc"
MORPHOLOGY_FILE = "cell1.asc"


def _is_torch_compiling():
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "is_compiling"):
        return compiler_mod.is_compiling()

    dynamo_mod = getattr(torch, "_dynamo", None)
    if dynamo_mod is not None and hasattr(dynamo_mod, "is_compiling"):
        return dynamo_mod.is_compiling()

    return False


def _spike_ste(v, v_th, surr_beta):
    s_hard = (v >= v_th).to(v.dtype)
    # In inference (no autograd), skip surrogate path for speed.
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

    # In inference for a single trace, use an event-driven sparse path:
    # segment voltage analytically until the next threshold crossing.
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


def ensure_neuron_mechanisms(root: Path):
    mech_lib = root / "x86_64" / "libnrnmech.so"
    if mech_lib.exists():
        return

    nrnivmodl = Path(sys.executable).with_name("nrnivmodl")
    if not nrnivmodl.exists():
        raise FileNotFoundError(f"Missing nrnivmodl at {nrnivmodl}")
    subprocess.run([str(nrnivmodl), "mod"], cwd=root, check=True)


def choose_sites_weighted(sections, count, rng):
    lengths = np.array([float(sec.L) for sec in sections], dtype=np.float64)
    probabilities = lengths / lengths.sum()
    indices = rng.choice(len(sections), size=count, replace=True, p=probabilities)
    return [(sections[i], float(rng.uniform(0.1, 0.9))) for i in indices]


def expected_event_count(total_ms, rate_hz):
    expected = total_ms * rate_hz / 1000.0
    return int(max(64, np.ceil(expected * 3.0)))


def build_neuron_runtime_state(duration_ms, base_seed):
    from neuron import h, load_mechanisms

    ensure_neuron_mechanisms(SHAI_ROOT)
    load_mechanisms(str(SHAI_ROOT))

    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(str(SHAI_ROOT / "models" / BIOPHYS_FILE))
    h.load_file(str(SHAI_ROOT / "models" / "L5PCtemplate.hoc"))

    cell = h.L5PCtemplate(str(SHAI_ROOT / "morphologies" / MORPHOLOGY_FILE))
    dend_sections = [sec for sec in cell.dend] + [sec for sec in cell.apic]
    site_rng = np.random.default_rng(SITE_SEED)
    sites = choose_sites_weighted(dend_sections, NUM_SYNAPSES, site_rng)

    interval_ms = 1000.0 / INPUT_RATE_HZ
    event_count = expected_event_count(duration_ms, INPUT_RATE_HZ)

    synapses = []
    netstims = []
    netcons = []
    channel_type = []

    for channel_idx, (section, location) in enumerate(sites):
        syn = h.Exp2Syn(section(location))
        is_exc = channel_idx < NUM_EXC
        if is_exc:
            syn.e = 0.0
            syn.tau1 = EXC_TAU1_MS
            syn.tau2 = EXC_TAU2_MS
            weight = EXC_WEIGHT
            channel_type.append("exc")
        else:
            syn.e = INH_REVERSAL_MV
            syn.tau1 = INH_TAU1_MS
            syn.tau2 = INH_TAU2_MS
            weight = INH_WEIGHT
            channel_type.append("inh")

        netstim = h.NetStim()
        netstim.interval = interval_ms
        netstim.number = event_count
        netstim.noise = 1.0
        netstim.start = 0.0
        netstim.noiseFromRandom123(int(base_seed), SAMPLE_INDEX + 1, channel_idx + 1)

        netcon = h.NetCon(netstim, syn)
        netcon.delay = 0.0
        netcon.weight[0] = weight

        synapses.append(syn)
        netstims.append(netstim)
        netcons.append(netcon)

    return {
        "h": h,
        "cell": cell,
        "duration_ms": float(duration_ms),
        "synapses": synapses,
        "netstims": netstims,
        "netcons": netcons,
        "channel_type": np.array(channel_type),
    }


def run_neuron_once(state):
    h = state["h"]
    h.dt = DT_MS
    h.steps_per_ms = int(round(1.0 / DT_MS))
    h.finitialize(V_INIT_MV)
    h.fcurrent()
    h.frecord_init()
    h.continuerun(state["duration_ms"])


# ==========================================================
# 你的模型代码（保持不变）
# ==========================================================

class BilinearModifiedLoop(nn.Module):
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

        with torch.no_grad():
            for i in range(min(self.in1_features, self.in2_features)):
                self.weight[:, i, i] = 0

    def forward(self, x1, x2):
        with torch.no_grad():
            for i in range(min(self.in1_features, self.in2_features)):
                self.weight[:, i, i] = 0
        return F.bilinear(x1, x2, self.weight, self.bias)



class BilinearModifiedMaskDynamic(nn.Module):
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

        mask = torch.ones(out_features, in1_features, in2_features, device=device)
        diagonal_size = min(in1_features, in2_features)
        diagonal_index = torch.arange(diagonal_size, device=device)
        mask[:, diagonal_index, diagonal_index] = 0
        self.register_buffer("offdiag_mask", mask, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)

        with torch.no_grad():
            self.weight.mul_(self.offdiag_mask)

    def forward(self, x1, x2):
        effective_weight = self.weight * self.offdiag_mask
        return F.bilinear(x1, x2, effective_weight, self.bias)


class BilinearModifiedMaskCached(BilinearModifiedMaskDynamic):
    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None):
        super().__init__(in1_features, in2_features, out_features, bias=bias, device=device)
        self._cached_effective_weight = None
        self._cache_signature = None

    def clear_inference_cache(self):
        self._cached_effective_weight = None
        self._cache_signature = None

    def _cache_enabled_for_this_forward(self):
        return (not self.training) and (not torch.is_grad_enabled())

    def _current_cache_signature(self):
        return int(self.weight._version)

    def _get_effective_weight(self):
        if not self._cache_enabled_for_this_forward():
            return self.weight * self.offdiag_mask

        signature = self._current_cache_signature()
        if self._cache_signature != signature or self._cached_effective_weight is None:
            self._cached_effective_weight = (self.weight * self.offdiag_mask).detach()
            self._cache_signature = signature

        return self._cached_effective_weight

    def forward(self, x1, x2):
        effective_weight = self._get_effective_weight()
        return F.bilinear(x1, x2, effective_weight, self.bias)


BilinearModifiedMask = BilinearModifiedMaskCached
BilinearModified = BilinearModifiedMaskCached

class DBNN_F(nn.Module):
    """
    FFT 版本：输出与 DBNN 中的 conv1d 实现严格对齐：
    y = conv1d(x, flip(kernels), groups=N, padding=T-1)[:, :, :T]
      ≡ full convolution of x with kernels（未翻转）取前 T 个点
    """
    def __init__(
        self,
        num_dimensions,
        time_dur,
        device,
        enable_inference_cache=True,
        bilinear_cls=BilinearModified,
        enable_spike=False,
        v_th=-55.0,
        reset_amp=15.0,
        tau_reset=10.0,
        surr_beta=10.0,
        homeostatic_ref_synapses=None,
        linear_alpha=0.0,
        quadratic_alpha=0.0,
    ):
        super(DBNN_F, self).__init__()
        self.num_dimensions = num_dimensions
        self.time_dur = time_dur
        self.device = device
        self.enable_inference_cache = enable_inference_cache
        self.enable_spike = bool(enable_spike)
        if homeostatic_ref_synapses is None:
            homeostatic_ref_synapses = num_dimensions
        ref = float(homeostatic_ref_synapses)
        nd = float(num_dimensions)
        self.linear_alpha = float(linear_alpha)
        self.quadratic_alpha = float(quadratic_alpha)
        self.register_buffer(
            "_linear_scale",
            torch.tensor((ref / nd) ** self.linear_alpha, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_quadratic_scale",
            torch.tensor((ref / nd) ** self.quadratic_alpha, device=device),
            persistent=False,
        )

        # parameters（和 DBNN 一致）
        self.tau_rise = nn.Parameter(torch.ones(num_dimensions, device=device) * 5)
        self.tau_decay = nn.Parameter(torch.ones(num_dimensions, device=device) * 20)
        self.omega    = nn.Parameter(torch.ones(num_dimensions, device=device) * 0.1)

        self.bilinear = bilinear_cls(num_dimensions, num_dimensions, 1, device=device).to(device)
        self.register_buffer(
            "_time_index",
            torch.arange(self.time_dur, device=device),
            persistent=False,
        )
        self._cached_kernel = None
        self._cached_k_fft = None
        self._cache_signature = None
        self.v_th = nn.Parameter(torch.tensor(float(v_th), device=device))
        self.reset_amp = nn.Parameter(torch.tensor(float(reset_amp), device=device))
        self.tau_reset = nn.Parameter(torch.tensor(float(tau_reset), device=device))
        self.surr_beta = float(surr_beta)

    def clear_inference_cache(self):
        self._cached_kernel = None
        self._cached_k_fft = None
        self._cache_signature = None
        if hasattr(self.bilinear, "clear_inference_cache"):
            self.bilinear.clear_inference_cache()

    def _cache_enabled_for_this_forward(self):
        return self.enable_inference_cache and (not self.training) and (not torch.is_grad_enabled())

    def _current_cache_signature(self, fft_size):
        return (
            fft_size,
            int(self.tau_rise._version),
            int(self.tau_decay._version),
            int(self.omega._version),
        )

    def create_kernels(self):
        # 和 DBNN 完全相同
        T = self._time_index
        tau_rise  = self.tau_rise.unsqueeze(1)   # (N,1)
        tau_decay = self.tau_decay.unsqueeze(1)  # (N,1)
        omega     = self.omega.unsqueeze(1)      # (N,1)

        kernels = omega * (1 - torch.exp(-T / tau_rise)) * torch.exp(-T / tau_decay)
        return kernels.unsqueeze(1)  # (N,1,T)

    def _get_kernels_and_frequency_response(self, fft_size):
        if not self._cache_enabled_for_this_forward():
            kernels = self.create_kernels()
            k = kernels.squeeze(1)
            k_fft = torch.fft.rfft(k, n=fft_size, dim=1)
            return kernels, k_fft

        signature = self._current_cache_signature(fft_size)
        if self._cache_signature != signature or self._cached_k_fft is None:
            kernels = self.create_kernels().detach()
            k = kernels.squeeze(1)
            self._cached_kernel = kernels
            self._cached_k_fft = torch.fft.rfft(k, n=fft_size, dim=1).detach()
            self._cache_signature = signature

        return self._cached_kernel, self._cached_k_fft

    def forward(self, x):
        """
        x: (B, N, T)
        目标：用 FFT 实现
        y = conv1d(x, flip(kernels), groups=N, padding=T-1)[:, :, :T]
          ≡ 对每个通道做 full convolution: x_i * k_i（未翻转），取前 T 个点
        """
        B, N, T = x.shape
        assert N == self.num_dimensions
        assert T == self.time_dur

        # ---- FFT 实现 full convolution ----
        # full conv 长度 = T + T - 1
        L_full = 2 * T - 1
        # 选择一个 >= L_full 的 FFT 长度（2 的幂，有利于加速）
        fft_size = 1 << (L_full - 1).bit_length()

        # ---- 生成 kernel（注意：这里不再 flip） ----
        _, K_fft = self._get_kernels_and_frequency_response(fft_size)  # (N,1,T), (N,F)

        # 对时间维做 FFT
        # X_fft: (B, N, F)
        X_fft = torch.fft.rfft(x, n=fft_size, dim=2)

        # 广播相乘： (B,N,F) * (1,N,F) -> (B,N,F)
        Y_fft = X_fft * K_fft.unsqueeze(0)

        # 反变换得到 full convolution，长度至少 L_full
        y_full = torch.fft.irfft(Y_fft, n=fft_size, dim=2)[..., :L_full]  # (B,N,L_full)

        # DBNN 里：conv1d(..., padding=T-1) 的结果就是这个 full conv，
        # 然后取前 T 个时间点
        y = y_full[..., :T]  # (B,N,T)

        # ---- 后续和 DBNN 完全一致 ----
        y_perm = y.permute(0, 2, 1)  # (B,T,N)
        bilinear_term = self.bilinear(y_perm, y_perm)         # (B,T,1)
        linear_term   = torch.sum(y_perm, dim=2).unsqueeze(-1)  # (B,T,1)
        bias = self.bilinear.bias.view(1, 1, 1) if self.bilinear.bias is not None else 0.0
        quadratic_term = bilinear_term - bias
        linear_scale = self._linear_scale.to(dtype=linear_term.dtype, device=linear_term.device)
        quadratic_scale = self._quadratic_scale.to(dtype=quadratic_term.dtype, device=quadratic_term.device)
        v_drive = (quadratic_term * quadratic_scale + linear_term * linear_scale + bias).squeeze(-1)  # (B,T)
        if not self.enable_spike:
            return v_drive
        return _apply_reset_closed_loop(
            v_drive,
            self.v_th,
            self.reset_amp,
            self.tau_reset,
            self.surr_beta,
        )


class DBNN_R(nn.Module):
    """
    递推版本：将每个维度的 kernel
    k[n] = omega * ((exp(-1/tau_decay))^n - (exp(-(1/tau_rise + 1/tau_decay)))^n)
    写成两个一阶状态的离散递推。
    """
    def __init__(
        self,
        num_dimensions,
        time_dur,
        device,
        enable_inference_cache=True,
        bilinear_cls=BilinearModified,
        scan_chunk_size=256,
        enable_spike=False,
        v_th=-55.0,
        reset_amp=15.0,
        tau_reset=10.0,
        surr_beta=10.0,
        homeostatic_ref_synapses=None,
        linear_alpha=0.0,
        quadratic_alpha=0.0,
    ):
        super(DBNN_R, self).__init__()
        self.num_dimensions = num_dimensions
        self.time_dur = time_dur
        self.device = device
        self.enable_inference_cache = enable_inference_cache
        self.scan_chunk_size = int(scan_chunk_size)
        self.enable_spike = bool(enable_spike)
        if homeostatic_ref_synapses is None:
            homeostatic_ref_synapses = num_dimensions
        ref = float(homeostatic_ref_synapses)
        nd_f = float(num_dimensions)
        self.linear_alpha = float(linear_alpha)
        self.quadratic_alpha = float(quadratic_alpha)
        self.register_buffer(
            "_linear_scale",
            torch.tensor((ref / nd_f) ** self.linear_alpha, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_quadratic_scale",
            torch.tensor((ref / nd_f) ** self.quadratic_alpha, device=device),
            persistent=False,
        )

        self.tau_rise = nn.Parameter(torch.ones(num_dimensions, device=device) * 5)
        self.tau_decay = nn.Parameter(torch.ones(num_dimensions, device=device) * 20)
        self.omega = nn.Parameter(torch.ones(num_dimensions, device=device) * 0.1)

        self.bilinear = bilinear_cls(num_dimensions, num_dimensions, 1, device=device).to(device)
        self._cached_coefficients = None
        self._cache_signature = None
        self._cached_scan_terms = {}
        self._cached_quadratic_weight = {}
        self.v_th = nn.Parameter(torch.tensor(float(v_th), device=device))
        self.reset_amp = nn.Parameter(torch.tensor(float(reset_amp), device=device))
        self.tau_reset = nn.Parameter(torch.tensor(float(tau_reset), device=device))
        self.surr_beta = float(surr_beta)

    def clear_inference_cache(self):
        self._cached_coefficients = None
        self._cache_signature = None
        self._cached_scan_terms = {}
        self._cached_quadratic_weight = {}
        if hasattr(self.bilinear, "clear_inference_cache"):
            self.bilinear.clear_inference_cache()

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
        if _is_torch_compiling() or not self._cache_enabled_for_this_forward():
            return self._compute_coefficients()

        signature = self._current_cache_signature()
        if self._cache_signature != signature or self._cached_coefficients is None:
            decay_a, decay_c, omega = self._compute_coefficients()
            self._cached_coefficients = (
                decay_a.detach(),
                decay_c.detach(),
                omega.detach(),
            )
            self._cache_signature = signature

        return self._cached_coefficients

    def _scan_dtype(self, dtype):
        if dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.float64
        return dtype

    def _get_scan_terms(self, decay, steps, dtype, device, cache_key):
        scan_dtype = self._scan_dtype(dtype)
        if _is_torch_compiling() or not self._cache_enabled_for_this_forward():
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

    def _compute_quadratic_weight(self, dtype, device):
        if _is_torch_compiling():
            if hasattr(self.bilinear, "offdiag_mask"):
                weight = self.bilinear.weight * self.bilinear.offdiag_mask
            else:
                weight = self.bilinear.weight
        elif hasattr(self.bilinear, "_get_effective_weight"):
            weight = self.bilinear._get_effective_weight()
        else:
            weight = self.bilinear.weight
        return weight.squeeze(0).to(dtype=dtype, device=device).contiguous()

    def _get_quadratic_weight(self, dtype, device):
        if _is_torch_compiling() or not self._cache_enabled_for_this_forward():
            return self._compute_quadratic_weight(dtype, device)

        signature = (
            self._current_cache_signature(),
            int(self.bilinear.weight._version),
            str(device),
            str(dtype),
        )
        cached = self._cached_quadratic_weight.get(signature)
        if cached is None:
            cached = self._compute_quadratic_weight(dtype, device).detach()
            self._cached_quadratic_weight[signature] = cached
        return cached

    def _state_scan_chunk(self, x_chunk, powers, inv_powers, state_prev):
        scan_dtype = powers.dtype
        x_scaled = x_chunk.to(dtype=scan_dtype) * inv_powers.unsqueeze(0)
        csum = x_scaled.cumsum(dim=2)
        state = powers.unsqueeze(0) * (state_prev.to(dtype=scan_dtype).unsqueeze(-1) + csum)
        return state.to(dtype=x_chunk.dtype)

    def forward(self, x):
        bsz, nd, total_steps = x.shape
        assert nd == self.num_dimensions
        assert total_steps == self.time_dur

        decay_a, decay_c, omega = self._get_coefficients()
        quadratic_weight = self._get_quadratic_weight(x.dtype, x.device)
        bias = self.bilinear.bias

        state_a = torch.zeros(bsz, nd, device=x.device, dtype=x.dtype)
        state_c = torch.zeros(bsz, nd, device=x.device, dtype=x.dtype)
        output = torch.empty(bsz, total_steps, device=x.device, dtype=x.dtype)

        omega_b = omega.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        chunk = max(1, self.scan_chunk_size)
        linear_scale = self._linear_scale.to(dtype=x.dtype, device=x.device)
        quadratic_scale = self._quadratic_scale.to(dtype=x.dtype, device=x.device)

        for start in range(0, total_steps, chunk):
            end = min(start + chunk, total_steps)
            steps = end - start
            x_chunk = x[:, :, start:end]  # (B, N, L)
            powers_a, inv_powers_a = self._get_scan_terms(decay_a, steps, x.dtype, x.device, "a")
            powers_c, inv_powers_c = self._get_scan_terms(decay_c, steps, x.dtype, x.device, "c")
            state_a_chunk = self._state_scan_chunk(x_chunk, powers_a, inv_powers_a, state_a)
            state_c_chunk = self._state_scan_chunk(x_chunk, powers_c, inv_powers_c, state_c)
            z_chunk = state_a_chunk - state_c_chunk
            y_chunk = omega_b * z_chunk

            if (not torch.is_grad_enabled()) and bsz == 1:
                y_ln = y_chunk[0].transpose(0, 1).contiguous()  # (L, N)
                linear = y_ln.sum(dim=1, keepdim=True).transpose(0, 1)
                quadratic = torch.sum(torch.mm(y_ln, quadratic_weight) * y_ln, dim=1, keepdim=True).transpose(0, 1)
            else:
                y_bt_n = y_chunk.permute(0, 2, 1).contiguous()  # (B, L, N)
                linear = y_bt_n.sum(dim=2)
                quadratic = ((y_bt_n @ quadratic_weight) * y_bt_n).sum(dim=2)
            out_chunk = linear * linear_scale + quadratic * quadratic_scale
            if bias is not None:
                out_chunk = out_chunk + bias
            output[:, start:end] = out_chunk

            state_a = state_a_chunk[:, :, -1]
            state_c = state_c_chunk[:, :, -1]

        if not self.enable_spike:
            return output
        return _apply_reset_closed_loop(
            output,
            self.v_th,
            self.reset_amp,
            self.tau_reset,
            self.surr_beta,
        )

class DBNN(nn.Module):
    def __init__(self, num_dimensions, time_dur, device, bilinear_cls=BilinearModified):
        super(DBNN, self).__init__()
        self.num_dimensions = num_dimensions
        self.time_dur = time_dur
        self.device = device
        # intinial values
        self.tau_rise = nn.Parameter(torch.ones(num_dimensions).to(self.device) * 5)
        self.tau_decay = nn.Parameter(torch.ones(num_dimensions).to(self.device) * 20)
        self.omega = nn.Parameter(torch.ones(num_dimensions).to(self.device) * 0.1)

        self.bilinear = bilinear_cls(num_dimensions, num_dimensions, 1, device=self.device).to(self.device)

    def create_kernels(self):
        T = torch.arange(self.time_dur).to(self.device)
        N = self.num_dimensions
        net_tau_rise = self.tau_rise.unsqueeze(1)  # (N, 1)
        net_tau_decay = self.tau_decay.unsqueeze(1)  # (N, 1)
        net_omega = self.omega.unsqueeze(1)  # (N, 1)

        kernels = net_omega * (1 - torch.exp(-T / net_tau_rise)) * torch.exp(-T / net_tau_decay)
        return kernels.unsqueeze(1)
        
    def forward(self, x):

        kernels = self.create_kernels()
        kernel_flipped = torch.flip(kernels, dims=[2])

        # Convolve using the kernel (perform manual convolution)
        y = torch.nn.functional.conv1d(x, kernel_flipped, groups=self.num_dimensions, padding=self.time_dur - 1)[:, :, :self.time_dur][:,:,:self.time_dur]
        y_permuted = y.permute(0, 2, 1)
        bilinear_term = self.bilinear(y_permuted, y_permuted)
        linear_term = torch.sum(y_permuted, dim=2).unsqueeze(-1)
        output = bilinear_term + linear_term
        return output.squeeze(-1)


# ==========================================================
# 输入生成（Poisson）
# ==========================================================
def generate_poisson_input(nd, td, freq, device, batch_size=1):
    dt = 1.0
    p = (freq / 1000.0) * dt
    return (torch.rand(batch_size, nd, td, device=device) < p).float()


# ==========================================================
# warm-up 逻辑（CPU / CUDA 都执行）
# ==========================================================
def warmup(model, nd, td, freq, device):
    print(f"      → Warm-up on {device} ...")
    x = generate_poisson_input(nd, td, freq, device=device)
    _ = model(x)
    if "cuda" in str(device):
        torch.cuda.synchronize()


# ==========================================================
# 单设备运行：cpu / cuda
# ==========================================================
def run_one_setting_on_device(nd, td, freq, device_name, n_repeat=3):
    device = torch.device(device_name)
    # model = DBNN(nd, td, device).to(device)
    model = DBNN_F(nd, td, device).to(device)
    model.eval()

    # ---------- warm-up ----------
    warmup(model, nd, td, freq, device)

    run_times = []

    for i in range(n_repeat):
        x = generate_poisson_input(nd, td, freq, device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.time()
        with torch.inference_mode():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()

        elapsed = t1 - t0
        run_times.append(elapsed)

        print(f"      → [{device_name}] Run {i+1}/{n_repeat}: {elapsed:.6f} s")

    avg = float(np.mean(run_times))
    print(f"      → [{device_name}] Average runtime = {avg:.6f} s\n")

    return avg, run_times


# ==========================================================
# sweep 所有参数组合
# ==========================================================
def sweep_parameters(nd_list, td_list, freq_list, outfile="results/json/dbnn_runtime_cpu_cuda.json"):
    results = {}
    total = len(nd_list) * len(td_list) * len(freq_list)
    pbar = tqdm(total=total, desc="DBNN Parameter Sweep")

    for nd in nd_list:
        for td in td_list:
            for fq in freq_list:

                key = f"dim_{nd}_dur_{td}_freq_{fq}"

                print(f"\n>>> Running setting: dim={nd}, dur={td}, freq={fq}")

                cpu_avg, cpu_runs = run_one_setting_on_device(nd, td, fq, "cpu")
                cuda_avg, cuda_runs = run_one_setting_on_device(nd, td, fq, "cuda:7")

                results[key] = {
                    "cpu": {"avg_runtime_sec": cpu_avg, "runs": cpu_runs},
                    "cuda": {"avg_runtime_sec": cuda_avg, "runs": cuda_runs}
                }

                pbar.update(1)

    pbar.close()

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n全部结果已保存到：{outfile}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DBNN variants and NEURON runtime.")
    parser.add_argument("--mode", choices=["sweep", "duration_benchmark"], default="sweep")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "dbnn_checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "simulate_time_compare.json",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=FIGURES_DIR / "runtime_comparison_neuron_dbnn_dbnnr.png",
    )
    parser.add_argument("--gpu-device", default="cuda:7")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--num-dimensions", type=int, default=500)
    parser.add_argument("--input-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--durations-sec",
        type=float,
        nargs="+",
        default=[0.5, 1, 2, 4, 8, 16, 32, 64],
    )
    return parser.parse_args()


def encode_inputs_signs(channel_type):
    return np.where(np.asarray(channel_type) == "exc", 1.0, -1.0).astype(np.float32)


def benchmark_neuron_duration(duration_ms, repeats, warmup_runs, seed):
    state = build_neuron_runtime_state(duration_ms, base_seed=seed + int(duration_ms))
    for _ in range(warmup_runs):
        run_neuron_once(state)

    runtimes = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_neuron_once(state)
        t1 = time.perf_counter()
        runtimes.append(t1 - t0)

    return {
        "avg_runtime_sec": float(np.mean(runtimes)),
        "std_runtime_sec": float(np.std(runtimes)),
        "runs": runtimes,
    }


def instantiate_from_checkpoint(model_cls, checkpoint_path, time_dur, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    channel_type = checkpoint["channel_type"]
    num_dimensions = len(channel_type)
    model = model_cls(
        num_dimensions=num_dimensions,
        time_dur=time_dur,
        device=device,
        enable_spike=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, channel_type


def make_poisson_input(num_dimensions, time_dur, freq_hz, device, seed, signed=False, channel_type=None):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    dt_ms = 1.0
    p = (freq_hz / 1000.0) * dt_ms
    x = (torch.rand((1, num_dimensions, time_dur), generator=gen, device=device) < p).float()
    if signed and channel_type is not None:
        signs = torch.from_numpy(encode_inputs_signs(channel_type)).to(device=device, dtype=x.dtype)
        x = x * signs.view(1, num_dimensions, 1)
    return x


def benchmark_model_runtime(model, x, device, repeats, warmup_runs):
    for _ in range(warmup_runs):
        with torch.inference_mode():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    runtimes = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        runtimes.append(t1 - t0)

    return {
        "avg_runtime_sec": float(np.mean(runtimes)),
        "std_runtime_sec": float(np.std(runtimes)),
        "runs": runtimes,
    }


def save_duration_benchmark_plot(results, output_path):
    duration_keys = sorted(results["results"].keys(), key=float)
    durations = np.array([float(k) for k in duration_keys], dtype=np.float64)
    neuron = np.array([results["results"][k]["NEURON"]["avg_runtime_sec"] for k in duration_keys], dtype=np.float64)
    dbnn_f = np.array([results["results"][k]["DBNN_F_gpu"]["avg_runtime_sec"] for k in duration_keys], dtype=np.float64)
    dbnn_r = np.array([results["results"][k]["DBNN_R_gpu"]["avg_runtime_sec"] for k in duration_keys], dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(durations, neuron, marker="o", linewidth=2.0, label="NEURON")
    plt.plot(durations, dbnn_f, marker="s", linewidth=2.0, label="DBNN_F (GPU)")
    plt.plot(durations, dbnn_r, marker="^", linewidth=2.0, label="DBNN_R (GPU)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Simulation Duration (s)")
    plt.ylabel("Average Runtime (s)")
    plt.title("Runtime Comparison: NEURON vs DBNN_F vs DBNN_R")
    plt.xticks(durations, [str(int(d)) if float(d).is_integer() else str(d) for d in durations])
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_duration_benchmark(args):
    results = {
        "checkpoint": str(args.checkpoint),
        "gpu_device": args.gpu_device,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "durations_sec": args.durations_sec,
        "results": {},
    }

    gpu_device = torch.device(args.gpu_device)
    for duration_sec in args.durations_sec:
        duration_ms = int(round(float(duration_sec) * 1000.0))
        time_dur = duration_ms + 1
        print(f"\n>>> Benchmarking duration={duration_sec}s (T={time_dur})")

        model_f, channel_type = instantiate_from_checkpoint(DBNN_F, args.checkpoint, time_dur, gpu_device)
        model_r, _ = instantiate_from_checkpoint(DBNN_R, args.checkpoint, time_dur, gpu_device)
        x_gpu = make_poisson_input(
            len(channel_type),
            time_dur,
            args.input_rate_hz,
            gpu_device,
            seed=args.seed + duration_ms,
            signed=True,
            channel_type=channel_type,
        )

        neuron_stats = benchmark_neuron_duration(duration_ms, args.repeats, args.warmup_runs, args.seed)
        dbnn_f_stats = benchmark_model_runtime(model_f, x_gpu, gpu_device, args.repeats, args.warmup_runs)
        dbnn_r_stats = benchmark_model_runtime(model_r, x_gpu, gpu_device, args.repeats, args.warmup_runs)

        results["results"][str(duration_sec)] = {
            "duration_ms": duration_ms,
            "time_dur": time_dur,
            "NEURON": neuron_stats,
            "DBNN_F_gpu": dbnn_f_stats,
            "DBNN_R_gpu": dbnn_r_stats,
            "speedup": {
                "neuron_over_dbnn_f": neuron_stats["avg_runtime_sec"] / dbnn_f_stats["avg_runtime_sec"],
                "neuron_over_dbnn_r": neuron_stats["avg_runtime_sec"] / dbnn_r_stats["avg_runtime_sec"],
                "dbnn_r_over_dbnn_f": dbnn_f_stats["avg_runtime_sec"] / dbnn_r_stats["avg_runtime_sec"],
            },
        }

        print(
            f"    Completed duration {duration_sec}s | "
            f"NEURON={neuron_stats['avg_runtime_sec']:.6f}s "
            f"DBNN_F_gpu={dbnn_f_stats['avg_runtime_sec']:.6f}s "
            f"DBNN_R_gpu={dbnn_r_stats['avg_runtime_sec']:.6f}s"
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"    Partial results saved to {args.output}")

    save_duration_benchmark_plot(results, args.plot_output)
    print(f"\nSaved duration benchmark to {args.output}")
    print(f"Saved duration benchmark plot to {args.plot_output}")
    return results


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "sweep":
        num_dimensions_list = [10, 100, 1000]
        time_dur_list = [5000]
        freq_list = [10]
        sweep_parameters(num_dimensions_list, time_dur_list, freq_list)
    else:
        run_duration_benchmark(args)
