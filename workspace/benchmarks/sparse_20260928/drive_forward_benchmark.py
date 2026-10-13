"""Swap only the complete drive formula inside one fixed inference pipeline.

The synaptic scan, event reset, input packing and precision are identical.
All drive variants return their result directly, so the original PyTorch
expressions do not incur an artificial copy to the native output buffer.
Frozen source/DBNN.py and deployed backends.py are left unchanged.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment import HERE, ROOT, DBNN_R, accuracy, case_data, resource_snapshot, stamp
from backends import Prepared, native, subset_state
from quadratic_benchmark import capture, digest, measure_device, measure_wall, save

CASES = [(a, 6001, b) for b in [1, 8] for a in [8, 32, 128, 500]]
METHODS = ['original_F', 'original_R', 'cublas', 'cutlass', 'direct']


class DrivePrepared(Prepared):
    def __init__(self, state, x, scan, drive):
        super().__init__(state, x, scan, drive)
        self.w3 = self.W.unsqueeze(0)
        self.bias_tensor = torch.tensor([self.bias], device=x.device, dtype=x.dtype)

    def drive_only(self, y):
        if self.quad == 'original_F':
            flat = y.reshape(-1, self.K)
            q = F.bilinear(flat, flat, self.w3, self.bias_tensor)
            return q.reshape(self.B, self.T) + y.sum(-1)
        if self.quad == 'original_R':
            flat = y.reshape(-1, self.K)
            q = ((flat @ self.W) * flat).sum(-1).reshape(self.B, self.T)
            return y.sum(-1) + q + self.bias
        return super().drive_only(y)

    @torch.inference_mode()
    def __call__(self):
        self.stream = torch.cuda.current_stream(self.x.device).cuda_stream
        drive = self.drive_only(self.synaptic())
        native().call('dbnn_reset_event', drive, self.out, self.alpha_pows,
                      self.B, self.T, self.alpha, self.amp, self.vth, self.stream)
        return self.out


def checked(out, ref, vth):
    result = accuracy(out, ref, vth)
    if (not result['finite'] or result['max_abs_mv'] > 1e-3
            or result['spike_mismatch_bins'] != 0):
        raise RuntimeError(f'Output/reset equivalence failed: {result}')
    return result


def run_case(index, args):
    A, T, B = CASES[index]
    sc, xc, ids, _ = case_data(500, A, T, B)
    sp = {k: v.cuda() for k, v in subset_state(sc, ids).items()}
    x = xc[:, ids].contiguous().cuda()
    models = {m: DrivePrepared(sp, x, args.scan, m) for m in METHODS}
    ref = models['original_R']().clone()
    vth = models['original_R'].vth
    # Check against the actual original model, including its native chunking.
    original = DBNN_R(500, T, torch.device('cuda'))
    original.load_state_dict({k: v.cuda() for k, v in sc.items()})
    original.eval()
    frozen_ref = original(xc.cuda()).clone()
    errors = {m: checked(fn().clone(), ref, vth) for m, fn in models.items()}
    frozen_errors = {m: checked(fn().clone(), frozen_ref, vth)
                     for m, fn in models.items()}
    del original, frozen_ref
    graphs = {m: capture(fn, 1) for m, fn in models.items()}
    unroll = 8
    repeated = {m: capture(fn, unroll) for m, fn in models.items()}
    saved = x.clone()
    x.zero_()
    zero_ref = models['original_R']().clone()
    zero_checks = {}
    for mode, group in [('single', graphs), ('unrolled', repeated)]:
        zero_checks[mode] = {m: checked(fn().clone(), zero_ref, vth)
                             for m, fn in group.items()}
    x.copy_(saved)
    graph_errors = {m: checked(fn().clone(), ref, vth) for m, fn in graphs.items()}
    unrolled_errors = {m: checked(fn().clone(), ref, vth) for m, fn in repeated.items()}
    fns = {f'{m}/eager': fn for m, fn in models.items()}
    fns.update({f'{m}/graph': fn for m, fn in graphs.items()})
    for fn in fns.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    before = resource_snapshot()
    wall = measure_wall(fns, args.repeats, args.seed + index * 37)
    device = measure_device(repeated, unroll, args.device_repeats,
                            args.seed + index * 37 + 3000)
    after = resource_snapshot()
    y = models['original_R'].synaptic().reshape(B*T, A).contiguous()
    rec = {
        'index': index, 'N': 500, 'A': A, 'T': T, 'B': B,
        'scan': args.scan, 'reset': 'native CUDA event reset',
        'X_sha256': digest(x), 'Y_sha256': digest(y),
        'W_sha256': digest(models['original_R'].W),
        'bias_mv': models['original_R'].bias, 'device_graph_unroll': unroll,
        'wall': wall, 'device_per_call': device,
        'errors_vs_same_pipeline_original_R': errors,
        'errors_vs_frozen_full_original_R': frozen_errors,
        'graph_errors': graph_errors, 'unrolled_errors': unrolled_errors,
        'changed_input_graph_passed': True, 'zero_input_errors': zero_checks,
        'resources_before': before, 'resources_after': after,
    }
    print('  forward device us: ' + ', '.join(
        f'{m}={v["median_ms"]*1000:.2f}' for m, v in device.items()), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=HERE/'drive_forward.json')
    ap.add_argument('--repeats', type=int, default=80)
    ap.add_argument('--device-repeats', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20928)
    ap.add_argument('--scan', choices=['cuda', 'triton'], default='cuda')
    ap.add_argument('--indices', nargs='*', type=int, default=list(range(8)))
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library('cublas')
    result = {
        'started': stamp(), 'scope': 'core forward; same scan and event reset, only drive changes',
        'formula': 'drive=sum(Y*(Y@W),-1)+sum(Y,-1)+bias',
        'baseline_scope': 'Original F/R drive expressions on packed full-window Y; same optimized scan/reset. Original R temporal chunks excluded from timed baseline.',
        'exclusions': 'packing, coefficient setup, compilation, graph capture, host transfer',
        'wall_protocol': 'Randomized eager/single-Graph interleaved calls; synchronize, perf_counter, call, synchronize.',
        'device_protocol': 'CUDA events around one Graph with 8 complete forwards; divided by 8. Same fixed input, hot-cache throughput.',
        'repeats': args.repeats, 'device_repeats': args.device_repeats, 'seed': args.seed,
        'scan': args.scan, 'methods': METHODS, 'precision': 'FP32, TF32 disabled',
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'physical_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'baseline_sha256': hashlib.sha256((ROOT/'source/DBNN.py').read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256((ROOT/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
        'resources_before': resource_snapshot(), 'cases': [],
    }
    save(args.output, result)
    with torch.no_grad():
        for index in args.indices:
            print('CASE', index, CASES[index], flush=True)
            result['cases'].append(run_case(index, args))
            save(args.output, result)
            gc.collect()
            torch.cuda.empty_cache()
    result['finished'] = stamp()
    result['resources_after'] = resource_snapshot()
    save(args.output, result)
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
