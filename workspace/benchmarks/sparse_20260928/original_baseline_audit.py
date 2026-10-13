"""Paired, source-faithful baselines: change one stage of the frozen forward.

No source/DBNN.py edits. Generated variants are saved alongside measurements so
that the changed statements, including the retained R chunk loop, are auditable.
All methods use the same already packed input. Packing is measured separately
in confirmation.json, and is not attributed again to a kernel.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
import textwrap
import types
from pathlib import Path

import torch
from experiment import (HERE, ROOT, case_data, accuracy, measure_group,
                        resource_snapshot, stamp, DBNN_F, DBNN_R)
import DBNN as original
from backends import Prepared, capture, native, subset_state


def replace_once(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after, 1)


def variant_source(cls, stage):
    source = textwrap.dedent(inspect.getsource(cls.forward))
    if stage == 'filter':
        if cls is DBNN_F:
            begin = source.index('    kernels = self.create_kernels()')
            end = source.index('    bilinear_term =')
            source = source[:begin] + '    y_perm = self._audit_filter()\n' + source[end:]
        else:
            source = replace_once(source,
                '    decay_a, decay_c, omega = self._get_coefficients()\n',
                '    audit_y = self._audit_filter()\n')
            for statement in [
                '    state_a = torch.zeros(bsz, nd, device=x.device, dtype=x.dtype)\n',
                '    state_c = torch.zeros(bsz, nd, device=x.device, dtype=x.dtype)\n',
                '    omega_b = omega.unsqueeze(0).unsqueeze(-1)\n',
                '        state_a = state_a_chunk[:, :, -1]\n',
                '        state_c = state_c_chunk[:, :, -1]\n',
            ]:
                source = replace_once(source, statement, '')
            begin = source.index('        x_chunk =')
            end = source.index('        if (not torch.is_grad_enabled())')
            source = source[:begin] + '        y_chunk = audit_y[:, start:end, :].permute(0, 2, 1)\n\n' + source[end:]
    elif stage == 'drive':
        if cls is DBNN_F:
            begin = source.index('    bilinear_term =')
            end = source.index('    return _apply_reset_closed_loop')
            source = source[:begin] + '    v_drive = self._audit_drive(y_perm)\n' + source[end:]
        else:
            # Retain the original 256-step loop and its filter-state updates.
            source = replace_once(source, '    quadratic_weight = self._get_quadratic_weight(x.dtype, x.device)\n', '')
            source = replace_once(source, '    bias = self.bilinear.bias\n', '')
            begin = source.index('        if (not torch.is_grad_enabled())')
            end = source.index('        output[:, start:end] = out_chunk')
            source = source[:begin] + '        out_chunk = self._audit_drive(y_chunk.permute(0, 2, 1))\n' + source[end:]
    elif stage == 'reset':
        source = replace_once(source, 'return _apply_reset_closed_loop(', 'return self._audit_reset(')
    elif stage == 'get_drive':
        begin = source.index('    return _apply_reset_closed_loop(')
        output = 'v_drive' if cls is DBNN_F else 'output'
        source = source[:begin] + f'    return {output}\n'
    else:
        raise ValueError(stage)
    return source


def attach(model, source):
    env = vars(original).copy()
    exec(compile(source, '<audited-one-stage-variant>', 'exec'), env)
    model.forward = types.MethodType(env['forward'], model)


class Reset:
    def __init__(self, state, B, T, device):
        self.B, self.T = B, T
        self.out = torch.empty(B, T, device=device)
        self.alpha = float(torch.exp(-1 / state['tau_reset'].abs().clamp(min=1e-4)))
        self.powers = torch.tensor(self.alpha, device=device).pow(torch.arange(T, device=device, dtype=torch.float32))
        self.amp, self.vth = abs(float(state['reset_amp'])), float(state['v_th'])
        self.stream = torch.cuda.current_stream().cuda_stream

    def __call__(self, drive, *ignored_original_parameters):
        native().call('dbnn_reset_event', drive.contiguous(), self.out, self.powers,
                      self.B, self.T, self.alpha, self.amp, self.vth, self.stream)
        return self.out


class Drive:
    def __init__(self, state, x, chunk, quad):
        self.prepared = {}
        B, A, T = x.shape
        for length in sorted({min(chunk, T), T % chunk} - {0}):
            dummy = torch.empty(B, A, length, device=x.device)
            self.prepared[length] = Prepared(state, dummy, 'cuda', quad)

    def __call__(self, y):
        # Layout conversion occurs in the timed call, as in the original R.
        return self.prepared[y.shape[1]].drive_only(y.contiguous())


def check_original_cache():
    """Read-only behavioral audit of cache reuse; no timings or source patches."""
    m = DBNN_R(4, 10, torch.device('cpu')).eval()
    rows = []
    with torch.no_grad():
        for i in range(3):
            da, _, _ = m._get_coefficients()
            powers, _ = m._get_scan_terms(da, 10, torch.float32, torch.device('cpu'), 'a')
            before = int(m.bilinear.weight._version)
            signature = (before, 'cpu', 'torch.float32')
            hit = signature in m._cached_quadratic_weight
            m._get_quadratic_weight(torch.float32, torch.device('cpu'))
            rows.append(dict(call=i+1, weight_version_before=before,
                weight_version_after=int(m.bilinear.weight._version), weight_cache_hit=hit,
                weight_cache_entries=len(m._cached_quadratic_weight),
                coefficient_pointer=da.data_ptr(), powers_pointer=powers.data_ptr()))
    return dict(device='cpu', scope='cache logic diagnostic, not a speed benchmark', calls=rows,
                coefficients_reused=len({r['coefficient_pointer'] for r in rows})==1,
                powers_reused=len({r['powers_pointer'] for r in rows})==1,
                explanation='W cache key is computed before _effective_weight zeroes the diagonal and increments weight._version; the next lookup misses.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--output', type=Path, default=HERE/'original_baseline_audit.json')
    ap.add_argument('--limit', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    native()
    sources = {f'{name}_{stage}': variant_source(cls, stage)
               for name, cls in [('F', DBNN_F), ('R', DBNN_R)]
               for stage in ['filter', 'drive', 'reset', 'get_drive']}
    source_file = HERE/'original_baseline_variants.txt'
    source_file.write_text('\n\n'.join(f'# {k}\n{v}' for k, v in sources.items()))
    report = dict(created=stamp(), baseline_sha256=hashlib.sha256((ROOT/'source/DBNN.py').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((ROOT/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
        generated_variants_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        resources_before=resource_snapshot(), exclusive_resources=False,
        scope='synchronized wall time; unmodified original forward vs one-stage replacement; packed input; no Graph for one-stage variants',
        notes=['Original R inference cache stays enabled and warm.',
               'Drive replacement in R retains its original 256-step chunk loop.',
               'Triton filter computes whole-window states, then R retains its 256-step drive loop.',
               'Source F/R at B>1 also measured via a Python loop over B=1 calls.',
               'Optimized eager vs Graph is an internal scheduling ablation, not an original-code baseline.',
               'All drive comparisons include the quadratic, linear and bias terms.'],
        cases=[])
    for ci, (A, B) in enumerate([(32, 1), (500, 1), (32, 8), (500, 8)][:args.limit]):
        T = 6001
        print(f'CASE {ci}: A={A}, B={B}, T={T}', flush=True)
        sc, xc, ids, _ = case_data(500, A, T, B)
        state = {k:v.cuda() for k, v in subset_state(sc, ids).items()}
        x = xc[:, ids].contiguous().cuda()
        record = dict(index=ci, N=500, A=A, B=B, T=T,
            input_sha256=hashlib.sha256(xc[:,ids].contiguous().numpy().tobytes()).hexdigest(),
            active_ids=ids.tolist(), resources_before=resource_snapshot())
        with torch.no_grad():
            methods, owners = {}, []
            for name, cls in [('F', DBNN_F), ('R', DBNN_R)]:
                def model():
                    m = cls(A, T, torch.device('cuda'))
                    m.load_state_dict(state); m.eval(); owners.append(m)
                    return m
                m = model()
                methods[f'original_{name}'] = lambda m=m:m(x)
                if B > 1:
                    methods[f'original_{name}_trace_loop'] = lambda m=m:torch.cat([m(x[b:b+1]) for b in range(B)])
                for stage, quad in [('filter', None), ('drive', 'cublas'), ('drive', 'direct'), ('reset', None)]:
                    m = model(); attach(m, sources[f'{name}_{stage}'])
                    if stage == 'filter':
                        p = Prepared(state, x, 'triton', 'cublas')
                        m._audit_filter = p.synaptic
                    elif stage == 'drive':
                        m._audit_drive = Drive(state, x, 256 if name == 'R' else T, quad)
                    else:
                        m._audit_reset = Reset(state, B, T, x.device)
                    key = f'{name}_only_{stage}' + (f'_{quad}' if quad else '')
                    methods[key] = lambda m=m:m(x)
            # These are explicitly separate internal and final-pipeline controls.
            p = Prepared(state, x, 'triton', 'cublas')
            methods['optimized_eager'] = p
            methods['optimized_graph'] = capture(p)
            reference = methods['original_R']().clone()
            errors = {k:accuracy(fn().clone(), reference, p.vth) for k, fn in methods.items()}
            for key, er in errors.items():
                assert er['finite'] and er['max_abs_mv'] < 1e-3 and er['spike_mismatch_bins'] == 0, (key, er)
            for fn in methods.values():
                for _ in range(2): fn()
            record['forward'] = measure_group(methods, args.repeats)
            record['errors_vs_original_R'] = errors
            # Reset module: call the original function itself on exactly the same u.
            m = DBNN_R(A, T, torch.device('cuda')); m.load_state_dict(state); m.eval()
            attach(m, sources['R_get_drive']); u = m(x).clone()
            reset = Reset(state, B, T, x.device)
            params = (m.v_th, m.reset_amp, m.tau_reset, m.surr_beta)
            reset_methods = {'original_reset':lambda:original._apply_reset_closed_loop(u, *params),
                             'event_cuda':lambda:reset(u)}
            if B > 1:
                reset_methods['original_reset_trace_loop'] = lambda:torch.cat([
                    original._apply_reset_closed_loop(u[b:b+1], *params) for b in range(B)])
            ref = reset_methods['original_reset']().clone()
            record['reset_events_per_trace'] = (ref >= p.vth).sum(1).tolist()
            record['drive_sha256'] = hashlib.sha256(u.cpu().numpy().tobytes()).hexdigest()
            record['reset_errors'] = {k:accuracy(fn().clone(), ref, p.vth) for k, fn in reset_methods.items()}
            for er in record['reset_errors'].values():
                assert er['finite'] and er['max_abs_mv'] < 1e-3 and er['spike_mismatch_bins'] == 0, er
            for fn in reset_methods.values():
                for _ in range(2): fn()
            record['reset_module'] = measure_group(reset_methods, args.repeats)
        record['resources_after'] = resource_snapshot()
        report['cases'].append(record)
        report['updated'] = stamp()
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({k:round(v['median_ms'],4) for k,v in record['forward'].items()}), flush=True)
        print('RESET', json.dumps({k:round(v['median_ms'],4) for k,v in record['reset_module'].items()}), flush=True)
        del methods, owners, state, x, p, m, reset, reset_methods
        gc.collect(); torch.cuda.empty_cache()
    report['cache_diagnostic'] = check_original_cache()
    report['finished'] = stamp()
    report['resources_after'] = resource_snapshot()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('COMPLETE', str(args.output), flush=True)


if __name__ == '__main__':
    main()
