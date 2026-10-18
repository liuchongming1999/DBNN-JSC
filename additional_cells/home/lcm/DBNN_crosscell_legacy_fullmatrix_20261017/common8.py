"""Immutable manifests, primary metric conventions, and exact causal reset."""
from pathlib import Path
import ctypes
import fcntl
import hashlib
import json
import subprocess
from functools import lru_cache
import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def match_events(ref, pred, tol=5.):
    ref, pred = sorted(map(float, ref)), sorted(map(float, pred))
    i = j = 0
    errors = []
    while i < len(ref) and j < len(pred):
        if pred[j] < ref[i]-tol-1e-6:
            j += 1
        elif ref[i] < pred[j]-tol-1e-6:
            i += 1
        else:
            errors.append(pred[j]-ref[i])
            i += 1
            j += 1
    return dict(tp=len(errors), fp=len(pred)-len(errors), fn=len(ref)-len(errors),
                matched_errors_ms=errors)


def event_aggregate(rows):
    tp, fp, fn = [int(sum(r[k] for r in rows)) for k in ('tp', 'fp', 'fn')]
    errors = [e for r in rows for e in r['matched_errors_ms']]
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                median_abs_error_ms=float(np.median(np.abs(errors))) if errors else None)


@lru_cache(maxsize=2048)
def _mask(n, spikes, dt):
    keep=np.ones(n,bool)
    times=np.arange(n)*dt
    for event in spikes:
        keep &= np.abs(times-event)>10.
    keep.flags.writeable=False
    return keep


def mask(n,spikes,dt=1.):
    return _mask(n,tuple(map(float,spikes)),float(dt))


def metric(ref, pred, spikes, dt=1.):
    keep = mask(len(ref), spikes, dt)
    r, e = np.asarray(ref, float)[keep], (np.asarray(pred, float)-ref)[keep]
    assert np.isfinite(ref).all() and np.isfinite(pred).all()
    if len(e)==0:
        return dict(n=0,sse=0.,reference_ss=0.,error_ss=0.,voltage_metric_defined=False)
    return dict(n=len(e), sse=float(e@e), reference_ss=float(np.sum((r-r.mean())**2)),
                error_ss=float(np.sum((e-e.mean())**2)))


def aggregate(rows):
    n, sse = sum(r['n'] for r in rows), sum(r['sse'] for r in rows)
    rss, ess = sum(r['reference_ss'] for r in rows), sum(r['error_ss'] for r in rows)
    return dict(n=n, sse=sse, pooled_rmse_mv=float(np.sqrt(sse/n)) if n else None,
                reference_ss=rss, error_ss=ess,
                within_trace_weighted_ve=1-ess/rss if rss else None)


def reset_function(folder):
    folder = Path(folder)
    src, lib = folder/'seq_reset.c', folder/'seq_reset.so'
    with (folder/'reset_build.lock').open('w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(['cc', '-O3', '-shared', '-fPIC', str(src), '-lm', '-o', str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    fn = dll.dbnn_reset
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_double, ctypes.c_double, ctypes.c_double]
    fn.restype = ctypes.c_size_t

    def reset(drive, th, amp, tau):
        d = np.ascontiguousarray(drive, dtype=np.float64)
        v, ev = np.empty_like(d), np.empty_like(d)
        n = fn(d.ctypes.data, v.ctypes.data, ev.ctypes.data, len(d), th, amp, tau)
        return v, ev[:n]
    return reset
