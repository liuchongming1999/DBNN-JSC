"""Frozen-event scoring with an inclusive, numerically stable time boundary.

Times are in milliseconds. The 1e-6-ms margin is one nanosecond, far below
the smallest integration/output interval in these experiments.
"""
import numpy as np

BOUNDARY_EPS_MS = 1e-6


def match_events(reference, prediction, tolerance_ms=5., boundary_eps_ms=BOUNDARY_EPS_MS):
    reference = sorted(map(float, reference))
    prediction = sorted(map(float, prediction))
    i = j = 0
    errors = []
    bound = float(tolerance_ms) + float(boundary_eps_ms)
    while i < len(reference) and j < len(prediction):
        if prediction[j] < reference[i] - bound:
            j += 1
        elif reference[i] < prediction[j] - bound:
            i += 1
        else:
            errors.append(prediction[j] - reference[i])
            i += 1
            j += 1
    return dict(tp=len(errors), fp=len(prediction)-len(errors),
                fn=len(reference)-len(errors), matched_errors_ms=errors)


def aggregate_events(records):
    tp, fp, fn = [sum(r[k] for r in records) for k in ('tp', 'fp', 'fn')]
    errors = [e for r in records for e in r['matched_errors_ms']]
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                median_abs_error_ms=float(np.median(np.abs(errors))) if errors else None)
