# Data and code for the 27 October manuscript revision

Fixed tag: `jsc-v4-revision-20261027`.
The primary model, thresholds, input membership, learned baselines, twelve
validation-selected cross-cell models, main results and main figure are unchanged.
This update adds reproducibility materials for previously completed experiments.

| Manuscript location | Released evidence | Recompute/check |
|---|---|---|
| Main comparison, Fig. 1, Table 1 | `paper/evidence/main379_results.json`; 0.0125 ms reference and all thirteen rows' arrays | `python scripts/verify_release.py --primary-only` |
| Appendix I; Section 5.2 full-call discussion | `paper/evidence/workflow_cost_summary.json`, raw timing sources and workflow tables | `python paper/scripts/build_workflow_cost_appendix.py` in the reproduction copy |
| Appendix K, Table 14 | All 637 original candidates and 120 fixed stress inputs; raw arrays indexed in `docs/supplement_asset_map_20261027.json` | `python scripts/verify_supplements.py` |
| Appendix L, Table 15 | Original engineering campaign scripts, complete repeated timings, compact records | `python paper/scripts/build_engineering_ablation_appendix.py --check` |
| Appendix E.3 | Actual LSTM source/configuration hashes and description audit | `python scripts/verify_supplements.py` |
| Cross-cell comparison | All twelve locked conditions and all 2,184 fixed inputs | `python scripts/verify_release.py` |

## Input scope and supplementary accuracy

The 379-input primary panel was selected from 637 original candidates using
the original 0.1 ms reference: at most three spikes and no interspike interval
below 50 ms. The 258 excluded cases all exceed three spikes; one also violates
the interval rule. Refining the evaluation reference to 0.0125 ms does not
change the panel. All 637 candidates and all 120 independently prescribed
stress inputs are retained in Appendix K without additional output selection.

| Input set | Inputs | Masked voltage RMSE (mV) | F1, inclusive ±5 ms |
|---|---:|---:|---:|
| Primary panel | 379 | 0.440 | 0.535 |
| All original candidates | 637 | 0.439 | 0.511 |
| Excluded candidates | 258 | 0.438 | 0.499 |
| Prespecified stress panel | 120 | 3.585 | 0.225 |
| Increased stationary excitation | 24 | 1.851 | 0.218 |
| Periodic excitatory packets | 24 | 5.537 | 0.164 |
| Shared excitatory-rate modulation | 24 | 2.575 | 0.310 |
| Partly synchronous excitation | 24 | 4.221 | 0.315 |
| Excitation-inhibition balance | 24 | 1.917 | 0.212 |

The 379 rows are a subset of the 637, not another independent test panel.
The five stress families contain 24 inputs each (three prescribed levels,
eight fixed seeds per level). No old TCN/ELM ranking from the exploratory
stress campaign is imported into the corrected primary comparison.

## Timing and the meaning of the comparisons

The main prepared-computation campaign uses seven paired GPU repetitions
and three CPU/solver repetitions. It gives about 16,404 times versus the
eight-core 0.5 ms NEURON result at no larger voltage RMSE, and about 32 times
versus whole-trajectory TCN, the fastest compared learned surrogate.
The stricter comparison requiring both RMSE and F1 to be no worse uses the
0.25 ms NEURON result, about 32,799 times. These are measured configurations,
not extrapolations to every neuron model or every workload.

The complete-call ratio of about 234 comes from a separate campaign:
about 636.27 seconds for NEURON divided by 2.72 seconds
for resident DBNN with cached recurring layouts. Input processing, transfers
and output extraction are included there. The main figure still uses the
prepared boundary; these two campaign ratios are not interchangeable.

Engineering controls use the same frozen DBNN and all 379 primary inputs,
five warm-ups and 30 calls per variant. Six cumulative stages and six
single-removal/control variants yield 136,440 timed calls. The combined
implementation is 4.28 times faster than its serial GPU reference; the
individual gains depend on the surrounding implementation. The training
description reports the proposed fitting procedure and full-process results;
it does not claim an independently measured benefit for each training component.

| Implementation step | Cumulative time (ms) | Time with this optimization removed (ms) |
|---|---:|---:|
| Serial GPU reference, all sites | 136.263 | — |
| Active-site restriction | 133.829 | 33.506 |
| Parallel filtering | 146.566 | 35.430 |
| Fused voltage calculation | 127.695 | 46.012 |
| Event-based reset | 49.223 | 111.662 |
| CUDA Graph replay | 31.869 | 49.598 |

The repeated full implementation control in the removal campaign is 31.875 ms.

## What can be verified offline

`verify_release.py` checks the current file manifest, primary/cross-cell arrays,
all supplementary arrays, repeated timings, source hashes and unchanged model
identity. `reproduce_paper.py` creates a new `build/` copy for plots and tables;
it does not edit the frozen data. The older dated manifests and verification
logs remain historical. The manuscript prose/PDF and internal submission
materials are not part of this public code/data release.

The complete [input/output index](supplement_asset_map_20261027.json) maps
all 757 supplementary inputs to their original arrays and checksums.
