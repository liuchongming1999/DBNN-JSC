# Reproducing the current DBNN study

## Frozen scope and software

The `jsc-v4-revision-20261027` snapshot uses the existing trained full-matrix mouse DBNN,
unchanged thresholds/reset, the original 379 inputs, and a common 0.0125 ms
NEURON reference. It incorporates the already-completed corrected Branch-ELM
training and whole-trajectory TCN inference. No training or tuning was performed
to make this release. `docs/reference_environment.json` records the experiment
software/hardware; optimized inference used PyTorch 2.8.0, CUDA 12.8, Triton 3.4.0,
NEURON 9.0.2, an RTX 5090, and AMD EPYC 9554 CPUs.

`docs/release_manifest_20261018.json` hashes the released files.
`docs/current_asset_map_20261018.json` maps original paths to repository paths.
Original absolute paths are retained in frozen provenance; `scripts/release_paths.py`
resolves these inside a clone. Neither `/home/lcm` nor the author's server is
needed for offline verification. The older `export_manifest.json` and
`additional_artifacts.json` describe the 13 October export, not the current
verification contract.

The authoritative file-integrity contract is `docs/release_manifest_20261027.json`.
The 13/18 October manifests and verification logs are retained as historical
records; they are not the verification contract for this revision.

## 1. Recompute published results from raw arrays

Only NumPy is required; no GPU, NEURON installation, training, or authentication:

```bash
pip install numpy
python scripts/verify_release.py
```

This checks every release-file hash, original 117/32 fitting/validation split,
379-input disjointness, current common reference, all 13 primary method/solver
rows, per-input voltage error and one-to-one events, timing medians and sums,
complete ELM budget, all twelve frozen cross-cell checkpoints, and all 2,184
fixed cross-cell input/reference/prediction pairs. It also verifies all 757
supplementary inputs from their saved rasters and voltage/event arrays,
recomputes the masked SSE, and checks all 136,440 engineering timing calls.
Original recorded GPU/FFT
and graph parity results are also checked. This command never fits anything.
Output: `build/release_verification.json`. Use `--primary-only` if only the
379-input comparison is needed. `--skip-hashes` skips only file-integrity reads,
not numerical recomputation.

The primary raw predictions are under
`workspace/benchmarks/random_init_main379_20261012/predictions/`.
The common reference and solver-step trajectories are under
`workspace/benchmarks/solver_main379_20261016/outputs/reference1/dt*/`.
NEURON/CoreNEURON parity and actual engine timings are archived under their
respective engine directories. Error statistics use their checked equivalent
same-step single-core trajectories. The copied manuscript aggregate is
`paper/evidence/main379_results.json`.

The TCN whole-trajectory implementation is checked against the unchanged frozen
predictions; rounding can differ below the archived numerical-equivalence
threshold, while event indices agree. The published voltage statistics use the
frozen arrays and the timing uses the equivalent whole-trajectory implementation.
Both sets are released. Corrected ELM predictions are under
`workspace/benchmarks/adviser6_elm_author_aligned_20261016/elm_evaluation/`.
The final seven-repeat paired GPU measurements are in that directory's sibling
`final_gpu_recheck_20261017/` and `paper/evidence/baseline_final_20261017/paired_gpu.json`.

## 2. Build and run the unchanged primary DBNN

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/prepare_workspace.py
# CPU build: Linux x86-64 with AVX-512 and MKL-enabled PyTorch.
python scripts/build_native.py --workspace build/workspace --cpu-only
# GPU build: compatible CUDA compiler and separately downloaded CUTLASS 3.9.2.
python scripts/build_native.py --workspace build/workspace --cutlass /path/to/cutlass-3.9.2
python scripts/predict.py --workspace build/workspace --backend gpu --case confirmatory_0532
python scripts/predict.py --workspace build/workspace --backend cpu --threads 1 --case confirmatory_0532
python scripts/predict.py --workspace build/workspace --backend cpu --threads 8 --case confirmatory_0532
```

`prepare_workspace.py` copies primary sources/data to a new `build/workspace`
and rewrites the original primary-root prefix only in that working copy.
It does not touch frozen inputs, models, or records. GPU defaults to `sm_120`;
`build_native.py --cuda-arch` accepts another supported architecture. Main
inference is FP32 IEEE with TF32 disabled. CPU/GPU roundoff can differ; the
entry point verifies voltage agreement below 0.001 mV and identical event indices
against the frozen prediction. Output NPZ and newly measured latency go to build/.
New latency measurements do not replace the paper's recorded measurements.

Rasters store event multiplicities and active IDs. The DBNN adapter uses +1 for
an excitatory event (IDs 0–249), -1 for an inhibitory event (IDs 250–499), and
0 for no event; event counts are retained when more than one event occupies a
bin. Each complete call starts with zero filter/reset state. Surrogate time
constants are expressed in ms; the primary sampling interval is 1 ms.

## 3. Main DBNN training and validation

The current selected checkpoint was randomly initialized and fitted on 117
trajectories, with 32 separate complete trajectories for validation. All sources
last 6 s. Random crops last 500 ms; the first 150 ms supplies filter history,
and the remaining 350 ms supplies the masked loss. The loss combines
trajectory-balanced voltage MSE and first differences. Reference events provide
teacher reset corrections during fitting; autonomous output events are used
in validation and inference. Validation calibration and subsequent fitting use
rebuilt teacher targets before final validation-only checkpoint/reset selection.
The selected reset is threshold -58 mV, amplitude 4 mV, recovery time 40 ms.

The frozen `protocol.json`, `primary_history.json`, `refine_history.json`, and
`final_selection.json` beside the primary checkpoint record exact choices.
The source is `workspace/benchmarks/diagonal_1ms_20261009/train_dbnn_diagonal.py`;
its historical filename does not mean this release fixes the diagonal to zero.
Optional retraining, in a NEW output directory, uses the recorded budget:

```bash
ROOT="$PWD/build/workspace"
mkdir -p build/training
cc -O3 -shared -fPIC "$ROOT/benchmarks/fair_window_20261005/seq_reset.c" -lm -o build/training/seq_reset.so
CUDA_VISIBLE_DEVICES=0 python "$ROOT/benchmarks/diagonal_1ms_20261009/train_dbnn_diagonal.py" \
  --root "$ROOT" --work "$PWD/build/training" --dt 1 --window 500 --burn 150 \
  --seed 291049 --allow-diagonal --epochs 2000 --refine-epochs 400 --lr 0.003
```

The released models remain authoritative for reproducing the reported values;
new independent fitting can differ by hardware/library nondeterminism and must
be evaluated separately. Do not overwrite the locked release with a new model.

## 4. Baseline implementations and fitting

All primary learned methods use the same 117 fit/32 validation sources. Source
versions and original-dataset replay records are retained. Retrieve pinned
third-party code under its original license:

```bash
python scripts/fetch_external.py --workspace build/workspace
```

- **Branch-ELM:** corrected geometry-only input routing covers all 45 virtual
  branches. Each batch samples eight distinct trajectories; 500 ms windows
  start no earlier than 500 ms. The fixed public Python defaults use seed
  930291, 35 x 10,000 updates, and validation-only output calibration and event
  threshold. The recorded complete checkpoint uses the full 350,000 budget.
  The Python implementation differs from some paper/JAX descriptions; these
  differences are disclosed in `resolved_args.json` and the manuscript.
- **TCN:** the 1 ms checkpoint and event threshold are unchanged. The current
  comparison times one complete trajectory forward, with archived voltage and
  event parity against its original overlapping-window implementation.
  The 20-component dendritic PCA fit uses fitting data only. The PCA transform
  and all 149 projected training/validation arrays are included under
  `additional_cells/data/lcm/DBNN_baselines_20260930/tcn_aux/pca_1ms/`.
  `sample_tcn_aux.py` and `prepare_pca.py` can regenerate auxiliary recordings
  using the upstream neuron model if raw dendritic traces are needed.
- **LSTM:** the published single-layer 50-unit architecture is implemented here
  in PyTorch with a two-output linear readout and overlapping-window inference.
  Input/readout weights use Glorot uniform initialization, recurrent weights
  are orthogonal, and one trainable gate bias has forget-gate value one and
  other values zero; the redundant PyTorch bias is fixed at zero. The fixed
  run uses seed 930291, batch size eight, Adam at initial learning rate 5e-4
  with per-update cosine decay, and 30 epochs of 11,400 updates. Its 500 ms
  windows exclude the first 150 ms from loss. Event BCE and voltage MSE
  have equal weight, with voltage scaled by 0.1. This is an architecture
  adaptation, not a claim to reproduce every detail of an unpublished author
  implementation or an independently replayed public LSTM checkpoint.
  Appendix E.3 and `paper/evidence/lstm_implementation_clarification_20261027.json`
  record the actual settings and source hashes; the model and results are unchanged.
- **NEURON/CoreNEURON:** the sweep covers 0.5, 0.25, and 0.1 ms solver steps,
  with a 1 ms observation grid; the common accuracy reference uses 0.0125 ms.
  CPU engine voltage/event parity is checked in the saved per-input records.
  **Neuron_Reduce** retains its locked morphology adaptation and solver setup.

The portable optional ELM training launcher preserves the archived defaults
and requires a new output folder:

```bash
python scripts/train_branch_elm.py --workspace build/workspace --output build/new_elm_training
```

For TCN/LSTM, the released `train_default.py`, exact `complete.json` configs,
source versions, and original replay gates are under the primary baseline
workspace. TCN uses `--method tcn --dt 1`; LSTM uses `--method lstm --dt 1`.
Pass a fresh work/output directory, the relocated primary root, and the released
PCA data directory (its parent must contain `tcn_aux`). Historical campaign
scripts have completion guards and absolute-path provenance: copy them into
a new work area before launching independent fitting. These training commands
are not run by release verification.

## 5. Separately fitted cells

The authoritative final protocol and model lock are under
`additional_cells/home/lcm/DBNN_crosscell_final_selection_test_20261017/`.
All twelve selected weights/reset settings were fixed before this replay.
Each condition keeps all 182 prelocked inputs without output-based screening.
The panel had already been used to evaluate other fitting recipes; this is
explicitly a fixed-panel reevaluation, not a newly unseen panel. A trainable
diagonal permits a validation-selected zero-valued winner; no test performance
is used to replace models or thresholds.

Released files include all selected checkpoints, original test rasters and
references, fitting/validation inputs and references, per-input predicted voltage
and events, full timing and numerical-equivalence records, starting checkpoints,
validation comparisons and histories, and the final candidate-selection code.
The selected model mapping is in `models_locked.json`; job manifests and reset
settings are preserved verbatim. Cross-cell training sources are under
`additional_cells/home/lcm/DBNN_crosscell_candidate_completion_20261017/`
and the parent legacy fitting directories. Cell/source accessions and model
conditions remain in the original per-job `reference_condition.json` records.

A portable reference forward for one locked model (Torch/NumPy only):

```bash
python scripts/replay_crosscell.py --cell granule --profile ampa --case new_granule_0000 --device cpu
# Use --device cuda:0 for the original FP32 FFT arithmetic path.
```

This computes the same fixed DBNN with FFT filtering and sequential FP64 reset,
checks against the saved prediction, and writes to build/. It is a reference
replay, not the optimized timing implementation. The actual optimized evaluator,
including its predeclared FP64-reset equivalence fallback, is preserved in the
final evaluation directory. All biological models are independently fitted;
this is not transfer of the primary mouse checkpoint to another cell.

## 6. Figures, uncertainty, and diagnostics

```bash
pip install matplotlib
python scripts/reproduce_paper.py
```

This creates a fresh `build/paper_reproduction/`, recalculates current event
statistics and paired bootstrap comparisons, rebuilds cross-cell tables and
figures, and reproduces the frozen validation/reset diagnostics. Original
release assets stay unchanged. Figures use Matplotlib and embedded vector fonts.
The copied manuscript source itself is not published by this code/data update.

Primary input bootstrap uses 10,000 resamples of the 374 stochastic complete
trajectories, retaining five deterministic controls; this measures input
variability for the fixed models, not variability across retraining seeds.
Cross-cell uncertainty uses the stored 10,000 trajectory-resample rule.
Subthreshold masks and spike matching are different: voltage excludes ±10 ms
of reference events, whereas F1 uses inclusive ±5 ms one-to-one event matching
with 1e-6 ms floating-point allowance at the boundary. No test thresholds are
recalibrated by any regeneration entry point.

Finite-strength analysis, isolated-response controls, and fitted-network error
budgets are in paper/evidence and theory_data with their source scripts. Practical
call boundaries and one-time costs are archived separately from prepared
computation. Raw timing is hardware-specific; the paper's primary GPU comparison
uses seven paired repetitions on a monitored device, while CPU/solver records
use three. Graph capture, packing, upload/download, initialization, and training
are outside the headline prepared interval. Cost tables make those boundaries
explicit and estimate amortization separately.


## 7. New supplementary results: Appendices K and L

```bash
python scripts/verify_supplements.py
python paper/scripts/build_unfiltered_stress_appendix.py --check
python paper/scripts/build_engineering_ablation_appendix.py --check
```

These commands need only NumPy and the standard library. The raw-array mapping
is `docs/supplement_asset_map_20261027.json`; every source file has a checksum.
The first command reloads all 757 input rasters, common reference trajectories,
saved accuracy predictions and separate GPU timing exports. It
recomputes the voltage-error sample counts and squared-error sums, verifies
native event lists and reproduces Appendix K. It requires GPU parity on the
primary379 and reports any discrepancies on the additional stronger inputs. No GPU or fitting
is needed to check the saved outputs. The saved accuracy traces use the
recorded FP32 filters/readout with sequential FP64 reset, as in the original
accuracy export. The optimized FP32 GPU timing traces are retained separately. Under some
stronger supplementary inputs their event sequences differ from the accuracy
traces. Those differences are reported, not assumed to pass parity and not used
to select or replace any accuracy output. Appendix K reports accuracy only;
none of these supplementary GPU timings replaces a main comparison timing.

Appendix L retains the full original `engineering/complete.json` and
`engineering_rollback/complete.json`, the 30 wall/device readings per case and
variant, and the two original campaign scripts. Its compact table can be
recomputed without executing a GPU benchmark. A new timing run requires the
documented hardware/software and a fresh relocated workspace. Campaign scripts
retain the original CPU affinity and completion/output paths as provenance;
adapt those paths in a new working copy before an independent timing run.
New timings must not overwrite or be substituted for the archived measurements.

The manuscript's code/data availability statement identifies the fixed tag;
the current release does not publish the manuscript text, submission documents,
cover letter, or internal review checklist.

The array audit reports three stress inputs with different event indices between
the original accuracy export and the optimized GPU timing export. All 637
original candidate inputs, including the full primary379, pass the required
voltage/event equivalence check. All supplementary accuracy values are
recomputed from their original accuracy arrays; no outputs are selected or
replaced on the basis of those numerical differences. The exact case IDs and
discrepancies are written to `build/supplement_verification.json`.
