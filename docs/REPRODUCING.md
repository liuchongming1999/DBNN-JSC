# Reproducing the DBNN study

## 1. What is frozen

The main DBNN checkpoint was fitted from independent random initialization on
117 training and 32 validation trajectories, each 6 s long. Training windows
are 500 ms, with a 150 ms input-history prefix and a 350 ms loss interval.
No input in the independent 379-case panel participates in fitting, reset
calibration, or checkpoint selection. Run `scripts/verify_release.py` to check
input disjointness and the raw-array primary DBNN statistics.

The original training source is
`workspace/benchmarks/diagonal_1ms_20261009/train_dbnn_diagonal.py`.
Its dated name reflects an earlier comparison; this release's main model uses
a trainable full quadratic matrix. The frozen `protocol.json` beside the
checkpoint gives the exact seed, initialization, losses, optimizer schedule,
candidate selection and validation reset parameters. The selected reset is
θ = −58 mV, a = 4 mV and τ = 40 ms.

## 2. Workspace and native builds

The `workspace/` snapshot preserves original absolute paths in historical JSON
and scripts so its hashes can be audited. Do not edit it in place.

```bash
python scripts/prepare_workspace.py
```

This creates `build/workspace` and records each path rewrite in
`build/relocation.json`. All following commands use this working copy.

CPU build (Linux x86-64 with AVX-512; PyTorch must export MKL):

```bash
python scripts/build_native.py --workspace build/workspace --cpu-only
```

GPU build additionally requires the CUDA toolkit, a compiler compatible with
that toolkit, and CUTLASS headers:

```bash
git clone --depth 1 --branch v3.9.2 https://github.com/NVIDIA/cutlass.git build/cutlass-3.9.2
python scripts/build_native.py --workspace build/workspace --cutlass build/cutlass-3.9.2
```

Default CUDA architecture is `sm_120` (RTX 5090). Use `--cuda-arch` for another
GPU supported by your installed toolkit. The recorded numerical policy uses
FP32 IEEE with TF32 disabled. CPU FMA and GPU reductions may differ slightly
in rounding, and the inference examples verify voltage agreement and identical
event indices against the frozen prediction.

## 3. A complete prediction and latency measurement

```bash
python scripts/predict.py --workspace build/workspace --case confirmatory_0532 --backend gpu
python scripts/predict.py --workspace build/workspace --case confirmatory_0532 --backend cpu --threads 1
python scripts/predict.py --workspace build/workspace --case confirmatory_0532 --backend cpu --threads 8
```

`--output` sets the NPZ path; `--repeats` changes timing repetitions. Input data
contain `ids` (active site IDs) and `x` (unsigned binary event raster). The
adapter assigns +1 to excitatory events (IDs 0–249), −1 to inhibitory events
(IDs 250–499), and 0 when no event occurs. Time constants stay in milliseconds.
Each whole-window call starts with zero filter/reset state.

The recorded GPU path uses active-site packing, a parallel affine scan,
fused quadratic/linear/bias calculation, event reset, cached workspaces and
CUDA Graph replay. CPU uses site-parallel filtering, MKL or fused quadratic
calculation, and event reset. The example selects from the paper's independent
shape policy and checks equivalence against the saved oracle.

The timing interval begins after input/model preparation. It includes the
complete forward and GPU synchronization. CPU packing, upload, download,
compilation and scoring are outside it. An additional NVIDIA-event measurement
is retained in the original timing records where available.

For all 379 inputs use the original portable working-copy runner:

```bash
ROOT="$PWD/build/workspace"
PANEL="$ROOT/benchmarks/spike_panel_20261009"
CP="$ROOT/benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt"
SHA=737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d
python "$ROOT/benchmarks/random_init_main379_20261012/benchmark_direct.py" \
  --root "$ROOT" --work "$PANEL" --method gpu --repeats 3 \
  --checkpoint "$CP" --checkpoint-sha256 "$SHA" \
  --predictions "$ROOT/benchmarks/random_init_main379_20261012/predictions" \
  --output "$PWD/build/benchmark-gpu" --cpus 0
```

For CPU use `--method cpu1 --cpus 0` or `--method cpu8 --cpus 0,1,2,3,4,5,6,7`.
Choose permitted physical cores from your machine's affinity mask. The command
runs per-input numerical validation before accepting timing results.

## 4. Train from scratch

The frozen pipeline uses AdamW, cosine decay, clipping and EMA. Its voltage
loss combines trajectory-balanced masked MSE and a first-difference term.
Reference events define the teacher reset correction; full validation
trajectories select checkpoints and reset. Primary optimization uses 2,000
epochs, followed by 400 refinement epochs from its own fitted parameters.
The initialization is random; no previously trained surrogate is required.

```bash
ROOT="$PWD/build/workspace"
mkdir -p build/training
# Build the small training reset routine into the new output directory.
gcc -O3 -shared -fPIC "$ROOT/benchmarks/fair_window_20261005/seq_reset.c" -lm -o build/training/seq_reset.so
CUDA_VISIBLE_DEVICES=0 python "$ROOT/benchmarks/diagonal_1ms_20261009/train_dbnn_diagonal.py" \
  --root "$ROOT" --work "$PWD/build/training" --dt 1 --window 500 --burn 150 \
  --seed 291049 --allow-diagonal --epochs 2000 --refine-epochs 400 --lr 0.003
```

Use `--smoke --epochs 1 --refine-epochs 1` for a short execution check. Smoke
models are not the published checkpoint. Seeded numerical results may still
vary across GPU/library versions; evaluate any new checkpoint separately.

## 5. Biological models and comparison methods

The main cell is ModelDB 180373, Shai et al. (2015), with `cell1.asc`,
`L5PCbiophys5.hoc`, full active channels, 500 synaptic sites, and the frozen
AMPA-like/ inhibitory Exp2Syn configuration. The model and site hashes are in
`docs/model_source_hashes.json` and the panel's `site_metadata.json`.
Download upstream models using `scripts/fetch_external.py`; install/compile
NEURON and CoreNEURON as described by their upstream distributions before
regenerating solver trajectories. The original run scripts and records are in
`workspace/benchmarks/spike_panel_20261009` and
`workspace/benchmarks/neuron_single_expanded_20260929`.

TCN, Branch-ELM and LSTM 1 ms checkpoints and frozen prediction arrays are
included. Their author-default fitting routines and pinned upstream commits
are provided. TCN also uses dendritic PCA supervision; prepare the auxiliary
arrays with the supplied `sample_tcn_aux.py` and `prepare_pca.py` before a full
retraining run. The upstream original-dataset reproduction evidence is
`paper/evidence/baseline_reproduction_and_training.json`.

Additional-cell source scripts, referenced checkpoints, evaluation inputs and
reference arrays are included in `additional_cells/`. The mapping from the
original source paths to released paths is `docs/additional_artifacts.json`.

Further-cell results are in `paper/evidence/crosscell_models.json` and
`crosscell_events.json`. These are cell-specific fits and configurations;
ModelDB accessions and their original sampling intervals are recorded in the
files. The primary 379-case protocol is specific to the mouse cell.

The input-design manifest records the historical checkpoint used when the
inputs were generated. For the model reported in the current manuscript, use
the checkpoint SHA in the repository README and the frozen random-init
`protocol.json`; the input rasters themselves are unchanged.

## 6. Numerical error analysis and figures

Recorded numerical scripts live in `workspace/benchmarks/theory_20261005`.
`run_passive.py --out <new-directory>` recomputes the passive finite-strength
controls. Pair-interaction and perturbation-order checks use the companion
Python scripts and saved NPZ trajectories. Keep newly generated output separate
from the frozen archive.

```bash
python paper/scripts/build_figures.py
```

This reconstructs six scientific figures and numerical table exports from
released evidence arrays. It needs no NEURON simulation or GPU training.

## 7. Inspect input-level results

- `workspace/benchmarks/spike_panel_20261009/main379/manifest.json`: selected
  independent inputs and complete event-generating conditions.
- `workspace/benchmarks/random_init_main379_20261012/records`: DBNN per-input
  masked SSE/sample count, TP/FP/FN and matched timing errors.
- `.../predictions`: predicted drive, voltage, and reference/model events.
- `.../gpu`, `.../cpu1`, `.../cpu8`: repetition-level runtime and equivalence.
- `workspace/benchmarks/spike_panel_20261009/main379/neuron8` and `core1`:
  sharded 0.5 ms solver curves and timing records.
- `paper/evidence/main379_results.json`: the frozen all-method aggregation,
  input-family and duration strata, and trajectory-level confidence intervals.

Changing masks, tolerances or aggregation conventions constitutes a new
analysis: preserve the frozen figures and report the new convention explicitly.
