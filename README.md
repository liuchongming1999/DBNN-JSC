# DBNN: fast quadratic surrogates for single-neuron simulation

This repository contains the code and numerical artifacts used for the DBNN
study prepared for *Journal of Scientific Computing*. DBNN combines one
causal double-exponential filter per synaptic site, a quadratic voltage
calculation, and an event-triggered reset. The same fitted model is evaluated
on CPU (one or eight cores) and GPU.

Authors: Chongming Liu, Jingyang Ma, Yeqiang Liao, Songting Li, and Douglas Zhou.
Chongming Liu and Jingyang Ma contributed equally. Correspondence: Douglas Zhou.

## Start here

```bash
git clone https://github.com/liuchongming1999/DBNN-JSC.git
cd DBNN-JSC
python -m venv .venv
source .venv/bin/activate
# For the measured CUDA 12.8 configuration:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/verify_release.py
```

The last command verifies released file hashes, the separation of training and
evaluation inputs, and recomputes DBNN RMSE and spike counts from the saved
voltage arrays. It does not train a new model.

| Primary evaluation | Frozen value |
|---|---:|
| Independent inputs | 379 |
| Reference spikes | 637 |
| Total simulated duration | 2,497 s |
| DBNN pooled subthreshold RMSE | 0.43739656 mV |
| DBNN event F1 | 0.47754811 |

The published checkpoint has SHA-256
`737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d`.
The quantitative source for all methods is `paper/evidence/main379_results.json`.

![Accuracy–runtime comparison](paper/figures/frontier.png)

## Contents

- `workspace/`: frozen source scripts, training/evaluation inputs and references,
  selected checkpoints, per-input predictions, and measured timing records.
  Dated directory names preserve the original experiment layout for auditing;
  they are not names of different DBNN architectures.
- `scripts/`: portable entry points for validation, workspace preparation,
  native compilation, and prediction.
- `paper/`: frozen numerical evidence, plotting code, and publication figures.
- `theory_data/`: saved numerical trajectories used by the theoretical figures.
- `additional_cells/`: further-cell fitting code, referenced checkpoints and test arrays.
- `docs/`: reproduction instructions, model sources, environment and file hashes.

## Run the implementations

Read [the reproduction guide](docs/REPRODUCING.md) for full commands, the
training pipeline, upstream models, numerical comparisons and timing scope.

```bash
python scripts/prepare_workspace.py
# Compile the recorded native code; a CUDA toolkit and CUTLASS 3.9.2 are needed.
python scripts/build_native.py --workspace build/workspace --cutlass /path/to/cutlass-3.9.2
python scripts/predict.py --workspace build/workspace --backend gpu
python scripts/predict.py --workspace build/workspace --backend cpu --threads 1
python scripts/predict.py --workspace build/workspace --backend cpu --threads 8
# Regenerate publication figures and numerical table exports from frozen data:
python paper/scripts/build_figures.py
```

The measured optimized CPU implementation requires Linux x86-64, AVX-512 and
an MKL-enabled PyTorch installation. The reference GPU environment is an RTX
5090 with CUDA 12.8. All working-copy preparation, binaries and newly generated
outputs are kept in `build/`, leaving the frozen files unchanged.

## Evaluation and timing

The reference NEURON integrates at 0.1 ms; learned models output at 1 ms;
comparison NEURON/CoreNEURON integrate at 0.5 ms. Voltage errors use the common
1 ms observation grid, excluding reference spikes ±10 ms. Event matching is
one-to-one within ±5 ms, on the full trajectory.

The main timing is complete prepared computation for one input: filtering,
voltage calculation and reset (and equivalent solver computation). Input
creation, model setup, compilation, external packing, uploads and downloads
are outside this boundary. Times are sums of per-input medians of three
repetitions on a shared server. Affinity and GPU synchronization are recorded;
absolute timings depend on hardware and resource contention.

## Third-party sources

Neuron models and comparison methods retain their original authorship and
licenses. Upstream repositories, pinned baseline commits and ModelDB accessions
are listed in [THIRD_PARTY.md](docs/THIRD_PARTY.md). Third-party model/source
packages are downloaded separately, rather than relicensed by this repository.
