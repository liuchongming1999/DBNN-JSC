# Original models and software

This repository's DBNN source and numerical results are authored by the study
team. Third-party code is not relicensed. Retrieve upstream source under its
original terms and cite the following work when reproducing comparisons.

| Component | Upstream source / recorded version |
|---|---|
| Mouse V1 L5PC | Shai et al. (2015), ModelDB https://modeldb.science/180373 |
| NEURON / CoreNEURON | https://github.com/neuronsimulator/nrn ; measured NEURON 9.0.2 |
| TCN biological-neuron model | Beniaguev et al. (2021), https://github.com/SelfishGene/neuron_as_deep_net ; commit `074c4666300a8ad246601dab179a97a6942f0f29` |
| Branch-ELM | Spieler et al. (2024), https://github.com/AaronSpieler/elmneuron ; commit `52e68a6d39523ac6613a586699b116e8e606dda3` |
| Neuron_Reduce | Amsalem et al. (2020), https://github.com/orena1/neuron_reduce ; commit `94b1850607f9c62a205ad8fb695f2fd91d84d87d` |
| CUTLASS headers | NVIDIA https://github.com/NVIDIA/cutlass ; v3.9.2 |
| PyTorch / Triton | https://github.com/pytorch/pytorch ; PyTorch 2.8.0, Triton 3.4.0 |

Further-cell accessions, model variants and synaptic configurations are retained
in `paper/evidence/crosscell_models.json` and `crosscell_events.json`.
ModelDB source files are fetched separately. Reference input/output arrays in
this release are simulation results generated for the study.

The project does not bundle the external original-dataset training/test dumps
for TCN/Branch-ELM. The original public dataset retrieval and reproduction
scripts, upstream commits, and completed numerical checks are supplied.
