# Frozen published-study results (18 October 2026)

All values below are copied from the approved, already-validated manuscript evidence.
They are not newly trained or selected results. The common reference is NEURON
0.0125 ms; all 379 original inputs remain.

| Method | RMSE (mV) | F1 ±5 ms | Sum of per-input median time (s) |
|---|---:|---:|---:|
| DBNN GPU | 0.440363586 | 0.535269710 | 0.032121312 |
| DBNN CPU, 1 core | 0.440363586 | 0.535269710 | 13.163915677 |
| DBNN CPU, 8 cores | 0.440363586 | 0.535269710 | 3.052005773 |
| TCN (whole trajectory) | 0.865142597 | 0.212593876 | 1.026231913 |
| Branch-ELM (corrected) | 0.755133923 | 0.415897800 | 36.010105701 |
| LSTM (unchanged) | 0.750120636 | 0.274666667 | 1.567096189 |
| Neuron_Reduce | 4.255388794 | 0.071243218 | 78.990257690 |
| NEURON, 8 cores, 0.5 ms | 0.391582463 | 0.148698885 | 526.926292349 |
| NEURON, 8 cores, 0.25 ms | 0.240936228 | 0.823717949 | 1053.548456030 |
| NEURON, 8 cores, 0.1 ms | 0.132283502 | 0.956719818 | 2597.580093382 |
| CoreNEURON, 1 core, 0.5 ms | 0.391582463 | 0.148698885 | 612.181931620 |
| CoreNEURON, 1 core, 0.25 ms | 0.240936228 | 0.823717949 | 1206.052699221 |
| CoreNEURON, 1 core, 0.1 ms | 0.132283502 | 0.956719818 | 2997.099399915 |

The timing boundary is prepared complete computation, not cold process launch.
Headline speed ratios use the fastest measured eight-core solver setting with
no larger voltage RMSE than DBNN. Finer steps can provide higher spike accuracy.

## Independently fitted cell models

These twelve models were selected on validation only and evaluated on the same
previously fixed 182-input panel per condition. The panel was previously used
for other fitting recipes; all inputs are retained without outcome filtering.

| Cell | Excitatory profile | Inputs | RMSE (mV) | F1 ±5 ms |
|---|---|---:|---:|---:|
| granule | ampa | 182 | 1.135096728 | 0.410708944 |
| granule | ampa_nmda | 182 | 1.229963049 | 0.487653992 |
| basket | ampa | 182 | 0.839315506 | 0.334401169 |
| basket | ampa_nmda | 182 | 0.773023334 | 0.415551073 |
| golgi | ampa | 182 | 0.390627808 | 0.010075567 |
| golgi | ampa_nmda | 182 | 0.512767886 | 0.005934718 |
| hay_l5 | ampa | 182 | 0.398468168 | 0.438596491 |
| hay_l5 | ampa_nmda | 182 | 1.129153561 | 0.103345555 |
| human_l23 | ampa | 182 | 0.264810048 | 0.436974790 |
| human_l23 | ampa_nmda | 182 | 0.260062578 | 0.480127186 |
| ca1 | ampa | 182 | 0.059108545 | 0.784313725 |
| ca1 | ampa_nmda | 182 | 0.239651267 | 0.496000000 |

See `paper/evidence/crosscell_final_20261017/` for intervals, counts, selected
checkpoint hashes, validation choices, and original protocol. These are not
one-model cross-cell transfer results. A validation winner can have zero-valued
diagonal entries inside the full-matrix model class.

Older `crosscell_models.json` / `crosscell_events.json` files are historical
31-input evidence retained for provenance; use the final182-input records above.
