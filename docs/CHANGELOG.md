# Release history

## 18 October 2026

- Synchronize with the final approved primary and cross-cell manuscript evidence.
- Preserve the primary checkpoint, weights, reset, seeds, and 379-input membership.
- Publish complete 0.0125 ms reference trajectories and recorded solver-step comparisons.
- Add the completed 350,000-update geometry-corrected Branch-ELM model, validation
  threshold, prediction arrays, paired GPU timings, author-default scripts, and logs
  expressed as structured histories/configurations.
- Add whole-trajectory TCN predictions and voltage/event parity; retain the same
  TCN/LSTM checkpoints and thresholds and the unchanged LSTM inference protocol.
- Publish all twelve final validation-selected cross-cell models, 2,184 input/reference/
  prediction pairs, training/validation data, selection records and source code.
- Include frozen PCA fitting/validation targets needed for the TCN recipe.
- Replace the default verifier with portable raw-array verification and explicit
  current file hashes; keep prior versions in Git history.
- Update figures, table sources, paired-statistics scripts, cost/diagnostic data,
  correspondence (Songting Li and Douglas Zhou), and reproduction instructions.
- In the private manuscript project, move the existing AI disclosure into Methods
  Section 3.2. No scientific wording, model, or result is changed by that move.

## 13 October 2026

Initial public code/data export, commit c47bb90cfc00222311fb6964744cd6d4351d8a25.
Its main reference step was 0.1 ms, with 637 reference spikes and older baseline
implementations. Use the latest release for the current paper; do not mix
headlines or verification rules from the two snapshots.
