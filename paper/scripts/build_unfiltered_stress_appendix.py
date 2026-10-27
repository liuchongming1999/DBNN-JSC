"""Recompute the fixed DBNN's supplementary results from archived case records."""

import gzip
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/unfiltered_stress_dbnn_records_20261016.json.gz"
data = json.loads(gzip.decompress(SOURCE.read_bytes()))
records = data["records"]
manifest = {row["case_id"]: row for row in data["input_manifest"]}
assert len(records) == len(manifest) == 757
assert {row["case_id"] for row in records} == set(manifest)
assert data["protocol"]["models_frozen_before_new_outputs"]
assert data["protocol"]["reference_dt_ms"] == 0.0125
assert data["protocol"]["dbnn_checkpoint_sha256"] == (
    "737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d"
)


def match_events(reference, predicted):
    """Apply the paper's chronological one-to-one 5 ms matching rule."""
    assert reference == sorted(reference) and predicted == sorted(predicted)
    i = j = tp = 0
    tolerance = data["protocol"]["event_tolerance_ms"]
    margin = data["protocol"]["event_boundary_eps_ms"]
    while i < len(reference) and j < len(predicted):
        delta = predicted[j] - reference[i]
        if abs(delta) <= tolerance + margin:
            tp += 1
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return tp, len(predicted) - tp, len(reference) - tp


for row in records:
    case = manifest[row["case_id"]]
    assert row["cohort"] == case["cohort"]
    assert row["original_main379"] == case["original_main379"]
    counts = match_events(row["reference_events_ms"], row["predicted_events_ms"])
    score = row["event_scores"]["5"]
    assert counts == (score["tp"], score["fp"], score["fn"]), row["case_id"]
    assert math.isclose(row["rmse_mv"] ** 2 * row["n"], row["sse"], rel_tol=1e-12)

full = [row for row in records if row["cohort"] == "full637"]
primary = [row for row in full if row["original_main379"]]
excluded = [row for row in full if not row["original_main379"]]
stress = [row for row in records if row["cohort"] == "stress120"]
assert (len(full), len(primary), len(excluded), len(stress)) == (637, 379, 258, 120)
assert all(manifest[row["case_id"]]["screening_reference_spikes"] > 3 for row in excluded)
assert sum(manifest[row["case_id"]]["screening_minimum_isi_ms"] < 50 for row in excluded) == 1


def summarize(rows):
    tp = sum(row["event_scores"]["5"]["tp"] for row in rows)
    fp = sum(row["event_scores"]["5"]["fp"] for row in rows)
    fn = sum(row["event_scores"]["5"]["fn"] for row in rows)
    return {
        "cases": len(rows),
        "rmse_mv": math.sqrt(sum(row["sse"] for row in rows) / sum(row["n"] for row in rows)),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "reference_events": sum(row["reference_count"] for row in rows),
        "predicted_events": sum(row["predicted_count"] for row in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


groups = [
    ("primary379", "Primary panel", primary),
    ("full637", "All original candidates", full),
    ("excluded258", "Excluded candidates", excluded),
    ("stress120", "All stress inputs", stress),
]
for family, label in [
    ("tonic_strength", "Increased stationary excitation"),
    ("event_packets", "Periodic excitatory packets"),
    ("shared_rate_modulation", "Shared excitatory-rate modulation"),
    ("synchronous_events", "Partly synchronous excitation"),
    ("ei_balance", "Excitation--inhibition balance"),
]:
    rows = [row for row in stress if row["family"] == family]
    assert len(rows) == 24
    assert all(sum(row["level"] == level for row in rows) == 8 for level in range(3))
    groups.append((family, label, rows))

results = {key: summarize(rows) for key, _, rows in groups}
assert math.isclose(results["primary379"]["rmse_mv"], 0.4403635862353741, abs_tol=1e-12)
assert results["primary379"]["tp"] == 387
assert results["primary379"]["fp"] == 379
assert results["primary379"]["fn"] == 293
lines = [
    r"\begin{tabularx}{\textwidth}{Xrrr}",
    r"\toprule",
    r"Input set & Inputs & RMSE (mV) & $F_1$\\",
    r"\midrule",
]
for i, (key, label, _) in enumerate(groups):
    if i == 4:
        lines.append(r"\midrule")
    result = results[key]
    lines.append(f"{label} & {result['cases']} & {result['rmse_mv']:.3f} & {result['f1']:.3f}" + r" \\")
lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
table = "\n".join(lines)
summary = json.dumps({
    "protocol": data["protocol"],
    "source_hashes": data["source_hashes"],
    "all_757_native_event_lists_rematched": True,
    "primary379_reproduced": True,
    "groups": results,
}, indent=2) + "\n"
outputs = {
    ROOT / "sections/unfiltered_stress_table.tex": table,
    ROOT / "evidence/unfiltered_stress_results.json": summary,
}
for path, content in outputs.items():
    if "--check" in sys.argv:
        assert path.read_text() == content, path
    else:
        path.write_text(content)
print("Verified 757 frozen-model inputs; pooled RMSE and event matching reproduced.")
