"""Verify archived paired timings and generate the fixed-model engineering table."""

import gzip
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
data = json.loads(gzip.decompress(
    (ROOT / "evidence/engineering_ablation_records_20261016.json.gz").read_bytes()
))
panel = json.loads(gzip.decompress(
    (ROOT / "evidence/unfiltered_stress_dbnn_records_20261016.json.gz").read_bytes()
))
primary = {row["case_id"]: row for row in panel["input_manifest"] if row["original_main379"]}
assert len(primary) == 379
summary = {}
for name, campaign in data["campaigns"].items():
    protocol = campaign["protocol"]
    records = campaign["records"]
    assert campaign["finished"] and campaign["checkpoint_unchanged"]
    assert protocol["checkpoint_sha256"] == panel["protocol"]["dbnn_checkpoint_sha256"]
    assert protocol["repeats"] == 30 and protocol["warmups"] == 5
    assert len(records) == 379 and {row["case_id"] for row in records} == set(primary)
    result = {}
    for variant in protocol["variants"]:
        elapsed = []
        device = []
        max_discrepancy = 0.0
        squared_error = samples = tp = fp = fn = 0
        for row in records:
            assert row["input_sha256"] == primary[row["case_id"]]["input_sha256"]
            assert row["checkpoint_sha256"] == protocol["checkpoint_sha256"]
            values = row["results"][variant]
            for clock in ["wall", "device"]:
                readings = values[clock]["runs_ms"]
                assert len(readings) == 30
                assert all(math.isfinite(x) and x >= 0 for x in readings)
                assert math.isclose(statistics.median(readings), values[clock]["median_ms"], abs_tol=1e-12)
            parity = values["parity"]
            assert parity["finite"] and parity["events_equal"] and parity["passed"]
            max_discrepancy = max(max_discrepancy, parity["max_abs_mv"])
            elapsed.append(values["wall"]["median_ms"])
            device.append(values["device"]["median_ms"])
            metrics = values["metrics"]
            squared_error += metrics["sse"]
            samples += metrics["samples"]
            tp += metrics["events"]["tp"]
            fp += metrics["events"]["fp"]
            fn += metrics["events"]["fn"]
        assert (tp, fp, fn) == (387, 379, 293)
        assert max_discrepancy < 2e-5
        time_ms = sum(elapsed)
        archived = campaign["summary"][variant]
        assert math.isclose(time_ms, archived["wall_ms"], abs_tol=1e-9)
        assert math.isclose(math.sqrt(squared_error / samples), archived["rmse_mv"], abs_tol=1e-12)
        result[variant] = {
            "cases": 379,
            "sum_per_input_wall_median_ms": time_ms,
            "sum_per_input_device_median_ms": sum(device),
            "max_voltage_discrepancy_mv": max_discrepancy,
            "all_event_sequences_equal": True,
            "rmse_mv": math.sqrt(squared_error / samples),
            "tp_fp_fn": [tp, fp, fn],
        }
    summary[name] = result

cumulative = summary["cumulative"]
removal = summary["removal"]
rows = [
    ("Serial reference, all sites", "G0_dense_serial", None),
    ("Active-site restriction", "G1_active_serial", "R0_no_packing"),
    ("Parallel filtering", "G2_parallel_filter", "R1_serial_filter"),
    ("Fused voltage calculation", "G3_fused_drive", "R2_unfused_drive"),
    ("Event-based reset", "G4_event_reset", "R3_sequential_reset"),
    ("CUDA Graph replay", "G5_cuda_graph", "R5_no_graph"),
]
lines = [
    r"\begin{tabularx}{\textwidth}{Xrr}",
    r"\toprule",
    r"Implementation step & Cumulative (ms) & Removal (ms)\\",
    r"\midrule",
]
for label, add_key, remove_key in rows:
    add = cumulative[add_key]["sum_per_input_wall_median_ms"]
    removed = f"{removal[remove_key]['sum_per_input_wall_median_ms']:.3f}" if remove_key else "---"
    lines.append(f"{label} & {add:.3f} & {removed}" + r" \\")
lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
table = "\n".join(lines)
speedup = (cumulative["G0_dense_serial"]["sum_per_input_wall_median_ms"] /
           cumulative["G5_cuda_graph"]["sum_per_input_wall_median_ms"])
assert math.isclose(speedup, 4.275773210488186, abs_tol=1e-12)
output = {
    "source_hashes": data["source_hashes"],
    "checkpoint_sha256": panel["protocol"]["dbnn_checkpoint_sha256"],
    "inputs": 379,
    "calls_per_variant_per_input": 30,
    "warmups": 5,
    "all_input_hashes_match_primary_panel": True,
    "all_archived_call_medians_and_parity_records_verified": True,
    "overall_cumulative_speedup": speedup,
    "complete_removal_control_ms": removal["R4_complete_graph"]["sum_per_input_wall_median_ms"],
    "summary": summary,
}
outputs = {
    ROOT / "sections/engineering_ablation_table.tex": table,
    ROOT / "evidence/engineering_ablation_results.json": json.dumps(output, indent=2) + "\n",
}
for path, content in outputs.items():
    if "--check" in sys.argv:
        assert path.read_text() == content, path
    else:
        path.write_text(content)
print("Verified 12 implementation variants on 379 inputs with 30 calls each; fixed-model parity retained.")
