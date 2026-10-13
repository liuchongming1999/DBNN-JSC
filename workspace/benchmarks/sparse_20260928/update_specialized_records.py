"""Record the completed specialized-kernel run without changing older evidence."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((HERE / 'quad_confirm.json').read_text())
    source = json.loads((HERE / 'quad_source_replace.json').read_text())
    qa = json.loads((HERE / 'quad_report_verification.json').read_text())
    report = ROOT / 'DBNN_L5PC_surrogate_report_20260928.html'
    assert result['finished'] and source['finished'] and qa['passed']
    assert digest(report) == qa['sha256']
    assert digest(ROOT / 'source/DBNN.py') == result['baseline_sha256']
    assert digest(ROOT / 'source/dbnn_checkpoint.pt') == result['checkpoint_sha256']
    record = dict(
        completed=True, shapes=len(result['cases']), first_pass_samples=50,
        second_pass_samples=60, original_control_samples=30,
        original_drive_replacement_shapes=len(source['cases']),
        physical_gpu='6', exclusive_resources=False,
        precision='IEEE FP32; frozen-checkpoint inference',
        policy='benchmarks/sparse_20260928/quad_backend_policy.json',
        entrypoint='benchmarks/sparse_20260928/quad_optimized_backend.py::make_optimized_prepared',
        result='benchmarks/sparse_20260928/quad_confirm.json',
        original_drive_control='benchmarks/sparse_20260928/quad_source_replace.json',
        synthetic_scope='A>500 is computational parameter expansion, not dense-site fitted biology.',
        heldout_samples=10, heldout_max_voltage_error_mv=max(c['error']['max_abs_mv'] for c in result['heldout_forward_checks']),
        heldout_reset_mismatch_bins=max(c['error']['spike_mismatch_bins'] for c in result['heldout_forward_checks']),
        profiler='Nsight hardware counters unavailable: ERR_NVGPUCTRPERM; compiler metadata retained.',
        report_sha256=qa['sha256'], updated=datetime.now().astimezone().isoformat(),
    )
    status_path = ROOT / 'status_20260928.json'
    status = json.loads(status_path.read_text())
    status['quadratic_specialized_optimization'] = record
    status['updated'] = record['updated']
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    manifest_path = ROOT / 'manifest_20260928.json'
    manifest = json.loads(manifest_path.read_text())
    files = {v['path']: v for v in manifest['files']}
    changed = {report, status_path, HERE / 'README.md', HERE / 'build_report.py',
               HERE / 'original_layer_guide.py', Path(__file__).resolve(), HERE / 'verify_specialized_report.py'}
    for pattern in ['quad_*', 'quadratic_tiled*', 'figures/quad_specialized_*']:
        changed.update(p for p in HERE.glob(pattern) if p.is_file())
    for p in changed:
        name = str(p.relative_to(ROOT))
        files[name] = dict(path=name, bytes=p.stat().st_size, sha256=digest(p))
    manifest['files'] = [files[k] for k in sorted(files)]
    manifest['quadratic_specialized_report_updated'] = record['updated']
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
