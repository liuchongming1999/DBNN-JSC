"""Attach the completed precision campaign to the ongoing project manifest."""
import hashlib,json
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    last=json.loads((HERE/'tf32_repeat.json').read_text())
    first=json.loads((HERE/'tf32_evaluation.json').read_text())
    q=json.loads((HERE/'tf32_quad_sweep.json').read_text())
    f=json.loads((HERE/'tf32_layer_sweep.json').read_text())
    api=json.loads((HERE/'tf32_api_validation.json').read_text())
    qa=json.loads((HERE/'tf32_report_verification.json').read_text())
    assert all(d.get('finished') for d in [last,first,q,f,api]) and api['passed'] and qa['passed']
    assert sha(ROOT/'source/DBNN.py')==last['baseline_sha256']
    assert sha(ROOT/'source/dbnn_checkpoint.pt')==last['checkpoint_sha256']
    report=ROOT/'DBNN_L5PC_surrogate_report_20260928.html'
    assert sha(report)==qa['sha256']
    for n,h in last['source_sha256'].items():assert sha(HERE/n)==h,(n,'changed since timing')
    record=dict(completed=True,shapes=13,quad_shape_config_attempts=sum(len(c['specs']) for c in q['cases']),
        filter_shape_config_attempts=sum(len(c['specs']) for c in f['cases']),
        first_repetitions=50,second_repetitions=60,original_paired_repetitions=30,
        heldout_traces_per_shape=10,api_dispatches=26,arbitrary_quadratic_checks=20,arbitrary_filter_checks=6,
        physical_gpu=6,exclusive_resources=False,profiles=['strict','approximate_0p01mv'],
        strict_max_voltage_mv=1e-3,approximate_max_voltage_mv=1e-2,reset_mismatch_bins_allowed=0,
        scope='FP32 tensors; TF32/TF32x3 only in selected matrix products; index gather, state carry and reset remain FP32.',
        entrypoint='benchmarks/sparse_20260928/tf32_pipeline.py::make_precision_prepared',
        policy='benchmarks/sparse_20260928/tf32_backend_policy.json',
        result='benchmarks/sparse_20260928/tf32_repeat.json',
        report_sha256=qa['sha256'],updated=datetime.now().astimezone().isoformat())
    status_path=ROOT/'status_20260928.json';status=json.loads(status_path.read_text())
    status['tf32_five_layer_optimization']=record;status['updated']=record['updated']
    status['current_tf32_benchmark']=record['result']
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    manifest_path=ROOT/'manifest_20260928.json';manifest=json.loads(manifest_path.read_text())
    files={d['path']:d for d in manifest['files']}
    changed={report,status_path,HERE/'README.md',HERE/'build_report.py',HERE/'original_layer_guide.py',
             HERE/'verify_tf32_report.py',Path(__file__).resolve()}
    for pattern in ['tf32_*','figures/tf32_*']:changed.update(p for p in HERE.glob(pattern) if p.is_file())
    for p in changed:
        key=str(p.relative_to(ROOT));files[key]=dict(path=key,bytes=p.stat().st_size,sha256=sha(p))
    manifest['files']=[files[k] for k in sorted(files)];manifest['tf32_five_layer_report_updated']=record['updated']
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
