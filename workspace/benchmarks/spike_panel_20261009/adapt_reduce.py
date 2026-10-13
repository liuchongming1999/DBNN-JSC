"""Make a separately versioned panel adapter of the verified reduction script."""
from pathlib import Path
import hashlib
import json


def main():
    here=Path(__file__).resolve().parent;src=here/'benchmark_reduce_source.py';text=src.read_text()
    replacements={
        "str(prepare(work))":"str(work/'compat')",
        "gate=json.loads((work/'original/reduce_legacy/complete.json').read_text());assert gate['passed']":
        "gate=json.loads((work/'original/reduce_legacy/complete.json').read_text());assert gate['passed'];compat_audit=json.loads((work/'compat/audit.json').read_text());assert hashlib.sha256((work/'compat/neuron_reduce/subtree_reductor_func.py').read_bytes()).hexdigest()==compat_audit['compat_sha256']",
        "a=p.parse_args()":
        "p.add_argument('--panel-work',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()",
        "root=a.root.resolve();work=a.work.resolve();out=work/('publication' if a.publication else 'aligned')/f'reduce_{a.dt:g}ms';out.mkdir(parents=True,exist_ok=True)":
        "root=a.root.resolve();work=a.work.resolve();panel=a.panel_work.resolve();out=panel/'methods/reduce1'/f'shard{a.shard:02d}';out.mkdir(parents=True,exist_ok=True)",
        "src=root/'benchmarks/dt_resolution_20260929/test';manifest=json.loads((src/'manifest.json').read_text());rows=manifest['cases']":
        "src=panel/'confirmatory';manifest=json.loads((src/'manifest.json').read_text());rows=manifest['cases'][a.shard::8]",
        "metric(ref['voltage'][::2],pv,ref['spike_times'],1.)":"metric(ref['voltage'],pv,ref['spike_times'],1.)",
        "full_main31=not a.limit and len(records)==31":"full_panel_shard=not a.limit and len(records)==len(rows)"
    }
    for before,after in replacements.items():
        assert text.count(before)==1,before;text=text.replace(before,after)
    dest=here/'evaluate_reduce_raw.py';dest.write_text(text)
    (here/'reduce_adapter_lineage.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
       adapter_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),changes=list(replacements),
       fixed_dt_ms=1,training_or_reduction_parameter_changes=False),indent=2)+'\n')


if __name__=='__main__':main()
