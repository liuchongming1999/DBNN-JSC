"""Collect completed timing and cross-cell evidence without rerunning experiments."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_publication(work):
    work = Path(work)
    out = dict(state=read(work/'publication/state.json'), dbnn={}, solvers={}, reduction={})
    for backend in ['gpu', 'cpu1', 'cpu8']:
        path = work/'publication/dbnn'/backend/'complete.json'
        out['dbnn'][backend] = read(path)
    for method, folder, prefix in [('neuron', 'neuron8', 'nt8'), ('core', 'core1', 'nt1')]:
        for dt in [1., .5, .25]:
            paths = sorted((work/'publication'/folder).glob(f'{prefix}_dt{dt:g}_*.json'))
            records = [read(p) for p in paths]
            out['solvers'][f'{method}_{dt:g}'] = dict(
                status=read(work/'publication'/folder/f'status_{prefix}_dt{dt:g}.json'),
                records=records, source_sha256={str(p):digest(p) for p in paths})
    for dt in [1., .5, .25]:
        out['reduction'][str(dt)] = read(work/'publication'/f'reduce_{dt:g}ms/complete.json')
    return out


def compact_selection(data):
    if data is None:
        return None
    out = {k:data[k] for k in ['finished', 'checkpoint_sha256', 'protocol', 'selected'] if k in data}
    if 'selected' in out:
        out['selected'] = dict(out['selected'])
        val = out['selected'].get('validation')
        if val:
            out['selected']['validation'] = {k:v for k,v in val.items() if k != 'cases'}
    return out


def collect_cells():
    models = {}
    sources = {}
    projects = [('DBNN_multineuron_20261002', '/data/lcm/DBNN_multineuron_20261002'),
                ('DBNN_noncerebellar_20261002', '/data/lcm/DBNN_noncerebellar_20261002')]
    for project, data_root in projects:
        work = Path('/home/lcm')/project
        spec = importlib.util.spec_from_file_location(project+'_models', work/'models.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, model_spec in module.REGISTRY.items():
            base = work/'results'/name
            paths = []
            def get(p):
                p = Path(p)
                if p.exists(): paths.append(p)
                return read(p)
            row = dict(project=str(work), data_root=data_root, spec=model_spec,
                       state=get(base/'state.json'), operating_point=get(base/'screen/operating_point.json'),
                       coverage=get(base/'screen/sites.json'),
                       reference_convergence=get(base/'screen/reference_convergence.json'),
                       reference_step_audit=get(base/'ampa/reference_step_audit/complete.json'),
                       profiles={})
            if row['coverage']:
                row['coverage'] = {k:v for k,v in row['coverage'].items() if k != 'sites'}
            for profile in ['ampa', 'ampa_nmda']:
                out = base/profile
                panel = Path(data_root)/name/profile/'test'
                manifest = get(panel/'manifest.json')
                q = dict(manifest=manifest, selection=get(panel/'selection.json'),
                         gpu=get(out/'gpu/complete.json'), training={}, solvers={}, invalid={},
                         screen=get(base/'screen'/f'{profile}.json'))
                if q['screen']:
                    q['screen'] = {k:v for k,v in q['screen'].items() if k != 'records'}
                expected = {r['case_id']:r['input_sha256'] for r in manifest['cases']}
                assert len(expected) == 31, (name, profile, 'reference panel count')
                assert q['gpu']['finished'] and len(q['gpu']['records']) == 31
                assert {r['case_id'] for r in q['gpu']['records']} == set(expected)
                for record in q['gpu']['records']:
                    assert record['input_sha256'] == expected[record['case_id']]
                    for variant in record['variants'].values():
                        assert variant['numerical']['finite'] and variant['numerical']['events_equal']
                        assert len(variant['wall']['runs_ms']) == 30
                for dt in ([1., .5, .25] if profile == 'ampa' else [1.]):
                    p = out/f'dt{dt}ms/final_selection.json'
                    q['training'][str(dt)] = compact_selection(get(p))
                    assert q['training'][str(dt)]['finished']
                    checkpoint = out/f'dt{dt}ms/best_checkpoint.pt'
                    assert digest(checkpoint) == q['training'][str(dt)]['checkpoint_sha256']
                    assert q['gpu']['runtime']['repeats'] == 30
                for p in sorted((out/'solvers').glob('*/complete.json')):
                    if '_smoke' in p.parent.name: continue
                    solver = get(p)
                    assert solver['finished'] and len(solver['records']) == 31
                    assert {r['case_id'] for r in solver['records']} == set(expected)
                    for record in solver['records']:
                        assert record['input_sha256'] == expected[record['case_id']]
                        assert len(record['runs_ms']) == 5
                    q['solvers'][p.parent.name] = solver
                for p in sorted((out/'solvers').glob('*/invalid.json')):
                    if '_smoke' not in p.parent.name: q['invalid'][p.parent.name] = get(p)
                row['profiles'][profile] = q
            for p in paths: sources[str(p)] = digest(p)
            models[name] = row
    return dict(collected=time.strftime('%Y-%m-%dT%H:%M:%S%z'), models=models,
                source_sha256=sources, scope='Six independently fitted models; 31 selected reference windows per synapse condition; AMPA 1/.5/.25ms, AMPA+NMDA 1ms; shared-server prepared complete compute.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, str(a.work/'collect_snapshot.py'), '--work', str(a.work), '--data', str(a.data)], check=True)
    cells = collect_cells()
    save(a.work/'snapshot/multineuron_final.json', cells)
    print(json.dumps(dict(collected=cells['collected'], models=list(cells['models']), publication=read(a.work/'publication/state.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
