#!/usr/bin/env python3
"""Validate the 8/16/32-core extension against the same-dt saved voltages."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE/'results'
RUNS = RES/'multisplit_highcore_results'


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(RES/'selected_manifest.json')
    ids = [r['case_id'] for r in manifest['cases']]
    assert len(ids) == len(set(ids)) == 31
    meta = {r['case_id']:r for r in manifest['cases']}
    refs = {cid:np.load(RES/'reference_results'/f'{cid}_reference.npz')['voltage'].astype(float).ravel()
            for cid in ids}
    previous = read(RES/'multisplit_analysis.json')
    dbnn_ms = previous['dbnn_total_steady_ms']
    campaign = read(RUNS/'campaign_status.json')
    assert campaign['finished'] and len(campaign['records']) == 9
    runner_sha = hashlib.sha256((HERE/'run_multisplit_grid.py').read_bytes()).hexdigest()
    rows,aggregates,runtimes = [],[],{}
    for dt in [.25,.5,1.]:
        for nt in [8,16,32]:
            name = f'nt{nt}_dt{dt:g}'
            status = read(RUNS/f'status_{name}.json')
            assert status['finished'] and status['completed'] == status['total'] == 31
            group = []
            for cid in ids:
                rec = read(RUNS/f'{name}_{cid}.json')
                ctrl = read(RUNS/f'nt8_dt{dt:g}_{cid}.json')
                assert rec['finished'] and rec['validation']['passed']
                assert rec['source_script_sha256'] == ctrl['source_script_sha256'] == runner_sha
                assert rec['threads'] == nt and rec['dt_ms'] == dt
                assert rec['input_sha256'] == meta[cid]['input_sha256']
                assert rec['ids_sha256'] == meta[cid]['ids_sha256']
                topo = rec['runtime']['physical_cores']
                assert len(topo) == len({(r['socket'],r['core']) for r in topo}) == nt
                assert len({r['socket'] for r in topo}) == 1
                assert rec['runtime']['affinity'][:8] == ctrl['runtime']['affinity']
                assert all(rec['runtime'][k] == ctrl['runtime'][k] for k in
                           ['model','thread_busywait','input_protocol','event_detector_metadata_delay_ms','compartments'])
                with np.load(RUNS/rec['voltage_file']) as f:
                    voltage = f['voltage'].astype(float).ravel()
                with np.load(RUNS/ctrl['voltage_file']) as f:
                    control = f['voltage'].astype(float).ravel()
                es,cs = np.asarray(rec['spike_times_ms']),np.asarray(ctrl['spike_times_ms'])
                events_equal = es.shape == cs.shape and bool(np.allclose(es,cs,rtol=0,atol=1e-5))
                delta = float(np.max(np.abs(voltage-control)))
                assert events_equal and delta <= 1e-3
                assert rec['timing']['steady_ms']['n'] == 2 and np.isfinite(voltage).all()
                row = dict(case_id=cid,threads=nt,dt_ms=dt,
                           duration_s=meta[cid]['duration_ms']/1000.,A=meta[cid]['A'],n=len(voltage),
                           rmse_mv=float(np.sqrt(np.mean((voltage-refs[cid])**2))),
                           reference_spikes=meta[cid]['spike_count'],observed_spikes=rec['spike_count'],
                           steady_ms=rec['timing']['steady_ms']['median_ms'],
                           steady_runs_ms=rec['timing']['steady_ms']['runs_ms'],
                           solver_cpu_ms=rec['timing']['solver_cpu_ms']['median_ms'],
                           control_steady_ms=ctrl['timing']['steady_ms']['median_ms'],
                           voltage_max_abs_vs_8core_mv=delta,events_equal_to_8core=events_equal,
                           voltage_max_abs_vs_original_mv=rec['validation']['voltage_max_abs_mv'],
                           repeat_voltage_max_abs_mv=rec['validation']['repeat_voltage_max_abs_mv'],
                           source_file=f'{name}_{cid}.json')
                group.append(row);rows.append(row)
                runtimes[name] = rec['runtime']
            total = sum(r['steady_ms'] for r in group)
            control_total = sum(r['control_steady_ms'] for r in group)
            ratios = [r['control_steady_ms']/r['steady_ms'] for r in group]
            previous_8 = next(r for r in previous['aggregates'] if r['dt_ms'] == dt and r['threads'] == 8)
            aggregates.append(dict(threads=nt,dt_ms=dt,cases=31,
                total_steady_ms=total,control_8core_ms=control_total,speedup_vs_8core=control_total/total,
                wall_time_factor_vs_8core=total/control_total,
                pooled_rmse_mv=float(np.sqrt(sum(r['n']*r['rmse_mv']**2 for r in group)/sum(r['n'] for r in group))),
                equal_trajectory_rmse_mv=float(np.sqrt(np.mean([r['rmse_mv']**2 for r in group]))),
                time_ratio_vs_dbnn=total/dbnn_ms,
                cases_faster_than_8core=sum(r['steady_ms'] < r['control_steady_ms'] for r in group),
                per_case_speedup_median=float(np.median(ratios)),
                per_case_speedup_min=float(min(ratios)),per_case_speedup_max=float(max(ratios)),
                voltage_max_abs_vs_8core_mv=max(r['voltage_max_abs_vs_8core_mv'] for r in group),
                voltage_max_abs_vs_original_mv=max(r['voltage_max_abs_vs_original_mv'] for r in group),
                all_events_equal_to_8core=all(r['events_equal_to_8core'] for r in group),
                event_count_equal_to_reference=sum(r['observed_spikes']==r['reference_spikes'] for r in group),
                cpu_time_over_wall_time=sum(r['solver_cpu_ms'] for r in group)/total,
                prior_8core_total_ms=previous_8['total_steady_ms'],
                control_8core_retest_factor=control_total/previous_8['total_steady_ms'],
                affinity=runtimes[name]['affinity'],multisplit=runtimes[name]['multisplit']))
    out = dict(complete=True,configurations=9,total_records=len(rows),cases_per_config=31,repeats=2,
               aggregates=aggregates,records=rows,runtime_by_configuration=runtimes,campaign=campaign,
               dbnn_total_steady_ms=dbnn_ms,
               protocol=dict(control='same campaign 8-core run at each dt',threads=[8,16,32],
                             dt_ms=[.25,.5,1.],physical_cores=True,single_socket_per_run=True,
                             timer='sum of 31 per-case median solver + soma recording + 1 ms output interpolation wall times',
                             busywait=0,shared_server=True,compartments=runtimes['nt8_dt0.25']['compartments']),
               fastest_in_extension={str(dt):min((r for r in aggregates if r['dt_ms']==dt),key=lambda r:r['total_steady_ms'])
                                     for dt in [.25,.5,1.]},
               source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [HERE/'run_multisplit_grid.py',HERE/'multisplit_highcore_campaign.py',RES/'selected_manifest.json']})
    (RES/'multisplit_highcore_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for r in aggregates:
        print(f"dt={r['dt_ms']:g} nt={r['threads']} time={r['total_steady_ms']/1000:.3f}s "
              f"vs8={r['speedup_vs_8core']:.3f}x factor={r['wall_time_factor_vs_8core']:.3f} "
              f"rmse={r['pooled_rmse_mv']:.6f} delta={r['voltage_max_abs_vs_8core_mv']:.3g}")


if __name__=='__main__':
    main()
