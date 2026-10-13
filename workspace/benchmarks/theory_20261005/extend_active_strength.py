"""Independent weak-input checks and a physical Taylor control for active L5PC.

The measured bilinear coefficients are loaded verbatim, never refitted. The
Taylor coefficients are extracted from separate small positive conductance
scales of the complete NEURON model, including its original voltage-gated
channels. They are held fixed over all plotted test strengths. Two extraction
scales, two polynomial degrees, and a half integration step check stability.
All simulation work is performed on the supplied experiment-server root.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

import numpy as np


PROFILES = ("proximal", "distributed")
TARGET_PEAKS_MV = np.array([.21, .3, .5, .75, 1., 1.5, 2., 2.5, 3., 3.5, 3.95])
CALIBRATION_NODES = np.array([.1, .2, .3, .4, .55, .7, .85, 1.])
EXTRACTION_SCALES = (.08, .04)


def rms(x):
    """Normalized trapezoidal time RMS, matching the original control."""
    w = np.ones(x.shape[-1]); w[[0, -1]] = .5; w /= w.sum()
    return float(np.sqrt(w @ np.square(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root / "benchmarks/theory_20261005"))
    import run_l5pc as runner

    source = json.loads((args.original / "l5pc_finite_strength.json").read_text())
    fits = {r["profile"]: r for r in source["fit_profiles"] if r["profile"] in PROFILES}
    bias = fits[PROFILES[0]]["holding_current_na"]
    assert all(r["holding_current_na"] == bias for r in fits.values())
    started = time.perf_counter()
    newly_simulated = 0

    # Reuse the recorded original voltages without rerunning or changing them.
    def old(name, e, i):
        with np.load(args.original / (runner.key(name, e, i) + ".npz")) as f:
            return {k: np.array(f[k], copy=True) for k in f.files}

    low_strengths = {}
    for name in PROFILES:
        base = old(name, 0., 0.)["voltage_mv"]
        peak = (old(name, .25, 0.)["voltage_mv"] - base).max()
        # This selects new input amplitudes; it does not fit a prediction.
        levels = np.round(TARGET_PEAKS_MV * .25 / peak, 8)
        assert not any(np.isclose(e, [.25, .5, 1.]).any() for e in levels)
        low_strengths[name] = levels.tolist()

    # A zero-input run at the matching step provides the common baseline.
    scales = sorted(set(float(s*q) for s in EXTRACTION_SCALES
                        for q in CALIBRATION_NODES))
    tasks = []
    for name in PROFILES:
        tasks.append((name, 0., 0.))
        tasks += [(name, e, e) for e in scales]
        for e in low_strengths[name]:
            tasks += [(name, e, 0.), (name, 0., e), (name, e, e)]

    def collect(jobs, dt, tag):
        nonlocal newly_simulated
        result = {}
        cache = args.out / tag; cache.mkdir(exist_ok=True)
        remaining = []
        for job in jobs:
            p = cache / (runner.key(*job) + ".npz")
            if p.exists():
                with np.load(p) as f:
                    r = {k: np.array(f[k], copy=True) for k in f.files}
                r["profile"] = job[0]
                result[runner.key(*job)] = r
            else:
                remaining.append(job)
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=runner.init,
                                 initargs=(str(args.root), -70., dt, bias)) as pool:
            futures = [pool.submit(runner.simulate, job) for job in remaining]
            for f in as_completed(futures):
                r = f.result(); k = runner.key(r["profile"], r["strength_e"], r["strength_i"])
                np.savez_compressed(cache / (k + ".npz"),
                                    **{k:v for k,v in r.items() if k != "profile"})
                result[k] = r; newly_simulated += 1
                if len(result) % 16 == 0:
                    print(f"{tag}: {len(result)}/{len(jobs)} trajectories available", flush=True)
        return result

    coarse = collect(tasks, .1, "dt01")

    def taylor(results, name, scale, degree):
        base = results[runner.key(name, 0., 0.)]["voltage_mv"]
        responses = np.array([results[runner.key(name, float(scale*q), float(scale*q))]
                              ["voltage_mv"] - base for q in CALIBRATION_NODES])
        design = np.stack([CALIBRATION_NODES ** j for j in range(1, degree+1)], axis=1)
        coefficients = np.linalg.lstsq(design, responses, rcond=None)[0]
        fit_gap = design @ coefficients - responses
        return (coefficients[0]/scale, coefficients[1]/scale**2,
                float(np.max(np.abs(fit_gap))))

    # Keep the smaller extraction interval as the reported physical expansion.
    expansion = {name: taylor(coarse, name, .04, 5) for name in PROFILES}
    controls = []
    raw = {}
    rows = []
    originals = {(r["profile"], r["strength"]):r for r in source["rows"]}
    for name in PROFILES:
        bg = coarse[runner.key(name, 0., 0.)]["voltage_mv"]
        ref0 = old(name, 0., 0.)
        assert np.max(np.abs(bg - ref0["voltage_mv"])) < 1e-9
        assert np.array_equal(coarse[runner.key(name, 0., 0.)]["ids_e"], fits[name]["ids_e"])
        assert np.array_equal(coarse[runner.key(name, 0., 0.)]["ids_i"], fits[name]["ids_i"])
        v1, v2, residual = expansion[name]
        raw[name + "_time_ms"] = ref0["time_ms"]
        raw[name + "_V1_mv"] = v1; raw[name + "_V2_mv"] = v2
        raw[name + "_baseline_mv"] = bg
        for scale, degree in [(.08, 5), (.04, 4)]:
            other1, other2, gap = taylor(coarse, name, scale, degree)
            controls.append(dict(profile=name, control="extraction_interval" if scale==.08 else "polynomial_degree",
                                 scale=scale, degree=degree,
                                 V1_rms_gap_mv=rms(v1-other1), V2_rms_gap_mv=rms(v2-other2),
                                 V1_relative_rms_gap=rms(v1-other1)/rms(v1),
                                 V2_relative_rms_gap=rms(v2-other2)/rms(v2),
                                 extraction_fit_max_abs_gap_mv=gap))
        c = fits[name]["fixed_c_per_mv"]
        levels = sorted(low_strengths[name] + [r["strength"] for r in source["rows"] if r["profile"]==name])
        for e in levels:
            is_new = e in low_strengths[name]
            def get(se, si):
                return coarse[runner.key(name, se, si)] if is_new else old(name, se, si)
            er, ir, jr = get(e, 0.), get(0., e), get(e, e)
            ve, vi, full = [r["voltage_mv"]-bg for r in (er, ir, jr)]
            additive = ve + vi; bilinear = additive + c*ve*vi
            first = e*v1; second = first + e*e*v2
            denom = rms(full)
            ep, ip = float(ve.max()), float(-vi.min())
            events = sum(len(r["spikes_ms"]) for r in (er, ir, jr))
            row = dict(profile=name, strength=float(e), epsp_peak_mv=ep,
                       ipsp_abs_peak_mv=ip, joint_response_rms_mv=denom,
                       all_reference_events=events, subthreshold=(events==0),
                       physiological_band=(events==0 and .2<=ep<=10 and ip<=3),
                       common_band=(events==0 and .2<=ep<=6 and ip<=3),
                       independent_low_strength_test=is_new,
                       peak_total_input_conductance_bound_ns=float(e*(32*.188+32*.08)))
            for method, predicted in [("additive",additive), ("fixed_bilinear",bilinear),
                                      ("physical_first",first), ("physical_second",second)]:
                row[method+"_rmse_mv"] = rms(predicted-full)
                row[method+"_relative_rms"] = row[method+"_rmse_mv"]/max(denom,1e-15)
            if not is_new:
                before = originals[name, e]
                assert abs(row["fixed_bilinear_relative_rms"]-before["fixed_bilinear_relative_rms"]) < 1e-12
            else:
                assert row["subthreshold"]
                assert not any(np.isclose(e, np.ravel(fits[name]["fit_strength_pairs"])))
            rows.append(row)
            if events==0 and ep<=10:
                raw[name+f"_strength_{e:g}_responses_mv"] = np.array([ve,vi,full,additive,bilinear,first,second])

    # Half-step runs use the identical current and coefficients. Cover both a
    # weak and a stronger plotted input for each spatial configuration.
    representative = {name:[low_strengths[name][4], 1.5] for name in PROFILES}
    fine_jobs = []
    for name in PROFILES:
        fine_jobs.append((name,0.,0.))
        fine_jobs += [(name,e,e) for e in scales]
        for e in representative[name]:
            fine_jobs += [(name,e,0.),(name,0.,e),(name,e,e)]
    fine = collect(fine_jobs, .05, "dt005")
    step_checks = []
    for name in PROFILES:
        f1, f2, gap = taylor(fine, name, .04, 5)
        v1, v2, _ = expansion[name]
        controls.append(dict(profile=name,control="integration_half_step", scale=.04,degree=5,
                             V1_rms_gap_mv=rms(v1-f1),V2_rms_gap_mv=rms(v2-f2),
                             V1_relative_rms_gap=rms(v1-f1)/rms(v1),
                             V2_relative_rms_gap=rms(v2-f2)/rms(v2),
                             extraction_fit_max_abs_gap_mv=gap))
        fbase = fine[runner.key(name,0.,0.)]["voltage_mv"]
        for e in representative[name]:
            ve,vi,full = [fine[runner.key(name,se,si)]["voltage_mv"]-fbase
                          for se,si in [(e,0.),(0.,e),(e,e)]]
            den=rms(full); before=next(r for r in rows if r["profile"]==name and r["strength"]==e)
            bi = rms(ve+vi+fits[name]["fixed_c_per_mv"]*ve*vi-full)/den
            ph = rms(e*f1+e*e*f2-full)/den
            original_volt = coarse[runner.key(name,e,e)]["voltage_mv"] if e in low_strengths[name] else old(name,e,e)["voltage_mv"]
            step_checks.append(dict(profile=name,strength=e,
                                    max_abs_joint_voltage_gap_mv=float(np.max(np.abs(fine[runner.key(name,e,e)]["voltage_mv"]-original_volt))),
                                    bilinear_change_percentage_points=100*(bi-before["fixed_bilinear_relative_rms"]),
                                    physical_second_change_percentage_points=100*(ph-before["physical_second_relative_rms"])))

    # Control reporting uses the same fixed coefficients to predict held-out
    # amplitudes; stability is judged on voltages as well as coefficient RMS.
    stability = []
    for name in PROFILES:
        v1,v2,_=expansion[name]
        eligible=[r for r in rows if r["profile"]==name and r["subthreshold"] and r["epsp_peak_mv"]<=10]
        for scale,degree in [(.08,5),(.04,4)]:
            c1,c2,_=taylor(coarse,name,scale,degree)
            differences=[rms(r["strength"]*(v1-c1)+r["strength"]**2*(v2-c2)) for r in eligible]
            stability.append(dict(profile=name,scale=scale,degree=degree,
                                  max_test_prediction_rms_gap_mv=max(differences)))
    data = dict(finished=True,execution_host=platform.node(),created_local_time=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                wall_s=time.perf_counter()-started,newly_simulated_trajectories=newly_simulated,
                original_reference_trajectories_reused=True,readout_refitted=False,
                original_source_sha256=hashlib.sha256((args.original/"l5pc_finite_strength.json").read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fixed_coefficients_per_mv={k:fits[k]["fixed_c_per_mv"] for k in PROFILES},
                holding_current_na=bias,model=source["model"],synapses=source["synapses"],
                solver_dt_ms=.1,observation_ms=.1,warmup_ms=500.,duration_ms=300.,
                low_test_strengths=low_strengths,
                taylor=dict(definition="V1=d(V_epsilon-V_0)/d epsilon at 0; V2=0.5*d2(V_epsilon-V_0)/d epsilon2 at 0; prediction=epsilon*V1+epsilon^2*V2",
                            common_E_I_scale=True,reported_extraction_max_strength=.04,
                            extraction_nodes=CALIBRATION_NODES.tolist(),degree=5,
                            extraction_uses_positive_strengths_only=True,
                            extraction_strengths=scales,held_fixed_over_test_strengths=True,
                            includes_original_active_channel_response=True,
                            numerical_extraction_controls=controls,
                            held_out_prediction_stability=stability),
                half_step_checks=step_checks,rows=rows)
    (args.out/"l5pc_strength_extension.json").write_text(json.dumps(data,indent=2)+"\n")
    np.savez_compressed(args.out/"l5pc_strength_extension.npz",**raw)
    print(json.dumps({"finished":True,"wall_s":data["wall_s"],"new_simulations":newly_simulated,
                      "new_test_peaks":{k:[r["epsp_peak_mv"] for r in rows if r["profile"]==k and r["independent_low_strength_test"]] for k in PROFILES},
                      "coefficient_controls":controls,"prediction_stability":stability,
                      "half_step_checks":step_checks},indent=2),flush=True)


if __name__=="__main__":
    main()
