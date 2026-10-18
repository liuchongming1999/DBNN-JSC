"""Frozen primary DBNN on the exact existing Fig3(c) holding-current cases.

The holding current and graded conductance amplitudes are not inputs to the
primary checkpoint's binary-raster training. These are explicitly labeled
protocol diagnostics, not a replacement test set or a new main accuracy claim.
No weights, bias, reset, or thresholds are fitted here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import sys

import numpy as np
from scipy.signal import fftconvolve
import torch

CP_SHA = '737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); args=ap.parse_args()
    assert platform.node() == 'ps'
    root=args.root.resolve(); out=args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    start=time.monotonic()
    cp=root/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
    assert sha(cp)==CP_SHA
    torch.set_num_threads(1)
    state=torch.load(cp, map_location='cpu', weights_only=False)['model_state']
    state={k:v.double().numpy() for k,v in state.items()}
    bias=float(state['bilinear.bias'].item())
    weight=state['bilinear.weight'][0]
    theta=float(state['v_th'].item()); amp=float(state['reset_amp'].item())
    tau=float(state['tau_reset'].item()); alpha=np.exp(-1/tau)
    def infer(ids, signed):
        n=signed.shape[-1]; age=np.arange(n)[None]
        tr=state['tau_rise'][ids,None]; td=state['tau_decay'][ids,None]
        omega=state['omega'][ids,None]
        k=omega*(1-np.exp(-age/tr))*np.exp(-age/td)
        y=fftconvolve(signed,k,axes=-1,mode='full')[:,:n].T
        first=y.sum(1); second=np.einsum('ni,ij,nj->n',y,weight[np.ix_(ids,ids)],y,optimize=True)
        drive=bias+first+second
        v=np.empty(n); events=[]; r=0.
        for j in range(n):
            v[j]=drive[j]+r
            fire=v[j]>=theta
            if fire:events.append(float(j))
            r=alpha*r-(amp if fire else 0.)
        return y,first,second,drive,v,np.array(events)
    # Validate this independent FP64 evaluation against saved primary outputs,
    # using actual binary rasters before any graded-amplitude diagnostics.
    panel=read(root/'benchmarks/solver_main379_20261016/manifest.json')['cases']
    parity=[]
    for row in [panel[0],next(r for r in panel if r['A']==64)]:
        f=np.load(row['input_file']); ids=f['ids'].astype(int)
        x=f['x']*np.where(ids[:,None]<250,1.,-1.)
        _,_,_,drive,v,ev=infer(ids,x)
        old=np.load(root/'benchmarks/random_init_main379_20261012/predictions'/f"{row['case_id']}.npz")
        p=dict(case_id=row['case_id'],max_drive_gap_mv=float(np.max(abs(drive-old['drive']))),
               max_voltage_gap_mv=float(np.max(abs(v-old['voltage']))),
               events_equal=bool(np.array_equal(ev,old['predicted_events_ms'])))
        assert p['max_voltage_gap_mv']<.002 and p['events_equal'],p
        parity.append(p)
    original=root/'benchmarks/theory_20261005/finite_strength_20261009/l5pc_hold70'
    extension=root/'benchmarks/theory_20261005/active_strength_extension_20261012'
    summary=read(extension/'l5pc_strength_extension.json')
    source=read(original/'l5pc_finite_strength.json')
    fits={r['profile']:r for r in source['fit_profiles']}
    raw={}; records=[]; sources=[]
    for row in summary['rows']:
        name=row['profile']; e=row['strength']; fit=fits[name]
        ids=np.array(fit['ids_e']+fit['ids_i'],int)
        assert len(ids)==64 and len(set(ids))==64
        signed=np.zeros((64,301)); signed[:32,20]=e; signed[32:,20]=-e
        y,first,second,drive,v,events=infer(ids,signed)
        y0,_,_,_,v0,events0=infer(ids,np.zeros_like(signed))
        assert len(events0)==0
        folder=extension/'dt01' if row['independent_low_strength_test'] else original
        refpath=folder/f'{name}_E{e:g}_I{e:g}.npz'
        erpath=folder/f'{name}_E{e:g}_I0.npz'; irpath=folder/f'{name}_E0_I{e:g}.npz'
        bgpath=original/f'{name}_E0_I0.npz'
        ref=np.load(refpath); bg=np.load(bgpath)['voltage_mv'][::10]
        target=ref['voltage_mv'][::10]-bg
        ve=np.load(erpath)['voltage_mv'][::10]-bg
        vi=np.load(irpath)['voltage_mv'][::10]-bg
        c=fit['fixed_c_per_mv']; law=ve+vi+c*ve*vi
        relpred=v-v0
        den=rms(target)
        rec=dict(profile=name,strength=e,epsp_peak_mv=row['epsp_peak_mv'],
                 ipsp_abs_peak_mv=row['ipsp_abs_peak_mv'],
                 existing_figure_case=bool(row['subthreshold'] and row['epsp_peak_mv']<=10),
                 input_encoding='Graded signed impulse amplitude +/-epsilon at the original 64 IDs; extension of the linear filter equations, not a binary-raster production call',
                 predicted_events_ms=events.tolist(),reference_events_ms=ref['spikes_ms'].tolist(),
                 response_rms_mv=den,reference_holding_background_mv=float(bg[0]),
                 frozen_checkpoint_bias_mv=bias,
                 isolated_sum_rmse_mv=rms(target-ve-vi),
                 measured_isolated_bilinear_rmse_mv=rms(target-law),
                 actual_dbnn_above_own_background_rmse_mv=rms(target-relpred),
                 actual_dbnn_drive_above_own_background_rmse_mv=rms(target-first-second),
                 actual_dbnn_absolute_rmse_mv=rms(ref['voltage_mv'][::10]-v))
        for key in ['isolated_sum','measured_isolated_bilinear','actual_dbnn_above_own_background','actual_dbnn_drive_above_own_background']:
            rec[key+'_relative_rms']=rec[key+'_rmse_mv']/max(den,1e-30)
        records.append(rec)
        key=f'{name}_{e:.8g}'
        raw[key+'_reference_above_background']=target
        raw[key+'_features']=y; raw[key+'_linear']=first; raw[key+'_bilinear']=second
        raw[key+'_drive']=drive; raw[key+'_closed_loop_voltage']=v
        raw[key+'_frozen_dbnn_above_background']=relpred
        raw[key+'_isolated_sum']=ve+vi; raw[key+'_measured_bilinear']=law
        for p in [refpath,erpath,irpath,bgpath]:
            sources.append(dict(path=str(p),sha256=sha(p)))
    assert sha(cp)==CP_SHA
    np.savez_compressed(out/'arrays.npz',**raw)
    result=dict(finished=True,execution_host=platform.node(),wall_s=time.monotonic()-start,
                checkpoint=str(cp),checkpoint_sha256=CP_SHA,checkpoint_unchanged=True,
                all_parameters_and_reset_frozen=True,source_sha256=sha(__file__),parity=parity,
                reference_sources=sources,observation_interval_ms=1.,solver_interval_ms=.1,
                reference_holding_current_na=summary['holding_current_na'],records=records,
                scope='Exact pre-existing Fig3(c) protocol diagnostic, not the independent primary evaluation. All archived strength rows retained, existing plotted subset marked only.',
                interpretation=['The neuron receives a somatic holding current not encoded in DBNN; the checkpoint was not trained for this holding-current protocol.',
                                'Graded per-event conductance strengths are represented by graded filter input amplitudes. The primary training raster is binary.',
                                'The above-background error subtracts the solver background and the DBNN own no-input background separately; the absolute error is also retained.',
                                'A mismatch here cannot be identified solely with the passive second-order representation residual.',
                                'No refitting, threshold calibration, case selection, or weight replacement performed.'])
    (out/'complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('ACTIVE_FROZEN_COMPLETE',len(records),'EXISTING_FIGURE',sum(r['existing_figure_case'] for r in records),flush=True)


if __name__=='__main__':main()
