"""Collect actual process/artifact state, without reading dense reference arrays."""
import argparse,hashlib,json,os,time,subprocess
from pathlib import Path
from common import write

def read(p):
    p=Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ValueError,TypeError,ProcessLookupError):return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    rows={}
    for name in ['granule','golgi','basket']:
        out=a.work/'results'/name;state=read(out/'state.json') or {}
        state['process_alive']=alive(state.get('pid'))
        r=dict(state=state,operating_point=read(out/'screen/operating_point.json'),coverage=read(out/'screen/sites.json'),
               screens={},adapter_checks={},sampling={},training={},gpu={},solvers={})
        if r['coverage']:r['coverage'].pop('sites',None)
        for profile in ['article_expsyn','ampa','ampa_nmda']:
            f=read(out/'screen'/(profile+'.json'))
            prog=read(out/'screen'/(profile+'_progress.json'))
            r['screens'][profile]=dict(complete=bool(f and f['finished']),result=f,progress=prog)
        for engine in ['neuron','core','multisplit']:
            r['adapter_checks'][engine]=read(out/'adapter_checks'/(engine+'_parity.json'))
        r['training_preflight']=read(out/'training_preflight.json')
        r['input_strength_correction']=read(out/'input_strength_correction.json')
        r['mechanism_source_audit']=read(a.work/'mechanisms'/name/'source_audit.json')
        model_id={'granule':265584,'golgi':266806,'basket':2018018}[name]
        git=a.work/'vendor'/str(model_id)
        if (git/'.git').exists():r['vendor_commit']=subprocess.check_output(['git','-C',str(git),'rev-parse','HEAD'],text=True).strip()
        else:r['vendor_commit']=(read(a.work/'vendor_versions.json') or {}).get(name,{}).get('commit')
        r['inference_preflight']=read(out/'inference_preflight.json')
        r['training_regime_revision']=read(out/'training_regime_revision.json')
        parallel=read(out/'parallel_nmda_state.json')
        if parallel:parallel['process_alive']=alive(parallel.get('pid'))
        r['parallel_nmda_state']=parallel
        r['reference_step_audit']=read(out/'ampa/reference_step_audit/complete.json')
        r['archived_6s_only_result']=read(out/'diagnostics/low_spike_6s_only/gpu/complete.json')
        for profile in ['ampa','ampa_nmda']:
            r['sampling'][profile]={}
            for kind in ['train_1.0ms','train_0.5ms','train_0.25ms','test']:
                panel=a.data/name/profile/kind
                r['sampling'][profile][kind]=dict(complete=read(panel/'complete.json') is not None,progress=read(panel/'status.json'),selection=read(panel/'selection.json'))
            r['training'][profile]={}
            for dt in [1.,.5,.25]:
                fit=out/profile/f'dt{dt}ms';sel=read(fit/'final_selection.json')
                r['training'][profile][str(dt)]=dict(complete=bool(sel and sel['finished']),progress=read(fit/'progress.json'),selection=sel)
            r['gpu'][profile]=read(out/profile/'gpu/complete.json')
            r['solvers'][profile]={}
            r.setdefault('solver_progress',{})[profile]={}
            for f in (out/profile/'solvers').glob('*/progress.json'):
                r['solver_progress'][profile][f.parent.name]=read(f)
            for f in (out/profile/'solvers').glob('*/complete.json'):
                if '_smoke' not in f.parent.name:r['solvers'][profile][f.parent.name]=read(f)
            for f in (out/profile/'solvers').glob('*/invalid.json'):
                if '_smoke' not in f.parent.name:r['solvers'][profile][f.parent.name]=dict(invalid=read(f))
        from paired_conditions import paired_conditions
        r['paired_ampa_nmda']=paired_conditions(a.work,a.data,name)
        rows[name]=r
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.work.glob('*.py')}
    noncerebellar=Path('/home/lcm/DBNN_noncerebellar_20261002')
    extension=read(noncerebellar/'snapshot.json')
    if extension:
        for name,r in extension['models'].items():
            r['state']=read(noncerebellar/'results'/name/'state.json')
            if r['state']:r['state']['process_alive']=alive(r['state'].get('pid'))
            r['adapter_checks']={engine:read(noncerebellar/'results'/name/'adapter_checks'/(engine+'_parity.json')) for engine in ['neuron','core','multisplit']}
    baseline=read(Path('/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/baselines_20260930/aligned/elm_0.25ms/progress.json'))
    write(a.work/'snapshot.json',dict(collected=time.strftime('%Y-%m-%dT%H:%M:%S%z'),models=rows,code_sha256=hashes,
                                    noncerebellar=extension,baseline_elm_025=baseline,
                                    resource='Shared 5090 server. GPU4/5/6: cerebellar main controllers; GPU1/7: independent NMDA fit/evaluation; GPU3: non-cerebellar extension; old ELM GPU2 and old publication benchmark GPU0 retained. CPU multisplit48–55, independent CoreNEURON CPU56–60, cerebellar reference workers0–47/96–127, non-cerebellar CPU64–71. Physical-core pools do not overlap.'))

if __name__=='__main__':main()
