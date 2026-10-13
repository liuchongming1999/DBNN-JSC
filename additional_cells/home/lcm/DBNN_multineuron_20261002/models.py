"""Isolated ModelDB adapters; original morphology/channels retained."""
from pathlib import Path
import importlib, os, sys
import numpy as np

REGISTRY = {
    'granule': dict(modeldb=265584, variant='01_GrC_2020_regular', module='Cereb_GrC_regular', cls='Grc_regular', args=[0], celsius=32., v_init=-70., article_label='GrC_1', author='Masoli et al. (2020)'),
    'golgi': dict(modeldb=266806, variant='Morphology_1', module='Golgi2020_morpho_1', cls='Golgi_morpho_1', args=[], celsius=32., v_init=-65., article_label='GoC_1', author='Masoli et al. (2020)'),
    'basket': dict(modeldb=2018018, variant='', module='basket', cls='Basket', args=[], celsius=32., article_label='BC_2', author='Masoli et al. (2025)'),
}

def build(work, name, profile='ampa', sites_file=None):
    import neuron
    from neuron import h, coreneuron
    work=Path(work).resolve(); spec=REGISTRY[name]
    src=work/'vendor'/str(spec['modeldb'])/spec['variant']
    mech=work/'mechanisms'/name
    neuron.load_mechanisms(str(mech))
    os.environ['CORENEURONLIB']=str(mech/'x86_64/libcorenrnmech.so')
    os.chdir(src);sys.path.insert(0,str(src))
    h.load_file('stdrun.hoc')
    cell=getattr(importlib.import_module(spec['module']),spec['cls'])(*spec['args'])
    soma=cell.soma[0]; dend=list(cell.dend)
    h.celsius=spec['celsius'];h.CVode().active(0);h.secondorder=0
    h.dt=.025;h.steps_per_ms=40
    coreneuron.enable=False
    pc=h.ParallelContext();pc.nthread(1);pc.thread_busywait(0);pc.set_maxstep(10.)
    clamp=h.IClamp(soma(.5));clamp.delay=0.;clamp.dur=1e9;clamp.amp=0.
    if sites_file is None:
        # 250 E and 250 I positions each cover the complete dendritic length.
        # Distinct positions may share an original electrical segment; report it.
        lens=np.asarray([s.L for s in dend]);cum=np.cumsum(lens)
        positions=[]
        for typ,phase in [('exc',.25),('inh',.75)]:
            q=(np.arange(250)+phase)/250*cum[-1]
            for val in q:
                i=int(np.searchsorted(cum,val));prev=0. if i==0 else cum[i-1]
                positions.append(dict(section_index=i,x=float(np.clip((val-prev)/lens[i],.001,.999)),type=typ))
    else:
        import json
        positions=json.loads(Path(sites_file).read_text())['sites']
    h.distance(0.,.5,sec=soma)
    for pos in positions:pos['distance_um']=float(h.distance(pos['x'],sec=dend[pos['section_index']]))
    syns=[];ncs=[];nmda=[];nmda_ncs=[]
    for pos in positions:
        sec=dend[pos['section_index']]; x=pos['x'];exc=pos['type']=='exc'
        if profile=='article_expsyn':
            s=h.ExpSyn(sec(x));s.tau=4. if exc else 5.;s.e=0. if exc else -75.
        else:
            s=h.Exp2Syn(sec(x));s.tau1=.5 if exc else 1.;s.tau2=5. if exc else 10.;s.e=0. if exc else -75.
        nc=h.NetCon(None,s);nc.delay=10.;nc.weight[0]=0.;syns.append(s);ncs.append(nc)
        if exc and profile=='ampa_nmda':
            ns=h.DBNN_NMDA(sec(x)); nn=h.NetCon(None,ns);nn.delay=10.;nn.weight[0]=0.;nmda.append(ns);nmda_ncs.append(nn)
        else:nmda.append(None);nmda_ncs.append(None)
    spike_nc=h.NetCon(soma(.5)._ref_v,None,sec=soma);spike_nc.threshold=-20.
    vv=h.Vector().record(soma(.5)._ref_v);tt=h.Vector().record(h._ref_t);sp=h.Vector();spike_nc.record(sp)
    return dict(h=h,cell=cell,soma=soma,dend=dend,pc=pc,clamp=clamp,syns=syns,ncs=ncs,nmda=nmda,nmda_ncs=nmda_ncs,
                vv=vv,tt=tt,sp=sp,spike_nc=spike_nc,sites=positions,profile=profile,spec=spec)

def run(env, duration_ms, ids=(), x=None, input_dt=1., weight=0., inhibition_ratio=.5, holding=0., dt=.025, warmup=2000., engine='neuron', site_weights=None):
    """Exact event replay; timer covers solve only, schedule/copies reported separately."""
    import time
    from neuron import coreneuron
    h=env['h'];pc=env['pc'];coreneuron.enable=False
    for nc in env['ncs']:nc.weight[0]=0.
    for nc in env['nmda_ncs']:
        if nc is not None:nc.weight[0]=0.
    env['clamp'].amp=float(holding)
    h.dt=float(dt);h.steps_per_ms=1/dt;h.finitialize(env['spec'].get('v_init',-65.))
    key=(float(holding),float(dt),float(warmup));cache=env.setdefault('warm_states',{})
    if key in cache:
        cache[key].restore(1);h.fcurrent();h.frecord_init()
    else:
        h.fcurrent();h.frecord_init();h.continuerun(warmup)
        saved=h.SaveState();saved.save();cache[key]=saved
    for j,site in enumerate(ids):
        w=float(site_weights[j]) if site_weights is not None else float(weight)*(1. if site<250 else inhibition_ratio)
        env['ncs'][int(site)].weight[0]=w
        if env['nmda_ncs'][int(site)] is not None:env['nmda_ncs'][int(site)].weight[0]=w*.5
    start=time.perf_counter()
    if x is not None:
        for j,site in enumerate(ids):
            for k in np.flatnonzero(x[j]):
                for _ in range(int(x[j,k])):
                    t=float(warmup+k*input_dt);env['ncs'][int(site)].event(t)
                    if env['nmda_ncs'][int(site)] is not None:env['nmda_ncs'][int(site)].event(t)
    schedule_ms=(time.perf_counter()-start)*1000
    coreneuron.enable=engine=='core';coreneuron.gpu=False
    start=time.perf_counter()
    if engine in ('core','multisplit'):pc.psolve(warmup+duration_ms)
    else:h.continuerun(warmup+duration_ms)
    solver_ms=(time.perf_counter()-start)*1000
    coreneuron.enable=False
    times=env['tt'].as_numpy().copy()-warmup;voltage=env['vv'].as_numpy().copy()
    spikes=env['sp'].as_numpy().copy()-warmup
    spikes=spikes[(spikes>=0)&(spikes<=duration_ms)]
    assert np.isfinite(voltage).all(), 'Non-finite voltage'
    return dict(time_ms=times,voltage=voltage,spike_times=spikes,solver_ms=solver_ms,schedule_ms=schedule_ms)

def register_core_sources(env):
    """Register all voltage-source detectors, including the vendor GoC detector.

    The native GoC class retains its soma(1) NetCon for recording. CoreNEURON
    exports every such source, even when it has no target, so this source needs
    its own gid as well as our common soma(.5) detector. No channel or event
    parameters are changed by this bookkeeping operation.
    """
    if env.get('core_sources_registered'):
        return
    pc=env['pc']
    pc.set_gid2node(0,0);pc.cell(0,env['spike_nc'])
    native=getattr(env['cell'],'nc_spike',None)
    if native is not None:
        pc.set_gid2node(1,0);pc.cell(1,native)
    env['core_sources_registered']=True
