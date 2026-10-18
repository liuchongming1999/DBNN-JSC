"""Paper-sized vector figures and tables from frozen, audited result files.

Run on the author's workspace; numerical experiments are read, never rerun.
The supplied PDF figures make the LaTeX project independently compilable.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent / 'theory_data'
E = ROOT / 'evidence'
F = ROOT / 'figures'
D = json.loads((E / 'main379_results.json').read_text())
P = json.loads((E / 'passive_finite_strength.json').read_text())
PAIR = json.loads((E / 'pair_order_checks.json').read_text())
ACTIVE = json.loads((E / 'l5pc_strength_extension.json').read_text())

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'lines.linewidth': 1.25,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 220,
})
WIDTH = 119 / 25.4
COLORS = dict(gpu='#008779', cpu1='#8A6BB5', cpu8='#346AAA',
              tcn='#CB513D', elm='#475668', lstm='#218FAF',
              neuron8='#D68A28', core1='#BD4C73', reduce='#97713D')
NAMES = dict(gpu='DBNN GPU', cpu1='DBNN CPU, 1 core', cpu8='DBNN CPU, 8 cores',
             tcn='TCN GPU', elm='Branch-ELM GPU', lstm='LSTM GPU',
             neuron8='NEURON, 8 cores', core1='CoreNEURON, 1 core',
             reduce='Neuron_Reduce, 1 core')
TABLE_CITATIONS = dict(tcn='beniaguev2021', elm='spieler2024',
                       lstm='hochreiter1997,spieler2024',
                       neuron8='hines1997,hines2008', core1='kumbhar2019',
                       reduce='amsalem2020')

def save(fig, name):
    fig.savefig(F / (name+'.pdf'), bbox_inches=None)
    fig.savefig(F / (name+'.png'), bbox_inches=None)
    numbered={'frontier':1,'pair_order':2,'finite_strength':3,
              'voltage_examples':5,'events':6,'duration':7}
    if name in numbered:
        import shutil
        shutil.copy2(F/(name+'.pdf'),F/f'Fig{numbered[name]}.pdf')
    plt.close(fig)

def grid(ax):
    ax.grid(which='major', color='#dbe2e8', lw=.6, alpha=.7)
    ax.set_axisbelow(True)

def frontier():
    fig, ax = plt.subplots(figsize=(WIDTH, 4.55))
    fig.subplots_adjust(left=.16, bottom=.13, top=.96, right=.96)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(.025,7.3); ax.set_ylim(8,2.0e6)
    ax.set_xticks([.05,.1,.2,.5,1,2,5]); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Pooled subthreshold RMSE (mV)')
    ax.set_ylabel('Sum of per-input median computation times (ms)')
    grid(ax)
    ax.text(.025,.024,'← Lower voltage error  ·  Lower compute time ↓',
            transform=ax.transAxes,fontsize=7,color='#718493',
            ha='left',va='bottom',zorder=6)

    # Pixel offsets keep leader lengths and text spacing consistent on log axes.
    # All arrows and leaders terminate at the measured operating points.
    placements = {
        'gpu': (17, 0, 'left'),
        'cpu8': (17, -26, 'left'),
        'cpu1': (17, -18, 'left'),
        'lstm': (20, 10, 'left'),
        'tcn': (13, 2, 'left'),
        'elm': (18, 2, 'left'),
        'neuron8': (-20, -15, 'right'),
        'core1': (-20, 15, 'right'),
        'reduce': (0, 30, 'right'),
    }
    direct_names = dict(
        gpu='DBNN GPU\nOurs', cpu1='DBNN CPU, 1 core\nOurs',
        cpu8='DBNN CPU, 8 cores\nOurs',
        tcn='TCN GPU\nBeniaguev et al. 2021',
        elm='Branch-ELM GPU\nSpieler et al. 2024',
        lstm='LSTM GPU\nSpieler et al. 2024',
        neuron8='NEURON, 8 cores\nHines et al. 2008',
        core1='CoreNEURON, 1 core\nKumbhar et al. 2019',
        reduce='Neuron_Reduce, 1 core\nAmsalem et al. 2020')
    points={k:(D['models'][k]['accuracy']['pooled_rmse_mv'],
               D['models'][k]['timing']['wall_ms']) for k in COLORS}
    speed_ratio=points['neuron8'][1]/points['gpu'][1]
    # An unobstructed curve connects the actual source and destination, with
    # enough marker clearance to distinguish it from the short label leaders.
    comparison=ax.annotate('',xy=points['gpu'],xytext=points['neuron8'],
        arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3,rad=.26',
                        color='#66788B',lw=1.05,mutation_scale=10,
                        shrinkA=5,shrinkB=5),zorder=2)
    for k in COLORS:
        x,y=points[k]
        marker = 'o' if k in ('gpu','cpu1','cpu8') else ('s' if k in ('tcn','elm','lstm') else ('v' if k=='neuron8' else '^'))
        ax.scatter(x,y,s=35,marker=marker,c=COLORS[k],edgecolor='white',lw=.7,zorder=5)
        dx,dy,ha=placements[k]
        title,source=direct_names[k].split('\n')
        box=VPacker(children=[
            TextArea(title,textprops=dict(fontsize=8,color=COLORS[k],weight='bold')),
            TextArea(source,textprops=dict(fontsize=7,color=COLORS[k]))],
            align=ha,pad=0,sep=2)
        label=AnnotationBbox(box,(x,y),xybox=(dx,dy),xycoords='data',
                            boxcoords='offset points',frameon=False,
                            box_alignment=(0 if ha=='left' else 1,.5),
                            arrowprops=dict(arrowstyle='-',lw=.6,color=COLORS[k],
                                            alpha=.65,shrinkA=3,shrinkB=5),zorder=6)
        ax.add_artist(label)


    # Position the ratio beside the middle of the displayed quadratic Bezier,
    # rather than using a free-floating box that hides part of the arrow.
    fig.canvas.draw()
    verts=comparison.arrow_patch.get_path().vertices
    t=.51
    mid=(1-t)**2*verts[0]+2*(1-t)*t*verts[1]+t*t*verts[2]
    ratio_anchor=ax.transData.inverted().transform(mid)
    ax.annotate(f'{speed_ratio:,.0f}'+r'$\times$'+'\nTime ratio',
                xy=ratio_anchor,xytext=(-11,0),textcoords='offset points',
                ha='right',va='center',fontsize=8,color='#4F6173',
                linespacing=1.35,zorder=6)
    save(fig,'frontier')

def duration():
    """Runtime and pooled voltage error on the same duration strata."""
    fig,(ax,err)=plt.subplots(2,1,figsize=(WIDTH,4.8),sharex=True,
                             gridspec_kw=dict(height_ratios=[1.4,1]))
    fig.subplots_adjust(left=.16,bottom=.10,top=.80,right=.97,hspace=.27)
    ts=sorted(int(k) for k in D['durations'])
    keys=['gpu','cpu8','cpu1','lstm','tcn','elm','neuron8','core1','reduce']
    for k in keys:
        ds=[D['durations'][str(t)][k] for t in ts]
        y=np.array([z['median_ms'] for z in ds])
        lo=np.array([z['q1_ms'] for z in ds]);hi=np.array([z['q3_ms'] for z in ds])
        ax.plot(ts,y,'o-',ms=3,color=COLORS[k],label=NAMES[k])
        ax.fill_between(ts,lo,hi,color=COLORS[k],alpha=.12,lw=0)
    ax.set_yscale('log');ax.set_ylim(.025,12000)
    ax.set_ylabel('Median computation time (ms)')
    ax.set_title('(a) Computation time',loc='left',pad=5)
    grid(ax)
    fig.legend(*ax.get_legend_handles_labels(),fontsize=8,ncol=2,
               frameon=False,loc='upper center',bbox_to_anchor=(.54,.995),
               handlelength=1.2,columnspacing=.8,labelspacing=.28)
    y16=D['durations']['16']['gpu']['median_ms']
    ratio16=D['durations']['16']['neuron8']['median_ms']/y16
    ax.annotate(f'{ratio16:,.0f}'+r'$\times$'+' at 16 s',xy=(16,y16),
                xytext=(-8,20),textcoords='offset points',ha='right',
                color=COLORS['gpu'],fontsize=8,
                arrowprops=dict(arrowstyle='->',color=COLORS['gpu'],lw=.7))
    # Identical CPU/GPU DBNN predictions and NEURON/CoreNEURON predictions
    # share one error curve. Recompute pooling from per-input SSE and counts.
    for k in ['gpu','lstm','tcn','elm','neuron8','reduce']:
        ys=[]
        for t in ts:
            rec=[r for r in D['models'][k]['records'] if r['duration_s']==t]
            rmse=np.sqrt(sum(r['voltage']['sse'] for r in rec)/
                         sum(r['voltage']['n'] for r in rec))
            frozen=D['models'][k]['groups']['duration_s'][str(t)]['pooled_rmse_mv']
            assert np.isclose(rmse,frozen,rtol=1e-12,atol=1e-12)
            ys.append(rmse)
        err.plot(ts,ys,'o-',ms=3,color=COLORS[k])
    err.set_xscale('log',base=2);err.set_xticks(ts)
    err.set_xticklabels([str(t) for t in ts])
    err.set_yscale('log');err.set_ylim(.12,6.5)
    err.set_yticks([.2,.5,1,2,5]);err.yaxis.set_major_formatter(ScalarFormatter())
    err.set_xlabel('Simulated duration (s)')
    err.set_ylabel('Pooled subthreshold RMSE (mV)')
    err.set_title('(b) Voltage prediction error',loc='left',pad=5)
    err.tick_params(which='minor',labelleft=False)
    grid(err)
    save(fig,'duration')

def events():
    """Aggregate event comparison; input-family results are in the main table."""
    keys=['gpu','elm','neuron8','lstm','tcn','reduce']
    fig,ax=plt.subplots(figsize=(WIDTH,2.2))
    fig.subplots_adjust(left=.33,right=.97,bottom=.25,top=.96)
    for i,k in enumerate(keys):
        a=D['models'][k]['accuracy'];f=a['events']['f1'];ci=a['bootstrap_95ci']['f1']
        ax.errorbar(f,i,xerr=[[f-ci[0]],[ci[1]-f]],fmt='o',color=COLORS[k],ms=4,capsize=2)
    ax.set_yticks(range(len(keys)));ax.set_yticklabels([NAMES[k] for k in keys]);ax.invert_yaxis()
    ax.set_xlim(0,.65);ax.set_xlabel('Event F1 (one-to-one matching, ±5 ms)');grid(ax)
    save(fig,'events')

def examples():
    import hashlib
    selection=json.loads((E/'selected_examples.json').read_text())
    cases=[e['case_id'] for e in selection]
    src=E/'examples'
    chosen={e['case_id']:e for e in selection}
    fig=plt.figure(figsize=(WIDTH,3.0))
    outer=fig.add_gridspec(1,2,left=.145,right=.985,bottom=.16,top=.865,wspace=.47)
    for j,cid in enumerate(cases):
        d=np.load(src/(cid+'.npz'))
        ref=np.load(src/(cid+'_native0.0125.npz'))
        sol=np.load(src/(cid+'_native0.25.npz'))
        raster=np.load(src/(cid+'_input.npz'))
        x,ids=raster['x'],raster['ids']
        assert x.shape==(500,8001) and set(ids.tolist())==set(range(500))
        assert hashlib.sha256(x.tobytes()).hexdigest()==chosen[cid]['dbnn_record']['input_sha256']
        sub=outer[0,j].subgridspec(3,1,height_ratios=[.85,1.5,.72],hspace=.09)
        inputs=fig.add_subplot(sub[0])
        voltage=fig.add_subplot(sub[1],sharex=inputs)
        events=fig.add_subplot(sub[2],sharex=inputs)
        row,step=np.nonzero(x);sites=ids[row];time_s=step/1000
        for take,col in [(sites<250,'#c7665c'),(sites>=250,'#4c8ca8')]:
            inputs.scatter(time_s[take],sites[take],marker='|',s=.7,
                           linewidths=.16,color=col,alpha=.55,rasterized=True)
        inputs.axhline(249.5,color='white',lw=.8)
        inputs.set_ylim(-5,505);inputs.set_yticks([0,250,500])
        inputs.set_ylabel('Input site',fontsize=7)
        inputs.text(.99,.94,'Inhibitory',transform=inputs.transAxes,
                    ha='right',va='top',fontsize=6.2,color='#3c7189',
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1))
        inputs.text(.99,.44,'Excitatory',transform=inputs.transAxes,
                    ha='right',va='top',fontsize=6.2,color='#a1433b',
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1))
        n=len(d['reference_events'])
        inputs.set_title(f'({chr(97+j)}) {n}-spike response',
                         loc='left',fontsize=8,pad=5)
        # Native solver samples preserve actual action-potential peaks.
        voltage.plot(ref['time_ms']/1000,ref['voltage'],c='#35424C',lw=.65,
                     label='Reference, 0.0125 ms',zorder=2)
        voltage.plot(sol['time_ms']/1000,sol['voltage'],c=COLORS['neuron8'],
                     lw=.65,alpha=.9,label='NEURON, 0.25 ms',zorder=2)
        voltage.plot(d['time_ms']/1000,d['dbnn_voltage'],c=COLORS['gpu'],
                     lw=.7,label='DBNN, 1 ms',zorder=3)
        voltage.set_ylabel('V (mV)',fontsize=7)
        if n:
            voltage.set_ylim(-83,38);voltage.set_yticks([-80,-40,0])
        else:
            voltage.set_ylim(-83,-62);voltage.set_yticks([-80,-72,-64])
        grid(voltage)
        tracks=[('reference_events','Reference','#35424C'),
                ('neuron_events','NEURON',COLORS['neuron8']),
                ('dbnn_events','DBNN',COLORS['gpu'])]
        for i,(key,label,col) in enumerate(tracks):
            y=2-i;events.axhline(y,lw=.5,color='#dbe2e8')
            events.vlines(d[key]/1000,y-.22,y+.22,lw=1.3,color=col)
        events.set_ylim(-.4,3.25)
        events.set_yticks([0,1,2],['DBNN','NEURON','Reference'],fontsize=6.5)
        events.tick_params(axis='y',length=0,pad=3)
        events.spines['left'].set_visible(False)
        if n:
            assert len(d['dbnn_events'])==len(d['neuron_events'])==n==3
            assert np.all(np.abs(d['dbnn_events']-d['reference_events'])<=5)
            assert np.all(np.abs(d['neuron_events']-d['reference_events'])<=5)
            for rt,pt in zip(d['reference_events'],d['dbnn_events']):
                events.plot([rt/1000,pt/1000],[2,0],color='#a7b5bd',lw=.65,ls=':',zorder=0)
        else:
            assert len(d['dbnn_events'])==len(d['neuron_events'])==0
            events.text(4,2.75,'No output events',ha='center',va='center',
                        fontsize=6.5,color='#596e7e')
        for ax in [inputs,voltage,events]:
            ax.set_xlim(0,8);ax.set_xticks([0,2,4,6,8]);ax.tick_params(labelsize=7)
        for ax in [inputs,voltage]:
            ax.tick_params(axis='x',bottom=False,labelbottom=False)
            ax.spines['bottom'].set_visible(False)
        events.set_xlabel('Time (s)',fontsize=7,labelpad=2)
        if j==0:
            handles,labels=voltage.get_legend_handles_labels()
            fig.legend(handles,labels,ncol=3,frameon=False,fontsize=6.8,
                       loc='upper center',bbox_to_anchor=(.56,.997),
                       handlelength=1.45,columnspacing=.75)
    save(fig,'voltage_examples')

def pair_order():
    # Blue and green identify the input cases in every panel; gray denotes
    # the fitted readout coefficient, not a second biological input case.
    fig,axs=plt.subplots(3,1,figsize=(WIDTH,5.65),
                         gridspec_kw={'height_ratios':[1.1,.72,1.32]})
    fig.subplots_adjust(left=.19,right=.97,bottom=.09,top=.975,hspace=.55)
    n=np.load(WORK/'实验脚本/dbnn_theory_20261005/results/pair_order_trajectories.npz')
    t=n['time_ms'];kp=n['fitted_kappa_per_mv']
    blue=COLORS['cpu8'];green=COLORS['gpu'];gray='#68757E'
    ax=axs[0];sel=(t<=60)&np.isfinite(kp)
    ax.plot(t[sel],kp[sel],c=blue,lw=1.5,
            label=r'Dynamic input: fitted $\widehat\kappa_{EI}(t)$')
    ax.axhline(PAIR['dynamic']['best_cross_coefficient_per_mv'],
               c=gray,lw=1.1,ls='--',
               label=r'Dynamic input: fixed $c^*_{EI}$')
    ax.set_xlim(0,60);ax.set_xticks([0,20,40,60])
    ax.set_xlabel('Time after input onset (ms)')
    ax.set_ylabel(r'$\kappa_{EI}$ (mV$^{-1}$)')
    grid(ax);ax.legend(frameon=False,fontsize=8,loc='upper left',
                      labelspacing=.15,handlelength=1.65)
    ax.text(-.19,1.035,'(a)',transform=ax.transAxes,fontweight='bold')

    ax=axs[1]
    k0=PAIR['steady']['kappa_per_mv'];horizon=PAIR['steady']['horizon_ms']
    ax.plot([0,horizon],[k0,k0],c=green,lw=1.7,
            label=r'Constant input: $\kappa_{EI}=c^*_{EI}$')
    ax.set_xlim(0,horizon);ax.set_xticks([0,100,200,300])
    ax.set_ylim(.06,.12);ax.set_yticks([.06,.08,.10,.12])
    ax.set_xlabel('Time within equilibrium observation (ms)')
    ax.set_ylabel(r'$\kappa_{EI}$ (mV$^{-1}$)')
    grid(ax);ax.legend(frameon=False,fontsize=8,loc='upper left',
                      handlelength=1.65)
    ax.text(.98,.11,r'$\kappa_{EI}=0.0857143$ mV$^{-1}$',
            transform=ax.transAxes,ha='right',fontsize=8,color=green)
    ax.text(-.19,1.055,'(b)',transform=ax.transAxes,fontweight='bold')

    ax=axs[2];rows=PAIR['rows'];eps=np.array([r['epsilon'] for r in rows])
    ax.axvspan(1e-3,1e-2,color=gray,alpha=.08,lw=0,zorder=0)
    ax.loglog(eps,[r['dynamic_pair_rmse_mv'] for r in rows],'o-',
              ms=2.6,color=blue,label='Dynamic input: slope 2.000')
    ax.loglog(eps,[r['steady_pair_rmse_mv'] for r in rows],'s-',
              ms=2.6,color=green,label='Constant input: slope 2.999')
    ax.set_xlabel(r'Common conductance scale $\varepsilon$')
    ax.set_ylabel('Pair-interaction RMSE (mV)');grid(ax)
    ax.legend(frameon=False,fontsize=8,loc='lower right',
              handlelength=1.65,labelspacing=.15)
    ax.text(-.19,1.035,'(c)',transform=ax.transAxes,fontweight='bold')
    save(fig,'pair_order')

def finite():
    from matplotlib.lines import Line2D
    fig,axs=plt.subplots(3,1,figsize=(WIDTH,6.05),
                         gridspec_kw={'height_ratios':[1,1,1.18]})
    fig.subplots_adjust(left=.17,right=.97,bottom=.09,top=.855,hspace=.68)
    band_color='#eaf0f5'
    for ax in axs:
        ax.axvspan(.2,6,color=band_color,zorder=0)
        ax.axvline(6,color='#aab9c5',lw=.6,ls=(0,(3,3)),zorder=1)
        ax.axhline(5,color='#AE6F42',lw=.8,ls=':',zorder=2)
        ax.set_xlim(0,10)
        ax.set_xticks([0,2,4,6,8,10])
        ax.set_xlabel('Isolated EPSP peak (mV)',labelpad=3)
        ax.set_ylabel('Relative RMS error (%)')
        grid(ax)
    for ax,pid,title,ymax in zip(axs[:2],['ampa_pair','tonic_pair'],
            ['(a) Synchronous 5/10-ms pulses; passive',
             '(b) Distributed tonic input; passive'],[36,48]):
        rows=sorted([r for r in P['rows'] if r['profile']==pid and
                     r['epsp_peak_mv']<=10],key=lambda r:r['epsp_peak_mv'])
        assert all(r['ipsp_abs_peak_mv']<=3 for r in rows)
        for field,col,sty,label in [
                ('first_relative_rms',COLORS['cpu8'],'--','First-order response'),
                ('fixed_dbnn_relative_rms',COLORS['gpu'],'-','Fixed bilinear readout'),
                ('exact_second_relative_rms','#657786',':','Physical second-order expansion')]:
            ax.plot([r['epsp_peak_mv'] for r in rows],
                    [100*r[field] for r in rows],sty,c=col,label=label)
        ax.set_ylim(-.6,ymax)
        ax.set_title(title,loc='left',fontsize=8,pad=5)
        ax.text(.985,5.7,'5%',transform=ax.get_yaxis_transform(),
                ha='right',va='bottom',fontsize=7,color='#936037')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7.5,loc='upper center',
               bbox_to_anchor=(.57,.998),labelspacing=.2)
    fig.text(.57,.905,'Shaded: EPSP 0.2-6 mV; all shown |IPSP| ≤ 3 mV',
             ha='center',fontsize=7.1,color='#596e7e')

    # The active-cell baseline retains complete isolated finite responses.
    # Shapes identify spatial profiles; colors identify the two approximations.
    ax=axs[2]
    # The reported active-cell comparison uses proximal and distributed inputs.
    markers={'proximal':'o','distributed':'s'}
    for pid,marker in markers.items():
        rows=sorted([r for r in ACTIVE['rows'] if r['profile']==pid and
                     r['subthreshold'] and r['epsp_peak_mv']<=10],
                    key=lambda r:r['epsp_peak_mv'])
        assert all(r['ipsp_abs_peak_mv']<=3 for r in rows)
        for field,col,sty in [
                ('additive_relative_rms',COLORS['cpu8'],'--'),
                ('fixed_bilinear_relative_rms',COLORS['gpu'],'-')]:
            ax.plot([r['epsp_peak_mv'] for r in rows],
                    [100*r[field] for r in rows],ls=sty,color=col,
                    marker=marker,ms=3.2,mew=.6,lw=1.1)
    ax.set_ylim(-.3,12)
    ax.set_yticks([0,5,10])
    ax.set_title('(c) L5PC with active channels; subthreshold',
                 loc='left',fontsize=8,pad=5)
    ax.text(.985,5.65,'5%',transform=ax.get_yaxis_transform(),
            ha='right',va='bottom',fontsize=7,color='#936037')
    legend=[Line2D([],[],color=COLORS['cpu8'],ls='--',label='Isolated sum'),
            Line2D([],[],color=COLORS['gpu'],ls='-',label='Fixed bilinear')]
    legend += [Line2D([],[],color='#596672',ls='',marker=m,ms=3.5,
                      label=p.capitalize()) for p,m in markers.items()]
    ax.legend(handles=legend,frameon=False,fontsize=6.8,ncol=2,
              loc='upper left',handlelength=1.7,columnspacing=.9,
              labelspacing=.3,borderaxespad=.3)
    save(fig,'finite_strength')

def family_table():
    groups=D['models']['gpu']['groups']['family']
    names={'gated_excitation':'Excitatory gates','rate_steps':'Rate changes',
           'stationary':'Stationary','zero':'Zero input'}
    rows=[]
    for k,name in names.items():
        a=groups[k];e=a['events']
        f1=f"{e['f1']:.3f}" if e['tp']+e['fp']+e['fn'] else '--'
        rows.append(f"{name} & {a['cases']} & {a['reference_events']} & "
                    f"{e['tp']} & {e['fp']} & {e['fn']} & "
                    f"{a['pooled_rmse_mv']:.3f} & {f1} \\\\")
    (ROOT/'sections/family_table.tex').write_text(
        '\\begin{tabular}{lrrrrrrr}\n\\toprule\n'
        'Input family & Inputs & Ref. & TP & FP & FN & RMSE (mV) & $F_1$\\\\\n'
        '\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')

def tables():
    family_table()
    rows=[]
    for k in COLORS:
        v=D['models'][k];a=v['accuracy'];ev=a['events'];tm=v['timing']['wall_ms']
        name=NAMES[k].replace('_',r'\_')
        if k in TABLE_CITATIONS:
            name += r' \cite{' + TABLE_CITATIONS[k] + '}'
        rows.append(f"{name} & {a['pooled_rmse_mv']:.3f} & {ev['f1']:.3f} & {tm:.2f} \\\\")
    (ROOT/'sections/main_table.tex').write_text('\\begin{tabular}{lrrr}\n\\toprule\nMethod & RMSE (mV) & Event $F_1$ & Time (ms)\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
    rows=[]
    for k in ['gpu','elm','lstm','tcn','neuron8','reduce']:
        a=D['models'][k]['accuracy'];v=a['events'];lo,hi=a['bootstrap_95ci']['f1']
        name=NAMES[k].replace('_', r'\_')
        rows.append(f"{name} & {v['tp']} & {v['fp']} & {v['fn']} & {v['precision']:.3f} & {v['recall']:.3f} & [{lo:.3f}, {hi:.3f}] \\\\")
    (ROOT/'sections/event_table.tex').write_text('\\begin{tabular}{lrrrrrl}\n\\toprule\nMethod & TP & FP & FN & $P$ & $R$ & F1 interval\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
    rows=[]
    for t in sorted(int(k) for k in D['durations']):
        v=D['durations'][str(t)];ratio=v['neuron8']['median_ms']/v['gpu']['median_ms']
        rows.append(f"{t} & {v['gpu']['cases']} & {v['gpu']['median_ms']:.4f} & {v['cpu8']['median_ms']:.3f} & {v['neuron8']['median_ms']:.2f} & {ratio:,.0f} \\\\")
    (ROOT/'sections/duration_table.tex').write_text('\\begin{tabular}{rrrrrr}\n\\toprule\n$T$ (s) & Inputs & DBNN GPU & DBNN CPU 8 & NEURON 8 & Ratio\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
    # Dataset counts are generated from the frozen panel rather than a plan.
    from collections import Counter
    records=D['models']['gpu']['records']
    audit={f:dict(Counter(str(r[f]) for r in records)) for f in ['family','A','site_mode','reference_spikes','duration_s','rate_hz']}
    audit['duration_s_total']=sum(r['duration_s'] for r in records)
    audit['reference_spikes_total']=sum(r['reference_spikes'] for r in records)
    audit['gpu_kernel_max_abs_mv']=max(r['validation']['max_abs_mv'] for r in D['models']['gpu']['timing']['records'])
    audit['cpu_kernel_max_abs_mv']={k:max(r['validation']['max_abs_mv'] for r in D['models'][k]['timing']['records']) for k in ['cpu1','cpu8']}
    (E/'paper_metric_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
    if D['reference_dt_ms']==.0125:
        from build_refined_figures import frontier,duration,events,tables
    frontier();duration();events();examples();pair_order();finite();tables()

    from build_trained_diagnostics import main as trained_diagnostics
    trained_diagnostics()
