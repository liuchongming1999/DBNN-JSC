"""Main figures/tables using the completed common-reference solver sweep."""
from pathlib import Path
import copy,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import build_figures as b

R=b.ROOT;D=b.D;S=json.loads((R/'evidence/spike_diagnostics_20261016.json').read_text())
STEPS=[.5,.25,.1];MARKERS={.5:'o',.25:'D',.1:'^'}
def key(engine,dt):return engine if dt==.5 else f'{engine}_dt{dt:g}'
def point(k):return D['models'][k]['accuracy']['pooled_rmse_mv'],D['models'][k]['timing']['wall_ms']

def frontier():
    fig,ax=plt.subplots(figsize=(b.WIDTH,4.75))
    fig.subplots_adjust(left=.16,right=.975,top=.975,bottom=.19)
    ax.set(xscale='log',yscale='log',xlim=(.04,8.5),ylim=(14,1.1e7),
           xlabel='Pooled subthreshold RMSE (mV)',ylabel='Sum of per-input median computation times (ms)')
    ax.set_xticks([.05,.1,.2,.5,1,2,5]);ax.xaxis.set_major_formatter(ScalarFormatter());b.grid(ax)
    anchors={}
    for engine in ['neuron8','core1']:
        ps=[point(key(engine,dt)) for dt in STEPS]
        ax.plot(*zip(*ps),color=b.COLORS[engine],lw=1.4,ls='-' if engine=='neuron8' else '--',zorder=2)
        for dt,(x,y) in zip(STEPS,ps):
            ax.scatter(x,y,s=31,marker=MARKERS[dt],c=b.COLORS[engine],edgecolors='white',lw=.6,zorder=5)
        anchors[engine]=ps[0] if engine=='neuron8' else ps[-1]
    for k in ['gpu','cpu1','cpu8','tcn','elm','lstm','reduce']:
        anchors[k]=point(k);x,y=anchors[k]
        ax.scatter(x,y,s=34,marker='s' if k in ['gpu','cpu1','cpu8','tcn','elm','lstm'] else 'v',
                   c=b.COLORS[k],edgecolors='white',lw=.6,zorder=5)
    labels=dict(gpu=('DBNN GPU\nOurs',(1.0,42),'left'),
       cpu8=('DBNN CPU, 8 cores\nOurs',(.60,6500),'left'),
       cpu1=('DBNN CPU, 1 core\nOurs',(.60,18000),'left'),
       lstm=('LSTM GPU\nSpieler et al. 2024',(1.05,2400),'left'),
       tcn=('TCN GPU\nBeniaguev et al. 2021',(1.25,650),'left'),
       elm=('Branch-ELM GPU\nSpieler et al. 2024',(1.1,50000),'left'),
       reduce=('Neuron_Reduce, 1 core\nAmsalem et al. 2020',(6.0,6.5e5),'right'),
       neuron8=('NEURON, 8 cores\nHines et al. 2008',(.38,1.7e5),'left'),
       core1=('CoreNEURON, 1 core\nKumbhar et al. 2019',(.29,6.5e6),'left'))
    annotations=[]
    for k,(label,xy,ha) in labels.items():
        annotations.append(ax.annotate(label,xy=anchors[k],xytext=xy,ha=ha,va='center',fontsize=7,
            weight='bold',linespacing=1.35,color=b.COLORS[k],
            bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=1.2),
            arrowprops=dict(arrowstyle='-',color=b.COLORS[k],lw=.6,alpha=.7,shrinkA=3,shrinkB=5),zorder=6))
    start=point('neuron8');end=point('gpu');ratio=start[1]/end[1]
    arrow=ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3,rad=.54',
        color='#66788B',lw=1,shrinkA=5,shrinkB=5,mutation_scale=10),zorder=1)
    fig.canvas.draw();v=arrow.arrow_patch.get_path().vertices;t=.55
    center=ax.transData.inverted().transform((1-t)**2*v[0]+2*t*(1-t)*v[1]+t*t*v[2])
    ax.annotate(f'{ratio:,.0f}×\nTime ratio',xy=center,xytext=(-7,0),textcoords='offset points',
                ha='right',va='center',fontsize=7,color='#4f6173',linespacing=1.3,zorder=6)
    ax.text(.022,.015,'← Lower voltage error  ·  Lower compute time ↓',transform=ax.transAxes,
            fontsize=6.6,color='#718493')
    handles=[Line2D([],[],marker=MARKERS[x],color='#667581',ls='None',ms=4,label=f'{x:g} ms solver step') for x in STEPS]
    handles.append(Line2D([],[],marker='s',color='#667581',ls='None',ms=4,label='1 ms learned-model sampling'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.005),ncol=2,frameon=False,
               fontsize=6.5,handletextpad=.4,columnspacing=1.2,labelspacing=.4)
    b.save(fig,'frontier')

def events():
    keys=['gpu','elm','lstm','tcn','neuron8','neuron8_dt0.25','neuron8_dt0.1','reduce']
    fig,ax=plt.subplots(figsize=(b.WIDTH,2.95));fig.subplots_adjust(left=.40,right=.97,bottom=.18,top=.97)
    for i,k in enumerate(keys):
        a=D['models'][k]['accuracy'];f=a['events']['f1'];lo,hi=a['bootstrap_95ci']['f1']
        color=b.COLORS[k.split('_dt')[0]]
        ax.errorbar(f,i,xerr=[[f-lo],[hi-f]],fmt=MARKERS.get(D['models'][k]['interval_ms'],'s'),
                    color=color,ms=3.6,capsize=2)
    labels=['DBNN','Branch-ELM','LSTM','TCN','NEURON/CN, 0.5 ms','NEURON/CN, 0.25 ms','NEURON/CN, 0.1 ms','Neuron_Reduce']
    ax.set_yticks(range(len(keys)));ax.set_yticklabels(labels,fontsize=7);ax.invert_yaxis()
    ax.set(xlim=(0,1.02),xlabel='Event F1 (one-to-one matching, ±5 ms)');b.grid(ax);b.save(fig,'events')

def duration():
    # Use the same 0.5 ms solver setting as the main voltage-RMSE comparison.
    saved=copy.deepcopy(b.D);saved_names=b.NAMES.copy()
    b.D=copy.deepcopy(D)
    for engine in ['neuron8','core1']:
        b.D['models'][engine]=D['models'][key(engine,.5)]
        for t in b.D['durations']:b.D['durations'][t][engine]=D['durations'][t][key(engine,.5)]
        b.NAMES[engine]+=', 0.5 ms'
    b.duration();b.D=saved;b.NAMES=saved_names

def tables():
    b.tables()
    rows=[]
    keys=['gpu','cpu8','cpu1','tcn','elm','lstm','reduce']+[key(e,dt) for e in ['neuron8','core1'] for dt in STEPS]
    for k in keys:
        m=D['models'][k];engine=k.split('_dt')[0];name=b.NAMES[engine].replace('_',r'\_')
        if engine in b.TABLE_CITATIONS:name+=r' \cite{'+b.TABLE_CITATIONS[engine]+'}'
        ev=m['accuracy']['events'];h=m['interval_ms']
        rows.append(f"{name} & {h:g} & {m['accuracy']['pooled_rmse_mv']:.3f} & {ev['f1']:.3f} & {m['timing']['wall_ms']/1000:.5g} \\\\")
    (R/'sections/main_table.tex').write_text('\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Interval (ms) & RMSE (mV) & $F_1$ & Time (s)\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
    rows=[]
    for k in ['gpu','elm','lstm','tcn','neuron8','neuron8_dt0.25','neuron8_dt0.1','reduce']:
        a=D['models'][k]['accuracy'];v=a['events'];lo,hi=a['bootstrap_95ci']['f1']
        name=b.NAMES[k.split('_dt')[0]].replace('_',r'\_')
        if k.startswith('neuron8'):name=f"NEURON/CN, {D['models'][k]['interval_ms']:g} ms"
        rows.append(f"{name} & {v['tp']} & {v['fp']} & {v['fn']} & {v['precision']:.3f} & {v['recall']:.3f} & [{lo:.3f}, {hi:.3f}] \\\\")
    (R/'sections/event_table.tex').write_text('\\begin{tabular}{lrrrrrl}\n\\toprule\nMethod & TP & FP & FN & $P$ & $R$ & F1 interval\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')

if __name__=='__main__':frontier();duration();events();b.examples();tables()
