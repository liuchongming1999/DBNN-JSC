"""Static paper figures using only the completed 379-input paired experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,NullLocator
from matplotlib.lines import Line2D

COL={'gpu':'#008577','cpu1':'#8974b1','cpu8':'#386fa6','neuron8':'#d68b32','core1':'#b75073','tcn':'#cb5038','elm':'#536b7c','lstm':'#2195b0','reduce':'#a7783e'}
LABEL={'gpu':'DBNN GPU','cpu1':'DBNN CPU · 1 core','cpu8':'DBNN CPU · 8 cores','neuron8':'NEURON · multisplit 8 cores','core1':'CoreNEURON CPU · 1 core','tcn':'TCN GPU','elm':'Branch-ELM GPU','lstm':'LSTM GPU','reduce':'Neuron_Reduce CPU · 1 core'}
AUTHOR={'gpu':'This work','cpu1':'This work','cpu8':'This work','neuron8':'Hines et al. 2008','core1':'Kumbhar et al. 2019','tcn':'Beniaguev et al. 2021','elm':'Spieler et al. 2024','lstm':'Hochreiter & Schmidhuber 1997','reduce':'Amsalem et al. 2020'}
ORDER=['gpu','cpu8','cpu1','neuron8','core1','tcn','elm','lstm','reduce']


def save(f,out,name):
    f.savefig(out/f'{name}.png',dpi=180);f.savefig(out/f'{name}.svg');plt.close(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.data.read_text());models=d['models'];out=a.output;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#233d4c','axes.labelcolor':'#233d4c',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#acbdc7','xtick.color':'#586f80','ytick.color':'#586f80'})
    f,ax=plt.subplots(figsize=(12.5,9.6));f.subplots_adjust(left=.11,right=.97,top=.84,bottom=.14)
    xs=[models[k]['accuracy']['pooled_rmse_mv'] for k in ORDER];ys=[models[k]['timing']['wall_ms'] for k in ORDER]
    ax.set(xscale='log',yscale='log',xlim=(min(xs)/1.8,max(xs)*1.6),ylim=(min(ys)*.55,max(ys)*2.1),
           xlabel='Masked pooled RMSE (mV)',ylabel='Sum of 379 median compute wall times (ms)')
    for k in ORDER:
        m=models[k];ax.scatter(m['accuracy']['pooled_rmse_mv'],m['timing']['wall_ms'],s=135,marker='o' if m['interval_ms']==1 else 'D',
                              color=COL[k],edgecolor='white',lw=1.4,zorder=5)
    ax.grid(color='#e2eaef',lw=.8);ax.set_axisbelow(True);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    ax.set_xticks([.25,.4,.6,1,2,4,6])
    ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
    offsets={'gpu':(34,25),'cpu1':(34,20),'cpu8':(34,20),'neuron8':(35,-22),'core1':(35,20),
             'tcn':(32,-14),'elm':(32,15),'lstm':(32,-10),'reduce':(-28,15)}
    for k in ORDER:
        m=models[k];x=m['accuracy']['pooled_rmse_mv'];y=m['timing']['wall_ms'];dx,dy=offsets[k]
        ax.annotate(LABEL[k]+'\n'+AUTHOR[k],xy=(x,y),xytext=(dx,dy),textcoords='offset points',ha='left' if dx>0 else 'right',va='center',
                    fontsize=10.6,fontweight='bold',color=COL[k],linespacing=1.45,
                    bbox=dict(boxstyle='round,pad=.22',fc='white',ec='none',alpha=.96),
                    arrowprops=dict(arrowstyle='-',color=COL[k],alpha=.55,lw=1.1,shrinkA=5,shrinkB=8),zorder=7)
    source=models['neuron8'];target=models['gpu'];sx=source['accuracy']['pooled_rmse_mv'];sy=source['timing']['wall_ms'];tx=target['accuracy']['pooled_rmse_mv'];ty=target['timing']['wall_ms']
    # The arrow follows the measured working points; no displacement of error values.
    ax.annotate('',xy=(tx,ty*1.12),xytext=(sx,sy*.85),arrowprops=dict(arrowstyle='-|>',color='#405b6d',lw=1.6,connectionstyle='arc3,rad=.34',mutation_scale=18),zorder=3)
    ax.text(.065,.52,f'{sy/ty:,.0f}× faster\nNEURON → DBNN GPU',transform=ax.transAxes,ha='left',va='center',fontweight='bold',fontsize=11,color='#405b6d',
            bbox=dict(boxstyle='round,pad=.4',fc='white',ec='#b8c8d0',lw=.9),zorder=8)
    ax.text(.015,.025,'← Lower voltage error   ·   Lower compute time ↓',transform=ax.transAxes,fontsize=10,color='#6c8190')
    f.text(.11,.95,'Accuracy–runtime trade-off',fontsize=20,fontweight='bold');f.text(.11,.913,'379 independent evaluation windows · 637 reference spikes · Reference spike ±10 ms excluded',fontsize=11,color='#647d8c')
    f.text(.11,.879,'Fixed trained models · 3 repeats per window · Prepared B=1 computation · Shared server',fontsize=10,color='#738896')
    shapes=[Line2D([0],[0],color='#5d7180',marker='o',ls='none',ms=9,label='Learned models: 1 ms input/output'),Line2D([0],[0],color='#5d7180',marker='D',ls='none',ms=9,label='Numerical solvers: 0.5 ms integration')]
    f.legend(handles=shapes,loc='lower center',bbox_to_anchor=(.54,.055),ncol=2,frameon=False,fontsize=10)
    f.text(.54,.024,'All voltage errors are scored on the same 1-ms observation grid',ha='center',color='#738896',fontsize=9)
    save(f,out,'frontier_379')
    f,ax=plt.subplots(figsize=(12.5,7.6));f.subplots_adjust(left=.10,right=.97,top=.84,bottom=.25);T=np.array([1,2,4,8,16])
    for k in ORDER:
        med=np.array([d['durations'][str(t)][k]['median_ms'] for t in T]);lo=np.array([d['durations'][str(t)][k]['q1_ms'] for t in T]);hi=np.array([d['durations'][str(t)][k]['q3_ms'] for t in T])
        ax.plot(T,med,'o-',color=COL[k],lw=2,ms=5,label=LABEL[k]);ax.fill_between(T,lo,hi,color=COL[k],alpha=.07)
    ax.set(yscale='log',xlim=(.6,20),xlabel='Simulated physical duration (s)',ylabel='Per-window median compute wall time (ms)');ax.set_xticks(T);ax.grid(axis='y',color='#e1eaef');ax.set_axisbelow(True)
    ns=d['durations']['16']['neuron8']['median_ms'];gs=d['durations']['16']['gpu']['median_ms'];ax.annotate('',xy=(17,gs),xytext=(17,ns),arrowprops=dict(arrowstyle='<->',color='#405b6d',lw=1.5))
    ax.text(17.3,np.sqrt(ns*gs),f'{ns/gs:,.0f}×\nat 16 s',ha='left',va='center',fontsize=11,fontweight='bold',color='#405b6d',bbox=dict(fc='white',ec='none',alpha=.95))
    f.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.055),ncol=3,frameon=False,fontsize=9)
    f.text(.10,.95,'Runtime across simulation durations',fontsize=18,fontweight='bold');f.text(.10,.907,'All 379 windows grouped by duration · Shading: interquartile range across inputs',fontsize=10.5,color='#647d8c')
    f.text(.10,.026,'Each window: median of 3 prepared full-compute repeats; groups retain different input rates and active-site counts.',fontsize=9,color='#738896')
    save(f,out,'duration_379')
    keys=['gpu','neuron8','core1','tcn','elm','lstm','reduce'];f,ax=plt.subplots(figsize=(12.5,5.6));yy=np.arange(len(keys))[::-1]
    for y,k in zip(yy,keys):
        e=models[k]['accuracy']['events'];v=e['f1'];ci=models[k]['accuracy']['bootstrap_95ci']['f1'];ax.barh(y,v,color=COL[k],height=.52)
        ax.errorbar(v,y,xerr=np.array([[max(0,v-ci[0])],[max(0,ci[1]-v)]]),fmt='none',color='#344f5d',capsize=4,lw=1.3)
        ax.text(max(v,ci[1])+.025,y,f'{v:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]',va='center',fontsize=10)
    ax.set_yticks(yy,[LABEL[k] for k in keys],fontweight='bold');ax.set_xlim(0,1.08);ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Event F1 · ±5-ms one-to-one matching');ax.grid(axis='x',color='#e5edef');ax.set_axisbelow(True)
    f.subplots_adjust(left=.30,right=.98,top=.79,bottom=.15);f.text(.04,.93,'Spike matching on independent inputs',fontsize=18,fontweight='bold');f.text(.04,.875,'379 windows · 637 reference events · 95% confidence intervals from trajectory bootstrap',color='#647d8c',fontsize=10.5)
    save(f,out,'spike_f1_379')


if __name__=='__main__':main()
