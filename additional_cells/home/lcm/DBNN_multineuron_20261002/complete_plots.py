"""Plots from locked 31-window outputs; no new fitting or selection of tests."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def duration_plot(model,r,profile,figure):
    g=r['gpu'][profile]
    if not g:return ''
    dts=sorted(map(float,g['summary']),reverse=True);fig,axes=plt.subplots(1,len(dts),figsize=(4*len(dts),4.2),squeeze=False)
    for ax,dt in zip(axes[0],dts):
        sets=[('DBNN GPU','#00897B',{z['case_id']:z['variants'][str(dt)]['wall']['median_ms'] for z in g['records']})]
        for engine,label,color in [('multisplit','NEURON 8 cores','#D28B36'),('core','CoreNEURON 1 core','#B95271')]:
            solver=r['solvers'][profile].get(f'{engine}_dt{dt}')
            if solver and solver.get('finished'):sets.append((label,color,{z['case_id']:z['median_ms'] for z in solver['records']}))
        durations=sorted({z['duration_s'] for z in g['records']})
        for label,color,times in sets:
            vals=[];lo=[];hi=[]
            for t in durations:
                a=[times[z['case_id']] for z in g['records'] if z['duration_s']==t]
                vals.append(np.median(a));lo.append(np.quantile(a,.25));hi.append(np.quantile(a,.75))
            ax.plot(durations,vals,'o-',color=color,lw=2,ms=5,label=label);ax.fill_between(durations,lo,hi,color=color,alpha=.1)
        ax.set_yscale('log');ax.set_title(f'{dt:g} ms');ax.set_xlabel('Simulation duration (s)');ax.set_xticks(durations)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_axisbelow(True)
    axes[0,0].set_ylabel('Median complete compute wall time (ms)');axes[0,-1].legend(frameon=False,fontsize=8)
    fig.tight_layout();return figure(fig,'按1、2、4、8、16s分组的完整计算耗时，中位数与Q1–Q3。')

def voltage_plot(model,r,profile,work,data,figure):
    g=r['gpu'][profile]
    if not g:return ''
    manifest=data/model/profile/'test/manifest.json'
    if not manifest.exists():return ''
    rows={z['case_id']:z for z in json.loads(manifest.read_text())['cases']}
    dt=min(map(float,g['summary']));examples=[]
    for count in range(4):
        eligible=[z for z in g['records'] if rows[z['case_id']]['reference_spike_count']==count]
        if not eligible:continue
        def rank(z):
            m=z['variants'][str(dt)];events=m['events'];return (z['duration_s']!=8,events['fp']+events['fn'],m['metrics']['rmse_mv'])
        examples.append((count,min(eligible,key=rank)))
    if not examples:return ''
    fig,axes=plt.subplots(len(examples),1,figsize=(11,2.5*len(examples)),squeeze=False)
    colors={1.:'#006D77',.5:'#2B9B92',.25:'#54B8AF'}
    for ax,(count,record) in zip(axes[:,0],examples):
        cid=record['case_id'];row=rows[cid];ref=np.load(row['reference_file']);t=ref['time_ms']/1000
        ax.plot(t,ref['voltage'],color='#233C4A',lw=.85,label='NEURON reference · dt 0.025 ms',zorder=2)
        for step in sorted(map(float,g['summary']),reverse=True):
            f=work/'results'/model/profile/'gpu'/f'dt{step}ms'/(cid+'.npz')
            if not f.exists():continue
            d=np.load(f);tt=np.arange(len(d['voltage']))*step/1000
            ax.plot(tt,d['voltage'],color=colors[step],lw=1,alpha=.9,label=f'DBNN {step:g} ms',zorder=3)
            ev=d['reset_events_ms']/1000
            if len(ev):ax.scatter(ev,np.interp(ev,tt,d['voltage']),s=20,color=colors[step],marker='v',zorder=5)
        for step,color in [(.25,'#D0A36E'),(.5,'#B9AB98'),(1.,'#9B927F')]:
            solver=r['solvers'][profile].get(f'multisplit_dt{step}')
            f=work/'results'/model/profile/'solvers'/f'multisplit_dt{step}'/(cid+'.npz')
            if solver and solver.get('finished') and f.exists():
                d=np.load(f);ax.plot(np.arange(len(d['voltage']))/1000,d['voltage'],color=color,lw=.7,ls='--',alpha=.7,label=f'NEURON dt {step:g} ms')
        for ev in ref['spike_times']/1000:ax.axvline(ev,color='#E2C2A1',lw=.8,alpha=.55)
        ax.set_xlim(0,row['duration_ms']/1000);ax.set_ylabel('Voltage (mV)');ax.set_title(f'{count} spike · {cid} · A={row["A"]} · {row["rate_hz"]} Hz/site',loc='left',fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.12);ax.set_axisbelow(True)
    axes[-1,0].set_xlabel('Time (s)');axes[0,0].legend(frameon=False,loc='upper right',fontsize=7,ncol=2)
    fig.tight_layout();return figure(fig,'0–3尖峰的实测电压示例。优先8秒，再按事件匹配和masked RMSE选择展示；整体指标仍包括全部31窗。')

def complete_plots(model,r,profile,work,data,figure):
    return voltage_plot(model,r,profile,work,data,figure)+duration_plot(model,r,profile,figure)
