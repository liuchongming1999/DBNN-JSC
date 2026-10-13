"""Honest voltage amplitudes and event rasters for four fully matched examples."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);selection=json.loads((a.data/'selected_examples.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#233e4c','axes.labelcolor':'#233e4c',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a9bbc3','xtick.color':'#5c7382','ytick.color':'#5c7382'})
    for ex in selection:
        cid=ex['case_id'];n=ex['reference_spikes'];d=np.load(a.data/'examples'/f'{cid}.npz');t=d['time_ms']/1000
        center=float(d['reference_events'][len(d['reference_events'])//2]) if n else 4000.
        lo,hi=center-(25 if n else 150),center+(65 if n else 150)
        f=plt.figure(figsize=(12.8,7.7));gs=f.add_gridspec(3,2,width_ratios=[2.4,1.2],height_ratios=[2.8,1.65,1.0],
            left=.085,right=.97,bottom=.10,top=.79,hspace=.27,wspace=.24)
        full=f.add_subplot(gs[0,0]);inputs=f.add_subplot(gs[1,0],sharex=full);events=f.add_subplot(gs[2,0],sharex=full);zoom=f.add_subplot(gs[:2,1])
        styles=[('reference_voltage','#293e4c','Reference: NEURON 0.1 ms',1.15),('neuron_voltage','#d78f35','NEURON: 0.5 ms integration',1.),('dbnn_voltage','#008577','DBNN: 1 ms output',1.3)]
        full.axvspan(lo/1000,hi/1000,color='#dce9ef',alpha=.6,zorder=0)
        for key,color,label,lw in styles:
            native=a.data/'examples'/f'{cid}_native{.1 if key=="reference_voltage" else .5:g}.npz'
            if key!='dbnn_voltage' and native.exists():
                nv=np.load(native);pt,pv=nv['time_ms'],nv['voltage']
            else:pt,pv=d['time_ms'],d[key]
            full.plot(pt/1000,pv,color=color,lw=lw,label=label,zorder=4 if key=='dbnn_voltage' else 2)
            zz=(pt>=lo)&(pt<=hi);zoom.plot(pt[zz]-center,pv[zz],color=color,lw=lw)
        full.set_xlim(0,8);full.set_ylabel('Soma voltage (mV)');full.tick_params(labelbottom=False)
        if n:full.set_ylim(-83,34)
        raster=np.load(a.data/'examples'/f'{cid}_input.npz')
        x,ids=raster['x'],raster['ids']
        digest=hashlib.sha256(x.tobytes()).hexdigest()
        assert digest==ex['dbnn_record']['input_sha256'], f'Input mismatch: {cid}'
        row,step=np.nonzero(x);sites=ids[row];input_time=step/1000
        assert x.shape==(500,8001) and set(ids.tolist())==set(range(500))
        for subset,color in [(sites<250,'#4c8ca8'),(sites>=250,'#b76b84')]:
            inputs.scatter(input_time[subset],sites[subset],marker='|',s=.8,
                           linewidths=.18,color=color,alpha=.55,rasterized=True)
        inputs.axhline(249.5,color='#ffffff',lw=1.0)
        inputs.set_ylim(-3,502);inputs.set_yticks([0,249,499]);inputs.set_ylabel('Input raster\nSite ID',fontsize=10)
        inputs.tick_params(labelbottom=False)
        inputs.text(.99,.93,'Inhibitory',transform=inputs.transAxes,ha='right',va='top',
                    color='#93526a',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=2))
        inputs.text(.99,.43,'Excitatory',transform=inputs.transAxes,ha='right',va='top',
                    color='#3c7189',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=2))
        for i,(key,label,color) in enumerate([('reference_events','Reference','#293e4c'),('neuron_events','NEURON','#d78f35'),('dbnn_events','DBNN','#008577')]):
            y=2-i;events.axhline(y,lw=.6,color='#e1eaee');events.vlines(d[key]/1000,y-.25,y+.25,lw=1.8,color=color)
        events.set(ylim=(-.55,2.55),xlabel='Physical time (s)');events.set_yticks([0,1,2],['DBNN','NEURON','Reference'],fontsize=9);events.tick_params(axis='y',length=0);events.spines['left'].set_visible(False)
        zp=zoom.get_position();zoom.set_position([zp.x0,.36,zp.width,.43])
        zoom.set_xlim(lo-center,hi-center);zoom.set_xlabel('Time from selected event (ms)' if n else 'Time from 4 s (ms)',fontsize=10)
        if n:zoom.set_ylim(-76,-54);zoom.set_title('Subthreshold detail',fontweight='bold',fontsize=11)
        else:
            zz=(d['time_ms']>=lo)&(d['time_ms']<=hi);v=np.concatenate([d[k][zz] for k,_,_,_ in styles]);zoom.set_ylim(v.min()-.2,v.max()+.2);zoom.set_title('Voltage detail',fontweight='bold',fontsize=11)
        for event in d['reference_events']:
            zoom.axvspan(event-center-10,event-center+10,color='#f7dfbd',alpha=.45,zorder=0)
        for ax in [full,zoom]:ax.grid(color='#e4edf1',lw=.7);ax.set_axisbelow(True)
        f.text(.085,.945,f'{n}-spike response',fontsize=18,fontweight='bold');f.text(.085,.904,f'{cid}  ·  8 s  ·  500 input sites  ·  Independent evaluation input',fontsize=10.8,color='#627d8c')
        f.legend(*full.get_legend_handles_labels(),ncol=3,loc='upper left',bbox_to_anchor=(.079,.888),frameon=False,fontsize=10)
        event_text=f'Both: {n}/{n} matched events\nNo extra or missed events' if n else 'No events in all three traces'
        text=f'DBNN RMSE: {ex["dbnn_record"]["voltage"]["rmse_mv"]:.3f} mV\nNEURON RMSE: {ex["neuron_record"]["voltage"]["rmse_mv"]:.3f} mV\n{event_text}\nMatching tolerance: ±5 ms'
        f.text(.71,.255,text,fontsize=10.6,color='#466170',va='top',linespacing=1.6)
        f.text(.085,.025,'Input raster: all events at 500 sites  ·  Output rows: recorded spikes / DBNN reset  ·  Shaded detail: reference spike ±10 ms',fontsize=9,color='#637d8b')
        f.savefig(a.output/f'voltage_{n}spike.png',dpi=175);f.savefig(a.output/f'voltage_{n}spike.svg');plt.close(f)


if __name__=='__main__':main()
