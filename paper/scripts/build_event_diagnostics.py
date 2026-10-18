"""Generate supplementary event/voltage figures and tables from frozen records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from event_metrics import match_events

R=Path(__file__).resolve().parents[1]
D=json.loads((R/'evidence/main379_results.json').read_text())
S=json.loads((R/'evidence/spike_diagnostics_20261016.json').read_text())
KEYS=['gpu','elm','lstm','tcn','neuron8','neuron8_dt0.25','neuron8_dt0.1','reduce']
NAMES=dict(gpu='DBNN',elm='Branch-ELM',lstm='LSTM',tcn='TCN',neuron8='NEURON / CoreNEURON',reduce='Neuron_Reduce')
COLORS=dict(gpu='#008779',elm='#475668',lstm='#218FAF',tcn='#CB513D',neuron8='#D68A28',reduce='#97713D')
NAMES.update({'neuron8':'NEURON/CN, 0.5 ms','neuron8_dt0.25':'NEURON/CN, 0.25 ms','neuron8_dt0.1':'NEURON/CN, 0.1 ms'})
COLORS.update({'neuron8_dt0.25':'#B16D20','neuron8_dt0.1':'#7C4B16'})
WIDTH=119/25.4
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'pdf.fonttype':42,'savefig.dpi':220})


def save(fig,name,number):
    fig.savefig(R/f'figures/{name}.pdf')
    fig.savefig(R/f'figures/{name}.png')
    (R/f'figures/Fig{number}.pdf').write_bytes((R/f'figures/{name}.pdf').read_bytes())
    plt.close(fig)


def main():
    ts=[1,2,5,10,20]
    fig,(ax,tm)=plt.subplots(2,1,figsize=(WIDTH,5.25))
    fig.subplots_adjust(left=.16,right=.97,bottom=.10,top=.80,hspace=.58)
    for key in KEYS:
        ax.plot(ts,[S['models'][key]['sensitivity'][str(t)]['f1'] for t in ts],
            'o-',color=COLORS[key],label=NAMES[key],lw=1.6 if key=='gpu' else 1,ms=3)
    ax.axvline(5,color='#a1adb6',ls='--',lw=.7,zorder=0)
    ax.set(xticks=ts,xlabel='Matching tolerance (±ms)',ylabel='Event F1',ylim=(0,1.02),xlim=(.4,20.7))
    ax.grid(axis='y',color='#e0e6ea',lw=.6);ax.set_axisbelow(True)
    ax.set_title('(a) Fixed predictions, different time tolerances',fontsize=8,loc='left',pad=6)
    fig.legend(*ax.get_legend_handles_labels(),ncol=2,fontsize=6.5,frameon=False,loc='upper center',
        bbox_to_anchor=(.57,.997),columnspacing=.9,labelspacing=.35,handlelength=1.8)
    for key in ['gpu','elm','neuron8_dt0.25','neuron8_dt0.1']:
        errors=[x for row in D['models'][key]['records'] for x in
                match_events(row['reference_events_ms'],row['predicted_events_ms'],20)['matched_errors_ms']]
        tm.hist(errors,bins=np.arange(-20,21,1),histtype='step',color=COLORS[key],
            lw=1.6 if key=='gpu' else 1,label=NAMES[key])
    tm.axvspan(-5,5,color='#718493',alpha=.08)
    tm.axvline(0,color='#a9b5be',lw=.6)
    tm.set(xlim=(-20,20),xlabel='Predicted − reference time (ms)',ylabel='Matched events')
    tm.set_title('(b) Timing offsets under ±20 ms matching',fontsize=8,loc='left',pad=6)
    tm.legend(frameon=False,fontsize=6.1,loc='upper left')
    save(fig,'event_tolerance_diagnostics',8)

    fig,ax=plt.subplots(figsize=(WIDTH,2.9))
    fig.subplots_adjust(left=.16,right=.97,bottom=.19,top=.73)
    for key in KEYS:
        vals=np.sort([r['voltage']['rmse_mv'] for r in D['models'][key]['records'] if r['family']!='zero'])
        ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=COLORS[key],
            label=NAMES[key],lw=1.6 if key=='gpu' else 1)
    ax.set(xscale='log',xlabel='Per-input masked RMSE (mV)',ylabel='Cumulative fraction',ylim=(0,1.02))
    ax.grid(color='#e0e6ea',lw=.6);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),ncol=2,fontsize=6.5,frameon=False,loc='upper center',
        bbox_to_anchor=(.57,1),columnspacing=.9,labelspacing=.35,handlelength=1.8)
    save(fig,'per_input_voltage_distribution',9)

    lines=[]
    for key in KEYS:
        a=S['models'][key];z=a['no_reference_spikes_stochastic'];c=a['count_error']
        name=NAMES[key].replace('_',r'\_')
        lines.append(f"{name} & {z['trajectories_with_predicted_events']}/{z['cases']} & {z['extra_events']} & "
            f"{c['mean_absolute_error']:.3f} & {c['exact_count_trajectories']}/379 \\\\")
    (R/'sections/spike_count_diagnostic_table.tex').write_text(
        '\\begin{tabular}{lrrrr}\n\\toprule\n'
        'Method & Inputs with events & Extra events & Count MAE & Exact counts\\\\\n'
        '\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n')
    if (R/'evidence/crosscell_final_20261017/source.json').exists():
        # The verified 182-input final-model panel supersedes the legacy 31 inputs.
        import subprocess,sys
        subprocess.run([sys.executable,str(R/'scripts/build_crosscell_final.py')],check=True)
    else:
        rows=[]
        for key in ['granule','basket','golgi','hay_l5','human_l23','ca1']:
            for profile in ['ampa','ampa_nmda']:
                cross=json.loads((R/'evidence/crosscell_events.json').read_text())
                job=next(j for j in cross['jobs'] if j['cell']==key and j['profile']==profile)
                a=job['evaluation']['1.0'];ev=a['events']
                name=dict(granule='Granule',basket='Basket',golgi='Golgi',hay_l5='Rat L5b',human_l23='Human L2/3',ca1='CA1')[key]
                excitation='E' if profile=='ampa' else 'E+N'
                rows.append(f"{name} & {excitation} & {a['metrics']['pooled_rmse_mv']:.3f} & {ev['tp']+ev['fn']} & {ev['f1']:.3f}\\\\")
        (R/'sections/crosscell_table.tex').write_text('\\begin{tabular}{llrrr}\n\\toprule\n'
            'Cell & Excitation & RMSE (mV) & Reference spikes & F1\\\\\n\\midrule\n'
            +'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')


if __name__=='__main__':
    main()
