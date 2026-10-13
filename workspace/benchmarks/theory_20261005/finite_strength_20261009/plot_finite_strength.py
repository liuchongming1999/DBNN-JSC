# -*- coding: utf-8 -*-
"""Plot saved remote results only; no local simulation or fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'legacy_fast_pair':'Fast pulses: 2 / 5 ms','ampa_pair':'Pulses: 5 / 10 ms','delayed_pair':'Inhibition delayed by 15 ms',
       'packet_pair':'20-event packets','tonic_pair':'Sustained distributed event train','poisson_pair':'Poisson event groups'}
BLUE='#3469a7';TEAL='#008577';GRAY='#87949d';ORANGE='#d28b36';RED='#ad5874'


def save(f,out,name):
    for ext in ['png','svg']:f.savefig(out/(name+'.'+ext),dpi=185,bbox_inches='tight')
    plt.close(f)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=json.loads((a.data/'passive_finite_strength.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.7,
      'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#203a4c','text.color':'#203a4c','axes.edgecolor':'#9fadb6',
      'xtick.color':'#647b87','ytick.color':'#647b87','savefig.facecolor':'white'})
    f,axes=plt.subplots(2,3,figsize=(12.8,8.0),sharex=True,sharey=True)
    for i,(ax,key) in enumerate(zip(axes.flat,LABEL)):
        rs=[r for r in d['rows'] if r['profile']==key and .2<=r['epsp_peak_mv']<=10 and r['ipsp_abs_peak_mv']<=3]
        x=[r['epsp_peak_mv'] for r in rs]
        ax.axhspan(.08,5,color='#edf7f3',zorder=0);ax.axvspan(0,6,color='#f5f8fa',zorder=-1)
        for field,col,ls,label in [('first_relative_rms',BLUE,'-','First order'),('exact_second_relative_rms',GRAY,'--','Full second-order response'),
            ('fixed_dbnn_relative_rms',TEAL,'-','Fixed quadratic readout'),('shared_readout_relative_rms',ORANGE,':','One readout across 5 families')]:
            rr=[r for r in rs if field in r]
            if rr:ax.plot([r['epsp_peak_mv'] for r in rr],[100*r[field] for r in rr],lw=2.1 if col in [BLUE,TEAL] else 1.4,color=col,ls=ls,label=label)
        ax.axhline(5,color=RED,ls='--',lw=1.2);ax.set_title(f'({chr(97+i)}) {LABEL[key]}',fontsize=11.1,fontweight='bold',pad=12)
        ax.set_yscale('log');ax.set_ylim(.08,100);ax.set_xlim(0,10);ax.set_xticks([0,2,4,6,8,10]);ax.grid(color='#dfe9ee',alpha=.75,linewidth=.6);ax.set_axisbelow(True)
        if i in [0,3]:ax.set_ylabel('Relative RMS error (%)')
        if i>=3:ax.set_xlabel('Isolated EPSP peak (mV)')
        if i==2:ax.text(.96,.46,'5% target',transform=ax.transAxes,ha='right',color=RED,fontsize=9)
    handles,labels=axes[1,0].get_legend_handles_labels();f.legend(handles,labels,ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(.52,.006),fontsize=10)
    f.subplots_adjust(left=.075,right=.98,bottom=.16,top=.88,wspace=.18,hspace=.36)
    f.text(.075,.964,'When is a quadratic response useful?',fontsize=18,fontweight='bold');f.text(.075,.923,'Passive compartment · Fixed kernels and readouts · Actual EPSP amplitude · Full-window relative RMS',color='#647b87',fontsize=10.7)
    save(f,a.out,'finite_strength_accuracy')

    f,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    for ax,key in zip(axes,['ampa_pair','tonic_pair']):
        rs=[r for r in d['rows'] if r['profile']==key and .2<=r['epsp_peak_mv']<=10 and r['ipsp_abs_peak_mv']<=3]
        x=[r['epsp_peak_mv'] for r in rs]
        for field,col,ls,label in [('second_representation_mv',BLUE,'-','Second-order representation residual'),
                                  ('higher_order_mv',ORANGE,'-','Exact second-order truncation remainder'),
                                  ('fixed_dbnn_rmse_mv',TEAL,'--','Actual fixed-quadratic error')]:
            ax.plot(x,[100*r[field]/r['response_rms_mv'] for r in rs],color=col,ls=ls,lw=2,label=label)
        ax.axhline(5,color=RED,ls=':',lw=1);ax.set_title(LABEL[key],fontsize=12,fontweight='bold');ax.set(xlabel='Isolated EPSP peak (mV)',ylabel='Error / true response RMS (%)',xlim=(0,10),ylim=(.05,35));ax.set_yscale('log');ax.grid(alpha=.18)
    f.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.5,-.10),fontsize=10)
    f.subplots_adjust(left=.08,right=.98,top=.83,bottom=.22,wspace=.28);f.text(.08,.95,'Second-order residual and higher-order remainder',fontsize=17,fontweight='bold')
    save(f,a.out,'finite_strength_error_budget')

    path=a.data/'l5pc_finite_strength.json'
    if path.exists():
        l=json.loads(path.read_text());f,axes=plt.subplots(1,3,figsize=(12.5,4.8),sharey=True)
        for ax,key in zip(axes,['proximal','distributed','distal']):
            rs=[r for r in l['rows'] if r['profile']==key and r['subthreshold'] and r['epsp_peak_mv']<=10]
            for field,col,label in [('additive_relative_rms',BLUE,'Sum of isolated responses'),('fixed_bilinear_relative_rms',TEAL,'Fixed fitted product correction')]:
                ax.plot([r['epsp_peak_mv'] for r in rs],[100*r[field] for r in rs],color=col,marker='o',ms=4,lw=2,label=label)
            ax.axhline(5,color=RED,ls='--',lw=1);ax.axhspan(.08,5,color='#edf7f3',zorder=0);ax.set_title(key.capitalize()+' sites',fontsize=12,fontweight='bold')
            ax.set(xlabel='Isolated EPSP peak (mV)',ylim=(.08,400),xlim=(0,10));ax.set_yscale('log');ax.grid(alpha=.18)
        axes[2].text(.97,.90,'Large error with\nsmall somatic EPSP',transform=axes[2].transAxes,ha='right',fontsize=9,color='#647b87')
        axes[0].set_ylabel('Full-window relative RMS error (%)');f.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01),fontsize=10)
        f.subplots_adjust(left=.075,right=.98,top=.80,bottom=.21,wspace=.18);f.text(.075,.95,'Finite E/I integration in the active mouse L5PC',fontsize=18,fontweight='bold');f.text(.075,.895,'Original channels and Exp2Syn kinetics · Held-out strengths · One fixed coefficient per spatial profile',color='#647b87',fontsize=10.5)
        save(f,a.out,'finite_strength_l5pc')
    print('Saved figures in',a.out)

if __name__=='__main__':main()
