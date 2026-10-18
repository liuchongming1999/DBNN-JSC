"""Paper figures/tables from frozen adviser-7 diagnostics; no simulation/fit.

The primary DBNN is unchanged. Passive controls use isolated-response fitted
kernels; active controls use the actual primary checkpoint's parameters.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]; E=R/'evidence'; F=R/'figures'
P=json.loads((E/'trained_passive_diagnostic.json').read_text())
A=json.loads((E/'trained_active_diagnostic.json').read_text())
V=json.loads((E/'trained_diagnostic_verification.json').read_text())
assert P['finished'] and A['finished'] and V['passed']
assert A['primary_checkpoint_unchanged']
assert A['protocol']['checkpoint_sha256']=='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
WIDTH=119/25.4
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,
    'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
    'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.65,'lines.linewidth':1.15,'pdf.fonttype':42,'savefig.dpi':220})
COLORS=['#3468a3','#008c7a','#bd753e','#bd4470']


def pool(rows,key):return float(np.sqrt(np.mean([r[key]**2 for r in rows])))
def group(d,e,family=None):
    return [r for r in d['records'] if r['split']=='test' and r['epsilon']==e and
            (family is None or r['family']==family)]


def main():
    levels=sorted({r['epsilon'] for r in P['records']})
    pr=P['profiles']; ar=A['protocol']['profiles']
    for rows in [pr,ar]:
        fit={r['input_sha256'] for r in rows if r['split']=='fit'}
        test={r['input_sha256'] for r in rows if r['split']=='test'}
        assert len(fit)==64 and len(test)==32 and not fit&test
    fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.8),sharey=True)
    fig.subplots_adjust(left=.135,right=.96,bottom=.30,top=.89,wspace=.18)
    configs=[(P,['first_order_contribution_mv','second_order_contribution_mv','higher_order_residual_mv','total_rmse_mv'],
              '(a) Passive L5PC control'),
             (A,['first_contribution_mv','second_contribution_mv','higher_order_residual_mv','main_drive_response_rmse_mv'],
              '(b) Frozen trained DBNN')]
    labels=[r'$\varepsilon\delta_1$',r'$\varepsilon^2\delta_2(W)$',r'$\|R_3\|_{\mathrm{RMS}}$','Total response RMSE']
    for ax,(d,keys,title) in zip(axs,configs):
        for key,label,col in zip(keys,labels,COLORS):
            ax.loglog(levels,[pool(group(d,e),key) for e in levels],'-o',ms=2.5,label=label,color=col)
        ax.set(xlim=(.008,10),ylim=(3e-10,2),xlabel=r'Input strength $\varepsilon$')
        ax.set_yticks([1e-9,1e-6,1e-3,1]);ax.set_title(title,loc='left',pad=6)
        ax.grid(which='major',color='#dbe2e8',lw=.5);ax.set_axisbelow(True)
    axs[0].set_ylabel('Pooled response error (mV)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.005),
               frameon=False,ncol=2,columnspacing=1.6,handlelength=2.3,labelspacing=.5)
    fig.savefig(F/'Fig4.pdf');fig.savefig(F/'trained_error_budget.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.75))
    fig.subplots_adjust(left=.12,right=.96,bottom=.30,top=.87,wspace=.42)
    distance=P['model_metadata']['distance_um']
    for idx,col,label,marker in [(range(8),'#c8534b','Excitatory','o'),(range(8,16),'#3468a3','Inhibitory','s')]:
        axs[0].plot([distance[j] for j in idx],[100*P['kernels'][j]['relative_rms'] for j in idx],
                    marker+'-',ms=3,lw=.8,color=col,label=label)
    axs[0].set(xlabel=r'Path distance from soma ($\mu$m)',ylabel='Isolated kernel RMS error (%)',ylim=(0,33),xticks=[0,600,1200])
    axs[0].set_title('(a) Passive one-kernel fits',loc='left',pad=6)
    axs[0].legend(frameon=False,fontsize=5.8,loc='upper left',handlelength=1.5)
    keys=['first_only_rmse_mv','main_drive_response_rmse_mv','projected_second_readout_rmse_mv','finite_voltage_conditional_readout_rmse_mv']
    names=['Filtered linear sum','Frozen primary readout',r'Fit to $V_2$','Fit to finite voltage']
    for key,label,col in zip(keys,names,[COLORS[0],COLORS[3],COLORS[1],'#8b76b1']):
        axs[1].loglog(levels,[pool(group(A,e),key) for e in levels],'-o',ms=2.4,label=label,color=col)
    axs[1].set(xlim=(.008,10),xlabel=r'Input strength $\varepsilon$',ylabel='Pooled response RMSE (mV)')
    axs[1].set_title('(b) Frozen active-cell features',loc='left',pad=6)
    for ax in axs:ax.grid(which='major',color='#dbe2e8',lw=.5);ax.set_axisbelow(True)
    fig.legend(*axs[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.005),
               frameon=False,ncol=2,columnspacing=1.3,handlelength=2,labelspacing=.5)
    fig.savefig(F/'Fig11.pdf');fig.savefig(F/'trained_diagnostic_controls.png');plt.close(fig)

    # Complete 16-site kernel errors; no selection by quality.
    iso={r['site_id']:r for r in A['isolated']};lines=[]
    for j,k in enumerate(P['kernels']):
        q=iso[k['site_id']];site=('E' if j<8 else 'I')+f" {k['site_id']}"
        lines.append(f"{site} & {distance[j]:.1f} & {k['target_rms_mv']:.5f} & {k['residual_rms_mv']:.5f} & {q['target_rms_mv']:.5f} & {q['rms_mv']:.5f} \\\\")
    (R/'sections/trained_kernel_table.tex').write_text(
        '\\begin{tabular}{lrrrrr}\n\\toprule\n'
        'Site & Distance & \\multicolumn{2}{c}{Passive fitted kernel} & \\multicolumn{2}{c}{Frozen trained kernel}\\\\\n'
        ' & ($\\mu$m) & $V_1$ RMS & Error RMS & $V_1$ RMS & Error RMS\\\\\n\\midrule\n'
        +'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n')
    lines=[];sweep=[]
    for e in [.01,.16,.5,1.,2.,4.,8.]:
        p=group(P,e);a=group(A,e)
        numbers=[pool(p,'fitted_features_first_only_rmse_mv'),pool(p,'total_rmse_mv')]+[
            pool(a,k) for k in ['first_only_rmse_mv','main_drive_response_rmse_mv','projected_second_readout_rmse_mv','finite_voltage_conditional_readout_rmse_mv']]
        lines.append(f'{e:g} & '+' & '.join(f'{x:.5f}' for x in numbers)+r' \\')
        sweep.append(dict(epsilon=e,passive_first_rmse_mv=numbers[0],passive_bilinear_rmse_mv=numbers[1],
                          active_first_rmse_mv=numbers[2],active_main_rmse_mv=numbers[3],
                          active_V2_projection_rmse_mv=numbers[4],active_voltage_fit_rmse_mv=numbers[5]))
    (R/'sections/trained_strength_table.tex').write_text(
        '\\begin{tabular}{r*{6}{r}}\n\\toprule\n'
        ' & \\multicolumn{2}{c}{Passive fitted features} & \\multicolumn{4}{c}{Frozen active-cell features}\\\\\n'
        '$\\varepsilon$ & Linear & Bilinear & Linear & Original $W$ & $V_2$ fit & Voltage fit\\\\\n\\midrule\n'
        +'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n')
    lines=[];control=[]
    for cell,d in [('Passive',P),('Active',A)]:
        for kind,name in [('half_extraction','Half strength'),('degree4','Degree 4'),('half_step','Half step')]:
            rows=[r for r in d['controls'] if r['control']==kind]
            vals=[max(r[k] for r in rows) for k in ['V1_rms_gap_mv','V2_rms_gap_mv','V2_relative_rms_gap']]
            lines.append(f'{cell} & {name} & {len(rows)} & {vals[0]:.2e} & {vals[1]:.2e} & {100*vals[2]:.3g} '+r' \\')
            control.append(dict(cell=cell,control=kind,cases=len(rows),max_V1_rms_gap_mv=vals[0],
                                max_V2_rms_gap_mv=vals[1],max_V2_relative_gap=vals[2]))
    (R/'sections/trained_control_table.tex').write_text(
        '\\begin{tabular}{llrrrr}\n\\toprule\nCell & Control & Inputs & $V_1$ (mV) & $V_2$ (mV) & Rel. $V_2$ (\\%)\\\\\n\\midrule\n'
        +'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n')
    # Independent recomputation of mismatch norms and their signed correlation.
    data=np.load(E/'trained_active_arrays.npz');i,j=np.triu_indices(16);cross={}
    for key in ['main_readout','second_projection_readout']:
        corr=[]
        for r in group(A,1.):
            name=r['name'];y=data[name+'_features'];phi=y[:,i]*y[:,j]
            e1=data[name+'_V1']-y.sum(1);e2=data[name+'_V2']-np.einsum('np,p->n',phi,data[key],optimize=False)
            assert abs(np.sqrt(np.mean(e1**2))-r['delta1_mv'])<1e-12
            corr.append(float(np.mean(e1*e2)))
        cross[key]=float(np.mean(corr))
    sl={}
    for key,d,value in [('passive_total',P,'total_rmse_mv'),('passive_remainder',P,'higher_order_residual_mv'),('active_response',A,'main_drive_response_rmse_mv')]:
        es=np.array(levels[:4]);v=np.array([pool(group(d,e),value) for e in es])
        sl[key]=float(np.polyfit(np.log(es),np.log(v),1)[0])
    audit=dict(passed=True,main_checkpoint_unchanged=True,fit_inputs=64,test_inputs=32,
               test_rasters_unique_and_disjoint_from_fit=True,candidate_sites=500,selected_sites=16,
               strengths=levels,sweep=sweep,controls=control,small_strength_slopes=sl,
               active_signed_first_second_correlation_mv2=cross,
               projection_is_fitting_set_only=True,
               source_sha256={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in [
                   'trained_passive_diagnostic.json','trained_active_diagnostic.json','trained_active_arrays.npz']})
    (E/'trained_diagnostic_paper_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(figures=['Fig4.pdf','Fig11.pdf'],tables=3,slopes=sl,correlations=cross),indent=2))


if __name__=='__main__':main()
