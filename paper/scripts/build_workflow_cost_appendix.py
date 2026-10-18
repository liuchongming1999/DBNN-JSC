"""Build appendix-only workflow tables and cumulative costs from frozen records."""
from pathlib import Path
import gzip,json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];E=R/'evidence'
S=json.loads((E/'workflow_cost_summary.json').read_text())
A=json.loads((E/'workflow_cost_offline_audit.json').read_text())
V=json.loads((E/'workflow_cost_integrity_validation.json').read_text())
raw=json.loads(gzip.decompress((E/'workflow_cost_records_20261016.json.gz').read_bytes()))
assert V['passed'] and S['finished'] and S['inputs']==379
assert S['frozen_checkpoint_sha256']==json.loads((E/'main379_results.json').read_text())['checkpoint_sha256']
assert len(raw['solver'])==379
assert all(x['validation']['events_equal'] and x['validation']['max_abs_mv']<.001 for x in raw['solver'])
solver_batches=[sum(x['records'][i]['resident_ms'] for x in raw['solver'])/1000 for i in range(10)]
neuron_panel=float(np.median(solver_batches))
assert np.isclose(neuron_panel,S['solver_panel_aggregation']['median_panel_s'],rtol=1e-12)
for method in ['gpu','cpu8','cpu1']:
    passes=raw['dbnn'][method];assert all(len(rs)==379 for rs in passes)
    maps=[{x['case_id']:x for x in rs} for rs in passes]
    for boundary in ['prepared','resident_reuse_layout','resident_new_layout']:
        total=sum(np.median(sum([m[cid]['timing'][boundary]['runs_ms'] for m in maps],[])) for cid in maps[0])/1000
        assert np.isclose(total,S['methods'][method]['boundaries'][boundary]['sum_input_medians_s'],rtol=1e-12)
    assert all(x['validation']['events_equal'] and x['validation']['max_abs_mv']<.001 for rs in passes for x in rs)
for method in ['gpu','cpu8']:
    totals=[sum(x['elapsed_ms'] for x in p['records'])/1000 for p in raw['stream'][method]]
    assert np.allclose(totals,S['stream'][method]['rounds_s'],rtol=1e-12)
    assert all(sum(x['layout_cache_hit'] for x in p['records'])==350 for p in raw['stream'][method])
offline=raw['data_generation']['observed_batch_elapsed_s']+float(np.median(raw['preparation_process_s']))+A['training_and_validation_calibration_elapsed_s']
assert np.isclose(offline,S['offline']['offline_parallel_generation_s'],rtol=1e-12)
for method in ['gpu','cpu8']:
    saving=(neuron_panel-S['stream'][method]['median_panel_s'])/379
    assert math.ceil(offline/saving)==S['breakeven'][method]['offline_parallel_generation_s']['calls']

def table(path,header,rows,columns):
    text='\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (R/'sections'/path).write_text(text)
rows=[]
for i,(boundary,name) in enumerate([('prepared','Prepared'),('resident_reuse_layout','Reused layout'),('resident_new_layout','Rebuilt layout')],1):
    ref=S['solver_total_s']['prepared_ms' if boundary=='prepared' else 'resident_ms']
    ts=[S['methods'][m]['boundaries'][boundary]['sum_input_medians_s'] for m in ['gpu','cpu8','cpu1']]
    rows.append(f'{i}. {name} & {ref:.2f} & {ts[0]:.4f} & {ts[1]:.3f} & {ts[2]:.3f} \\\\')
rows.append(f"4. Cached layouts & {neuron_panel:.2f} & {S['stream']['gpu']['median_panel_s']:.4f} & {S['stream']['cpu8']['median_panel_s']:.3f} & --- \\\\")
table('workflow_timing_table.tex','Boundary & NEURON & GPU & CPU, 8 cores & CPU, 1 core',rows,'lrrrr')
rows=[]
for method,name in [('neuron8','NEURON, 8 cores'),('cpu8','DBNN CPU, 8 cores'),('gpu','DBNN GPU')]:
    c=S['cold'][method];rt=c['runtime_call_ms']['median']/1000;total=c['fresh_process_total_ms']
    rows.append(f"{name} & {rt:.3f} & {total['median']/1000:.3f} & {total['min']/1000:.3f}--{total['max']/1000:.3f} \\\\")
table('workflow_first_call_table.tex','Method & Runtime & Process & Process range',rows,'lrrr')
rows=[f"Reference generation (16 workers) & {raw['data_generation']['observed_batch_elapsed_s']:.2f} \\\\",
      f"Startup and training-data preparation & {np.median(raw['preparation_process_s']):.2f} \\\\",
      f"Optimization, validation, and calibration & {A['training_and_validation_calibration_elapsed_s']:.2f} \\\\",
      r'\midrule',f'Combined one-time estimate & {offline:.2f} \\\\']
table('workflow_offline_table.tex','Component & Time (s)',rows,'lr')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(119/25.4,2.8));fig.subplots_adjust(left=.14,right=.98,bottom=.17,top=.98)
k=np.geomspace(1,10000,300);colors={'neuron8':'#d48b24','gpu':'#008d81','cpu8':'#386ba5'}
ax.plot(k,k*neuron_panel/379,lw=1.4,color=colors['neuron8'],label='NEURON, 8 cores')
for method,label in [('gpu','DBNN GPU, including one-time cost'),('cpu8','DBNN CPU, 8 cores, including one-time cost')]:
    own=S['stream'][method]['median_panel_s']
    ax.plot(k,offline+k*own/379,lw=1.4,color=colors[method],label=label)
    if method=='gpu':
        b=S['breakeven'][method]['offline_parallel_generation_s']['calls']
        ax.scatter(b,b*neuron_panel/379,s=15,color=colors[method],zorder=5)
        ax.annotate(f'{b} calls',xy=(b,b*neuron_panel/379),xytext=(18,550),fontsize=7,
             color=colors[method],arrowprops=dict(arrowstyle='->',lw=.7,color=colors[method]))
ax.set(xscale='log',yscale='log',xlabel='Number of complete calls',ylabel='Cumulative elapsed time (s)')
ax.grid(alpha=.16,lw=.6);ax.legend(loc='upper left',frameon=False,fontsize=6,handlelength=1.4)
fig.savefig(R/'figures/Fig10.pdf');fig.savefig(R/'figures/workflow_cost.png',dpi=220);plt.close(fig)
check=dict(passed=True,raw_record_bundle_sha256=hashlib.sha256((E/'workflow_cost_records_20261016.json.gz').read_bytes()).hexdigest(),
           primary_figure_and_headline_scope='prepared computation, unchanged',inputs=379,
           primary_checkpoint_matches=True,complete_call_gpu_ratio=S['stream']['gpu']['ratio_vs_solver_resident'],
           one_time_cost_s=offline,gpu_break_even_calls=137,cpu8_break_even_calls=138,
           raw_timings_reaggregated=True)
(E/'workflow_cost_verification.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check,indent=2))
