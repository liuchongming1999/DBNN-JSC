"""Publish completed baseline evaluations into the existing accuracy/runtime section.

The snapshot owns all new numbers. Unfinished evaluations are never plotted;
first engineering timings remain explicitly distinct from the final replay.
"""
import base64
import argparse
import copy
import hashlib
import html
import json
import math
import re
import shutil
import statistics
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.ticker import LogLocator, NullLocator, FuncFormatter
except ModuleNotFoundError:
    plt = None

HERE = Path(__file__).resolve().parent
TARGET = Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN/DBNN_L5PC_surrogate_report_20260928.html')
FIG = HERE / 'figures'
STEM = 'DBNN_resolution_frontier_20260929'
DT = {'B2': 1., 'B6': .5, 'B7': .25}
ORDER = ['gpu', 'cpu8', 'cpu1', 'neuron', 'core', 'tcn', 'elm', 'lstm', 'reduce']
COL = {'gpu': '#008675', 'cpu1': '#8870ad', 'cpu8': '#376aa5',
       'neuron': '#ce8730', 'core': '#b75170', 'tcn': '#c04f37',
       'elm': '#545d70', 'lstm': '#258eaf', 'reduce': '#a06e35'}
LABEL = {'gpu': 'DBNN GPU', 'cpu1': 'DBNN CPU, 1 core', 'cpu8': 'DBNN CPU, 8 cores',
         'neuron': 'NEURON, multisplit 8 cores', 'core': 'CoreNEURON CPU, 1 core',
         'tcn': 'TCN GPU', 'elm': 'Branch-ELM GPU', 'lstm': 'LSTM GPU',
         'reduce': 'Neuron_Reduce CPU, 1 core'}
ZH = {'gpu': 'DBNN GPU', 'cpu1': 'DBNN CPU 1 核', 'cpu8': 'DBNN CPU 8 核',
      'neuron': 'NEURON multisplit 8 核', 'core': 'CoreNEURON CPU 1 核',
      'tcn': 'TCN GPU', 'elm': 'Branch-ELM GPU', 'lstm': 'LSTM GPU',
      'reduce': 'Neuron_Reduce CPU 1 核'}
MARK = {1.: 'o', .5: 'D', .25: '^'}
REF = {
    'neuron': ('Hines et al. 2008', 'https://neuron.yale.edu/ftp/hines/tmp/multisplit.pdf', 'Fully implicit parallel simulation of single neurons'),
    'core': ('Kumbhar et al. 2019', 'https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00063/full', 'CoreNEURON: An Optimized Compute Engine for the NEURON Simulator'),
    'tcn': ('Beniaguev et al. 2021', 'https://doi.org/10.1016/j.neuron.2021.07.002', 'Single cortical neurons as deep artificial neural networks'),
    'elm': ('Spieler et al. 2024', 'https://proceedings.iclr.cc/paper_files/paper/2024/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html', 'The Expressive Leaky Memory Neuron'),
    'lstm': ('Hochreiter & Schmidhuber 1997', 'https://doi.org/10.1162/neco.1997.9.8.1735', 'Long Short-Term Memory'),
    'reduce': ('Amsalem et al. 2020', 'https://www.nature.com/articles/s41467-019-13932-6', 'An efficient analytical reduction of detailed nonlinear neuron models'),
}
if plt is not None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})


def esc(x): return html.escape(str(x))
def pct(x): return '—' if x is None else f'{100*x:.2f}%'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '<div class="resolution-table"><table><thead><tr>' + ''.join('<th>'+esc(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def get_points(payload, snapshot):
    refresh_publication(payload, snapshot)
    main = payload['current_main']
    fair = payload['compute_fair']
    expected = {r['case_id'] for r in main['cases']}
    inputs = {r['case_id']: r['input_sha256'] for r in fair['cases']}
    assert len(expected) == 31
    points = {}
    cpu_rows = []
    for backend in ['gpu', 'cpu8', 'cpu1']:
        points[backend] = []
        for ident, dt in DT.items():
            row = copy.deepcopy(main['backend_summary'][backend][ident])
            row.update(dt_ms=dt, phase='统一重测（30次）', repeats=30, checkpoint_sha256=main['checkpoint_sha256'][ident])
            points[backend].append(row)
    for method in ['neuron', 'core']:
        points[method] = []
        for dt in [1., .5, .25]:
            key = f'{method}_{dt:g}'
            counts = {r['solver_sources'][key]['timing']['solver_ms']['n'] for r in fair['cases']}
            assert len(counts) == 1
            points[method].append(dict(dt_ms=dt, metrics=copy.deepcopy(fair['summary'][key]),
                                      events=copy.deepcopy(fair['events'][key]),
                                      wall_ms=fair['totals_ms'][key], repeats=counts.pop(), phase='统一重测（30次）'))
    for method in ['tcn', 'elm', 'lstm']:
        rows = []
        for dt in [1., .5, .25]:
            experiment = snapshot['training'][f'{method}_{dt:g}ms']
            published = experiment.get('publication')
            e = published if published and published.get('finished') else experiment.get('evaluation')
            if not e or not e.get('finished'):
                continue
            assert e['checkpoint']['default_budget_completed']
            assert e['checkpoint']['finished']
            assert e['checkpoint']['updates'] == e['checkpoint']['total_updates']
            assert len(e['records']) == 31 and {r['case_id'] for r in e['records']} == expected
            assert e['runtime']['repeats'] == 30
            assert e['runtime']['checkpoint_sha256'] == e['checkpoint']['checkpoint_sha256']
            for r in e['records']:
                assert r['input_sha256'] == inputs[r['case_id']]
                assert len(r['backends']['gpu']['runs_ms']) == 30
            phase = '集中重测' if published and published.get('finished') else '首次工程测量'
            rows.append(dict(dt_ms=dt, metrics=copy.deepcopy(e['summary']['metrics']),
                             events=copy.deepcopy(e['summary']['events']), wall_ms=e['summary']['wall_ms']['gpu'],
                             phase=phase, checkpoint_sha256=e['runtime']['checkpoint_sha256'],
                             training_updates=e['checkpoint']['updates'],
                             timer=e['runtime']['timer'], repeats=e['runtime']['repeats'],
                             event_threshold=e['threshold']))
            cpu = {}
            for backend in ['cpu1', 'cpu8']:
                records = [r['backends'][backend] for r in e['records']]
                assert all(len(r['runs_ms']) == 30 for r in records)
                n = sum(r['metrics']['n'] for r in records)
                rmse = math.sqrt(sum(r['metrics']['sse'] for r in records)/n)
                time = sum(r['wall_ms'] for r in records)
                assert abs(time-e['summary']['wall_ms'][backend]) < 1e-7
                cpu[backend] = dict(pooled_rmse_mv=float(rmse), wall_ms=time)
            cpu_rows.append(dict(method=method, dt_ms=dt, phase=phase, backends=cpu))
        if rows:
            points[method] = rows
    rows = []
    for dt in [1., .5, .25]:
        e = snapshot['publication_results']['reduction'][str(dt)]
        if not e or not e.get('finished') or not e.get('full_main31'):
            continue
        assert len(e['records']) == 31 and {r['case_id'] for r in e['records']} == expected
        assert e['protocol']['repeats'] == 30
        rows.append(dict(dt_ms=dt, metrics=copy.deepcopy(e['summary']['metrics']),
                         events=copy.deepcopy(e['summary']['events']), wall_ms=e['summary']['wall_ms'],
                         phase='统一重测（30次）', timer=e['protocol']['scope'], repeats=e['protocol']['repeats']))
    if rows:
        assert snapshot['original']['reduce_legacy']['passed']
        points['reduce'] = rows
    for method in ['tcn', 'elm']:
        if method in points:
            assert snapshot['original'][method]['passed']
    return points, cpu_rows


def refresh_publication(payload, snapshot):
    """Replace current timing only after case, weight, repeat and output checks."""
    pub = snapshot['publication_results']
    assert pub['state']['finished'] and pub['state']['status'] == 'complete'
    main, fair = payload['current_main'], payload['compute_fair']
    inputs = {r['case_id']:r['input_sha256'] for r in fair['cases']}
    expected = set(inputs)
    assert len(expected) == 31
    if 'pre_publication_timing' not in payload:
        payload['pre_publication_timing'] = dict(backend_summary=copy.deepcopy(main['backend_summary']),
                                                solver_totals_ms=copy.deepcopy(fair['totals_ms']),
                                                duration_groups=copy.deepcopy(payload['resolution_frontier']['duration_groups']))
    for backend, data in pub['dbnn'].items():
        assert data['finished'] and len(data['records']) == 31
        assert {r['case_id'] for r in data['records']} == expected
        assert data['runtime']['repeats'] == 30
        for name in DT:
            assert data['runtime']['variants'][name]['checkpoint_sha256'] == main['checkpoint_sha256'][name]
            assert abs(data['summary'][name]['metrics']['pooled_rmse_mv'] - main['backend_summary'][backend][name]['metrics']['pooled_rmse_mv']) < 1e-5
            for key in ['tp', 'fp', 'fn']:
                assert data['summary'][name]['events'][key] == main['backend_summary'][backend][name]['events'][key]
        for row in data['records']:
            assert row['input_sha256'] == inputs[row['case_id']]
            for name in DT:
                variant = row['variants'][name]
                assert len(variant['wall']['runs_ms']) == 30
                assert variant['numerical']['passed'] and variant['numerical']['events_equal']
        main['backend_summary'][backend] = copy.deepcopy(data['summary'])
    main['summary'] = {name:copy.deepcopy(pub['dbnn']['gpu']['summary'][name]) for name in DT}
    main['cases'] = copy.deepcopy(pub['dbnn']['gpu']['records'])
    main['timing_phase'] = pub['state']
    lookup = {r['case_id']:r for r in fair['cases']}
    for key, data in pub['solvers'].items():
        assert data['status']['finished'] and data['status']['completed'] == 31
        assert len(data['records']) == 31 and {r['case_id'] for r in data['records']} == expected
        for row in data['records']:
            assert row['finished'] and row['input_sha256'] == inputs[row['case_id']]
            assert row['validation']['passed']
            assert row['validation']['voltage_max_abs_mv'] == 0
            assert len(row['timing']['solver_ms']['runs_ms']) == 30
            target = lookup[row['case_id']]
            target['timing_ms'][key] = row['timing']['solver_ms']['median_ms']
            target['solver_sources'][key] = row
        fair['totals_ms'][key] = sum(r['timing']['solver_ms']['median_ms'] for r in data['records'])
    fair['publication_timing_phase'] = pub['state']
    payload['resolution_frontier']['data'] = copy.deepcopy(pub['dbnn'])
    payload['resolution_frontier']['timing_phase'] = pub['state']


def quantile(values, fraction):
    values = sorted(values)
    x = (len(values)-1)*fraction
    lower = int(x)
    upper = min(lower+1, len(values)-1)
    return values[lower] + (values[upper]-values[lower])*(x-lower)


def duration_groups(payload, snapshot):
    data = snapshot['publication_results']['dbnn']
    lookup = {m:{r['case_id']:r for r in d['records']} for m,d in data.items()}
    fair = payload['compute_fair']
    solvers = {r['case_id']:r for r in fair['cases']}
    out = []
    for name, dt in DT.items():
        for group in fair['groups']:
            row = dict(config=name, dt_ms=dt, duration_s=group['duration_s'], n=group['n'])
            for method in ['gpu', 'cpu1', 'cpu8', 'neuron', 'core']:
                if method in data:
                    values = [lookup[method][cid]['variants'][name]['wall']['median_ms'] for cid in group['cases']]
                else:
                    values = [solvers[cid]['timing_ms'][f'{method}_{dt:g}'] for cid in group['cases']]
                row[method] = dict(median=statistics.median(values), q1=quantile(values,.25), q3=quantile(values,.75), sum=sum(values))
            for method in ['neuron', 'core']:
                row[f'ratio_{method}_gpu'] = row[method]['sum']/row['gpu']['sum']
                row[f'median_ratio_{method}_gpu'] = row[method]['median']/row['gpu']['median']
            out.append(row)
    payload['resolution_frontier']['duration_groups'] = out
    return out


def plot_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', color='#dce3e9', lw=.85)
    ax.grid(axis='x', which='major', color='#edf1f4', lw=.65)
    ax.tick_params(colors='#465568', labelsize=10, length=4)
    for name in ['left', 'bottom']:
        ax.spines[name].set_color('#9aa7b4')
        ax.spines[name].set_linewidth(.8)
    ax.xaxis.label.set_color('#24364a')
    ax.yaxis.label.set_color('#24364a')


def metric_x(row, squared=False):
    value = row['metrics']['pooled_rmse_mv']
    return value**2 if squared else value


def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/(stem+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(FIG/(stem+'.svg'), bbox_inches='tight')
    plt.close(fig)


def frontier(points):
    """One RMSE panel with curve labels, authors, and an interval-only legend."""
    assert plt is not None, 'Plot generation needs the benchmark Python runtime.'
    fig, ax = plt.subplots(figsize=(12.6, 10.0))
    fig.subplots_adjust(left=.105, right=.975, bottom=.145, top=.855)
    for method in ORDER:
        if method not in points:
            continue
        rows = points[method]
        ax.plot([metric_x(r) for r in rows], [r['wall_ms'] for r in rows],
                color=COL[method], lw=2.45, zorder=3,
                ls=(0, (5, 2.5)) if method in ['neuron', 'core', 'reduce'] else '-')
        for row in rows:
            ax.scatter(metric_x(row), row['wall_ms'], marker=MARK[row['dt_ms']],
                       s=103, color=COL[method], edgecolor='white', linewidth=1.25, zorder=5)
    xmin = min(metric_x(r) for rows in points.values() for r in rows)
    xmax = max(metric_x(r) for rows in points.values() for r in rows)
    ax.set(xscale='log', yscale='log', xlim=(xmin/1.45, xmax*1.7),
           ylim=(.9, 300000), xlabel='Masked pooled RMSE (mV)',
           ylabel='Sum of 31 median compute wall times (ms)')
    ax.set_xticks([.02, .05, .1, .2, .5, 1, 2, 5, 10])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    plot_style(ax)

    # Text positions are in plot coordinates; thin leaders identify close curves.
    positions = {
        'core': (.13, .96, 2),
        'neuron': (.045, .755, 1),
        'cpu1': (.405, .445, 0),
        'cpu8': (.415, .235, 0),
        'gpu': (.075, .145, 1),
        'elm': (.525, .80, 2),
        'tcn': (.635, .585, 2),
        'lstm': (.565, .435, 2),
        'reduce': (.715, .775, 2),
    }
    for method in ORDER:
        if method not in points:
            continue
        px, py, idx = positions[method]
        row = points[method][min(idx, len(points[method])-1)]
        name = LABEL[method].replace(', ', ' · ')
        author = REF[method][0] if method in REF else 'This work'
        ax.annotate(name+'\n'+author,
                    xy=(metric_x(row), row['wall_ms']), xycoords='data',
                    xytext=(px, py), textcoords='axes fraction',
                    ha='left', va='center', color=COL[method], fontsize=11.2,
                    linespacing=1.5, weight='bold',
                    bbox=dict(boxstyle='round,pad=.22', fc='white', ec='none', alpha=.94),
                    arrowprops=dict(arrowstyle='-', lw=1.0, color=COL[method], alpha=.65,
                                    shrinkA=6, shrinkB=7, connectionstyle='arc3,rad=0'), zorder=7)

    fig.canvas.draw()
    gpu, neuron = points['gpu'][0], points['neuron'][0]
    axes_xy = lambda x, y: ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
    source = axes_xy(metric_x(neuron), neuron['wall_ms'])
    target = axes_xy(metric_x(gpu), gpu['wall_ms'])
    arrow_path = MplPath([(source[0]-.007, source[1]-.013), (.215, .58), (.225, .22), (target[0]+.010, target[1]+.010)],
                         [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=arrow_path, transform=ax.transAxes, arrowstyle='-|>',
                                mutation_scale=17, color='#344958', lw=1.7, zorder=4))
    ax.text(.263, .275, f'{neuron["wall_ms"]/gpu["wall_ms"]:,.0f}× faster\nat 1 ms',
            transform=ax.transAxes, ha='center', va='center', fontsize=10.6, weight='bold',
            color='#344958', bbox=dict(boxstyle='round,pad=.30', fc='white', ec='#bac7ce', lw=.8), zorder=8)
    ax.text(.014, .025, '← Lower error   ·   Lower time ↓', transform=ax.transAxes,
            color='#687988', fontsize=10)
    fig.text(.105, .955, 'Accuracy–runtime trade-off', fontsize=19, weight='bold', color='#20364a')
    fig.text(.105, .916, '31 evaluation windows  ·  Reference spikes ±10 ms excluded  ·  Lower-left is better',
             fontsize=11.0, color='#617080')
    fig.text(.105, .884, '27 working points · 30 repeats per window · Prepared complete computation · Shared server',
             fontsize=9.7, color='#71808c')
    shape = [Line2D([0], [0], color='#586675', marker=MARK[dt], ls='none', ms=8,
                    label=f'{dt:g} ms') for dt in [1., .5, .25]]
    fig.legend(handles=shape, ncol=3, loc='lower center', bbox_to_anchor=(.54, .057),
               frameon=False, fontsize=11, columnspacing=2.6, handletextpad=.7)
    fig.text(.54, .030, 'Marker shape: model sampling interval / numerical-solver timestep',
             ha='center', fontsize=9.0, color='#6f7e8b')
    save(fig, STEM)


def durations_plot(groups):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.9), sharey=True)
    for ax, dt in zip(axes, [1., .5, .25]):
        rows = [r for r in groups if r['dt_ms'] == dt]
        xs = [r['duration_s'] for r in rows]
        for method in ['gpu', 'cpu8', 'cpu1', 'neuron', 'core']:
            ys = [r[method]['median'] for r in rows]
            ax.plot(xs, ys, color=COL[method], marker=MARK[dt], ms=6.5, mec='white', mew=.8,
                    lw=2, ls='--' if method in ['neuron', 'core'] else '-')
        ax.set(xscale='log', yscale='log', xlabel='Simulated duration (s)', title=f'{dt:g} ms interval',
               xlim=(.85, 42), ylim=(.006, 20000))
        ax.set_xticks(xs, [f'{r["duration_s"]:g}\nn={r["n"]}' for r in rows])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        plot_style(ax)
        at16 = next(r for r in rows if r['duration_s'] == 16)
        low, high, x = at16['gpu']['median'], at16['neuron']['median'], 27.
        ax.hlines([low, high], 16, x, color='#8596a7', ls=':', lw=.9)
        ax.annotate('', xy=(x,high), xytext=(x,low), arrowprops=dict(arrowstyle='<->',color='#536b7a',lw=1.1))
        ax.text(x, math.sqrt(low*high), f'{high/low:,.0f}×\nat 16 s', ha='center', va='center',
                fontsize=11.3, weight='bold', color=COL['gpu'], bbox=dict(boxstyle='round,pad=.3',fc='white',ec='none'))
        ax.annotate(f'{high:,.0f} ms', (x,high), xytext=(0,8), textcoords='offset points', ha='center', fontsize=8, color='#82602f')
        ax.annotate(f'{low:.3f} ms', (x,low), xytext=(0,-14), textcoords='offset points', ha='center', fontsize=8, color=COL['gpu'])
    axes[0].set_ylabel('Median window compute wall time (ms)', fontsize=11)
    handles = [Line2D([0],[0],color=COL[m],lw=2,label=LABEL[m]) for m in ['gpu','cpu8','cpu1','neuron','core']]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(.5,.040), frameon=False, fontsize=9.5)
    fig.suptitle('Runtime by simulation duration', fontsize=16, weight='bold', color='#20364a', y=.985)
    fig.text(.5,.927,'16 s annotations: NEURON multisplit 8 cores / DBNN GPU · ratio of plotted medians',ha='center',fontsize=10,color='#617080')
    fig.text(.5,.005,'30 repeats per window · Prepared complete computation · Shared server',ha='center',fontsize=8.5,color='#6f7e8b')
    fig.tight_layout(rect=[0,.155,1,.91])
    save(fig, 'resolution_durations')


def ve_plot(points):
    fig, axes = plt.subplots(1,2,figsize=(13,6.1),sharey=True)
    for ax,key,label in zip(axes,['mean_case_ve','within_trace_weighted_ve'],['1 − mean per-window VE','1 − variance-weighted VE']):
        for method in ['gpu','cpu8','cpu1','neuron','core']:
            rows=points[method]
            xs=[max(1e-8,1-r['metrics'].get(key,r['metrics'].get('within_trace_ve'))) for r in rows]
            ax.plot(xs,[r['wall_ms'] for r in rows],color=COL[method],lw=2)
            for x,r in zip(xs,rows):ax.scatter(x,r['wall_ms'],marker=MARK[r['dt_ms']],s=65,color=COL[method],edgecolor='white')
        ax.set(xscale='log',yscale='log',xlabel=label,ylim=(.9,300000))
        plot_style(ax)
    axes[0].set_ylabel('Sum of 31 median compute wall times (ms)')
    fig.legend(handles=[Line2D([0],[0],color=COL[m],lw=2,label=LABEL[m]) for m in ['gpu','cpu8','cpu1','neuron','core']],ncol=3,loc='lower center',frameon=False)
    fig.suptitle('VE aggregation diagnostics · current 30-repeat timings',color='#20364a',weight='bold')
    fig.tight_layout(rect=[0,.13,1,.94])
    save(fig,'resolution_ve')


def gpu_detail(points):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, squared, label in zip(axes, [False, True], ['Masked pooled RMSE (mV)', 'Masked pooled MSE (mV²)']):
        rows = points['gpu']
        ax.plot([metric_x(r, squared) for r in rows], [r['wall_ms'] for r in rows], color=COL['gpu'], lw=1.7)
        for r in rows:
            ax.scatter(metric_x(r, squared), r['wall_ms'], color=COL['gpu'], marker=MARK[r['dt_ms']],
                       edgecolor='white', s=90, label=f'{r["dt_ms"]:g} ms')
        ax.set(xlabel=label, ylabel='GPU compute wall sum (ms)')
        ax.margins(x=.18, y=.2)
        plot_style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc='lower center', frameon=False)
    fig.suptitle('DBNN GPU · same working points as the main frontier', fontsize=12, color='#20364a')
    fig.tight_layout(rect=[0, .10, 1, .94])
    save(fig, 'DBNN_completed_gpu_detail_20261001')


def image(stem, caption):
    blob = (FIG/(stem+'.png')).read_bytes()
    w, h = (int.from_bytes(blob[a:a+4], 'big') for a in [16, 20])
    return f'<figure><img loading="lazy" width="{w}" height="{h}" src="data:image/png;base64,{base64.b64encode(blob).decode()}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'


def duration_html(old_section, groups, snapshot):
    start = old_section.index('<span id="gpu-compute-optimization">')
    end = old_section.index('<span id="duration-timing">', start)
    s = old_section[start:end]
    s += '<span id="duration-timing"></span><h3>4.2 分模拟时长查看耗时与加速</h3>'
    s += image('resolution_durations', '每个时长组均包含原31窗中对应的全部输入。各窗重复30次，图中取组内窗口中位数；16 s箭头标注NEURON multisplit 8核／DBNN GPU的中位数比。')
    parts=[]
    for r in groups:
        if r['duration_s'] == 16:
            parts.append(f'{r["dt_ms"]:g} ms：NEURON {r["neuron"]["median"]:,.1f} ms → DBNN GPU {r["gpu"]["median"]:.3f} ms，约 <strong>{r["median_ratio_neuron_gpu"]:,.0f}×</strong>')
    s += '<p><strong>16 s窗口：</strong>'+'；'.join(parts)+'。倍率均由图中的组内中位数计算。</p>'
    s += table(['时间间隔','时长 / s','窗口数','GPU / ms','CPU1 / ms','CPU8 / ms','NEURON8 / ms','CoreNEURON1 / ms','NEURON/GPU','CoreNEURON/GPU'],
               [[f'{r["dt_ms"]:g} ms',f'{r["duration_s"]:g}',r['n'],*[f'{r[m]["median"]:.3f}' for m in ['gpu','cpu1','cpu8','neuron','core']],f'{r["median_ratio_neuron_gpu"]:,.0f}×',f'{r["median_ratio_core_gpu"]:,.0f}×'] for r in groups])
    s += '<p>主图使用31窗中位数的总时间比，本节使用各时长组内的中位数比。原始JSON同时保存组内四分位数和总时间，分组不改变输入或模型权重。</p>'
    rows=[]
    for backend,data in snapshot['publication_results']['dbnn'].items():
        for r in data['records']:
            for name,dt in DT.items():
                v=r['variants'][name];m=v['metrics'];e=v['events']
                rows.append([r['case_id'],ZH[backend],f'{dt:g} ms',r['duration_s'],r['A'],f'{m["rmse_mv"]:.4f}',pct(m['ve']),f'{e["tp"]}/{e["fp"]}/{e["fn"]}',f'{v["wall"]["median_ms"]:.4f}',str(v.get('backend',v.get('mode','—')))])
    s += '<details><summary>逐窗口的全部DBNN计算与误差记录</summary>'+table(['窗口','后端','间隔','T / s','位点A','RMSE','VE','TP/FP/FN','计算wall / ms','实测策略'],rows)+'</details>'
    s += '<button id="download-neuron">下载当前主线与历史研究 JSON</button> <a href="DBNN_resolution_frontier_20260929.svg" download>下载主图 SVG</a> <a href="DBNN_resolution_frontier_20260929.png" download>下载主图 PNG</a></section>'
    return s


def refresh_other_main_sections(text, payload, snapshot, points):
    gpu=snapshot['publication_results']['dbnn']['gpu']['summary']
    def replace_section(ident, new):
        nonlocal text
        pattern=r'<section id="'+ident+r'">.*?</section>'
        text,count=re.subn(pattern,lambda _:new,text,count=1,flags=re.S)
        assert count == 1
    s='<section id="conclusion"><span id="backup-neuron-main" class="expanded-anchor"></span><h2>主线：固定模型结构，三个分辨率的精度与速度</h2>'
    s+='<p>完整主动通道的L5PC、500个候选突触位点、31条输入，以NEURON dt=0.1 ms为参考。当前1／0.5／0.25 ms模型沿用此前锁定的权重，所有方法均已完成每窗30次的统一重测。</p>'
    s+=table(['采样间隔','掩码RMSE / mV','平均VE','Precision','Recall','31窗GPU wall / ms','同数字间隔NEURON8 / GPU'],
             [[f'{dt:g} ms',f'{gpu[name]["metrics"]["pooled_rmse_mv"]:.4f}',pct(gpu[name]['metrics']['mean_case_ve']),pct(gpu[name]['events']['precision']),pct(gpu[name]['events']['recall']),f'{gpu[name]["wall_ms"]:.3f}',f'{points["neuron"][i]["wall_ms"]/gpu[name]["wall_ms"]:,.0f}×'] for i,(name,dt) in enumerate(DT.items())])
    s+=f'<p>0.5 → 0.25 ms的pooled RMSE降低{100*(1-gpu["B7"]["metrics"]["pooled_rmse_mv"]/gpu["B6"]["metrics"]["pooled_rmse_mv"]):.2f}%，GPU计算wall time增加{100*(gpu["B7"]["wall_ms"]/gpu["B6"]["wall_ms"]-1):.1f}%。倍率比较相同数值的采样间隔／求解步长；各自实际误差见第4节横轴。</p>'
    s+='<p><a href="#accuracy-development">训练流程与原始checkpoint对照</a> → <a href="#voltage-consistency">电压与事件</a> → <a href="#expanded-frontier">精度—速度曲线</a>。不同脑区和细胞的完整结果见<a href="#multineuron-generalization">附录AD</a>。</p></section>'
    replace_section('conclusion',s)
    old=re.search(r'<section id="accuracy-development">.*?</section>',text,re.S)[0]
    old=re.sub(r'<div id="latest-timing-note".*?</div>','',old,flags=re.S)
    note='<div id="latest-timing-note" class="note"><strong>本次同步的是已锁定模型的重测记录。</strong>三个checkpoint、31组输入及评分规则与此前一致，RMSE四位小数和尖峰统计一致。新增结果为2026年10月2–3日完成的统一测速，每窗重复30次。旧批次有并发GPU负载，本次开始测速时所用GPU处于空闲状态；共享服务器的资源记录随JSON保存。</div>'
    old=old.replace('</h2>','</h2>'+note,1)
    pos=old.index('<h3>2.3 ')
    head,tail=old[:pos],old[pos:]
    improvement=[]
    for name in ['U0',*DT]:
        m=gpu[name]['metrics'];e=gpu[name]['events']
        improvement.append(['用户原始checkpoint' if name=='U0' else f'DBNN {DT[name]:g} ms',f'{m["pooled_rmse_mv"]:.4f}',pct(m['mean_case_ve']),pct(m['within_trace_weighted_ve']),f'{e["tp"]}/{e["fp"]}/{e["fn"]}',pct(e['precision']),pct(e['recall'])])
    tail=re.sub(r'<div class="resolution-table">.*?</div>',lambda _:table(['模型','31窗RMSE / mV','平均VE','方差加权VE','TP/FP/FN','Precision','Recall'],improvement),tail,count=1,flags=re.S)
    old=head+tail
    old=re.sub(r'<h3>2.4 .*?(?=</section>)','',old,flags=re.S)
    details='<h3>2.4 锁定权重后的统一测速</h3><p>GPU、CPU1和CPU8加载相同分辨率的同一checkpoint；完整计算包含滤波、二次项、线性项、偏置及事件reset。外部打包、上传、下载和编译在计时外完成。</p>'
    pub=snapshot['publication_results']['dbnn']
    details+=table(['间隔','参考尖峰数','TP/FP/FN','尖峰F1','GPU wall / ms','GPU设备 / ms','CPU1 wall / ms','CPU8 wall / ms'],
                   [[f'{dt:g} ms',gpu[name]['events']['tp']+gpu[name]['events']['fn'],f'{gpu[name]["events"]["tp"]}/{gpu[name]["events"]["fp"]}/{gpu[name]["events"]["fn"]}',pct(gpu[name]['events']['f1']),f'{gpu[name]["wall_ms"]:.3f}',f'{gpu[name]["device_ms"]:.3f}',f'{pub["cpu1"]["summary"][name]["wall_ms"]:.3f}',f'{pub["cpu8"]["summary"][name]["wall_ms"]:.3f}'] for name,dt in DT.items()])
    details+='<p>31窗包含25个无尖峰窗口、3个单尖峰窗口、2个双尖峰窗口和1个三尖峰窗口，共10个参考尖峰。各窗先以±5 ms容差一一匹配，再合计TP、FP、FN计算F1；无尖峰窗口中的额外预测计入FP。</p>'
    replace_section('accuracy-development',old.replace('</section>',details+'</section>'))
    ab=re.search(r'<section id="ve-frontier-diagnostics">.*?</section>',text,re.S)[0]
    ab=re.sub(r'<figure>.*?</figure>',lambda _:image('resolution_ve','原五种主线计算方式的VE诊断；误差使用相同31窗掩码，纵轴更新为本次30次统一重测结果。'),ab,count=1,flags=re.S)
    replace_section('ve-frontier-diagnostics',ab)
    next_section=re.search(r'<section id="next">.*?</section>',text,re.S)[0]
    maximum=max(r['variants'][name]['numerical']['max_abs_mv'] for d in pub.values() for r in d['records'] for name in DT)
    next_section=re.sub(r'最大电压差 [\d.eE+\-]+ mV',f'最大电压差 {maximum:.2e} mV',next_section)
    replace_section('next',next_section)
    return text


def build_section(old_section, points, cpu_rows, snapshot, groups):
    suffix = duration_html(old_section, groups, snapshot)
    complete = []
    for method in ['tcn', 'elm', 'lstm', 'reduce']:
        if method in points:
            complete.append(ZH[method]+'（'+'／'.join(f'{r["dt_ms"]:g}' for r in points[method])+' ms）')
    missing = [f'{LABEL[m]} {dt:g} ms' for m in ['tcn', 'elm', 'lstm'] for dt in [1., .5, .25]
               if not any(r['dt_ms'] == dt for r in points.get(m, []))]
    s = '<section id="comparison-protocol"><h2>4. 不同方法与采样间隔的速度—精度关系</h2>'
    s += '<p>每个工作点都对应同一组31条输入窗口，横轴使用共同1 ms观察网格与参考尖峰前后±10 ms的同一掩码，纵轴为逐窗完整计算wall time中位数之和，<strong>越靠左下越好</strong>。颜色区分方法，圆／菱形／三角表示1／0.5／0.25 ms；对DBNN和学习基线是模型采样间隔，对NEURON、CoreNEURON和Neuron_Reduce是求解步长。</p>'
    s += f'<div class="note"><strong>27个工作点的统一重测全部完成。</strong>本次纳入{esc("、".join(complete))}，并同步DBNN GPU／CPU1／CPU8、NEURON multisplit8和CoreNEURON CPU1。所有方法每窗均重复30次；学习模型使用此前锁定的权重与作者默认训练流程。证据快照：{esc(snapshot["created"])}。'
    if missing:
        s += '尚待完整评估：'+esc('、'.join(missing))+'。'
    s += '</div><span id="expanded-frontier"></span>'
    caption = '主图展示masked pooled RMSE与完整计算耗时，横纵轴均使用对数尺度。模型名及作者年份以粗体标在对应曲线附近；底部仅保留1／0.5／0.25 ms的形状图例。深灰箭头连接NEURON multisplit 8核与DBNN GPU的1 ms工作点，速度倍率为其31窗计算时间合计之比。TCN、Branch-ELM、LSTM显示GPU结果，Neuron_Reduce显示CPU单核结果。'
    s += image(STEM, caption)
    s += '<p>图中加速倍率比较相同数字的时间间隔，实际误差由横轴给出。学习基线沿用原作者损失、优化器与预算，电压仿射校准和事件阈值由验证集锁定；原始未校准指标及公开结果复核见<a href="#baseline-reproduction">附录AC</a>。VE曲线与聚合说明见<a href="#ve-frontier-diagnostics">附录AB</a>。下表保留MSE（RMSE²）供数值核对。</p>'
    refs = '；'.join(f'{esc(LABEL[m])}：<a href="{esc(REF[m][1])}" target="_blank" rel="noopener">{esc(REF[m][0])}</a>' for m in ['neuron', 'core', 'tcn', 'elm', 'lstm', 'reduce'])
    s += '<details><summary>图中作者与年份的文献来源</summary><p>'+refs+'。</p><p>LSTM的模型来源为Hochreiter与Schmidhuber的1997年论文；本次神经元拟合的训练配置参考<a href="https://arxiv.org/html/2306.16922v3" target="_blank" rel="noopener">Spieler et al. 2024的NeuronIO实验与Table S10</a>。DBNN标为This work。</p></details>'
    rows = []
    for method in ORDER:
        for r in points.get(method, []):
            method_label = ZH[method]+(' · '+REF[method][0] if method in REF else ' · This work')
            rows.append([method_label, f'{r["dt_ms"]:g} ms', f'{metric_x(r):.4f}', f'{metric_x(r, True):.5f}',
                         pct(r['events'].get('f1')), f'{r["wall_ms"]:,.3f}',
                         f'{r["device_ms"]:,.3f}' if r.get('device_ms') is not None else '—', r['phase']])
    s += table(['方法', '时间间隔', 'RMSE / mV', 'MSE / mV²', '事件F1', '完整计算wall / ms', 'CUDA设备 / ms', '计时阶段'], rows)
    if cpu_rows:
        s += '<details><summary>学习基线的CPU 1核／8核完整计算结果</summary><p>使用与主图相同的锁定权重和31窗输入。下表RMSE分别由两个CPU后端的电压输出汇总；完整推理含时间递归、重叠拼接与输出计算。CPU事件阈值重算随集中评估完成。</p>'
        s += table(['方法', '采样间隔', 'CPU1 RMSE / mV', 'CPU8 RMSE / mV', 'CPU1 wall / ms', 'CPU8 wall / ms', '阶段'],
                   [[LABEL[r['method']].replace(' GPU', ''), f'{r["dt_ms"]:g} ms',
                     f'{r["backends"]["cpu1"]["pooled_rmse_mv"]:.4f}', f'{r["backends"]["cpu8"]["pooled_rmse_mv"]:.4f}',
                     f'{r["backends"]["cpu1"]["wall_ms"]:,.3f}', f'{r["backends"]["cpu8"]["wall_ms"]:,.3f}', r['phase']]
                    for r in cpu_rows])
        s += '</details>'
    s += '<h3>4.1 统一计时范围与当前测量阶段</h3>'
    s += table(['方法', '计入完整计算', '计时外准备'], [
        ['DBNN GPU', 'CUDA Graph调用、滤波、二次项＋线性项＋偏置、事件reset，以及等待结果完成', '模型加载与编译、Graph捕获、active IDs与权重准备、CPU打包、H2D／D2H'],
        ['DBNN CPU 1／8核', '已准备输入的active-site读取、滤波、完整驱动与事件reset；输出写入预分配内存', '权重与系数准备、外部事件栅格打包、工作区分配'],
        ['TCN／Branch-ELM／LSTM', '完整forward（含递归／重叠与burn-in拼接）、验证校准、sigmoid输出；GPU计入完成同步', '模型加载、编译／Graph捕获、外部打包、输入上传与输出下载'],
        ['NEURON multisplit 8核／CoreNEURON 1核', '完整目标时段的求解调用及同步', '形态与通道构建、输入事件准备、500 ms预热、外部记录插值与数据整理'],
        ['Neuron_Reduce CPU 1核', '降阶后的完整目标时段求解及电压记录', '形态构建与降阶、事件准备、500 ms预热与输出插值']])
    s += '<p>各方法先取每窗30次计算的中位数，再对31窗求和。CPU为AMD EPYC 9554的1或8个物理核心，GPU为RTX 5090；DBNN三个分辨率在各后端内随机交错运行。2026年10月2–3日的重测使用锁定权重和原输入，求解器输出与原相同步长记录逐窗核对一致。主图采用包括完成同步的计算wall time，主表另保留CUDA设备时间。全部记录来自共享服务器，并保存各批次资源状态。</p>'
    return s + suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', type=Path)
    parser.add_argument('--assemble-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        bundle = json.loads(args.plot_only.read_text())
        points = bundle.get('points', bundle)
        frontier(points)
        if 'duration_groups' in bundle:
            durations_plot(bundle['duration_groups'])
            ve_plot(points)
        print(json.dumps({'generated': str(FIG/STEM)}, ensure_ascii=False))
        return
    snapshot_path = HERE/'snapshot/status.json'
    snapshot = json.loads(snapshot_path.read_text())
    old = TARGET.read_text()
    backup = HERE/'snapshot/report_before_latest_results.html'
    if not backup.exists():
        shutil.copy2(TARGET, backup)
    match = re.search(r'<script type="application/json" id="neuron-data">(.*?)</script>', old, re.S)
    payload = json.loads(match[1])
    points, cpu_rows = get_points(payload, snapshot)
    groups = duration_groups(payload, snapshot)
    (HERE/'plot_input.json').write_text(json.dumps(dict(points=points, duration_groups=groups), ensure_ascii=False, indent=2))
    if args.prepare_only:
        print(json.dumps(dict(prepared=str(HERE/'plot_input.json'), total_points=sum(map(len,points.values())), duration_groups=len(groups))))
        return
    if not args.assemble_only:
        frontier(points)
        durations_plot(groups)
        ve_plot(points)
    original_section = re.search(r'<section id="comparison-protocol">.*?</section>', old, re.S)[0]
    new_section = build_section(original_section, points, cpu_rows, snapshot, groups)
    evidence = dict(snapshot=snapshot['created'], snapshot_sha256=sha(snapshot_path),
                    scope='31 same windows; common 1ms observation grid; reference spikes ±10ms excluded',
                    aggregate='Pooled RMSE = sqrt(sum masked SSE / sum masked points); MSE = RMSE²',
                    timer='Sum of 31 median prepared-complete-computation wall times; all methods 30 repeats; shared server',
                    timing_phase=snapshot['publication'],points=points,learned_cpu=cpu_rows,duration_groups=groups)
    payload['current_comparison'] = evidence
    text = old.replace(original_section, new_section, 1)
    text = refresh_other_main_sections(text,payload,snapshot,points)
    payload['updated']=snapshot['created']
    text = text.replace('4. 三种分辨率的速度—精度', '4. 方法与采样的速度—精度')
    date = snapshot['created'][:10]
    time = snapshot['created'][11:16]
    text = re.sub(r'DBNN / L5PC SURROGATE DEVELOPMENT · \d{4}-\d{2}-\d{2}',
                  'DBNN / L5PC SURROGATE DEVELOPMENT · '+date, text)
    text = re.sub(r'<footer>.*?</footer>',
                  f'<footer>更新于 {date} {time} · L5PC 27个工作点均完成30次重测 · 31窗相同参考与掩码 · 六种细胞的推广结果见附录AD · 共享服务器实测。</footer>', text, count=1, flags=re.S)
    blob = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    text = re.sub(r'<script type="application/json" id="neuron-data">.*?</script>',
                  lambda _: '<script type="application/json" id="neuron-data">'+blob+'</script>', text, count=1, flags=re.S)
    ids = re.findall(r'\bid="([^"]+)"', text)
    assert len(ids) == len(set(ids))
    assert all(anchor in ids for anchor in re.findall(r'href="#([^"]+)"', text))
    preserved = original_section[original_section.index('<span id="gpu-compute-optimization">'):original_section.index('<span id="duration-timing">')]
    assert preserved in new_section
    source_file = HERE/'current_main_comparison.json'
    source_file.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    for stem in [STEM,'resolution_durations','resolution_ve']:
        for extension in ['png', 'svg']:
            shutil.copy2(FIG/(stem+'.'+extension), TARGET.parent/(stem+'.'+extension))
    TARGET.write_text(text)
    print(json.dumps(dict(report=str(TARGET), snapshot=snapshot['created'],
                          plotted_points={LABEL[m]: len(points[m]) for m in ORDER if m in points},
                          total_points=sum(map(len, points.values())), cpu_rows=len(cpu_rows),
                          duration_groups_updated=len(groups), implementation_preserved=True, anchors_valid=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
