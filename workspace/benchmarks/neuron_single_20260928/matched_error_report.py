"""Plots and main narrative for the matched-RMSE single-cell comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
R = H / 'matched_error_20260929'
FIG = H / 'figures'
NAMES = {'neuron': 'NEURON CPU 1T', 'core_cpu': 'CoreNEURON CPU 1T',
         'core_gpu': 'CoreNEURON GPU'}
COLORS = {'neuron': '#657887', 'core_cpu': '#3477bb', 'core_gpu': '#b77a2a', 'dbnn': '#087f69'}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.png', dpi=190, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def figures(d):
    budget = d['dbnn']['pooled_rmse_mv']
    latency = d['dbnn']['response']['median_ms']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for mode in NAMES:
        rows = [r for r in d['rows'] if r['mode'] == mode]
        if not rows:
            continue
        x = [r['dt_ms'] for r in rows]
        e = [r['pooled_rmse_mv'] for r in rows]
        y = [r['response']['median_ms'] for r in rows]
        style = dict(color=COLORS[mode], marker='o', ms=5, lw=1.2, label=NAMES[mode])
        ax[0].plot(x, e, **style)
        ax[1].plot(e, y, **style)
        selected = d['selected'][mode]
        ax[1].scatter([selected['pooled_rmse_mv']], [selected['response']['median_ms']],
                      s=100, facecolors='none', edgecolors=COLORS[mode], linewidths=1.8)
        for r in rows:
            if mode == 'neuron' and r['dt_ms'] in [.1, .5, 4.]:
                ax[1].annotate(f"{r['dt_ms']:g}", (r['pooled_rmse_mv'], r['response']['median_ms']),
                               xytext=(4, 6), textcoords='offset points', fontsize=7, color=COLORS[mode])
        shift = {'neuron': (-83, 12), 'core_cpu': (-83, -14), 'core_gpu': (-83, 10)}[mode]
        ax[1].annotate(f"dt={selected['dt_ms']:.3f}",
                       (selected['pooled_rmse_mv'], selected['response']['median_ms']),
                       xytext=shift, textcoords='offset points', fontsize=8, color=COLORS[mode],
                       arrowprops=dict(arrowstyle='-', color=COLORS[mode], lw=.6),
                       bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
    ax[0].axhline(budget, color=COLORS['dbnn'], ls='--', label=f'DBNN RMSE = {budget:.3f} mV')
    ax[0].set(xlabel='Solver step dt (ms)', ylabel='Pooled RMSE vs NEURON dt=0.1 ms (mV)',
              title='Accuracy as the solver step is relaxed')
    ax[0].set_xscale('log'); ax[0].set_yscale('symlog', linthresh=.03)
    ax[0].set_ylim(0, max(q['pooled_rmse_mv'] for q in d['rows'])*1.4)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].axvspan(0, budget, color=COLORS['dbnn'], alpha=.07)
    ax[1].axvline(budget, color=COLORS['dbnn'], ls='--', lw=1)
    ax[1].scatter([budget], [latency], marker='*', s=150, color=COLORS['dbnn'], zorder=5)
    ax[1].annotate(f'DBNN {latency:.3f} ms', (budget, latency), xytext=(-10, 12),
                   textcoords='offset points', ha='right', color=COLORS['dbnn'], fontsize=9)
    ax[1].set(xlabel='Pooled RMSE (mV; labels indicate dt in ms)',
              ylabel='Wall time per 6 s input (ms, log scale)', title='Measured accuracy–latency frontier')
    ax[1].set_xscale('symlog', linthresh=.03); ax[1].set_yscale('log')
    save(fig, 'matched_error_frontier')

    labels, vals, colors = [], [], []
    for mode in NAMES:
        r = d['selected'][mode]
        labels.append(NAMES[mode].replace(' CPU', '\nCPU').replace(' GPU', '\nGPU') + f"\ndt={r['dt_ms']:g} ms")
        vals.append(r['response']['median_ms']); colors.append(COLORS[mode])
    labels.append('DBNN\noptimized GPU'); vals.append(latency); colors.append(COLORS['dbnn'])
    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    bars = ax.bar(labels, vals, color=colors, width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.2, f'{v:,.3f} ms', ha='center', fontsize=9)
    ax.set_yscale('log'); ax.set_ylim(.4, max(vals)*3)
    ax.set_ylabel('Wall time per input (ms, log scale)')
    ax.set_title(f'Fastest measured configurations with pooled RMSE ≤ {budget:.3f} mV')
    save(fig, 'matched_error_speed')

    ref = np.load(R/'neuron_dt0.1.npz', allow_pickle=True)['voltage'][0]
    chosen = d['selected']['neuron']
    nv = np.load((R/chosen['source_file']).with_suffix('.npz'), allow_pickle=True)['voltage'][0]
    dv = np.concatenate([np.load(R/f'dbnn_sample{s}.npz')['voltage'] for s in range(3)])
    t = np.arange(6001)/1000.
    fig, ax = plt.subplots(3, 1, figsize=(9.7, 6.5), sharex=True)
    for s in range(3):
        ax[s].plot(t, ref[s], color='#303c46', lw=1, label='NEURON reference, dt=0.1 ms')
        ax[s].plot(t, nv[s], color='#bd873a', lw=.8, alpha=.85, label=f'NEURON dt={chosen["dt_ms"]:g} ms')
        ax[s].plot(t, dv[s], color=COLORS['dbnn'], lw=.8, alpha=.8, label='DBNN optimized')
        ax[s].set_ylabel('Voltage (mV)')
        ax[s].set_title(f'Input {s}: simulator RMSE {chosen["per_sample_rmse_mv"][s]:.3f}; DBNN {d["dbnn"]["per_sample_rmse_mv"][s]:.3f} mV', fontsize=9)
    ax[0].legend(frameon=False, ncol=3, fontsize=8)
    ax[-1].set_xlabel('Physical time (s)')
    save(fig, 'matched_error_voltage')


def sections(d, table, img):
    budget = d['dbnn']['pooled_rmse_mv']
    latency = d['dbnn']['response']['median_ms']
    selected = d['selected']
    rows = []
    for mode in NAMES:
        r = selected[mode]
        rows.append([NAMES[mode], f'{r["dt_ms"]:g}', f'{r["pooled_rmse_mv"]:.4f}',
                     f'{r["response"]["median_ms"]:,.3f}', f'{r["response"]["median_ms"]/latency:,.0f}×',
                     '满足' if r['within_every_sample_budget'] else '见逐条约束对照'])
    rows.append(['DBNN optimized GPU', '输出间隔 1 ms', f'{budget:.4f}', f'{latency:.3f}', '1×', '误差预算来源'])
    r = selected['neuron']; c = selected['core_cpu']; g = selected['core_gpu']
    main = f'''<section id="conclusion"><h2>主线：以相近的轨迹误差比较速度</h2>
<p>固定同一只小鼠 V1 L5PC、500 个输入位点和 3 条各 6 s 的输入。以 <strong>NEURON、dt=0.1 ms</strong> 为参考，先计算 DBNN 的误差，再放宽详细求解器的步长，在同一 RMSE 预算内选择最快的实测配置。</p>
<div class="stats"><div><span>DBNN 总体 RMSE</span><b>{budget:.3f} mV</b><span>3 × 6,001 个电压采样点</span></div><div><span>NEURON / DBNN</span><b>{r['response']['median_ms']/latency:,.0f}×</b><span>NEURON dt={r['dt_ms']:g} ms，误差预算内</span></div><div><span>DBNN 完整路径</span><b>{latency:.3f} ms</b><span>每条 6 s 输入；B=1</span></div></div>
<p>主表同时给出实际 RMSE 和时间；步长由本轮测量网格选择。当前使用共享服务器，CPU 绑定固定核心，GPU 6 上仍有其他任务。速度比对应这些资源条件、输入和已训练的 DBNN checkpoint。</p></section>'''
    methods = f'''<section id="neuron-comparison"><h2>1. 共同比较对象与运行形式</h2>
{table(['方法', '计算方式', '本轮变化量'], [['NEURON CPU 1T', '同一形态、离子通道和突触事件的固定步长求解', 'dt；其余生物物理参数固定'], ['CoreNEURON CPU 1T', '同一机制的优化 CPU 引擎', 'dt；与 NEURON 使用相同输入'], ['CoreNEURON GPU', 'NVHPC 25.3 / CUDA 12.8 / sm_120，RTX 5090', 'dt；cell_permute=2，warp_balance=4'], ['DBNN optimized GPU', '固定 checkpoint 的数据驱动代理模型', 'active-site packing、Triton 前缀扫描、完整驱动项融合、event reset、缓存 / CUDA Graph，FP32']])}
<p>DBNN 保持此前选定的全部优化。每次处理一条轨迹（B=1）；三条输入依次执行。NEURON / CoreNEURON 的显式输入事件时间固定，较大 dt 引入的电压与事件响应变化直接反映在误差中。</p></section>'''
    errors = []
    for s in range(3):
        errors.append([str(s), f'{d["dbnn"]["per_sample_rmse_mv"][s]:.4f}', f'{r["per_sample_rmse_mv"][s]:.4f}',
                       str(len(d['reference']['spikes'][s])), str(len(r['spikes'][s])),
                       str(len(d['dbnn']['threshold_events_ms'][s]))])
    voltage = f'''<section id="voltage-consistency"><h2>2. 固定参考轨迹与误差预算</h2>
<p>所有输出统一到物理时间 0–6,000 ms、间隔 1 ms 的 6,001 点网格。求解器在自身步长上记录电压，随后按时间线性插值到该网格；插值耗时计入结果取回。DBNN 的原生输出间隔为 1 ms。</p>
<p class="model-equations"><strong>RMSE(m) = √[ Σ<sub>s=0,1,2</sub> Σ<sub>k=0,…,6000</sub> (v<sub>m,s</sub>[k] − v<sub>ref,s</sub>[k])² / (3 × 6001) ]</strong></p>
<p>本轮 DBNN 的总体误差预算为 <strong>{budget:.6f} mV</strong>。这里先汇总全部采样点的平方误差再开方；每条轨迹的 RMSE 也单独列出。dt=0.1 ms 是本次约定的数值参考。</p>
{img('matched_error_voltage', '同一参考下：DBNN 与放宽步长后的 NEURON 胞体电压，包含有尖峰的输入')}
{table(['输入样本', 'DBNN RMSE / mV', f'NEURON dt={r["dt_ms"]:g} ms RMSE / mV', '参考尖峰数', '该步长尖峰数', 'DBNN 阈值事件数'], errors)}
<p>NEURON 尖峰数采用胞体上穿 −20 mV 的事件记录；DBNN 使用 checkpoint 的 reset 阈值 {d['dbnn']['reset_threshold_mv']:g} mV。样本 1 中参考轨迹有 {len(d['reference']['spikes'][1])} 次尖峰，匹配步长的 NEURON 有 {len(r['spikes'][1])} 次，DBNN 有 {len(d['dbnn']['threshold_events_ms'][1])} 次阈值事件。本报告的“同等精度”具体指全时段电压 RMSE，事件指标按各自定义并列展示。</p></section>'''
    strict_rows = []
    for mode in NAMES:
        sr = d['strict_selected'].get(mode)
        if sr:
            strict_rows.append([NAMES[mode], f'{sr["dt_ms"]:g}', f'{sr["pooled_rmse_mv"]:.4f}',
                                f'{sr["response"]["median_ms"]:,.3f}', f'{sr["response"]["median_ms"]/latency:,.0f}×'])
    grid = [[NAMES[q['mode']], f'{q["dt_ms"]:g}', f'{q["pooled_rmse_mv"]:.4f}',
             ' / '.join(f'{v:.3f}' for v in q['per_sample_rmse_mv']), f'{q["response"]["median_ms"]:,.3f}',
             '满足' if q['within_pooled_budget'] else '超出', '满足' if q['within_every_sample_budget'] else '超出'] for q in d['rows']]
    # JSON is embedded by the parent report; this section needs no remote dependencies.
    agreement = max(q['max_abs_mv'] for q in d['cpu_gpu_agreement'])
    protocol = f'''<section id="comparison-protocol"><h2>3. 同一 RMSE 预算内的速度比较</h2>
<p>筛选规则：对每个求解器，在已测试步长中保留 <strong>总体 RMSE ≤ {budget:.4f} mV</strong> 的配置，再取实测“求解＋电压取回”时间最短者。选取真实测量点，表中列出各自达到的精度。</p>
{table(['方法', '求解步长 / ms', '总体 RMSE / mV', '每条输入 wall time / ms', '时间 / DBNN', '每条误差也≤DBNN'], rows)}
{img('matched_error_speed', '总体 RMSE 预算内，各求解器最快的已测配置与完整优化 DBNN 的速度')}
{img('matched_error_frontier', '放宽步长后的误差变化与速度—精度关系；绿色区域为 DBNN 的总体误差预算')}
<p>含尖峰样本的全时段 RMSE 随步长可能非单调：步长同时影响尖峰时刻、峰值与采样点位置。因此扫描所有候选后按实测误差筛选，误差曲线保留这些波动。</p>
<h3>更严格的逐条输入约束</h3><p>进一步要求三条输入分别满足 RMSE<sub>solver,s</sub> ≤ RMSE<sub>DBNN,s</sub>，在这个集合中重新选择最快配置：</p>
{table(['方法', '求解步长 / ms', '总体 RMSE / mV', '每条输入 wall time / ms', '时间 / DBNN'], strict_rows)}
<h3>计时与复核口径</h3><ol><li>NEURON / CoreNEURON：预热一轮，再重复 3 轮；每轮依次处理三条输入。配对相加 solver_call 与 output_copy，求每轮单条平均后取中位数。胞体记录、CoreNEURON 在线数据交接以及输出插值都计入。</li><li>DBNN：每条输入重复 40 次；包含稠密输入整理、H2D、完整优化推理、reset 和 D2H；按相同输入平均后取中位数。各条中位数为 {'、'.join(f'{x:.3f}' for x in d['dbnn']['response']['per_sample_median_ms'])} ms。</li><li>编译、模型构建、权重准备、首次 Graph capture 以及 NEURON 静息预热和输入事件队列建立单独记录。主表比较已准备输入的稳态调用；原始记录也保留包含预热和事件队列建立的 trial_e2e。</li><li>全部使用相同输入 hash、位点集合和参考轨迹；已配对步长下 CoreNEURON GPU / CPU 电压最大差异为 {agreement:.3g} mV。</li></ol>
<p>当前匹配精度的配置中，CoreNEURON GPU / CPU 的 wall time 比为 <strong>{g['response']['median_ms']/c['response']['median_ms']:.1f}×</strong>。这个数值对应单个细胞、当前记录与在线交接方式以及共享 GPU 环境。</p>
<details><summary>展开完整步长扫描表</summary>{table(['方法', 'dt / ms', '总体 RMSE', '三条 RMSE / mV', 'wall time / ms', '总体约束', '逐条约束'], grid)}</details>
<p><button id="download-neuron">下载同精度比较数据 JSON</button></p></section>'''
    nxt = '''<section id="next"><h2>4. 当前范围与下一步</h2><p>本轮已完成同一误差预算下的实测比较，范围为固定小鼠 V1 L5PC、冻结 DBNN checkpoint 和三条 6 s 输入。后续增加输入样本，并在预留 CPU / GPU 资源下重复；同时以更小参考步长检查结论的稳定性。连续在线窗口、状态迁移和更密位点采样仍作为应用扩展。</p></section>'''
    return [main, methods, voltage, protocol, nxt]
