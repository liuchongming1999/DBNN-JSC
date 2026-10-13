"""Source-faithful forward replacements and actual original reset benchmarks."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def build_baseline_section(here, table, pic, save):
    audit = json.loads((here/'original_baseline_audit.json').read_text())
    assert audit.get('finished') and len(audit['cases']) == 4
    cases = audit['cases']
    stage_keys = ['filter', 'drive_cublas', 'drive_direct', 'reset']
    def ms(c, key):
        return c['forward'][key]['median_ms']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for ax, kind in zip(axes, ['F', 'R']):
        ratios = np.array([[ms(c, f'original_{kind}')/ms(c, f'{kind}_only_{k}') for k in stage_keys] for c in cases])
        im = ax.imshow(np.log2(ratios), cmap='RdYlGn', aspect='auto',
                       norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=8))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{ratios[i,j]:.2f}x', ha='center', va='center', fontsize=10,
                        color='white' if np.log2(ratios[i,j]) > 5 else '#182230')
        ax.set_xticks(range(4), ['Triton\nfilter', 'cuBLAS\nfull drive', 'Single-kernel\nfull drive', 'Event CUDA\nreset'], fontsize=8)
        ax.set_title(f'Unmodified {kind} / {kind} with one stage replaced', fontsize=11)
        ax.grid(False)
    axes[0].set_yticks(range(4), [f'A={c["A"]}, B={c["B"]}' for c in cases])
    fig.suptitle('Complete-forward wall time, original batch dispatch retained', fontsize=12)
    fig.tight_layout(rect=[0, 0, .94, .92])
    cax = fig.add_axes([.948, .17, .012, .63])
    fig.colorbar(im, cax=cax, label='log2(speed ratio)')
    save(fig, 'original_one_stage')

    fig, ax = plt.subplots(figsize=(10, 4.6))
    xs = np.arange(4); width = .25
    methods = [('Original reset: batch call', '#6b7787', 'original_reset'),
               ('Original reset: B=1 trace loop', '#3168c9', 'original_reset_trace_loop'),
               ('Event CUDA reset', '#11866b', 'event_cuda')]
    for offset, (name, color, key) in zip([-width, 0, width], methods):
        values = [c['reset_module'].get(key, c['reset_module']['original_reset'])['median_ms'] for c in cases]
        bars = ax.bar(xs+offset, values, width, label=name, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.1, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs, [f'A={c["A"]}, B={c["B"]}\n{sum(c["reset_events_per_trace"])} reset events total' for c in cases], fontsize=9)
    ax.set_yscale('log'); ax.set_ylim(.07, 900)
    ax.set_ylabel('Reset-only synchronized wall time (ms)')
    ax.set_title('Same fixed drive u[n]; directly call the original reset function')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout(); save(fig, 'original_reset_module')

    rows, details, summary = [], [], []
    for c in cases:
        condition = f'A={c["A"]}, B={c["B"]}'
        for kind in ['F','R']:
            before = ms(c, f'original_{kind}')
            rows.append([condition, kind, f'{before:.3f}'] + [
                f'{ms(c, f"{kind}_only_{k}"):.3f}（{before/ms(c, f"{kind}_only_{k}"):.2f}×）'
                for k in stage_keys])
        original_keys = [k for k in c['forward'] if k.startswith('original_')]
        best = min(original_keys, key=lambda k:ms(c,k))
        summary.append([condition, best, f'{ms(c,best):.3f}', f'{ms(c,"optimized_eager"):.3f}',
                        f'{ms(c,"optimized_graph"):.3f}', f'{ms(c,best)/ms(c,"optimized_graph"):.2f}×'])
        drows = []
        for scope in ['forward','reset_module']:
            for key, value in c[scope].items():
                drows.append([scope, key, f'{value["median_ms"]:.3f}', f'{value["p95_ms"]:.3f}',
                              f'{value["std_ms"]:.3f}', value['n']])
        details.append(f'<details><summary>{condition}：原 F/R、逐条调用与单模块替换的全部记录</summary>'+
                       table(['范围','方法','median / ms','p95 / ms','std / ms','次数'],drows)+'</details>')
    max_error = max(e['max_abs_mv'] for c in cases for field in ['errors_vs_original_R','reset_errors'] for e in c[field].values())
    mismatches = max(e['spike_mismatch_bins'] for c in cases for field in ['errors_vs_original_R','reset_errors'] for e in c[field].values())
    b1 = cases[0]; b8 = cases[2]
    content = f'''<section id="original-audit"><h2>2.1 逐层核对：直接从原 F / R 代码替换一个模块</h2>
<p>新增这组实验专门量化五层表中各改动的作用。两边都使用同一活跃位点输入、同一 checkpoint，
对照直接调用冻结的 <code>DBNN_F.forward / DBNN_R.forward</code>。原 R 的推理缓存保持开启并预热；
其 256 步驱动项循环保留。新版本每次仅替换滤波、完整 drive 或 reset 中的一项。</p>
<div class="note">4 种形状，每方法 30 次随机交错，物理 GPU {audit['physical_gpu']}、主机单线程绑定 CPU 48。
两端均采用 eager 调用；记录完整 forward 的同步 wall time。GPU 与其他任务共享。
完整 drive 包括二次项、线性项、偏置及必要布局转换。新实现的系数与工作区在准备阶段创建。</div>
<figure>{pic('original_one_stage','原始 F 和 R 分别仅替换滤波、完整驱动项和 reset 后的整体速度比')}<figcaption>
每格的分子都是对应原 F 或 R 的整批调用耗时，分母只替换列出的那一项。B=8 的原 reset 是 Python/PyTorch 时间格循环；
后面另列原 B=1 事件分支逐条调用的更快对照。单项比值分别反映同一起点上的变化。</figcaption></figure>
{table(['条件（6 s）','原版','原 forward / ms','只换滤波 / ms（速度比）','只换 cuBLAS drive / ms（速度比）','只换单核 drive / ms（速度比）','只换 reset / ms（速度比）'],rows)}
<p>A=32、B=1 时，只换滤波使原 F 从 {ms(b1,'original_F'):.3f} 降至 {ms(b1,'F_only_filter'):.3f} ms，
原 R 从 {ms(b1,'original_R'):.3f} 降至 {ms(b1,'R_only_filter'):.3f} ms。
完整驱动项融合也有形状依赖：A=500、B=1 时，给原 F 换单核 drive 后耗时增加，原表保留该结果。
B=8 时，滤波和 drive 的单独改动被原 reset 的较长耗时掩盖；整体提速需要一并处理 reset。</p>
<h3>reset：原函数的两种真实路径</h3>
<p>固定原 R 生成的同一条 u[n]，直接调用 <code>_apply_reset_closed_loop</code>。
B=1 原函数已按事件推进；B&gt;1 原函数逐时间格推进。第二个对照把 batch 拆开，逐条调用原 B=1 事件分支。
图内同时标注触发次数：前三组为零事件，A=500、B=8 合计 11 次事件；性能结论对应这一触发密度。</p>
<figure>{pic('original_reset_module','同一驱动曲线下原整批 reset、原事件分支逐条调用及事件 CUDA 的模块耗时')}<figcaption>
单独 reset 的同步 wall time。原函数包含输出分配及指数幂构造；事件 CUDA 复用工作区和指数幂。
B=1 时前两柱相同。模块数据与完整 forward 的倍数分开报告。</figcaption></figure>
<h3>把优化组合后，与原始代码中更快的调用方式比较</h3>
<p>同一轮内同时测原 F、原 R，以及 B&gt;1 时逐条调用它们的 B=1 分支，取较快者。
下表最终路径固定为 Triton＋cuBLAS＋事件 CUDA，并分别报告 eager / Graph；第 2 节仍保留原有 14 场景的完整探索与独立复测。</p>
{table(['条件','较快的原始调用','原代码 / ms','优化 eager / ms','优化 Graph / ms','原代码 / 优化 Graph'],summary)}
<p>全部新记录相对原输出的最大电压误差为 {max_error:.2e} mV，reset 时间格差为 {mismatches}。
该误差核对覆盖输出电压与阈值事件；代理相对 NEURON 的拟合精度继续见第 5 节。</p>
<details><summary>全部原代码配对记录与耗时分布</summary>{''.join(details)}</details>
<p class="muted">数据：<code>original_baseline_audit.json</code>；实验：<code>original_baseline_audit.py</code>；
逐语句替换后的 forward：<code>original_baseline_variants.txt</code>。原源文件及 checkpoint 的 SHA256 与第 2 节一致。</p>
</section>'''
    return content, audit
