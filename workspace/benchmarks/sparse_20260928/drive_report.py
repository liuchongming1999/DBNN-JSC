"""Show complete-drive fusion and its matched full-forward impact."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from quadratic_report import COLORS, LABELS

METHODS = ['original_F', 'original_R', 'cublas', 'cutlass', 'direct']


def build_drive_section(here, table, pic, save, quadratic_data):
    first = json.loads((here/'drive_forward.json').read_text())
    repeat = json.loads((here/'drive_forward_repeat.json').read_text())
    if not first.get('finished') or not repeat.get('finished'):
        raise RuntimeError('Full-drive forward comparisons must finish before publication')
    f1 = {c['index']: c for c in first['cases']}
    f2 = {c['index']: c for c in repeat['cases']}
    q1 = {c['index']-8: c for c in quadratic_data['first']['cases'] if c['T'] == 6001}
    q2 = {c['index']-8: c for c in quadratic_data['repeat']['cases']}
    assert set(f1) == set(f2) == set(q1) == set(q2) == set(range(8))
    for i in f2:
        for field in ['Y_sha256', 'W_sha256']:
            assert f1[i][field] == f2[i][field] == q1[i][field] == q2[i][field]

    def us(c, method, mode='device'):
        d = c['device_per_call'][method] if mode == 'device' else c['wall'][f'{method}/{mode}']
        return 1000*d['median_ms']

    def drive(i):
        return q2[i]['scopes']['drive']

    def ratio(c, method, mode='device'):
        return us(c, 'original_R', mode)/us(c, method, mode)

    names = {'original_F': 'F whole-window expression', 'original_R': 'R expression, whole-window GEMM',
             'cublas': 'cuBLAS + fused full reduction', 'cutlass': 'CUTLASS SIMT + fused reduction',
             'direct': 'One-kernel full drive'}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, B in zip(axs, [1, 8]):
        indices = [i for i in range(8) if f2[i]['B'] == B]
        for m in METHODS:
            ax.plot([f2[i]['A'] for i in indices], [us(drive(i), m) for i in indices],
                    'o--' if m == 'original_F' else 'o-', label=names[m],
                    color=COLORS[m], lw=1.8, markersize=4)
        ax.set_xscale('log', base=2)
        ax.set_xticks([8, 32, 128, 500], ['8', '32', '128', '500'])
        ax.set_yscale('log')
        ax.set_title(f'Complete drive: quadratic + linear + bias, B={B}', fontsize=11)
        ax.set_xlabel('Active sites A; T=6001')
        ax.set_ylabel('Steady device time / call (microseconds)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, .14, 1, 1])
    save(fig, 'drive_fusion_device')

    cols = ['cublas', 'cutlass', 'direct']
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.9), sharey=True)
    for ax, scope, title in zip(axs, ['drive', 'forward'],
                               ['Complete-drive module', 'Full forward: same scan and reset']):
        values = np.array([[ratio(drive(i) if scope == 'drive' else f2[i], m)
                            for m in cols] for i in range(8)])
        im = ax.imshow(np.log2(values), cmap='RdYlGn',
                       norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1.5), aspect='auto')
        for i in range(8):
            for j in range(3):
                ax.text(j, i, f'{values[i,j]:.2f}x', ha='center', va='center', fontsize=10,
                        color='white' if np.log2(values[i,j]) < -2 else '#183044')
        ax.set_xticks(range(3), ['cuBLAS\n+ full reduction', 'CUTLASS SIMT\n+ full reduction',
                                'One-kernel\nfull drive'], fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(False)
    axs[0].set_yticks(range(8), [f'A={f2[i]["A"]}, B={f2[i]["B"]}' for i in range(8)])
    fig.suptitle('Rebatched R expression / fused implementation  (>1 is faster)', fontsize=12)
    fig.tight_layout(rect=[0, 0, .94, .94])
    cax = fig.add_axes([.945, .15, .015, .68])
    fig.colorbar(im, cax=cax, label='log2(speed ratio)')
    save(fig, 'drive_fusion_forward_ratios')

    drive_rows, forward_rows, wall_rows, stability = [], [], [], []
    for i, c in sorted(f2.items()):
        condition = f'A={c["A"]}, B={c["B"]}'
        drive_rows.append([condition]+[f'{us(drive(i), m):.2f}' for m in METHODS])
        forward_rows.append([condition]+[f'{us(c, m):.2f}' for m in METHODS])
        wall_rows.append([condition]+[f'{us(c, m, "graph"):.2f}' for m in METHODS])
        stability.append([condition]+[
            f'{ratio(f1[i], m):.2f}× → {ratio(c, m):.2f}×' for m in cols])
    heads = ['条件（6 s）', 'F 整窗表达式', 'R 整窗表达式', 'cuBLAS＋融合归约',
             'CUTLASS＋融合归约', '单 kernel 全融合']
    detail = []
    for i, c in sorted(f2.items()):
        rows = []
        for m in METHODS:
            for mode in ['eager', 'graph']:
                s = c['wall'][f'{m}/{mode}']
                rows.append([names[m], mode, f'{s["median_ms"]*1000:.2f}',
                             f'{s["p95_ms"]*1000:.2f}', f'{s["std_ms"]*1000:.2f}', s['n']])
        detail.append(f'<details><summary>A={c["A"]}, B={c["B"]}：完整前向 wall 分布</summary>' +
                      table(['实现', '调用方式', 'median / μs', 'p95 / μs', 'std / μs', '次数'], rows)+'</details>')
    max_error = max(e['max_abs_mv'] for data in [f1, f2] for c in data.values()
                    for e in c['errors_vs_frozen_full_original_R'].values())
    reset_diff = max(e['spike_mismatch_bins'] for data in [f1, f2] for c in data.values()
                    for e in c['errors_vs_frozen_full_original_R'].values())
    content = f'''<section id="drive-analysis"><h2>4.1 内部消融：完整驱动项的不同实现</h2>
    <p class="note">本节两端均采用准备好的整窗布局；后半节还共用优化后的滤波与事件 reset。
    “R 整窗表达式”将原 R 的 mm＋sum 重排为一次整窗 GEMM，省去了原来的 256 步调用循环。
    这些记录用于内部算子选择。保留原分块、滤波和 reset 的直接配对见 <a href="#original-audit">第 2.1 节</a>。</p>
    <p><strong>后端选择应以完整驱动项和完整前向为依据。</strong>单看 q=yᵀWy 会漏掉线性求和与偏置的调度、访存成本。
    把三项一起计算后，A=32、B=1 的设备耗时为 R 整窗式 {us(drive(1),'original_R'):.2f} μs，
    cuBLAS＋融合归约 {us(drive(1),'cublas'):.2f} μs（{ratio(drive(1),'cublas'):.2f}×），
    单 kernel 全融合 {us(drive(1),'direct'):.2f} μs（{ratio(drive(1),'direct'):.2f}×）。
    整窗 <code>mm + sum</code> 表达式保留为算子对照，按形状筛选完整驱动项实现。</p>
    <div class="formula">v<sub>drive,r</sub> = ΣᵢYᵣᵢ(YW)ᵣᵢ + ΣᵢYᵣᵢ + b<br>
    = Σᵢ[Yᵣᵢ(YW)ᵣᵢ + Yᵣᵢ] + b</div>
    <p>Y 已包含每个位点的时间滤波和 ωᵢ。这里线性项是 ΣᵢYᵣᵢ，偏置来自 checkpoint 的 <code>bilinear.bias</code>。</p>
    {table(['实现','实际操作','融合范围'], [
        ['R 整窗表达式','Z=Y@W；q=(Y*Z).sum(-1)；linear=Y.sum(-1)；drive=linear+q+b','保留原表达式的多个 PyTorch 算子；统一用整窗 GEMM'],
        ['cuBLAS / CUTLASS＋融合归约','先求 Z=YW，再用一个 CUDA 核求 sum(Y*Z+Y)+b','逐元素乘、线性项、求和和偏置全部合并；矩阵乘法仍是单独阶段'],
        ['单 kernel 全融合','每个时间点一个 block，在共享内存读取 y，直接求双重求和、线性项和偏置','三项在一个 CUDA kernel 中完成；无需写出中间矩阵 Z']])}
    <p>全融合路径已在 <code>native_kernels.cu::direct_kernel</code> 中实现；
    <code>reduce_kernel</code> 是 GEMM 后的完整归约核。两者都包含线性项与偏置。
    本节把此前已保留的 drive 独立计时放到正文，并给出在优化路径内“仅更换 drive”的完整前向配对实验。</p>
    <h3>完整驱动项：相同 Y、W 和偏置下的设备耗时</h3>
    <p class="muted">单位 μs，来自 <code>quadratic_repeat.json / scopes.drive</code>。
    每方法 60 次设备测量，Graph 连续执行 16/32 次后摊销；固定输入、热缓存、FP32 且关闭 TF32。</p>
    {table(heads,drive_rows)}
    <figure>{pic('drive_fusion_device','包含线性项和偏置的完整驱动项设备耗时')}<figcaption>
    每条曲线都包含同一个完整公式。A 很小时，当前直接融合路径节省中间结果与调用；较大 A 和 batch 下，GEMM 的矩阵复用更有利。</figcaption></figure>
    <p>A=8、B=1 的全融合达到 {ratio(drive(0),'direct'):.2f}×；
    A=8、B=8 时，R 整窗表达式仍较快（{us(drive(4),'original_R'):.2f} μs，
    全融合 {us(drive(4),'direct'):.2f} μs）。A=500、B=8 时，cuBLAS＋归约相对 R 整窗表达式为
    {ratio(drive(7),'cublas'):.2f}×，相对两种整窗表达式中的较快者为
    {min(us(drive(7),'original_R'),us(drive(7),'original_F'))/us(drive(7),'cublas'):.2f}×。</p>
    <h3>优化路径内部：只替换 drive，完整推理保留多少收益</h3>
    <p>固定同一份活跃输入、CUDA/CUB 时间滤波、GPU 事件 reset、FP32 和 Graph 策略，
    只换上表中的 drive 实现。F/R 整窗表达式对照均使用相同活跃位点、优化后的滤波和 reset；
    对照表达式直接交给 reset，未额外添加输出复制。完整原始模型的速度对比保留在第 2 节。</p>
    <div class="note">8 种 6 s 形状，第一轮每方法 {first['repeats']} 次 wall / {first['device_repeats']} 次设备测量；
    换随机交错顺序后第二轮 {repeat['repeats']} / {repeat['device_repeats']} 次。
    下表为第二轮。主机单线程固定 CPU 48，物理 GPU {repeat['physical_visible_devices']}，GPU 仍与其他任务共享。</div>
    <p class="muted">设备摊销耗时 / μs：Graph 连续执行 8 次完整前向，CUDA events 计时除以 8。
    包含滤波、完整 drive、reset；输入打包、准备、编译、capture、主机传输在计时之外。</p>
    {table(heads,forward_rows)}
    <figure>{pic('drive_fusion_forward_ratios','完整驱动项融合速度比与完整推理速度比')}<figcaption>
    两列均以对应实验里的 R 整窗表达式为对照；左为独立 drive，右为仅替换 drive 的完整前向。
    两组来自不同测量时段，同一列内部随机交错；它们的比值分别解释，不跨组相乘。</figcaption></figure>
    <p>A=32、B=1：全融合在完整驱动项中为 {ratio(drive(1),'direct'):.2f}×，
    在完整前向中为 {ratio(f2[1],'direct'):.2f}×（{us(f2[1],'original_R'):.2f} → {us(f2[1],'direct'):.2f} μs）。
    A=500、B=8：cuBLAS＋融合归约的完整前向速度比为 {ratio(f2[7],'cublas'):.2f}×。
    固定滤波和 reset 后，这部分收益只归因于 drive 实现的替换。</p>
    <h3>实际单次调用：完整前向 Graph wall time</h3>
    <p class="muted">单位 μs。Graph 只执行一次前向，计时包含主机发起、同步及共享调度。
    设备摊销与这个单调用延迟分别报告，后端最终选择同时考察两者。</p>
    {table(heads,wall_rows)}
    <details><summary>完整前向设备速度比：第一轮 → 第二轮</summary>
    {table(['条件','R 整窗式 / cuBLAS＋归约','R 整窗式 / CUTLASS＋归约','R 整窗式 / 全融合'],stability)}
    <p class="muted">表中所有速度比均按“R 整窗式耗时 / 新实现耗时”计算，&gt;1 为加速。</p></details>
    <details><summary>完整前向：eager / Graph 的 wall 分布</summary>{''.join(detail)}</details>
    <h3>实现选择与数值一致性</h3>
    <p>两轮共 16 个配置记录、五种驱动项实现，相对冻结的完整原始 R 模型，最大电压差
    {max_error:.2e} mV，reset 时间格最大差为 {reset_diff}；每条 Graph 都通过零输入替换和恢复检查。
    Y/W 哈希与原来的独立模块实验逐配置一致。这里衡量实现误差，代理模型相对 NEURON 的拟合精度见第 5 节。</p>
    <p>下一步优先优化每个 block 处理多个时间点、归约的线程分配和 W 的跨时间复用。
    当前直接核按每个时间点独立读取 W；合并成一个 kernel 节省调用和中间矩阵，但大 A 下矩阵复用仍有改善空间。
    以完整 drive 为模块目标，完整前向为最终选择依据，保留 cuBLAS、R 整窗表达式和小形状全融合候选。</p>
    <p class="muted">新记录：<code>drive_forward.json</code>、<code>drive_forward_repeat.json</code>；
    脚本：<code>drive_forward_benchmark.py</code>。冻结源文件与生产后端默认值保持原样，当前完成受控比较。</p>
    </section>'''
    return content, {'first': first, 'repeat': repeat}
