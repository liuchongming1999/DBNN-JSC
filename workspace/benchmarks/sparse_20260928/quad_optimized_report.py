"""Explain the specialized contraction with original-code and cuBLAS controls."""
import json
import numpy as np
import matplotlib.pyplot as plt


def build_optimized_quad_section(here,table,pic,save):
    first=json.loads((here/'quad_optimized.json').read_text())
    final=json.loads((here/'quad_confirm.json').read_text())
    meta=json.loads((here/'quad_kernel_metadata.json').read_text())
    policy=json.loads((here/'quad_backend_policy.json').read_text())
    source_replace=json.loads((here/'quad_source_replace.json').read_text())
    assert first.get('finished') and final.get('finished') and len(final['cases'])==9
    assert source_replace.get('finished') and len(source_replace['cases'])==2
    assert source_replace['baseline_sha256']==final['baseline_sha256']
    old={c['index']:c for c in first['cases']};cs=final['cases']
    def value(c,scope,key):return c[scope][key]['median_ms']
    def ratio(c,scope):return value(c,scope,'cublas')/value(c,scope,'new_ieee')
    def shape(c):return f'A={c["A"]}, B={c["B"]}, {(c["T"]-1)/1000:g} s'+(' *' if c['A']>500 else '')
    names=[f'A{c["A"]}, B{c["B"]}\n{(c["T"]-1)/1000:g}s'+(' *' if c['A']>500 else '') for c in cs]
    xs=np.arange(len(cs));width=.25
    fig,ax=plt.subplots(figsize=(11.6,4.8))
    for delta,label,color,key in [(-width,'Previous one-row CUDA kernel','#7b8898','old_direct'),
                                  (0,'cuBLAS + complete reduction','#3168c9','cublas'),
                                  (width,'Optimized fused kernel (FP32)','#11866b','new_ieee')]:
        ax.bar(xs+delta,[1000*value(c,'module_device',key) for c in cs],width,label=label,color=color)
    ax.set_yscale('log');ax.set_xticks(xs,names,fontsize=8)
    ax.set_ylabel('Complete-drive device time per call (microseconds)')
    ax.set_title('Fixed configurations: independent 60-sample retest',pad=34)
    ax.legend(frameon=False,fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01))
    fig.tight_layout();save(fig,'quad_specialized_device')
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8));xs=np.arange(len(cs));width=.34
    for ax,scope,label in [(axes[0],'module_device','Complete-drive device time'),
                           (axes[1],'forward_graph_wall','Whole DBNN Graph wall time')]:
        v=[ratio(c,scope) for c in cs]
        ax.bar(xs,v,color='#11866b');ax.axhline(1,color='#657488',lw=1)
        for x,z in zip(xs,v):ax.text(x,z+.04,f'{z:.2f}x',ha='center',fontsize=8)
        ax.set_ylim(0,max(v)*1.18);ax.set_xticks(xs,names,fontsize=7)
        ax.set_ylabel('cuBLAS path time / new kernel path time');ax.set_title(label,fontsize=11)
    fig.tight_layout();save(fig,'quad_specialized_forward')
    module_rows=[];forward_rows=[];stability=[];resources=[];details=[];source_rows=[]
    for c in source_replace['cases']:
        for kind in ['F','R']:
            before=c['wall'][f'original_{kind}'];after=c['wall'][f'new_{kind}']
            source_rows.append([shape(c),kind,f'{before["median_ms"]:.3f}',f'{after["median_ms"]:.3f}',
                                f'{before["median_ms"]/after["median_ms"]:.2f}×',before['n']])
    for c in cs:
        i=c['index'];o=old[i]
        module_rows.append([shape(c),f'{value(c,"module_device","old_direct")*1000:.2f}',
            f'{value(c,"module_device","cublas")*1000:.2f}',f'{value(c,"module_device","new_ieee")*1000:.2f}',
            f'{ratio(c,"module_device"):.2f}×',o['selected']['new_ieee']['key']])
        p=c['original_paired_wall'];orig=c['original_method']
        forward_rows.append([shape(c),orig,f'{p[orig]["median_ms"]:.3f}',
            f'{p["optimized_cublas"]["median_ms"]:.3f}',f'{p["optimized_new_ieee"]["median_ms"]:.3f}',
            f'{p[orig]["median_ms"]/p["optimized_new_ieee"]["median_ms"]:.2f}×'])
        stability.append([shape(c),f'{o["module_device"]["cublas"]["median_ms"]/o["module_device"]["new_ieee"]["median_ms"]:.2f}× → {ratio(c,"module_device"):.2f}×',
            f'{o["forward_graph_wall"]["cublas"]["median_ms"]/o["forward_graph_wall"]["new_ieee"]["median_ms"]:.2f}× → {ratio(c,"forward_graph_wall"):.2f}×'])
        rows=[]
        for scope in ['module_device','forward_device','forward_graph_wall','original_paired_wall']:
            for method,s in c[scope].items():
                rows.append([scope,method,f'{s["median_ms"]:.4f}',f'{s["p95_ms"]:.4f}',f'{s["std_ms"]:.4f}',s['n']])
        details.append(f'<details><summary>{shape(c)}：耗时分布、配置与误差</summary><p><code>{json.dumps(c["spec"])}</code></p>'+
            table(['计时范围','方法','median / ms','p95 / ms','std / ms','次数'],rows)+
            f'<p>相对原代码最大电压差 {c["error_vs_original"]["max_abs_mv"]:.2e} mV，reset 差 {c["error_vs_original"]["spike_mismatch_bins"]} 格。</p></details>')
    for k in meta['kernels']:
        resources.append([f'A={k["A"]}, rows={k["M"]}',str(k['tile']),str(k['triangle']),k['blocks'],
                          k['registers_per_thread'],k['spill_count'],k['shared_bytes']])
    by_index={c['index']:c for c in cs};c500=by_index[5];c2048=by_index[8]
    max_physical=max(c['error_vs_original']['max_abs_mv'] for c in cs if c['A']<=500)
    max_all=max(c['error_vs_original']['max_abs_mv'] for c in cs)
    heldout=max(c['error']['max_abs_mv'] for c in final['heldout_forward_checks'])
    selected500=old[5]['intermediate_bytes'];old_z=selected500['cublas_Z'];new_p=selected500['new_ieee']
    thread_data=json.loads((here/'quad_tune_A500_B1.json').read_text())
    thread_us=1000*thread_data['timings']['thread']['median_ms']
    cub_us=1000*thread_data['timings']['cublas']['median_ms']
    content=f'''<section id="quad-specialized"><h2>4.3 新增二次型专用 kernel：跨时间块复用与三角收缩</h2>
<p><strong>本轮已实现并接入完整 DBNN，主结果采用 IEEE FP32。</strong>
A=500、B=8、6 s 时，完整驱动项相对同轮 cuBLAS＋归约达到 {ratio(c500,'module_device'):.2f}×；
完整 DBNN 的 Graph wall time 达到 {ratio(c500,'forward_graph_wall'):.2f}×。
A=2,048 的 6 s 合成计算规模测试中，对应为 {ratio(c2048,'module_device'):.2f}× 和 {ratio(c2048,'forward_graph_wall'):.2f}×。</p>
<div class="note">先探索候选与 tile 配置，再固定每种形状的 IEEE 实现，完成 50 次及独立第二轮 60 次配对。
模块使用 CUDA events 测量连续 Graph 的每次摊销时间；完整推理另报同步 wall time。
原 F/R 对照直接调用冻结源代码，与新路径在第二轮随机交错各 30 次。
物理 GPU 6、CPU 48 单线程，资源为共享状态。带 * 的 A&gt;500 使用参数扩展，仅评价计算规模。</div>
<h3>算的仍是同一个二次型，改变运算组织方式</h3>
<div class="formula">p 表示展平后的 (batch, time)；Y ∈ ℝ<sup>(BT)×A</sup><br>
q<sub>p</sub> = (Y W Yᵀ)<sub>pp</sub> = Σ<sub>j</sub> Y<sub>pj</sub> Σ<sub>i</sub>Y<sub>pi</sub> W<sub>ij</sub><br>
U<sub>ij</sub> = W<sub>ij</sub>＋W<sub>ji</sub>（i&lt;j）；U<sub>ii</sub>=W<sub>ii</sub>；U<sub>ij</sub>=0（i&gt;j）<br>
q<sub>p</sub> = Σ<sub>j</sub> Y<sub>pj</sub> Σ<sub>i≤j</sub>Y<sub>pi</sub> U<sub>ij</sub><br>
u<sub>p</sub> = q<sub>p</sub>＋Σ<sub>j</sub>Y<sub>pj</sub>＋b</div>
<p>原代码已经按时间点计算二次型，没有构造 (BT)×(BT) 的矩阵。
新增实现把中间的 YW 与后续加权求和在块内合并；三角重排在实数代数下保持同一函数，FP32 的求和顺序差异通过电压与事件检查约束。
原模型的 W 对角线为零；实现同时支持一般非零对角线。</p>
{table(['实现方向','实际操作','结果与采用方式'],[
['一个线程处理一个时间点','单线程累加完整二次型；先转置输入以合并相邻时间点的访存，转置也计时',f'500 位点、6 s、B=1 的探索为 {thread_us:.1f} μs；同轮 cuBLAS 为 {cub_us:.1f} μs。保留实测，未选为默认'],
 ['一个 warp 处理一个时间点','同一个 warp 协作累加，多个时间点共用一个 block；测试 4 / 8 / 16 个 warp', '已实现 CUDA 候选，参与 A≤500 的筛选'],
 ['多个时间点共同处理 W tile','自写 CUDA SIMT 和 Triton 分块矩阵乘法；累加器直接乘对应的 Y 并归约','保留严格 IEEE FP32 全矩阵路径及三角路径'],
 ['三角收缩','预先合并 Wᵢⱼ 与 Wⱼᵢ，跳过下三角 tile；一般约减少一半的主要乘加，边界块仍有填充开销','大形状主要采用该路径；A=500、B=1 选 64×64×32，B=8 选 128×64×32'],
 ['三角块对并行','每个上三角块对独立计算，再汇总每个时间点的标量部分和','小形状及部分 1 s 大维度形状获选；按形状选择'],
 ['TF32x3 Tensor Core','FP32 存储，显式采用 tf32x3 乘法模式','单独核对误差和计时；此次发布的策略全部使用 IEEE 路径']])}
<p>大形状通常由两个 kernel 完成：第一阶段在寄存器中融合矩阵块与二次型加权归约，第二阶段合并少量部分和并加偏置。
线性项一并计算。A=500、B=8 时，中间 YW 需要 {old_z/1024**2:.2f} MiB；所选新路径的部分和为 {new_p/1024**2:.2f} MiB，
减少约 {old_z/new_p:.1f} 倍。完整新后端已删除未使用的 YW 工作区；输入、权重和输出另计。</p>
<h3>完整驱动项：二次＋线性＋偏置</h3>
{table(['形状','上一版自写 CUDA / μs','cuBLAS＋归约 / μs','新 IEEE / μs','cuBLAS / 新','固定的实现'],module_rows)}
<figure>{pic('quad_specialized_device','九种形状下旧单核 cuBLAS 和新二次型 kernel 的完整驱动项耗时')}<figcaption>
第二轮固定配置，各方法 60 次，纵轴为对数。灰色是上一版自写的单时间点 CUDA 核；用户原代码配对另列于下表。
每个数值包含完整驱动项和必要的部分和归约；不含滤波、reset、参数准备与传输。</figcaption></figure>
<h3>原 F/R 仅替换驱动项：保留原滤波与 reset</h3>
{table(['形状','原实现','原 forward / ms','只换新二次型 / ms','整体速度比','每方法次数'],source_rows)}
<p>以上直接在冻结源代码的 drive 调用位置插入新实现；R 保留 256 步 chunk 循环，两边保留原来的滤波和 reset，布局转换也计时。
A=500、B=1 时，F 的完整 forward 从约 0.902 ms 降至 0.686 ms；R 的 5.139 → 5.220 ms 在这一轮没有收益。
B=8 的原整批调用仍主要耗时在 Python/PyTorch 逐时间格 reset，单独替换 drive 后整体变化很小。
这组结果说明新 kernel 的应用收益也取决于调用粒度和其他模块；下表再给出整窗 scan＋事件 reset 路径的配对结果。</p>
<h3>完整推理：和你的原始代码直接比较</h3>
{table(['形状','冻结的原代码调用方式','原代码 / ms','优化滤波＋cuBLAS / ms','优化滤波＋新二次型 / ms','原代码 / 新路径'],forward_rows)}
<p>三列在同一轮内交错计时，使用相同活跃集合。原代码保留 F/R 的计算流程；batch&gt;1 时，
原 B=1 事件分支逐条调用也参与前一轮筛选，并在这一轮固定复测。
“优化滤波＋cuBLAS”与“优化滤波＋新二次型”两列的滤波、reset 和 Graph 策略相同，量化本轮二次型改动的增量收益。</p>
<figure>{pic('quad_specialized_forward','新二次型相对 cuBLAS 的模块速度比与完整 DBNN 速度比')}<figcaption>
左为完整驱动项的设备摊销时间比；右为相同 scan/reset 下完整 DBNN 的单次 Graph wall time 比。
公式与 NEURON 拟合目标保持一致，模块加速和整模型加速分别给出。</figcaption></figure>
<details><summary>两轮固定配置的稳定性</summary>{table(['形状','模块速度比：第一轮 → 第二轮','完整 Graph wall：第一轮 → 第二轮'],stability)}</details>
<h3>线程分工和数值检查</h3>
<p>这张 RTX 5090 有 {meta['multiprocessors']} 个 SM。“每个线程一个时间点”的 6,001 点版本只发出 47 个 block；
新的 64×64 时间—位点分块发出 752 个 block，同时复用 W。
编译器元数据也帮助筛掉寄存器溢出较多的配置；下表是静态资源信息。</p>
{table(['形状','tile M/N/K','三角化','block 数','寄存器 / 线程','n_spills 指标','共享内存 / bytes'],resources)}
<p class="muted">Nsight 已尝试运行，服务器返回 <code>ERR_NVGPUCTRPERM</code>。因此没有给出实测 occupancy、带宽或 stall 占比。
寄存器与共享内存数据来自编译结果；后续仍可通过获得计数器访问权限继续定位瓶颈。</p>
<p>原 500 位点范围内，本轮最大电压差为 {max_physical:.2e} mV；包含合成大维度时为 {max_all:.2e} mV。
两种 500 位点获选配置还分别核对了全部 10 条测试输入，最大差为 {heldout:.2e} mV，reset 触发时间格差均为 0。
另外完成 {len(final['arbitrary_shape_checks'])} 组一般非对称 W、非零对角线、非整齐维度和尾块检查，以及 Graph 零输入替换与恢复检查。
零输入期望值取自 checkpoint 的偏置 {final['checkpoint_bias_mv']:.4f} mV。</p>
<h3>已经可以使用的入口与下一步</h3>
<p><code>quad_optimized_backend.py::make_optimized_prepared</code> 配合 <code>quad_backend_policy.json</code>，
按 A、T、B 选择已经独立复测的 IEEE 实现；当前 {sum(v['enabled'] for v in policy['shapes'].values())} 种形状通过启用条件。
未知形状保留 cuBLAS 路径。权重固定时，U 和工作区准备一次后复用，输入内容可更新。
当前入口用于推理，训练反向传播尚未实现。</p>
<p>后续优化重点是：在独占窗口复测；按需要扩展形状选择表；进一步减少部分和归约的调度成本；
结合性能计数器调整 tile、软件流水和寄存器压力。当前结果对应已测形状和这台 5090，可作为下一轮优化的起点。</p>
<p class="muted">实现依据：<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html">Triton 分块矩阵乘法与调度文档</a>；
<a href="https://triton-lang.org/main/python-api/generated/triton.language.dot.html">dot 的显式精度模式</a>。
自写 CUDA 代码采用同样的时间块复用与融合归约思想；比较结果来自本项目的实际记录。</p>
<details><summary>本轮全部分布与逐配置误差</summary>{''.join(details)}</details>
<p class="muted">代码：<code>quadratic_tiled.cu / quadratic_tiled.py</code>；
记录：<code>quad_optimized.json / quad_confirm.json / quad_source_replace.json</code>；
选择表：<code>quad_backend_policy.json</code>；编译资源：<code>quad_kernel_metadata.json</code>。</p>
</section>'''
    return content,dict(first=first,confirmation=final,policy=policy,kernel_metadata=meta,source_replacement=source_replace)
