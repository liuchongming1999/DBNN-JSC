"""Render the isolated quadratic-module measurements, without altering raw data."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

METHODS=['original_F','original_R','cublas','cutlass','direct','fastf32']
LABELS={'original_F':'F.bilinear expression','original_R':'Rebatched mm + sum',
        'cublas':'cuBLAS + fused reduction','cutlass':'CUTLASS SIMT + fused reduction',
        'direct':'Direct fused kernel','fastf32':'CUTLASS FastF32 (separate precision)'}
COLORS={'original_F':'#778795','original_R':'#183044','cublas':'#11866b',
        'cutlass':'#ce7c27','direct':'#3168c9','fastf32':'#b5474a'}


def build_quadratic_section(here,table,pic,save):
    source=here/'quadratic_isolated.json'
    if not source.exists():return '',None
    main=json.loads(source.read_text())
    if 'finished' not in main:raise RuntimeError('Isolated quadratic sweep is still incomplete')
    repeat_path=here/'quadratic_repeat.json'
    repeat=json.loads(repeat_path.read_text()) if repeat_path.exists() else None
    if repeat is None or 'finished' not in repeat:raise RuntimeError('Quadratic repeat must finish before publishing')
    first={c['index']:c for c in main['cases']};second={c['index']:c for c in repeat['cases']}
    assert set(first)==set(range(16)) and set(second)==set(range(8,16))
    def us(c,m,scope='quadratic',mode='device'):
        s=c['scopes'][scope]
        return 1000*(s['device_per_call'][m] if mode=='device' else s['wall'][f'{m}/{mode}'])['median_ms']
    def speed(c,m,scope='quadratic',mode='device'):
        return us(c,'original_R',scope,mode)/us(c,m,scope,mode)

    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    for ax,B in zip(axs,[1,8]):
        rows=sorted([c for c in second.values() if c['B']==B],key=lambda c:c['A'])
        for name in METHODS:
            x=[c['A'] for c in rows];y=[us(c,name) for c in rows]
            ax.plot(x,y,'o--' if name in ('original_F','fastf32') else 'o-',color=COLORS[name],
                    label=LABELS[name],lw=1.7,markersize=4)
        ax.set_xscale('log',base=2);ax.set_xticks([8,32,128,500],['8','32','128','500'])
        ax.set_yscale('log');ax.set_title(f'Isolated quadratic module, B={B}, T=6001')
        ax.set_xlabel('Active sites A');ax.set_ylabel('Steady device time / call (microseconds)')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',fontsize=8,frameon=False)
    fig.tight_layout(rect=[0,.13,1,1]);save(fig,'quadratic_isolated_device')

    cols=['cublas','cutlass','direct','fastf32'];rows=[second[i] for i in range(8,16)]
    fig,axs=plt.subplots(1,2,figsize=(11,5.9),sharey=True)
    for ax,mode,title in zip(axs,['device','graph'],['Steady device throughput ratio','Single Graph wall-time ratio']):
        values=np.array([[speed(c,m,mode=mode) for m in cols] for c in rows])
        im=ax.imshow(np.log2(values),cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=1),aspect='auto')
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j,i,f'{values[i,j]:.2f}x',ha='center',va='center',fontsize=9,
                        color='white' if np.log2(values[i,j])< -2 else '#182230')
        ax.set_xticks(range(4),['cuBLAS\n+ reduction','CUTLASS\nSIMT','Direct\nfused','FastF32*'],fontsize=9)
        ax.set_title(title,fontsize=12);ax.grid(False)
    axs[0].set_yticks(range(8),[f'A={c["A"]}, B={c["B"]}' for c in rows]);axs[0].set_ylabel('6 s input, same prepared Y and W')
    fig.suptitle('Rebatched mm + sum / new implementation   (>1 is faster)',fontsize=12)
    fig.tight_layout(rect=[0,0,.94,.94]);cax=fig.add_axes([.945,.15,.015,.68])
    fig.colorbar(im,cax=cax,label='log2(speed ratio)');save(fig,'quadratic_speed_ratios')

    primary=[];wall=[];repeat_rows=[];gemm=[]
    for i,c in sorted(second.items()):
        condition=f'A={c["A"]}, B={c["B"]}'
        primary.append([condition]+[f'{us(c,m):.2f}' for m in METHODS])
        wall.append([condition]+[f'{us(c,m,mode="graph"):.2f}' for m in METHODS])
        repeat_rows.append([condition]+[f'{speed(first[i],m):.2f}× → {speed(c,m):.2f}×' for m in ['cublas','cutlass','direct']])
        gemm.append([condition]+[f'{us(c,m,"gemm"):.2f}' for m in ['cublas','cutlass','fastf32']])
    errors=[]
    for m in METHODS:
        errors.append([LABELS[m]]+[f'{max(c["scopes"][s]["errors_vs_float64"][m]["max_abs"] for d in [first,second] for c in d.values()):.2e}'
                                   for s in ['quadratic','drive']])
    detailed=[]
    for i,c in sorted(first.items()):
        rows=[]
        for scope in ['gemm','quadratic','drive']:
            for m in c['scopes'][scope]['device_per_call']:
                rows.append([scope,LABELS[m],f'{us(c,m,scope,"eager"):.2f}',f'{us(c,m,scope,"graph"):.2f}',
                             f'{us(c,m,scope):.2f}',f'{c["scopes"][scope]["errors_vs_float64"][m]["max_abs"]:.2e}'])
        detailed.append(f'<details><summary>独立模块配置 #{i}：A={c["A"]}, B={c["B"]}, T={c["T"]}</summary>'+
                        table(['模块','实现','eager wall / μs','Graph wall / μs','设备摊销 / μs','max vs FP64'],rows)+'</details>')
    # This paragraph describes only these implemented candidates and measurements.
    key=second[9];large=second[15];tiny=second[8]
    summary=f'''<p><strong>独立测量后的判断：本轮 cuBLAS 加自定义归约的收益依赖形状；当前 CUTLASS 模板整体缺少优势。</strong>
    以 R 重排后的整窗 <code>mm + multiply + sum</code> 表达式为对照，A=32、B=1 的纯二次项设备耗时为
    {us(key,'original_R'):.2f} μs → cuBLAS {us(key,'cublas'):.2f} μs / CUTLASS SIMT {us(key,'cutlass'):.2f} μs。
    A=500、B=8 时，cuBLAS 相对 R 整窗表达式的速度比为 {speed(large,'cublas'):.2f}×；
    A=8、B=1 的完整融合核为 {speed(tiny,'direct'):.2f}×。下面保留所有形状及两轮结果。</p>
    <p>若两种整窗表达式按形状取较快者，A=500、B=8 时 cuBLAS 的速度比为
    {min(us(large,'original_F'),us(large,'original_R'))/us(large,'cublas'):.3f}×；
    A=500、B=1 时为 {min(us(second[11],'original_F'),us(second[11],'original_R'))/us(second[11],'cublas'):.3f}×。
    这些收益应按对应基线解读。</p>'''
    section=f'''<section id="quadratic-analysis"><h2>4.2 内部消融：纯二次项与 GEMM</h2>
    <p class="note">本节为统一整窗布局下的内部算子比较，R 的时间分块已被移出计时。
    原代码保持分块的直接比较见 <a href="#original-audit">第 2.1 节</a>。
    这里用 q=yᵀWy 定位二次计算的成本；加入线性项和偏置后，结合
    <a href="#drive-analysis">4.1 节完整 drive 与完整前向比较</a>选择后端。</p>
    {summary}
    <p>从原始 500 位点 checkpoint 和测试输入得到滤波输出 Y，固定同一份连续布局的
    Y∈ℝ<sup>(BT)×A</sup> 和去对角线后的 W，分别比较三层计算：</p>
    <div class="formula">矩阵乘法 Z = YW<br>纯二次项 qᵣ = Σⱼ Yᵣⱼ Zᵣⱼ<br>driveᵣ = qᵣ + Σⱼ Yᵣⱼ + b</div>
    <p>这里的 F 整窗表达式是 <code>F.bilinear</code> 表达式；R 整窗表达式是一次整窗矩阵乘法加逐元素乘、求和。
    原代码 R 在完整模型里的时间分块循环，以及滤波、reset、输入压缩、传输和权重准备均在模块计时之外。
    因而这是统一连续布局下的算子比较，R 表达式在此已重排为整窗 GEMM。</p>
    <div class="note">独立 sweep 覆盖 A=8/32/128/500、T=1001/6001、B=1/8，共 16 种形状，
    每方法 {main['repeats']} 次单调用 wall time、{main['device_repeats']} 次设备计时；
    随后对全部 8 种 6 s 形状换随机顺序复测，每方法 {repeat['repeats']} / {repeat['device_repeats']} 次。
    使用物理 GPU {repeat['physical_visible_devices']}，主机单线程绑定 CPU 48。GPU 仍为共享状态。
    以下主表来自第二轮，两种整窗表达式和所有新后端均在同一轮交错测量。</div>
    <h3>单次调用要等多久：Graph wall time</h3>
    <p class="muted">单位 μs。一次 Graph 包含一次模块调用，用 synchronize 前后包围 wall time。
    此值包含主机发起、同步和共享设备调度；各表达式 eager 的耗时保留在逐配置明细与下载 JSON。</p>
    {table(['条件（均为 6 s）','F 整窗表达式','R 整窗表达式','cuBLAS + 归约','CUTLASS SIMT','完整融合','FastF32*'],wall)}
    <p class="muted">共享调度对单调用 wall time 的影响较明显：A=32、B=1 时，R 整窗表达式的两轮中位数为
    {us(first[9],'original_R',mode='graph'):.2f} → {us(second[9],'original_R',mode='graph'):.2f} μs，
    cuBLAS 为 {us(first[9],'cublas',mode='graph'):.2f} → {us(second[9],'cublas',mode='graph'):.2f} μs。
    因此单次调用延迟仍需独占资源复测；下面同时给出复现更稳定的设备摊销计时。</p>
    <h3>设备计算能力：连续执行后的每次摊销时间</h3>
    <p class="muted">每个 Graph 连续执行同一模块 32 次（最大形状为 16 次），用 CUDA events 计时后除以次数；
    这是固定 Y/W、热缓存下的稳态设备吞吐口径。上表保留实际单调用延迟。FastF32* 始终作为独立精度模式。</p>
    {table(['条件（均为 6 s）','F 整窗表达式','R 整窗表达式','cuBLAS + 归约','CUTLASS SIMT','完整融合','FastF32*'],primary)}
    <figure>{pic('quadratic_isolated_device','二次模块独立设备耗时，比较整窗表达式 cuBLAS CUTLASS 和融合 kernel')}<figcaption>
    第二轮所有 6 s 形状。每个数值均只包含 q=yᵀWy；线性项、偏置、滤波与 reset 均不包含。纵轴为对数刻度。</figcaption></figure>
    <figure>{pic('quadratic_speed_ratios','二次模块相对整窗重排 mm 加求和的设备及 wall time 速度比')}<figcaption>
    分母均为对应列的新实现，分子为相同条件下的 R 整窗表达式。绿色 &gt;1 表示加速，红色 &lt;1 表示变慢。
    左右比较两个计时口径，展示设备端的变化在单次调用中能保留多少。</figcaption></figure>
    <h3>两轮测量是否一致</h3>
    {table(['条件','cuBLAS + 归约¹','CUTLASS SIMT¹','完整融合¹'],repeat_rows)}
    <p class="muted">¹ 每格为第一轮 → 第二轮；统一按“R 整窗表达式耗时 / 新实现耗时”定义，均为纯二次项的设备摊销时间。
    单独的微秒差异随共享负载变化；后端选择优先依据跨轮与跨形状的结果。</p>
    <details><summary>仅矩阵乘法 Z=YW 的计时，定位 GEMM 与归约的影响</summary>
    {table(['条件（6 s）','cuBLAS / μs','CUTLASS SIMT / μs','CUTLASS FastF32 / μs'],gemm)}
    <p>这些矩阵乘法已经使用相同输出工作区。cuBLAS 对照直接调用 <code>torch.mm(..., out=Z)</code>；
    CUTLASS 使用本轮已经实现的 C++ 3.9.2 SIMT / FastF32 模板。模板参数与硬件适配还有继续优化的空间。</p></details>
    <details><summary>精度检查：各方法相对 FP64 参考的最大绝对误差</summary>
    {table(['实现','纯二次项 max / mV','drive max / mV'],errors)}
    <p>最大值覆盖两轮共 24 个配置记录。所有结果有限，q 与 drive 均满足本轮 10⁻³ mV 的检查门槛。
    每个 Graph 均通过零输入替换和恢复后的校验，确保重放读取更新的输入。
    主组为 FP32 且禁用 TF32；FastF32 单列。本节测量止于 drive，完整模型的 reset 一致性见原有验证。</p></details>
    <p>此次新增的是独立量化，生产后端保持本轮已有实现。后续若继续优化二次模块，优先测试更高效的归约线程映射、
    GEMM 与归约的真正融合，以及跨多个时刻复用 W 的分块设计；用这些模块基准和完整前向共同筛选。</p>
    <details><summary>第一轮全部 16 种形状：矩阵乘法 / 纯二次项 / drive 明细</summary>{''.join(detailed)}</details>
    <p class="muted">记录：<code>quadratic_isolated.json</code>、<code>quadratic_repeat.json</code>；
    脚本：<code>quadratic_benchmark.py</code>，纯二次项核：<code>quadratic_only.cu</code>。</p>
    </section>'''
    return section,{'first':main,'repeat':repeat}
