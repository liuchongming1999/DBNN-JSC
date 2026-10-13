"""Advisor-facing five-layer precision audit with exact measurement boundaries."""
import json
import numpy as np
import matplotlib.pyplot as plt


def build_tf32_section(here,table,pic,save):
    data={k:json.loads((here/n).read_text()) for k,n in [
        ('quad_sweep','tf32_quad_sweep.json'),('layer_sweep','tf32_layer_sweep.json'),
        ('first','tf32_evaluation.json'),('repeat','tf32_repeat.json'),('policy','tf32_backend_policy.json'),
        ('api_validation','tf32_api_validation.json')]}
    assert all(data[k].get('finished') for k in ['quad_sweep','layer_sweep','first','repeat'])
    assert data['api_validation']['passed']
    cs=data['repeat']['cases'];qs={c['index']:c for c in data['quad_sweep']['cases']}
    ls={c['index']:c for c in data['layer_sweep']['cases']};first={c['index']:c for c in data['first']['cases']}
    assert len(cs)==13
    policy=data['policy']
    def key(c):return f'{c["A"]}:{c["T"]}:{c["B"]}'
    def entry(c,profile='strict'):return policy['profiles'][profile]['shapes'][key(c)]
    def label(c):return f'A={c["A"]}, B={c["B"]}, {(c["T"]-1)/1000:g} s'+(' *' if c['A']>500 else '')
    names=[f'A{c["A"]}/B{c["B"]}\n{(c["T"]-1)/1000:g}s'+('*' if c['A']>500 else '') for c in cs]
    def med(c,scope,k):return c[scope][k]['median_ms']
    def us(c,scope,k):return med(c,scope,k)*1000
    x=np.arange(len(cs));width=.2
    for scope,filename,title,methods in [
        ('filter_device','tf32_filter_device','Full temporal filtering: all phases included',[
            ('previous_ieee','FP32 scan 128','#7b8898'),('tuned_ieee','Tuned FP32 scan','#3168c9'),
            ('filter_tf32','TF32 block filter','#11866b'),('filter_tf32x3','TF32x3 block filter','#ce7c27')]),
        ('drive_device','tf32_drive_device','Complete quadratic + linear + bias',[
            ('ieee','Fused kernel (FP32)','#3168c9'),('tf32','Fused kernel (TF32)','#11866b'),
            ('tf32x3','Fused kernel (TF32x3)','#ce7c27'),('cublas_tf32','cuBLAS TF32 allowed','#7b8898')])]:
        fig,ax=plt.subplots(figsize=(12,4.8))
        for j,(k,name,color) in enumerate(methods):
            ax.bar(x+(j-1.5)*width,[us(c,scope,k) for c in cs],width,label=name,color=color)
        ax.set_xticks(x,names,fontsize=9);ax.set_yscale('log');ax.set_ylabel('Device time (microseconds)',labelpad=8)
        ax.set_title(title,pad=34);ax.legend(frameon=False,ncol=4,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01))
        fig.tight_layout(pad=1.6);save(fig,filename)
    fig,axes=plt.subplots(2,1,figsize=(12,6.8),sharex=True);width=.35
    for ax,scope,title in [(axes[0],'forward_device','Whole DBNN device time'),(axes[1],'forward_graph_wall','Whole DBNN single-replay wall time')]:
        for j,(profile,col,name) in enumerate([('strict','#11866b','<0.001 mV profile'),('approximate_0p01mv','#ce7c27','Opt-in <0.01 mV profile')]):
            vals=[med(c,scope,'previous_ieee')/med(c,scope,entry(c,profile)['name']) for c in cs]
            ax.bar(x+(j-.5)*width,vals,width,color=col,label=name)
        ax.axhline(1,color='#657488',lw=1);ax.set_xticks(x,names,fontsize=9)
        ax.set_ylabel('Baseline / selected profile');ax.set_title(title,fontsize=11)
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'tf32_whole_forward')
    c=next(c for c in cs if c['A']==500 and c['B']==8)
    fig,ax=plt.subplots(figsize=(9,4.6))
    draw=[('previous_ieee','FP32 + scan128'),('tuned_ieee','FP32 + tuned scan'),('quad_tf32','TF32 quadratic'),
          ('filter_tf32','TF32 filter'),('filter_tf32x3','TF32x3 filter'),('combined_tf32','TF32 both'),('quad_tf32_filter_x3','TF32 q + TF32x3 filter')]
    offsets=[(8,5),(8,-15),(15,23),(8,6),(8,10),(15,-16),(15,5)]
    for i,(name,desc) in enumerate(draw):
        e=c['heldout_accuracy'][name]['max_abs_mv'];t=us(c,'forward_device',name)
        ax.scatter(e,t,s=48,color='#11866b' if e<1e-3 else '#ce7c27')
        ax.annotate(desc,(e,t),xytext=offsets[i],textcoords='offset points',fontsize=9,
                    arrowprops=dict(arrowstyle='-',color='#7b8898',lw=.7))
    ax.axvline(1e-3,color='#657488',ls='--',label='Existing 0.001 mV gate')
    ax.set_xscale('log');ax.set_xlim(3e-6,6e-3);ax.set_ylim(438,618);ax.set_xlabel('Maximum voltage difference on all 10 held-out traces (mV)')
    ax.set_ylabel('Whole DBNN device time (microseconds)');ax.set_title('A=500, B=8, 6 s: speed and numerical deviation')
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'tf32_precision_tradeoff')
    module=[];filters=[];forward=[];quality=[];original=[];scheduling=[];controls=[];details=[];fit=[]
    aliases={'previous_ieee':'IEEE＋scan128','tuned_ieee':'IEEE＋调优scan','quad_tf32':'仅二次项TF32',
             'quad_tf32_strict':'仅二次项TF32（模块门限筛选）','quad_tf32x3':'仅二次项TF32x3',
             'quad_cublas_tf32':'cuBLAS允许TF32','filter_tf32':'IEEE二次项＋TF32滤波',
             'filter_tf32x3':'IEEE二次项＋TF32x3滤波','combined_tf32':'二次项/滤波均TF32',
             'quad_tf32_filter_x3':'TF32二次项＋TF32x3滤波','combined_x3':'二次项/滤波均TF32x3'}
    for c in cs:
        strict=entry(c)['name'];approx=entry(c,'approximate_0p01mv')['name']
        module.append([label(c),*[f'{us(c,"drive_device",k):.2f}' for k in ['ieee','tf32','tf32x3','cublas_tf32']],
                       f'{med(c,"drive_device","ieee")/med(c,"drive_device","tf32"):.2f}×'])
        filters.append([label(c),*[f'{us(c,"filter_device",k):.2f}' for k in ['previous_ieee','tuned_ieee','filter_tf32','filter_tf32x3']]])
        forward.append([label(c),f'{med(c,"forward_graph_wall","previous_ieee"):.3f}',f'{med(c,"forward_graph_wall","tuned_ieee"):.3f}',
                        aliases[strict],f'{med(c,"forward_graph_wall",strict):.3f}',f'{med(c,"forward_graph_wall",approx):.3f}',
                        f'{med(c,"forward_device","previous_ieee")/med(c,"forward_device",strict):.2f}× / {med(c,"forward_graph_wall","previous_ieee")/med(c,"forward_graph_wall",strict):.2f}×'])
        quality.append([label(c),aliases[strict],f'{c["heldout_accuracy"][strict]["max_abs_mv"]:.2e}',
                        aliases[approx],f'{c["heldout_accuracy"][approx]["max_abs_mv"]:.2e}',
                        c['heldout_accuracy'][strict]['spike_mismatch_bins'],c['heldout_accuracy'][approx]['spike_mismatch_bins']])
        if c['A']==500:
            er=c['heldout_accuracy'][strict];ea=c['heldout_accuracy'][approx]
            fit.append([label(c),f'{er["reference_neuron_pooled_rmse_mv"]:.6f}',f'{er["neuron_pooled_rmse_mv"]:.6f}',f'{ea["neuron_pooled_rmse_mv"]:.6f}'])
        orig=c['original_method'];paired=c['original_paired_wall']
        original.append([label(c),orig,f'{paired[orig]["median_ms"]:.3f}',
                         f'{paired["selected_strict"]["median_ms"]:.3f}',aliases[c['selected_strict']],
                         f'{paired[orig]["median_ms"]/paired["selected_strict"]["median_ms"]:.2f}×'])
        scheduling.append([label(c),aliases[strict],f'{med(c,"forward_eager_wall",strict):.3f}',f'{med(c,"forward_graph_wall",strict):.3f}',
                           f'{med(c,"forward_eager_wall",strict)/med(c,"forward_graph_wall",strict):.2f}×'])
        ctrl=ls[c['index']]['non_matmul_controls']
        for stage in ['pack_input','reset_same_drive']:
            d=ctrl[stage]['device'];controls.append([label(c),stage,ctrl['N'],f'{d["tf32_off"]["median_ms"]*1000:.2f}',
                f'{d["tf32_on"]["median_ms"]*1000:.2f}','逐位相同；算子没有矩阵乘法'])
        rows=[]
        for name in c['specs']:
            er=c['heldout_accuracy'][name]
            rows.append([aliases[name],f'{us(c,"forward_device",name):.2f}',f'{med(c,"forward_graph_wall",name):.4f}',
                         f'{c["forward_graph_wall"][name]["p95_ms"]:.4f}',f'{er["max_abs_mv"]:.2e}',er['spike_mismatch_bins'],
                         '通过' if er['strict_gate_passed'] else '超出原电压门限'])
        details.append('<details><summary>'+label(c)+'：所有组合与精度</summary>'+table(['组合','设备 / μs','Graph wall / ms','wall p95 / ms','max / mV','reset差','原门限'],rows)+'</details>')
    c500=next(c for c in cs if c['A']==500 and c['B']==8)
    candidate_total=sum(len(c['specs']) for c in data['quad_sweep']['cases'])
    filter_total=sum(len(c['specs']) for c in data['layer_sweep']['cases'])
    small=next(c for c in cs if c['A']==32 and c['B']==1)
    sc=entry(small)['name'];c500strict=entry(c500)['name'];c500approx=entry(c500,'approximate_0p01mv')['name']
    selected_count=sum(v['uses_tensor_core'] for v in policy['profiles']['strict']['shapes'].values())
    text=f'''<section id="tf32-five-layers"><h2>4.4 TF32 系统调优：五层分别评估，按形状和误差门限选择</h2>
<p><strong>已完成 13 种形状、{candidate_total} 个二次项候选配置及 {filter_total} 个滤波候选配置检查。</strong>
覆盖 A=8/32/128/500/1,024/2,048，B=1/2/4/8，窗口 0.3/1/6 s。
先完成模块初筛，再固定配置进行 50 次和独立第二轮 60 次测量；每种形状的全部组合均核对 10 条测试轨迹。
原有 IEEE 入口保留，新精度入口显式选择 profile。原门限方案中有 {selected_count} / 13 个形状使用了 Tensor Core。</p>
<div class="note">本节的 TF32 数值均是新实测。二次项包含线性项与偏置；Tensor Core 滤波包括块内 GEMM、块间状态扫描及最终输出合成。
主数据为共享物理 GPU 6、CPU 48 单线程。设备时间为重复 Graph 的摊销时间；完整应用另报单次同步 wall time。
权重 / 系数准备、编译、输入位点选择与传输位于 core 之外。* 表示 A&gt;500 的参数扩展计算规模测试。</div>
{table(['层次','实际探索','结论'],[
 ['位点集合','对相同已知 IDs 的 gather 分别打开 / 关闭 TF32；检查输出逐位一致','输入和权重仍为 FP32 存储。索引收集不调用矩阵乘法，TF32 没有直接加速作用；压缩 A 的收益继续保留。'],
 ['时间滤波','将短时间块的双指数卷积写成 Toeplitz GEMM；扫块长 16/32/64/128、每组 16/32 个块、TF32 与 TF32x3；另扫 FP32 scan 的 64/128/256/512 块长','中小规模和部分长窗口有收益；短窗口及部分大 batch 保留 scan。块间状态传递和输出合成仍为 FP32。'],
 ['二次耦合','普通 TF32 显式舍入、TF32x3；全矩阵 / 上三角；20 种矩阵 tile、7 种块对 tile、group 和舍入对照；cuBLAS 允许 TF32 另列','普通 TF32 出现加速，数值偏差随形状变化。按完整电压与 reset 门限选择，保持线性项、偏置和输出 FP32。'],
 ['reset','相同 u[n] 下比较 TF32 开关；所有上游组合另核对闭环事件','比较、最早事件归约、指数衰减保留 FP32。开关本身不改变 reset 算子；上游近似是否改变事件单独记录。'],
 ['调度','所有固定组合均测 eager、单次 Graph wall 与设备时间；选择策略按 A/T/B 存储','TF32 只改变采用它的矩阵运算。Graph 的收益来自减少主机提交，完整推理 wall time 用于评估最终延迟。']])}
<h3>时间滤波怎样改成 Tensor Core 计算</h3>
<div class="formula">aᵢ=e<sup>−1/τᵢ,decay</sup>，cᵢ=e<sup>−(1/τᵢ,rise＋1/τᵢ,decay)</sup><br>
Hᵢ[j,r]=ωᵢ(aᵢ<sup>r−j</sup>−cᵢ<sup>r−j</sup>)（j≤r；其余为 0）<br>
块内零状态响应：Zᵢ = Xᵢ Hᵢ<br>
块末状态：sᵢ,next = aᵢ<sup>L</sup>sᵢ,start＋Σ<sub>j=0…L−1</sub>aᵢ<sup>L−1−j</sup>xᵢ[j]<br>
完整输出：yᵢ[r]=Zᵢ[r]＋ωᵢ(aᵢ<sup>r+1</sup>sᵢ,start−cᵢ<sup>r+1</sup>tᵢ,start)</div>
<p>每个通道的多个时间块组成矩阵行，复用该通道的短卷积矩阵 Hᵢ；第二个状态 tᵢ 使用 cᵢ 的同类递推。
块末状态与块间前缀扫描均保留 FP32，只有 XᵢHᵢ 使用 TF32 / TF32x3。
该实现没有截断跨块的指数尾部，数学上仍表示同一零初态滤波；额外工作区和三阶段调度全部包含在计时中。</p>
{table(['形状','原 scan128 / μs','调优 scan / μs','TC TF32 / μs','TC TF32x3 / μs'],filters)}
<figure>{pic('tf32_filter_device','完整时间滤波的 FP32 scan 与 Tensor Core 候选计时')}<figcaption>
第二轮各 60 次。矩阵化滤波增加局部响应工作区及读写，并把原来的时间块循环分成更多独立任务；这两种变化共同决定不同形状的收益。</figcaption></figure>
<h3>二次项：普通 TF32、TF32x3 与 cuBLAS</h3>
{table(['形状','IEEE 专用 / μs','TF32 专用 / μs','TF32x3 专用 / μs','cuBLAS允许TF32 / μs','IEEE / TF32'],module)}
<figure>{pic('tf32_drive_device','完整二次线性偏置驱动项的精度和后端对照')}<figcaption>
TF32 列为初筛后固定的最快普通 TF32 候选，包含未通过原电压门限的情况；应用选择见下表。
TF32 的张量存储仍为 FP32，乘法操作数在核内显式舍入到 TF32；累加、线性项与偏置保留 FP32。</figcaption></figure>
<p>已检查获选自写核的 PTX 中存在 <code>mma.sync...tf32</code> 指令；普通 TF32 路径还检查了显式舍入指令。
部分较大 tile 超过 SM120 的共享内存上限，失败原因保留在 sweep 记录中；它们未进入计时选择。
cuBLAS 列表示开启其 TF32 许可，实际内核仍由库选择。Nsight 硬件性能计数器权限状态与上一节相同。</p>
<h3>完整推理与两种显式精度方案</h3>
<p><strong>默认的新 strict profile：</strong>最大电压差 &lt;10⁻³ mV，reset 时间格差为 0。
<strong>可选 approximate_0p01mv profile：</strong>最大电压差 &lt;0.01 mV，reset 时间格差为 0，需显式选用。
两者均要求在两轮 10 条测试轨迹检查中通过，并在第二轮确认设备收益、没有明显 wall time 回退。
这里的门限是已测输入上的准入条件。</p>
{table(['形状','IEEE＋scan128 / ms','IEEE＋调优scan / ms','strict 选择','strict / ms','0.01 mV方案 / ms','strict设备 / wall提升'],forward)}
<p class="muted">基准列固定 IEEE 二次项，先用 scan128、再调优 scan 块长，从而单列纯 FP32 调优收益。
原有 9 个形状复用了已验证 IEEE 候选；新增的 4 个形状使用本轮 IEEE 初筛结果。
所有列都包含完整滤波、驱动项和 reset，为单次 Graph 同步 wall time。</p>
<figure>{pic('tf32_whole_forward','完整推理的设备和 wall time 两种口径的精度方案速度比')}<figcaption>
两个精度 profile 都以相同 IEEE＋scan128 作为分母。设备时间和主机可见延迟分别给出，原代码真实调用的对照随后单列。</figcaption></figure>
{table(['形状','strict 组合','strict最大差 / mV','0.01 mV方案组合','该方案最大差 / mV','strict reset差','该方案reset差'],quality)}
<figure>{pic('tf32_precision_tradeoff','500 位点 Batch8 的速度与完整电压偏差关系')}<figcaption>
横轴是全部 10 条轨迹的最大电压差，纵轴是第二轮完整设备耗时。门限判断使用完整闭环输出；模块精度通过不自动等于整条路径通过。</figcaption></figure>
<details><summary>500 位点：同一 NEURON 测试数据上的代理拟合 RMSE</summary>
{table(['运行形状（均核对10条轨迹）','原R / mV','strict / mV','0.01 mV方案 / mV'],fit)}
<p>以原先采样的 NEURON 电压作为共同目标，合并全部时间点计算 RMSE。这里没有重新采样 NEURON，也没有重新训练 checkpoint。</p></details>
<details><summary>与冻结原 F/R 代码直接配对：每方法 30 次 wall time</summary>
{table(['形状','原代码实际调用','原代码 / ms','当轮strict候选 / ms','实际候选','整体速度比'],original)}
<p>原 F/R 整批与原 B=1 逐条调用先分别测量，然后固定较快原调用参与本轮配对。新路径的数字包含所有层优化；单独 TF32 增量由上面的 IEEE 对照给出。
此表保留每轮实际计时的 strict 候选，最终发布策略还经过两轮稳定性筛选。</p></details>
<details><summary>位点收集与 reset：TF32 开关控制</summary>
{table(['形状','操作','候选位点 N','开关关 / μs','开关开 / μs','输出与计算性质'],controls)}
<p>同一算子、同一输入和相同地址，随机交错 30 次设备计时。数值逐位相同；小幅时间差记录为测量波动，未认定为 TF32 加速。
此处只测输入 gather；固定 W<sub>AA</sub> 仍在准备阶段收集。TF32 未改变这些张量的存储位宽。</p></details>
<details><summary>Graph 调度：同一获选实现 eager 与单次 replay</summary>
{table(['形状','strict组合','eager / ms','Graph / ms','eager / Graph'],scheduling)}
<p>eager 各 20 次、Graph 各 60 次；均同步 wall time。Graph 的调度收益独立于 TF32 数值格式。</p></details>
<h3>部署入口与数据</h3>
<p><code>tf32_pipeline.py::make_precision_prepared(state, x, policy, profile='strict')</code>
读取 <code>tf32_backend_policy.json</code>。输入为相同活跃集合上的 B×A×T、CUDA FP32；参数与初始状态范围沿用本报告。
需要更快的近似方案时显式传入 <code>profile='approximate_0p01mv'</code>；未知形状回到原 IEEE 策略。
原来的 <code>make_optimized_prepared</code> 和冻结源代码保持原样。</p>
<p>下一步可围绕新滤波工作区和输出合成继续减少读写，尝试与二次项相邻阶段融合；在独占资源窗口复测全部表格。
精度进一步放宽时，应沿用完整电压、事件时刻与输入分布覆盖的检查。</p>
<p class="muted">依据：<a href="https://triton-lang.org/main/python-api/generated/triton.language.dot.html">Triton dot 精度模式</a>，
<a href="https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/_p_matmul.py">官方显式 TF32 舍入示例</a>。
代码和逐次计时保存在 <code>tf32_*.py / tf32_*.json</code>；全部数据同时内嵌于本页下载按钮。</p>
<details><summary>全部 13 种形状的组合、延迟分布与精度</summary>{''.join(details)}</details>
</section>'''
    data['summary']=dict(selected_tensor_shapes=selected_count,total_shapes=13,quad_candidates=candidate_total,filter_candidates=filter_total,
        sample500b8_strict=c500strict,sample500b8_approx=c500approx)
    return text,data
