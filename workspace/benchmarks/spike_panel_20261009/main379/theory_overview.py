# -*- coding: utf-8 -*-
"""Integrate the current theory overview without changing its source report."""
from pathlib import Path
import re,hashlib

THEORY_FILE='DBNN_理论分析路线_JSC_20261005.html'
STYLE='''
.theory-overview{margin-top:30px;padding-top:20px;border-top:2px solid #cde0e4;scroll-margin-top:28px}
.theory-overview h4{font-size:16px;color:#21566a;margin:24px 0 9px}
.theory-overview .equation{position:relative;background:#f5f9fb;border:1px solid #dbe7ed;border-radius:8px;margin:18px 0;padding:20px 56px 20px 16px;overflow-x:auto;scroll-margin-top:25px}
.theory-overview .equation math{display:block;font-size:1.06em;min-width:max-content}
.theory-overview .eq-no{position:absolute;right:12px;top:12px;color:#67818e;font-size:12px}
.theory-overview .analysis-framework,.theory-overview .theorem{padding:18px;background:#f0f6fa;border:1px solid #d5e5ed;border-radius:8px;margin:18px 0}
.theory-overview .framework-title,.theory-overview .theorem-title{font-weight:700;color:#21566a}
.theory-overview .finding{padding:16px 20px;background:#edf7f4;border-left:4px solid #087b68;margin:18px 0}
.theory-overview .flow{display:flex;gap:10px;align-items:center;background:#eef7f8;padding:16px;border-radius:8px;margin:20px 0}
.theory-overview .flow>div{flex:1;text-align:center}.theory-overview .flow small{display:block;font-size:12px}
.theory-overview .proof{background:#fcfbf5;padding:14px 18px;border:1px solid #e7e2cf;border-radius:8px;margin:18px 0}
.theory-overview .complete-theory{margin-top:24px}.theory-overview .complete-theory>summary{font-weight:650;color:#21566a}
@media(max-width:700px){.theory-overview .flow{display:block}.theory-overview .flow>span{display:none}.theory-overview .flow>div{margin:10px 0}.theory-overview .equation math{font-size:.95em}}
'''

def render(root):
    source=(root/THEORY_FILE).read_text()
    match=re.search(r'<section\b[^>]*id="overview"[^>]*>(.*?)</section>',source,re.S)
    assert match,'Theory overview missing'
    overview=match[1]
    equations={m[1]:m[0] for m in re.finditer(r'<div class="equation" id="(eq-1-\d+)">.*?</div>',overview,re.S)}
    def eq(number,label):
        block=equations[f'eq-1-{number}'].replace(f'id="eq-1-{number}"',f'id="theory-{label}"')
        return block.replace(f'(1.{number})',f'({label})')
    original_ids=set(re.findall(r'\bid="([^"]+)"',overview))
    full=re.sub(r'\bid="([^"]+)"',lambda m:'id="theory-detail-'+m[1]+'"',overview)
    full=re.sub(r'href="#([^"]+)"',lambda m:'href="'+('#theory-detail-'+m[1] if m[1] in original_ids else THEORY_FILE+'#'+m[1])+'"',full)
    full=re.sub(r'\(1\.(\d+)\)',r'(D\1)',full)
    full=re.sub(r'第\s*([123](?:\.\d+)?)\s*(章|节)',r'理论报告第 \1 \2',full)
    full=re.sub(r'<h2>.*?</h2>','<h4>完整推导：误差分解、逐阶匹配与 κ 方差</h4>',full,count=1,flags=re.S)
    full=re.sub(r'<h3([^>]*)>1\.(\d+) ',r'<h4\1>推导 \2 · ',full).replace('</h3>','</h4>')
    full=full.replace('<h4 id="theory-detail-model">','<h5 id="theory-detail-model">').replace('DBNN 模型与参数：基准输入的双指数滤波</h4>','DBNN 模型与参数：基准输入的双指数滤波</h5>')
    result=f'''<div class="theory-overview" id="theory-overview"><style id="theory-overview-style">{STYLE}</style><h3>2.1 理论分析：逐阶匹配与可达到误差</h3>
<p>双指数滤波描述逐路一阶响应，全矩阵 W 描述平方自作用与不同位点的交叉作用。下面分析这两层结构能拟合到什么程度，以及固定二次系数留下的误差由什么决定；随后第 3、4 章给出完整主动 L5PC 的实际预测与计算结果。</p>
<h4>先把完整电压误差按输入强度分开</h4>
<p>ξ 表示一套基准输入，包括位点、事件时刻、波形与相对权重；ε≥0 是共同缩放输入电导或事件权重的无量纲强度。ρ 是输入与观察时间上的归一化权重，∥f∥<sub>ρ</sub>=(E<sub>ρ</sub>f²)<sup>1/2</sup>。在指定初始化、有限观察时间、有界输入族以及真实响应具有一致三阶展开的条件下，相对静息电压 v<sub>ε</sub> 写为：</p>
{eq(1,'T1')}
<p>V<sub>1,i</sub> 是第 i 路真实一阶响应；V<sub>2,ii</sub>=V<sub>2,i</sub> 是平方自作用，i&lt;j 时 V<sub>2,ij</sub> 是混合二阶响应。yᵢ 是 DBNN 核对未乘 ε 的基准输入的滤波输出。对任意固定参数 b、核与 W，reset 前的模型输出及误差满足：</p>
{eq(2,'T2')}
<p>δ<sub>b</sub>、δ₁、δ₂(W) 分别为静息偏置、一阶总响应和二阶响应的 RMS 误差，单位均为 mV；输入缩放时分别以 1、ε、ε² 进入总误差。W 的单位是 mV⁻¹，对角允许非零。输入强度改变时，上式保持同一组模型参数。</p>
<h4>静息项和一阶项怎样匹配？</h4>
<p>取 b=v<sub>rest</sub>，即可使 δ<sub>b</sub>=0。若每个位点的真实一阶电压响应核恰好属于 DBNN 的双指数核族，则取相应的幅度和时间常数，可使 yᵢ=V<sub>1,i</sub>，从而 δ₁=0。这种双指数形式可由被动单室中单指数电导经膜电容、漏电导滤波得到，对应 ExpSyn 所采用的 AMPA 型电导近似。对有树突传播、多时间尺度的细胞，一阶核的可表示程度由 δ₁ 记录。</p>
<h4>固定二次矩阵的误差由 κ 的什么性质决定？</h4>
<p>在逐路一阶核已经匹配的条件下，令 p<sub>ij</sub>=V<sub>1,i</sub>V<sub>1,j</sub>=yᵢyⱼ；p<sub>ij</sub>≠0 处定义真实 κ<sub>ij</sub>=V<sub>2,ij</sub>/p<sub>ij</sub>。固定读出系数为 c<sub>ii</sub>=W<sub>ii</sub>，c<sub>ij</sub>=W<sub>ij</sub>+W<sub>ji</sub>（i&lt;j）。对 E/I 输入，混合项对应 SC=κ·EPSP·IPSP+O(ε³)，与 Li et al. (2014) 的小电导双线性分析衔接。</p>
<p>对 E<sub>ρ</sub>p<sub>ij</sub>²&gt;0 的单项，以响应乘积的平方能量作为权重；最佳固定系数和最小平方误差为：</p>
{eq(9,'T3')}
<p>μ<sub>ij</sub> 是输入与时间的联合加权分布。若 p=0 处的二阶目标也为零，右侧最后一项消失，得到<strong>单项最小 RMSE = RMS(p<sub>ij</sub>) × κ<sub>ij</sub> 的加权标准差</strong>，其中 RMS(p)=(E<sub>ρ</sub>p²)<sup>1/2</sup>。这样既考虑 κ 的变化，也考虑该变化对应的实际响应幅度。纯分流抑制可能出现一阶电压为零而混合二阶响应非零的情况，此时保留零乘积处的目标能量。</p>
<p>按 c<sub>ii</sub>* 设置 W<sub>ii</sub>，按 c<sub>ij</sub>*/2 设置对称的 W<sub>ij</sub>、W<sub>ji</sub>，可构造一个可行的全矩阵。记 δ<sub>2,*</sub>=min<sub>W</sub>δ₂(W)，其联合最优误差和完整电压误差满足：</p>
{eq(11,'T4')}
<div class="finding"><strong>何时能达到三阶精度？</strong> 若每个有效平方项、交叉项的 κ 在同一输入—时间范围内为常数，且零乘积处的目标为零，那么同一组 W 可以精确匹配二阶项，完整误差只剩 C₃ε³。更一般地，只要 V₂ 属于全部 yᵢyⱼ张成的固定乘积空间，联合二阶误差就为零。κ 的加权方差较小时，上式给出较小的二阶误差上界；不同项的联合拟合还允许残差相互补偿。</div>
<h4>这怎样指导当前训练与数值检验？</h4>
<p>静息偏置、响应核和二次读出分别对应 δ<sub>b</sub>、δ₁、δ₂；开放对角项使模型能够表示同一位点的平方响应。实际训练系数还可能偏离最佳固定系数，对单项有 e<sub>ij</sub>(c)²=e<sub>ij,*</sub>²+(E<sub>ρ</sub>p<sub>ij</sub>²)(c−c<sub>ij</sub>*)²。因此，核与读出的联合微调可以同时调整一阶匹配和二阶读出，具体改善程度由独立验证与评估统计确定。</p>
<p>数值上用拟合估计 κ，并用乘积能量加权，避免小响应尾部的直接相除放大数值误差。理论报告中的<a href="{THEORY_FILE}#steady-order-results">恒定电导稳态检验</a>与<a href="{THEORY_FILE}#pair-order-results">动态 E–I pair 检验</a>，分别验证二阶精确匹配后的三阶误差、以及固定系数残差保留时的二阶误差。</p>
<p class="note">此处分析 reset 前的响应表示。三阶余项常数与一阶匹配条件均随输入范围、初始化和观察区间确定；L5PC 的主动通道、闭环 reset 和尖峰时刻表现由后续独立电压与事件实验衡量。<a href="{THEORY_FILE}#overview">打开理论报告原文</a>。</p>
<p id="theory-finite-strength"><strong>有限输入强度的实际检验：</strong>理论报告新增<a href="{THEORY_FILE}#finite-strength">第 3.6 节：有限强度与 5% 误差</a>，把横轴换成实际 EPSP/IPSP 幅度，对照一阶、完整二阶与固定二次读出，并分析 κ 方差和高阶余项。另有原主动 L5PC 的未拟合强度检验，单独陈述完整单输入响应的乘积校正结果。</p>
<details class="complete-theory"><summary>展开完整推导：加权方差、联合最优矩阵与任意事件组合</summary><p>以下同步理论报告 overview 的完整内容，公式在本页记为 D1–D18；恒定电导、动态输入及其数值检验的章节链接指向理论报告。</p>{full}</details>
</div>'''
    return result,dict(theory_source=THEORY_FILE,theory_source_sha256=hashlib.sha256(source.encode()).hexdigest(),overview_integrated=True,theory_core_equations=4,full_overview_equations=len(equations))
