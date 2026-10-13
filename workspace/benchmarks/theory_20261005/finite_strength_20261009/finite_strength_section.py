# -*- coding: utf-8 -*-
"""Finite physiological amplitude section, built from completed 5090 outputs."""
from pathlib import Path
import json,math
import numpy as np

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'finite_strength_20261009/results'
NAMES={'legacy_fast_pair':'快速同步脉冲（2/5 ms）','ampa_pair':'同步脉冲（5/10 ms）','delayed_pair':'抑制延迟 15 ms',
       'packet_pair':'38 ms 内的 20 事件输入包','tonic_pair':'300 ms 持续事件流','poisson_pair':'100/80 Hz 聚合 Poisson 输入'}


def build(eq,theorem,proof,table,figure):
    d=json.loads((RESULTS/'passive_finite_strength.json').read_text());l=json.loads((RESULTS/'l5pc_finite_strength.json').read_text());control=json.loads((RESULTS/'l5pc_half_step_validation.json').read_text())
    assert d['finished'] and l['finished'] and control['finished']
    representatives=[];variance=[];slopes=[]
    for check in d['profile_checks']:
        key=check['config']['id'];rs=[r for r in d['rows'] if r['profile']==key and r['common_band']]
        r=min(rs,key=lambda r:abs(r['epsp_peak_mv']-5))
        representatives.append([NAMES[key],f'{r["epsp_peak_mv"]:.3f} / {r["ipsp_abs_peak_mv"]:.3f}',
            f'{100*r["first_relative_rms"]:.2f}%',f'{100*r["exact_second_relative_rms"]:.2f}%',f'{100*r["fixed_dbnn_relative_rms"]:.2f}%',
            f'{100*r["shared_readout_relative_rms"]:.2f}%' if 'shared_readout_relative_rms' in r else '单独核族'])
        info=check['pair_variance'];variance.append([NAMES[key],f'{info["weighted_kappa_variance_per_mv2"]:.6g}',
            f'{info["weighted_kappa_std_per_mv"]:.5f}',f'{r["second_to_higher_ratio"]:.3f}'])
        if key in ['ampa_pair','tonic_pair','poisson_pair']:
            valid=[r for r in rs if r['second_needed'] and r['quadratic_meets_5pct']]
            slope=float(np.polyfit(np.log([r['epsilon'] for r in valid]),np.log([r['fixed_dbnn_rmse_mv'] for r in valid]),1)[0])
            slopes.append([NAMES[key],f'{valid[0]["epsp_peak_mv"]:.2f}–{valid[-1]["epsp_peak_mv"]:.2f}',len(valid),f'{slope:.3f}'])
    actual=[]
    for r in l['rows']:
        if r['physiological_band']:
            actual.append([{'proximal':'近端','distributed':'分散','distal':'远端'}[r['profile']],r['strength'],
                f'{r["epsp_peak_mv"]:.3f} / {r["ipsp_abs_peak_mv"]:.3f}',
                f'{100*r["additive_relative_rms"]:.2f}%',f'{100*r["fixed_bilinear_relative_rms"]:.2f}%',
                f'{r["joint_response_rms_mv"]:.3f}',f'{r["fixed_bilinear_rmse_mv"]:.4f}'])
    previous=json.loads((RESULTS/'original_single_pulse_metric_review.json').read_text())
    count=d['grid_counts_not_population_probabilities']['common_band'];extended=d['grid_counts_not_population_probabilities']['amplitude_band']
    gap=max(r['half_step_max_voltage_gap_mv'] for r in d['profile_checks'])
    real_gap=max(v['max_absolute_gap_mv'] for c in control['checks'] for v in c['voltage_checks'])
    real_rel=max(abs(c['coarse_bilinear_relative_rms']-c['fine_bilinear_relative_rms']) for c in control['checks'])
    bias=l['fit_profiles'][0]['holding_current_na']
    settings=[]
    for c in d['profile_checks']:
        p=c['config'];settings.append([NAMES[p['id']],f'{p["tau_e"]:g}/{p["tau_i"]:g}',
            len(p['te']),len(p['ti']),f'{p["ge"]*200:g}/{p["gi"]*200:g}',f'{p["T"]:g}'])
    return '''<h3 id="finite-strength">3.6 有限输入强度：何时需要二阶，何时达到 5% 误差？</h3>
<p class="section-intro">前面的阶数试验回答 ε→0 时的误差规律。本节把横轴换成实际 EPSP/IPSP 幅度，检查有限电导下的一阶近似、完整二阶响应与固定二次读出。随后在原主动 L5PC 上检查有限强度的 E/I 乘积法则；数值求解均在 5090 完成。</p>
<p><strong>先核对原动态例：</strong>用原保存轨迹重新计算相对 RMS，在单路兴奋性指数脉冲 ε=1 时，实际 EPSP 峰值为 5.186 mV，一阶误差为 4.59%，原固定二次读出为 1.51%。因此，原例在这个强度点的一阶近似已经达到 5% 目标。本节进一步检查在哪些有限输入条件下，二次结构成为达到该目标的必要改进。</p>
<h4>先确定物理幅度与“5% 误差”的口径</h4>
<p>输入强度的实际影响取决于突触电导、漏电导、反转电位、作用时长和空间位置。<a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004014">Li et al. (2014)，Discussion</a> 以胞体 EPSP 约 6 mV 以内、IPSP 绝对幅度约 3 mV 以内作为其生理分析范围，并讨论了接近 10 mV 去极化时的二阶精度。这里据此标示常用与扩展幅度带，具体归属由每条参考响应测量。这些幅度带对应所讨论的工作区间；不同细胞、位点和背景电压还需结合各自响应确定。原单路脉冲与多路 E/I 条件的误差同时展示，幅度之外还比较输入形状与共同电导作用。</p>
<p>设 v(t) 为真实电压相对同一背景轨迹的变化，v̂(t) 为近似响应。取整个指定观察窗的归一化 RMS：</p>'''+eq(r'''\eta_{\mathrm{RMS}}=\frac{\|v-\widehat v\|_\rho}{\|v\|_\rho},\qquad
\eta_{\mathrm{peak}}=\frac{\max_t|v(t)-\widehat v(t)|}{\max_t|v(t)|},\qquad
\eta_{\mathrm{RMS}}\leq0.05''','FS1')+'''
<p>本节的 5% 目标使用 η<sub>RMS</sub>，分母是输入带来的响应幅度；同时保存 η<sub>peak</sub> 供最大误差口径复核。所有亚阈值曲线包含起始响应与衰减尾部。零输入及零响应不定义这个相对指标。</p>
<h4>一个可精确求解的有限强度参照：恒定电导</h4>
<p>单室兴奋性恒定电导 G、漏电导 g<sub>L</sub>，令 χ=G/g<sub>L</sub>，相对静息的反转电位为 E。稳态真实响应、一阶及二阶近似为：</p>'''+eq(r'''v=\frac{E\chi}{1+\chi},\qquad v^{(1)}=E\chi,\qquad
v^{(2)}=E\chi-E\chi^2,\qquad
\frac{|v-v^{(1)}|}{|v|}=\chi,\qquad
\frac{|v-v^{(2)}|}{|v|}=\chi^2''','FS2')+'''
<p>因此，χ≤0.05 时一阶已满足 5%；0.05&lt;χ≤√0.05≈0.224 时一阶超过 5%，而二阶仍满足 5%。取 E=70 mV，这个二阶必要区间对应约 3.33–12.79 mV 的真实 EPSP。这里 κ 恒定、二階表示精确，剩余绝对误差为 Eχ³/(1+χ)。</p>'''.replace('二階','二阶')+table(['真实 EPSP','G/gL','一阶相对误差','二阶相对误差'],[
['3.33 mV','0.0500','5.00%','0.25%'],['6.00 mV','0.09375','9.38%','0.88%'],['10.00 mV','0.1667','16.67%','2.78%']])+proof('有限强度公式的直接核查',
'''<p>稳态方程为 (g<sub>L</sub>+G)v=EG。将真实响应分别减去 Eχ 和 Eχ−Eχ²，得到 −Eχ²/(1+χ) 和 Eχ³/(1+χ)，除以 |v| 即得两个相对误差。对应固定二次系数为 −1/E，参数在整个 χ 区间保持不变。</p>''')+'''
<h4>动态输入：把弱输入扫描扩展到实际幅度</h4>
<p>使用被动单室 C=200 pF、g<sub>L</sub>=10 nS（膜时间常数 20 ms），兴奋/抑制反转电位相对静息为 +70/−10 mV。C、g<sub>L</sub> 是本节用于把归一化电导换算成 nS 的示例参数。每路事件带来指数电导；对应真实一阶电压核可用 DBNN 双指数核精确表示。每种输入固定一组核与包含平方项的二次矩阵，扫过 64 个 ε=0.05–12 的强度点，求完整非线性方程。</p>'''+table(['输入形状','E/I 衰减 / ms','E 事件','I 事件','基准每事件电导峰 / nS','观察 / ms'],settings)+'''
<p>持续事件流与 Poisson 流表示多位点输入合并后的事件组；事件组速率描述聚合驱动。强度同时缩放两路电导，幅度带由孤立 E/I 响应实测。图中蓝线为真实一阶核；灰虚线为完整物理二阶展开；绿线为一次拟合后固定的二次读出。二次系数仅针对真实 V₂ 拟合，扫描 ε 时保持固定。橙点线进一步用一组共同 W 覆盖全部五种 5/10 ms 核族输入，显示跨输入形状的实际要求。</p>'''+figure('finite_strength_accuracy','图 7 · 有限输入强度与 5% 误差。横轴为实际孤立 EPSP 峰值，显示 IPSP≤3 mV 的点；绿色区域为相对 RMS 误差≤5%。每条曲线保留完整观察窗。全物理二阶展开与固定二次乘积读出分别列示。')+'''
<p>下表对每种形状选取常用幅度带中最接近 EPSP=5 mV 的已计算点。每行的输入与参考完全相同：</p>'''+table(['输入形状','EPSP / |IPSP|（mV）','一阶','完整物理二阶','固定二次读出','共同 W'],representatives)+f'''
<p>本次常用幅度带共有 {count['grid_points']} 个网格点：一阶达标 {count['first_within_5pct']} 点；其余 {count['second_needed']} 点需要更高阶描述，其中固定二次读出达标 {count['fixed_quadratic_success']} 点。扩展到 EPSP≤10 mV 后，{extended['second_needed']} 个一阶未达标点中有 {extended['fixed_quadratic_success']} 点达到二次读出的 5% 目标。它们是预设幅度网格上的结果计数；各形状与完整数值保存在 JSON。</p>
<h4>需要二阶时，κ 是否更平坦、误差是否更接近三阶？</h4>
<p>对固定事件形状，共同幅度缩放同时给 V₂ 与一阶乘积带来 ε² 因子，所以理论二阶 κ 及其归一化加权方差随 ε 保持不变。输入持续时间、相对时间尺度、E/I 时差和空间传播会改变形状。下表按真实 E–I 混合二阶项的响应能量计算 κ 方差，并列出上述约 5 mV 样本的二阶表示残差／高阶余项 RMS 比：</p>'''+table(['输入形状','κ 加权方差 / mV⁻²','κ 加权标准差 / mV⁻¹','二阶表示残差／高阶余项'],variance)+'''
<p>每个 ε 下，用固定 W 的二阶残差 R₂=V₂−yᵀWy 和真实高阶余项 R₃=v−εV₁−ε²V₂，逐项保存：</p>'''+eq(r'''v-\widehat v_{\mathrm{DBNN}}=\epsilon^2R_2+R_3,\qquad
\|v-\widehat v_{\mathrm{DBNN}}\|_\rho^2=
\epsilon^4\|R_2\|_\rho^2+\|R_3\|_\rho^2+2\epsilon^2\langle R_2,R_3\rangle_\rho''','FS3')+figure('finite_strength_error_budget','图 8 · 固定二次读出的误差预算。两类输入均按真实响应 RMS 归一化。实际误差由二阶表示残差与高阶余项共同构成，平方误差保留两者的有符号交叉项。')+'''
<p>持续事件流的 κ 加权方差约为同步 5/10 ms 脉冲的 1/14；在约 5 mV 样本上，高阶余项明显占主要部分。以下在“一阶&gt;5%、固定二次≤5%”的有限幅度区间拟合绝对 RMSE 对 ε 的 log–log 斜率：</p>'''+table(['输入形状','拟合 EPSP 范围 / mV','强度点','有限区间斜率'],slopes)+'''
<div class="finding"><strong>有限强度的结论：</strong>二阶读出在一阶误差已明显超过 5% 的幅度区间仍能有效控制误差。持续、较平缓的驱动使二阶表示残差更小，有限区间内更接近三阶的误差规律；快速事件流保留更明显的二阶残差，也可满足具体的 5% 精度目标。渐近阶数与有限区间精度分别通过残差分解和实际误差计算。</div>
<h4>原主动 L5PC：有限 EPSP/IPSP 乘积法则的实际检验</h4>
<p>使用主实验相同的 Shai et al. (2015) 小鼠 V1 L5PC、cell1.asc、L5PCbiophys5.hoc 及原主动通道。保留 500 个候选位点与原 Exp2Syn：E/I 上升 0.5/1 ms，衰减 5/10 ms，反转电位 0/−80 mV，单事件基准峰电导 0.188/0.080 nS。分别选取 32 个 E 位点和 32 个 I 位点，覆盖近端、分散、远端三种集合；20 ms 同步触发，记录 300 ms。</p>
<p>先按原配置计算无持续驱动的基准，胞体约 −80.844 mV；抑制反转电位接近此工作点，孤立输入主要改变分流而不是产生明显负向 IPSP。再加入同一固定背景电流，使基准胞体电压为 −70.000 mV。背景电流仅设定工作点，通道与突触参数保持原值。</p>'''+f'''
<p>固定背景电流为 {bias:.6f} nA，预热 500 ms。每组位置只用 9 个较弱强度组合（E/I 分别为 0.25、0.5、1 倍）拟合一个共享 c；随后在 1.5、2、3、4、6、8、12、16 倍强度上测试。每种位置共 40 条 NEURON 轨迹，三种位置合计 120 条。</p>
<p>本组以完整单输入响应 V<sub>E</sub>、V<sub>I</sub> 为基底：相加基线为 V<sub>E</sub>+V<sub>I</sub>，乘积校正为 V<sub>E</sub>+V<sub>I</sub>+cV<sub>E</sub>V<sub>I</sub>。这些单输入轨迹包含各路自作用；其目标是检验有限强度下固定乘积系数的实际适用区间。当前学习 DBNN 的整体电压与 reset 表现仍由 379 条冻结模型实验报告。</p>
'''+figure('finite_strength_l5pc','图 9 · 原主动 L5PC 在 −70 mV 背景工作点的有限强度检验。只显示未发生动作电位且 EPSP≤10 mV 的未拟合强度；所有其他输入、尖峰与完整轨迹保留。一个固定 c 覆盖每组位点的输入幅度与完整时间窗。')+table(['空间集合','强度倍数','EPSP / |IPSP|（mV）','相加误差','固定乘积校正','联合响应 RMS / mV','校正 RMSE / mV'],actual)+'''
<p>近端与分散位点在 6–9 mV EPSP 范围中出现了“相加超过 5%、固定乘积校正仍低于 5%”的情况。远端输入出现明显不同的响应形状，即使胞体 EPSP 较小，固定系数的外推误差仍可能较大。表中同时列出绝对 RMSE 与联合响应 RMS，保留响应抵消对相对误差分母的影响。由此实际工作区间需要联合指定幅度、位点、事件形状与背景电压。</p>
<details><summary>复现设置、积分精度和完整数据</summary>'''+f'''
<p>被动方程以 RK4 0.01 ms 求解，与 0.02 ms 对照；0.1 ms 观察网格，事件跳变在时间区间起点处理，全部强度的最大电压差 {gap:.3g} mV。真实 L5PC 以 0.1 ms 积分、0.1 ms 记录；对三组代表性强度及其零/E/I/EI 轨迹再用 0.05 ms 对照，保持相同背景电流与固定读出，最大电压差 {real_gap:.4g} mV，相对校正误差最大变化 {100*real_rel:.4g} 个百分点，参考事件一致。</p>
<p>κ 的数值展示由多个强度的响应回归得到：逐时间拟合 SC 对 V<sub>E</sub>V<sub>I</sub>，固定系数由整个输入—时间集合共同拟合；小乘积尾部只按能量支撑筛选展示。有限强度的拟合系数记为 κ<sup>eff</sup>，与弱输入展开的真实二阶 κ 分开保存。</p>
<p><a href="DBNN_theory_data_20261005/passive_finite_strength.json">被动 384 个幅度点与固定矩阵</a> · <a href="DBNN_theory_data_20261005/passive_finite_strength.npz">被动完整轨迹与逐阶系数</a> · <a href="DBNN_theory_data_20261005/l5pc_finite_strength.json">L5PC 工作点、位点与全部未拟合强度</a> · <a href="DBNN_theory_data_20261005/l5pc_finite_strength_summary.npz">L5PC 响应与拟合 κ</a> · <a href="DBNN_theory_data_20261005/l5pc_half_step_validation.json">L5PC 半步核查</a> · <a href="DBNN_theory_data_20261005/original_single_pulse_metric_review.json">原单路例的相对误差复核</a> · <a href="DBNN_theory_data_20261005/l5pc_rest_finite_strength.json">原无背景驱动对照</a>。</p>
<p>远程原始目录：<code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/theory_20261005/finite_strength_20261009/</code>。背景驱动的正式 L5PC 数据位于 <code>l5pc_hold70</code>，无背景对照位于 <code>l5pc</code>。采样、分析脚本和全部强度轨迹保留；本节没有训练或替换任何 checkpoint。</p></details>
'''
