# -*- coding: utf-8 -*-
"""Build a local scientific diagnostic report from completed server outputs."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def rms(a):return float(np.sqrt(np.mean(np.square(a))))
def pool(rows,key):return float(np.sqrt(np.mean([r[key]**2 for r in rows])))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return f'{v:.6g}'


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args();out=a.data
    p=read(out/'passive_complete.json'); q=read(out/'active_error_budget_complete.json');f=read(out/'active_frozen_complete.json')
    assert p['finished'] and q['finished'] and f['finished']
    assert q['primary_checkpoint_unchanged'] and f['checkpoint_unchanged']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#243b4b',
                         'text.color':'#243b4b','axes.titlesize':11,'svg.fonttype':'none'})
    colors={'first':'#3468a3','second':'#008c7a','remainder':'#bf6e36','budget':'#9c80b5','main':'#bd4470'}
    def save(fig,name):
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    # Representative per-site kernels are fixed by distance quantile, not error.
    arr=np.load(out/'passive_arrays.npz');fig,axs=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
    for ax,j in zip(axs,[0,4,7]):
        k=p['kernels'][j];name=f'isolated_{j:03}'
        ax.plot(np.arange(501),arr[name+'_V1'],color=colors['first'],label='Physical first order')
        ax.plot(np.arange(501),arr[name+'_fit'],'--',color=colors['second'],label='One DBNN kernel')
        ax.set_xlim(0,180);ax.set_xlabel('Time (ms)');ax.set_ylabel('Somatic voltage (mV)')
        ax.set_title(f"Site {k['site_id']}: {p['model_metadata']['distance_um'][j]:.0f} um")
        ax.grid(alpha=.18)
    axs[0].legend(fontsize=8);save(fig,'kernel_fits')
    eps=sorted({r['epsilon'] for r in p['records']});pr=p['records'];qr=q['records']
    def vals(records,key,family=None):
        return np.array([pool([r for r in records if r['split']=='test' and r['epsilon']==e and (family is None or r['family']==family)],key) for e in eps])
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for key,label,col in [('first_order_contribution_mv','First-order mismatch',colors['first']),
                          ('second_order_contribution_mv','Second-order mismatch',colors['second']),
                          ('higher_order_residual_mv','Third and higher orders',colors['remainder']),
                          ('total_rmse_mv','Total response RMSE',colors['main'])]:
        axs[0].loglog(eps,vals(pr,key),'-o',ms=3,label=label,color=col)
    axs[0].set_title('Passive morphology: error contributions')
    for key,label,col in [('fitted_features_first_only_rmse_mv','Filtered linear sum',colors['first']),
                          ('total_rmse_mv','Fixed bilinear readout',colors['second']),
                          ('physical_second_rmse_mv','Physical second-order expansion',colors['remainder'])]:
        axs[1].loglog(eps,vals(pr,key),'-o',ms=3,label=label,color=col)
    axs[1].set_title('Passive morphology: voltage approximations')
    for ax in axs:ax.set_xlabel('Common input strength epsilon');ax.set_ylabel('Pooled held-out RMSE (mV)');ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
    save(fig,'passive_budget')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for key,label,col in [('first_contribution_mv','First-order mismatch',colors['first']),
                          ('second_contribution_mv','Main readout second-order residual',colors['second']),
                          ('higher_order_residual_mv','Third and higher orders',colors['remainder']),
                          ('main_drive_response_rmse_mv','Frozen DBNN drive RMSE',colors['main'])]:
        axs[0].loglog(eps,vals(qr,key),'-o',ms=3,label=label,color=col)
    axs[0].set_title('Actual frozen DBNN: original resting protocol')
    for key,label,col in [('first_only_rmse_mv','Frozen filtered linear sum',colors['first']),
                          ('main_drive_response_rmse_mv','Frozen primary readout',colors['main']),
                          ('projected_second_readout_rmse_mv','Diagnostic V2 projection',colors['second']),
                          ('finite_voltage_conditional_readout_rmse_mv','Diagnostic finite-voltage LS',colors['budget'])]:
        axs[1].loglog(eps,vals(qr,key),'-o',ms=3,label=label,color=col)
    axs[1].set_title('Counterfactual readouts; main model unchanged')
    for ax in axs:ax.set_xlabel('Common input strength epsilon');ax.set_ylabel('Pooled held-out response RMSE (mV)');ax.grid(alpha=.2,which='both');ax.legend(fontsize=7.5)
    save(fig,'active_budget')
    fig,axs=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,name,title in zip(axs,['proximal','distributed'],['Proximal inputs','Distributed inputs']):
        rr=sorted([r for r in f['records'] if r['profile']==name and r['existing_figure_case']],key=lambda r:r['epsp_peak_mv'])
        for key,label,col in [('isolated_sum_relative_rms','Complete isolated sum',colors['first']),
                              ('measured_isolated_bilinear_relative_rms','Measured-response bilinear law',colors['second']),
                              ('actual_dbnn_above_own_background_relative_rms','Frozen DBNN; no holding-current input',colors['main'])]:
            ax.plot([r['epsp_peak_mv'] for r in rr],[100*r[key] for r in rr],'-o',ms=3,label=label,color=col)
        ax.axhline(5,color='#87949c',ls=':',lw=1);ax.set_title(title);ax.set_xlabel('Isolated EPSP peak (mV)');ax.set_ylabel('Relative response RMSE (%)');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7.5);save(fig,'holding_protocol')
    def image(name,caption):
        svg=(out/(name+'.svg')).read_text()
        return '<figure><svg'+svg.split('<svg',1)[1]+'<figcaption>'+caption+'</figcaption></figure>'
    # Summaries recomputed from per-input SSE-equivalent RMS values, not from
    # averages of individual relative errors.
    ptest=[r for r in pr if r['split']=='test' and r['epsilon']==1.]
    qtest=[r for r in qr if r['split']=='test' and r['epsilon']==1.]
    pfirst=pool(ptest,'fitted_features_first_only_rmse_mv');psecond=pool(ptest,'total_rmse_mv');pden=pool(ptest,'true_response_rms_mv')
    qfirst=pool(qtest,'first_only_rmse_mv');qmain=pool(qtest,'main_drive_response_rmse_mv');qden=pool(qtest,'response_rms_mv')
    elements=[]
    elements.append('<h1>DBNN 理论与实际网络：第七条诊断</h1><p class="sub">2026-10-16 · 全部仿真与拟合在 5090 执行 · 主线模型、训练、正文和主图均未改动</p>')
    elements.append('<div class="note"><b>先区分三个问题。</b>真实形态中，一阶核能否被实际 DBNN 的一个双指数核准确表示？固定这些特征后，一个固定二阶矩阵还能留下多少表示误差？主线训练得到的矩阵与诊断拟合矩阵有何差别？这三件事分别测量，不能把第三件事直接等同于训练失败。</div>')
    elements.append('<h2>1. 实验对象与拟合、测试划分</h2><p>保留 Shai 小鼠 L5PC 的原始形态和 500 个候选突触位点。按胞体路径距离，在 E/I 两类中各选 8 个位点，共 16 个；选择只依赖几何信息。被动控制移除原模型除 pas 以外的膜密度机制，保留各段的形态、nseg、Ra、cm 和 g_pas，并将 e_pas 设为 −70 mV。主动诊断保留原通道，不添加保持电流，采用原流程的 500 ms 无输入预热。</p><p>突触均为原 Exp2Syn：E/I 基准峰电导 0.188/0.080 nS，rise 0.5/1 ms、decay 5/10 ms，反转电位 0/−80 mV。每段观察 500 ms，以 1 ms 取 501 个等权样本。主求解步长 0.05 ms，代表性输入以 0.025 ms 复核。64 组拟合输入包含同步、异步、20 Hz 随机流和事件包；32 组独立测试包含未见过的 E–I 延迟、26/37 ms 事件间隔以及独立随机流。所有测试输入都保留，不依据输出筛选。</p><p>每个位点仅一个双指数核，使用完整二阶矩阵；16 个位点对应 136 个独立对称系数。被动控制的一阶核仅用 16 条孤立脉冲拟合；主动诊断直接读取冻结主线 checkpoint 的对应核和矩阵。共同强度 ε 覆盖 0.01–8，所有拟合矩阵在整段强度扫描和所有测试时序上保持固定。</p>')
    elements.append('<h2>2. 误差项怎样实际测量</h2><p>用 7 个正强度点对完整电压作多项式回归，提取 V₁、V₂；不通过小电压乘积的逐点除法计算 κ。定义 y 为固定 DBNN 特征，p 为它们的两两乘积。逐条验证下面的向量等式，再分别计算各项 RMS：</p><div class="formula">真实响应 − DBNN drive 的相对背景响应<br>= ε(V₁ − Σᵢyᵢ) + ε²(V₂ − yᵀWy) + R₃。</div><p>因此，一阶与二阶误差可相互抵消；它们的 RMS 不能直接相加当成实测总误差。三项 RMS 之和是上界。主动模型另保存背景差、绝对 drive 误差和带冻结 reset 的闭环电压误差。</p><p>被动细胞的构造定理使用单室和单指数电导等条件；本实验采用多室形态及 Exp2Syn，故允许 δ₁ ≠ 0。主动模型中的 V₁、V₂ 和余项是数值诊断，没有把被动定理的稳定性常数移用于主动模型。</p>')
    elements.append('<p>汇总（pooled）RMSE 将同组所有输入和时间点等权合并。相对响应误差为合并误差 RMS / 合并真实响应 RMS，并非逐条相对误差的算术平均。文中电压 RMS 与 RMSE 均以 mV 计；均使用完整观察窗，不使用主线尖峰排除 mask。强度扫描中 ε=1 对应原基准单事件电导，其他 ε 是用于误差分析的连续电导缩放。</p>')
    elements.append('<h2>3. 被动真实形态：一阶误差不会自动消失</h2>')
    elements.append(image('kernel_fits','按预先指定的近、中、远三个 E 位点展示孤立一阶核；选取不依赖拟合误差。完整 16 个位点见下表。'))
    elements.append(table(['位点 ID','路径距离 (μm)','孤立 V₁ RMS (mV)','核拟合 RMS (mV)','相对 RMS'],[[k['site_id'],fmt(p['model_metadata']['distance_um'][i]),fmt(k['target_rms_mv']),fmt(k['residual_rms_mv']),f"{100*k['relative_rms']:.2f}%"] for i,k in enumerate(p['kernels'])]))
    elements.append('<p>多室传播和 Exp2Syn 的有限 rise time 带来多个时间尺度，一个双指数核一般不能精确重现全部波形。远端位点的相对核误差较高时，其绝对胞体响应也可能很小；两项同时报告，不单凭相对误差判断主线模型的性能。</p>')
    elements.append(image('passive_budget','32 组未见时序的等权 pooled RMS。二阶投影只在 64 组拟合输入上求解一次；红线为真实误差，其他误差贡献用于解释其来源。'))
    elements.append(f'<p>ε=1 时，测试集的滤波一阶和、固定二阶模型 RMSE 为 {pfirst:.6f} 和 {psecond:.6f} mV；以 pooled 真实响应 RMS 归一化，分别为 {100*pfirst/pden:.2f}% 和 {100*psecond/pden:.2f}%。这个控制说明二阶项确实能改善有限强度误差，同时也说明：一阶核未精确匹配时，总误差在 ε→0 不必呈三阶。所有强度结果如下。</p>')
    pt=[]
    for e in eps:
        rr=[r for r in pr if r['split']=='test' and r['epsilon']==e]
        den=pool(rr,'true_response_rms_mv')
        pt.append([e,fmt(den),fmt(pool(rr,'fitted_features_first_only_rmse_mv')),fmt(pool(rr,'total_rmse_mv')),f"{100*pool(rr,'total_rmse_mv')/den:.2f}%",fmt(pool(rr,'higher_order_residual_mv'))])
    elements.append(table(['ε','真实响应 RMS','一阶和 RMSE','固定二阶 RMSE','二阶相对 RMS','物理三阶以上余项 RMS'],pt))
    elements.append('<h2>4. 固定特征后的表示误差与读出拟合</h2><p>第一种诊断将 V₂ 投影到固定乘积特征，用来测二阶表示误差；第二种在 ε=0.5/1/2 的有限电压上拟合条件读出，用来测固定核下的有限幅度误差。这两个目标不同，最优矩阵也不必相同。被动控制另用固定设置的 AdamW 训练后者，与完全相同样本和 MSE 目标的最小二乘解比较。这是独立的条件读出实验，不是重新训练主线网络，也不代替原训练算法的收敛分析。</p>')
    elements.append(table(['被动拟合诊断','结果'],[['V₂ 投影拟合 RMS (mV)',fmt(p['physical_second_projection_fit']['training_rmse_mv'])],['投影秩 / 系数数',f"{p['physical_second_projection_fit']['rank']} / 136"],['保留奇异值最小 / 最大',fmt(p['physical_second_projection_fit']['retained_singular_ratio'])],['有限电压 LS 拟合 RMS (mV)',fmt(p['finite_voltage_projection_fit']['training_rmse_mv'])],['AdamW 与 LS 预测差 RMS (mV)',fmt(np.sqrt(p['fitting_set_orthogonality']['prediction_gap_mse']))]]))
    elements.append('<p>在这组条件读出问题中，梯度训练与最小二乘预测达到数值一致，剩余误差主要不能靠继续训练同一固定特征下的读出来消除。此结论只针对这里的条件问题。测试集上的投影误差单独实测，不称为测试集下界。</p>')
    elements.append('<h2>5. 实际冻结主线 DBNN：原始静息协议</h2><p>这里直接使用原 checkpoint 的核、矩阵、偏置、阈值和 reset 参数，无保持电流。作为诊断，另在 64 组拟合输入上计算 V₂ 投影读出与有限电压 LS 读出；它们仅保存在诊断数组中，并未写回 checkpoint。原训练包含轨迹平衡、差分、mask 和正则项，因此主线矩阵与这些诊断最优解的差距不能称为原训练目标的优化误差。</p>')
    elements.append(f'<p>主动模型关于相同无输入轨迹作强度展开；其记录起点和终点为 {q["background_start_mv"]:.6f}、{q["background_end_mv"]:.6f} mV，窗口内变化范围 {q["background_range_mv"]:.6g} mV。冻结偏置为 {q["background_bias_mv"]:.6f} mV，背景差 RMS 为 {qtest[0]["delta_background_mv"]:.6f} mV。下图先比较各自相对背景的响应；表中同时给出保留背景差和原 reset 的绝对电压误差。这样不会把零阶偏置差混入一阶核或二阶矩阵误差。</p>')
    elements.append(image('active_budget','主动模型的 32 组独立测试。左：冻结主线参数的误差分解；右：原矩阵与两种离线条件读出对照。所有曲线使用相同冻结特征，不改变主线模型。'))
    elements.append(f'<p>ε=1 时，冻结核的一阶和 RMSE 为 {qfirst:.6f} mV，原主线矩阵加入后的 drive 响应 RMSE 为 {qmain:.6f} mV。对应 pooled 相对响应误差为 {100*qfirst/qden:.2f}% 和 {100*qmain/qden:.2f}%。这些是这组 16 位点控制的结果，不替代主线 379 条输入的 RMSE/F1。</p>')
    qt=[]
    for e in [.01,.16,.5,1.,2.,4.,8.]:
        rr=[r for r in qr if r['split']=='test' and r['epsilon']==e]
        qt.append([e,fmt(pool(rr,'first_only_rmse_mv')),fmt(pool(rr,'main_drive_response_rmse_mv')),fmt(pool(rr,'projected_second_readout_rmse_mv')),fmt(pool(rr,'finite_voltage_conditional_readout_rmse_mv')),fmt(pool(rr,'main_closed_loop_absolute_rmse_mv'))])
    elements.append(table(['ε','一阶和','原矩阵 drive','V₂ 投影读出','有限电压 LS 读出','原闭环绝对电压'],qt))
    elements.append(f'<p><b>为什么二阶系数更准，总误差反而可能更大？</b>ε=1 时，V₂ 投影读出的测试 RMSE 为 {pool(qtest,"projected_second_readout_rmse_mv"):.6f} mV，高于原矩阵的 {qmain:.6f} mV。其二阶残差更小，但它不再补偿部分一阶核误差。原训练面向有限强度下的总电压误差，未要求读出逐项等于物理 V₂。因此，固定特征后的最小二阶表示误差与最佳有限幅度预测不是同一目标。有限电压条件读出在拟合强度附近可能改善结果，而向更强输入外推也可能变差；完整扫描均保留在表中。</p>')
    slope_eps=np.array(eps[:4])
    pslope=float(np.polyfit(np.log(slope_eps),np.log(vals(pr,'total_rmse_mv')[:4]),1)[0])
    prslope=float(np.polyfit(np.log(slope_eps),np.log(vals(pr,'higher_order_residual_mv')[:4]),1)[0])
    qslope=float(np.polyfit(np.log(slope_eps),np.log(vals(qr,'main_drive_response_rmse_mv')[:4]),1)[0])
    elements.append(f'<p>在预定的小强度区间 ε=0.01–0.08，pooled log–log 斜率：被动实际双指数特征总误差 {pslope:.3f}，被动物理三阶以上余项 {prslope:.3f}，冻结主动 DBNN 的响应误差 {qslope:.3f}。这分别对应一阶核失配带来的低强度误差，以及正确提取物理一、二阶系数后的高阶余项。斜率用于本控制的阶数核对，不向全部主实验输入外推。</p>')
    fam=[]
    for name in ['unseen_delay','unseen_spacing','independent_random']:
        pp=[r for r in pr if r['split']=='test' and r['epsilon']==1 and r['family']==name]
        qq=[r for r in qr if r['split']=='test' and r['epsilon']==1 and r['family']==name]
        fam.append([name,len(pp),fmt(pool(pp,'total_rmse_mv')),fmt(pool(qq,'main_drive_response_rmse_mv')),fmt(pool(qq,'finite_voltage_conditional_readout_rmse_mv'))])
    elements.append(table(['未见输入族 (ε=1)','条数','被动固定读出 RMSE','主动原矩阵 RMSE','主动条件 LS RMSE'],fam))
    elements.append('<h2>6. Fig. 3(c)：整合法则与实际滤波网络的区别</h2><p>重用现有完整参考轨迹和原来的两个系数，冻结主线网络对全部 38 个已存强度输入作推理；图中标记原 Fig. 3(c) 的 28 个展示点，不新增输出筛选。现有整合法则使用 NEURON 的完整孤立 E/I 响应；新曲线使用实际 DBNN 核和原矩阵。</p>')
    elements.append(image('holding_protocol','相同 1 ms 栅格重新计分。红线是冻结网络相对其自身无输入背景的响应误差；并非在 −70 mV 保持条件下训练的新模型。完整绝对电压误差亦保存在 JSON。'))
    elements.append(f'<p>这个旧实验有 {f["reference_holding_current_na"]:.6f} nA 的胞体保持电流，使参考背景接近 −70 mV；主线偏置为 {f["records"][0]["frozen_checkpoint_bias_mv"]:.6f} mV，输入栅格不编码该保持电流。此外，强度扫描将事件幅度连续缩放，而主线训练使用二值事件。因此，旧图的低误差验证了整合法则，不能直接当作冻结主线模型在这一工作点的端到端误差。新的无保持电流控制用于把这两件事分开。</p>')
    elements.append('<h2>7. 数值核验与保存位置</h2>')
    control_rows=[]
    for model,d in [('被动',p),('主动',q)]:
        for kind in ['half_extraction','degree4','half_step']:
            rr=[r for r in d['controls'] if r['control']==kind]
            control_rows.append([model,kind,len(rr),fmt(max(r['V1_rms_gap_mv'] for r in rr)),fmt(max(r['V2_rms_gap_mv'] for r in rr)),f"{100*max(r['V2_relative_rms_gap'] for r in rr):.4g}%"])
    elements.append(table(['模型','控制','条数','最大 V₁ RMS 差','最大 V₂ RMS 差','最大 V₂ 相对差'],control_rows))
    elements.append('<p>系数提取区间 0.08→0.04、多项式阶数 5→4，以及积分步长 0.05→0.025 ms 分别检查。实际主线推理另与两条已保存的二值输入结果核对，事件一致，电压差小于 0.002 mV。所有误差分解均逐条检查向量恒等式与三角不等式。以上数值控制结果完整列出；没有据此挑选测试结果。</p>')
    elements.append('<p>服务器：<code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/adviser7_theory_bridge_20261016</code>。本地保存完整逐条 JSON、核与响应数组、执行脚本和复现说明。原始逐强度 NEURON 轨迹保存在服务器 raw 子目录。论文及主线 HTML 未修改。</p>')
    elements.append('<ul>'+''.join(f'<li><a href="{n}">{n}</a></li>' for n in ['passive_complete.json','active_error_budget_complete.json','active_frozen_complete.json','passive_arrays.npz','active_error_budget_arrays.npz','active_frozen_arrays.npz','verification.json'])+'</ul>')
    body='\n'.join(elements)
    style='body{font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#243b4b;background:#f6f8fa;margin:0}main{max-width:1120px;margin:auto;padding:36px 48px;background:white}h1{font-size:30px}h2{font-size:23px;border-bottom:1px solid #dce4e8;padding:10px 0;margin-top:38px}.sub,figcaption{color:#5c7281;font-size:14px}.note,.formula{padding:18px 24px;background:#edf7f4;border-left:4px solid #008c7a}.formula{text-align:center;font:18px/2 Georgia,serif}figure{margin:26px 0}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border-bottom:1px solid #dce4e8;text-align:left;padding:9px 12px}th{background:#edf3f7}td{font-variant-numeric:tabular-nums}.scroll{overflow:auto}code{overflow-wrap:anywhere}a{color:#24628b}@media(max-width:700px){main{padding:20px}h1{font-size:24px}}'
    target=out/'DBNN_实际网络理论诊断_20261016.html'
    target.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DBNN 实际网络理论诊断</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>')
    # Verify scalar identities against archived feature arrays, independently
    # of the simulation loops that produced the reported JSON.
    pa=np.load(out/'passive_arrays.npz');qa=np.load(out/'active_error_budget_arrays.npz')
    i,j=np.triu_indices(16);checks=[]
    for model,d,ar,wkey in [('passive',p,pa,'physical_second_readout'),('active',q,qa,'main_readout')]:
        rows=[r for r in d['records'] if r['epsilon']==1.]
        for r in rows:
            name=r['name'];y=ar[name+'_features'];v1=ar[name+'_V1'];v2=ar[name+'_V2'];phi=y[:,i]*y[:,j]
            assert abs(r['delta1_mv']-rms(v1-y.sum(1)))<1e-12
            reported=r['delta2_fixed_mv'] if model=='passive' else r['delta2_main_mv']
            prediction=np.einsum('np,p->n',phi,ar[wkey],optimize=False)
            assert np.isfinite(prediction).all()
            assert abs(reported-rms(v2-prediction))<1e-12
        checks.append(dict(model=model,independently_recomputed_inputs=len(rows)))
    verify=dict(passed=True,local_host=platform.node(),source_results_finished=True,
                primary_checkpoint_unchanged=q['primary_checkpoint_unchanged'] and f['checkpoint_unchanged'],
                metric_checks=checks,active_inference_parity=f['parity'],
                passive_fit_orthogonality_gap=p['fitting_set_orthogonality']['identity_abs_gap'],
                main_fit_V2_orthogonality_gap=q['fitting_set_V2_orthogonality']['identity_abs_gap'],
                maximum_passive_decomposition_identity_gap_mv=max(r['vector_identity_max_gap_mv'] for r in pr),
                maximum_active_decomposition_identity_gap_mv=max(r['vector_identity_gap_mv'] for r in qr),
                test_inputs_retained={'passive':32,'active_original_rest':32},
                manuscript_modified=False,main_report_modified=False,
                note='Visual plot QA is recorded separately after rendering; not inferred from file creation.',
                files={x.name:sha(x) for x in out.iterdir() if x.suffix in ['.json','.npz'] and x.name!='verification.json'})
    (out/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n')
    print(target)
    print(json.dumps(dict(passive_first_rmse=pfirst,passive_bilinear_rmse=psecond,active_first_rmse=qfirst,active_main_rmse=qmain),indent=2))


if __name__=='__main__':main()
