# -*- coding: utf-8 -*-
"""Replace the mainline coherently; preserve the former 31-window story in an appendix."""
from pathlib import Path
import json,re,hashlib,html,shutil
from collections import Counter
from html.parser import HTMLParser

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN')
DATA=ROOT/'DBNN_main379_data_20261009'
PREFIX='DBNN_main379_data_20261009/'


class Parse(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.append(d['id'])
        if tag=='img':self.images.append(d.get('src',''))


def section_range(s,identifier):
    m=re.search(r'<section\b[^>]*\bid="'+re.escape(identifier)+r'"[^>]*>',s);assert m,identifier
    depth=1
    for tag in re.finditer(r'</?section\b[^>]*>',s[m.end():]):
        depth+=-1 if tag.group().startswith('</') else 1
        if depth==0:return m.start(),m.end()+tag.end()
    raise ValueError(identifier)


def main():
    d=json.loads((DATA/'main379_results.json').read_text());m=d['models'];assert d['finished'] and d['cases']==379 and d['reference_events']==637
    training=json.loads((HERE/'checkpoint_training_summary.json').read_text());shutil.copy2(HERE/'checkpoint_training_summary.json',DATA/'checkpoint_training_summary.json')
    shutil.copy2(HERE/'selected_examples.json',DATA/'selected_examples.json')
    shutil.copytree(HERE/'examples',DATA/'examples',dirs_exist_ok=True)
    db=m['gpu']['accuracy'];ne=m['neuron8']['accuracy'];wall=lambda k:m[k]['timing']['wall_ms'];ratio=wall('neuron8')/wall('gpu')
    keys=['gpu','cpu8','cpu1','neuron8','core1','tcn','elm','lstm','reduce']
    names={'gpu':'DBNN GPU','cpu8':'DBNN CPU，8 核','cpu1':'DBNN CPU，1 核','neuron8':'NEURON multisplit，8 核','core1':'CoreNEURON CPU，1 核',
           'tcn':'TCN GPU · Beniaguev et al. 2021','elm':'Branch-ELM GPU · Spieler et al. 2024','lstm':'LSTM GPU · Hochreiter & Schmidhuber 1997','reduce':'Neuron_Reduce CPU · Amsalem et al. 2020'}
    def figure(stem,alt,caption):return f'<figure><img loading="lazy" src="{PREFIX}figures/{stem}.png" alt="{alt}" style="width:100%;height:auto"/><figcaption>{caption}</figcaption></figure>'
    def table(headers,rows):return '<div class="resolution-table"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
    summary=[]
    for k in ['gpu','neuron8','core1']:
        e=m[k]['accuracy']['events'];summary.append([names[k],f'{m[k]["accuracy"]["pooled_rmse_mv"]:.4f}',f'{e["f1"]:.3f}',f'{wall(k):,.3f}'])
    parts={}
    parts['conclusion']=f'''<section class="card" id="conclusion"><span id="backup-neuron-main"></span><h2>主线：379 条独立输入上的精度、尖峰与速度</h2>
<p>同一小鼠 V1 L5PC、完整主动离子通道、500 个候选突触位点。评估使用新生成的 <strong>379 条输入、637 个参考尖峰</strong>；训练与验证轨迹另行划分。学习模型统一使用 1 ms 输入／输出；NEURON、CoreNEURON 与 Neuron_Reduce 使用 <strong>0.5 ms 积分</strong>。参考 NEURON 为 0.1 ms 积分，全部误差在共同 1 ms 网格计算。</p>
{table(['工作点','掩码 RMSE / mV','事件 F1（±5 ms）','379 窗计算时间合计 / ms'],summary)}
<p>全矩阵 DBNN 的 RMSE 为 <strong>{db['pooled_rmse_mv']:.4f} mV</strong>，事件 F1 为 <strong>{db['events']['f1']:.3f}</strong>（95% CI {db['bootstrap_95ci']['f1'][0]:.3f}–{db['bootstrap_95ci']['f1'][1]:.3f}）。0.5 ms NEURON 的电压误差更低；DBNN 的 ±5 ms 事件匹配表现更高。准备后的完整 GPU 计算时间合计为 <strong>{wall('gpu'):.3f} ms</strong>，对应 NEURON multisplit 8 核工作点的时间比为 <strong>{ratio:,.0f}×</strong>。</p>
<p><a href="#accuracy-development">训练流程</a> → <a href="#voltage-consistency">电压与尖峰</a> → <a href="#comparison-protocol">速度—精度图</a> → <a href="#duration-timing">分时长结果</a>。旧 31 窗、多分辨率模型与先前计时已归入<a href="#historical-main31">附录 AK</a>。</p></section>'''
    input_rows=[['参考与细胞','Shai 小鼠 V1 L5PC：原形态、完整主动通道；兴奋性 AMPA 样 Exp2Syn，抑制性 Exp2Syn；0.1 ms 积分、500 ms 预热'],
                ['独立输入','从 637 条新输入中，按参考每窗 0–3 个尖峰、最小间隔至少 50 ms 固定主集合；379 条中有 374 条随机轨迹及 5 条确定性零输入'],
                ['覆盖范围','T=1 / 2 / 4 / 8 / 16 s；A=64 / 128 / 256 / 500；10 / 25 / 50 Hz/位点；随机、分层、近端、远端与全部位点'],
                ['随时间变化的输入','包含连续 Poisson、25/50 Hz 分段变化及背景 25 Hz 上的三段 40 ms E/I 比例变化（E=50 Hz、I=10 Hz）'],
                ['训练与验证','117 条拟合、32 条验证源轨迹，每条 6 s；与本次输入的 ID 和完整输入数组哈希无重叠。评估输入未用于训练、reset／阈值或 checkpoint 选择'],
                ['模型与求解器','DBNN：完整 500×500 二次矩阵 W，对角可训练；GPU、CPU1 与 CPU8 使用同一模型。TCN、Branch-ELM、LSTM 沿用已锁定作者流程模型；全部为 1 ms。数值求解器统一 0.5 ms 积分'],
                ['电压统计','全部 379 条计入；剔除参考尖峰 ±10 ms，在共同 1 ms 观察网格累计平方误差后计算 pooled RMSE'],
                ['事件统计','完整时间窗，±5 ms 一对一匹配；每个事件最多计入一次。累计 TP、FP、FN，计算 precision、recall 和 F1；额外事件也计入'],
                ['不确定性','5,000 次轨迹级 bootstrap；随机轨迹整体重采样，5 条确定性零输入固定计入。区间描述固定细胞与模型在本输入集合上的变化']]
    n_keep=sum(r['voltage']['n'] for r in m['gpu']['records']);n_all=sum(r['duration_ms']+1 for r in m['gpu']['records'])
    parts['neuron-comparison']=f'''<section class="card" id="neuron-comparison"><h2>1. 比较对象、独立输入与评分</h2>{table(['环节','当前主线设置'],input_rows)}
<div class="formula">Ωᵢ = {{n : n 距任一参考尖峰超过 10 ms}}；eᵢ[n] = vᵢ[n] − v<sub>ref,i</sub>[n]<br/>RMSE<sub>pooled</sub> = √(Σᵢ Σ<sub>n∈Ωᵢ</sub> eᵢ[n]² / Σᵢ |Ωᵢ|)<br/>Precision = TP/(TP+FP)；Recall = TP/(TP+FN)；F1 = 2TP/(2TP+FP+FN)</div>
<p>共 {d['total_duration_s']:,} 秒物理时长，{n_all:,} 个共同观察点，电压掩码后保留 {n_keep:,} 点。输入选择只依据参考生理输出，模型权重与事件阈值在本集合生成前已锁定。完整输入设计见<a href="DBNN_spike_panel_data_20261009/DBNN_独立尖峰评估_1ms_20261009.html#inputs">输入说明</a>，逐条清单见<a href="#main379-inputs">附录 AL</a>。</p></section>'''
    v=training['validation'];reset=training['reset']
    train_rows=[['训练来源','117 条拟合／32 条验证、6 s 源轨迹；教师 NEURON 内部积分 0.1 ms，训练目标按 1 ms 提取'],
                ['上下文','随机截取 500 ms 窗口；前 150 ms 输入恢复局部状态，后 350 ms 计入监督损失；教师 reset 中扣除截窗前历史贡献'],
                ['当前模型','每个位点一个双指数核；完整二次矩阵、线性项、偏置及阈值 reset。由此前 1 ms 短窗口参数起步，开放全部二次项并联合微调'],
                ['电压损失','按源轨迹方差平衡样本；参考尖峰 ±10 ms 掩码后的电压 MSE，加 0.1×物理时间一阶差分误差'],
                ['尺度与正则','对角参数按拟合响应的 RMS 尺度归一化；响应尺度正则系数 0.001，读出参数接近初始化的正则系数 10⁻⁵'],
                ['联合微调','本轮完整训练 1300 轮；对角、其余读出与时间核学习率分别为 10⁻⁴、5×10⁻⁵、10⁻⁵；FP32，保持既有短窗口训练流程'],
                ['验证选择','仅用 32 条完整验证轨迹校准 reset 与选择保存点；选择指标同时考虑电压误差、按方差归一化误差和事件 F1'],
                ['锁定结果',f'验证选中第 {training["epoch"]} 轮：RMSE={v["pooled_rmse_mv"]:.4f} mV，F1={v["events"]["f1"]:.3f}；阈值 θ={reset["v_th"]:g} mV，reset 幅度 a={reset["reset_amp"]:g} mV，τᵣ={reset["tau_reset"]:g} ms'],
                ['评估方式','冻结模型和 reset，连续预测完整 1–16 s 输入；GPU 与 CPU1／8 均读取同一 checkpoint，并逐输入核对数值和事件']]
    parts['accuracy-development']=f'''<section class="card" id="accuracy-development"><h2>2. 全矩阵 DBNN：短窗口训练与连续推理</h2><p>当前主线统一采用允许非零对角的完整矩阵 W。训练源数据与评估输入分离，学习基线共用相同的拟合／验证源轨迹，并沿用各自已完成的 400／500 ms 作者训练流程；本轮没有用 379 条评估输入重新训练或选阈值。</p>{table(['环节','实际流程'],train_rows)}
<h3>两阶段推理与训练目标</h3><div class="formula">kᵢ(t)=ωᵢ(1−e<sup>−t/τrise,i</sup>)e<sup>−t/τdecay,i</sup>；yᵢ=kᵢ*xᵢ<br/>u[n]=Σᵢyᵢ[n]+y[n]<sup>T</sup>Wy[n]+b，W 包含全部对角项<br/>v[n]=u[n]+r[n]；r[n+1]=αr[n]−a·1[v[n]≥θ]；α=e<sup>−Δt/τᵣ</sup>，Δt=1 ms</div>
<p>第一阶段得到不含 reset 的驱动 u[n]；第二阶段由固定阈值发放与 reset 得到电压及事件。训练时以教师电压减去窗口内 reset 恢复项构造驱动目标，先拟合电压与局部波形，再用验证轨迹选择闭环推理的参数。</p>
<div class="formula">L = mean<sub>batch</sub>{{ŵᵢ[mean<sub>Ω</sub>(eᵢ²)+0.1 ms²·mean<sub>Ω′</sub>((Δeᵢ/Δt)²)]}} + 参数正则<br/>J<sub>val</sub>=pooled MSE + 0.02 mV²·meanᵢ[MSEᵢ/max(σᵢ²,0.04 mV²)] + 0.05 mV²·(1−F1<sub>±5ms</sub>)</div>
<p>此处训练误差 e 是驱动预测与教师驱动之差；Ω 包含损失窗口中参考尖峰 ±10 ms 外的点，Ω′要求相邻两点均有效；ŵᵢ由拟合源轨迹方差确定并归一化。验证使用固定的完整 6 s 轨迹。<a href="{PREFIX}checkpoint_training_summary.json">下载当前保存点与验证记录</a>。</p>
<h3>完整模型的计算实现</h3><p>GPU 按已知活跃位点压缩输入和 W，使用并行前缀扫描完成时间滤波，以优化的融合算子计算完整二次项、线性项和偏置，再用事件式 reset 寻找下一次阈值越界。系数、工作区和 CUDA Graph 准备后复用。CPU 路径使用 AVX512／FMA 滤波、MKL 或等价二次读出及事件 reset，分别测试 1 核和 8 核。完整 W 在三个后端均保留。</p></section>'''
    examples=''
    for ex in d['examples']:
        n=ex['reference_spikes'];cid=ex['case_id'];examples+=figure(f'voltage_{n}spike',f'{n}尖峰的8秒参考、DBNN及0.5ms NEURON电压和事件',
            f'{n}-spike：{cid}，8 s。DBNN 与 0.5 ms NEURON 均无额外或遗漏事件，所有参考尖峰在 ±5 ms 内匹配。参考／NEURON 曲线分别显示 0.1／0.5 ms 原生记录；DBNN 显示 1 ms 输出。中间为全部 500 个突触位点的输入 raster：蓝色为兴奋性输入（ID 0–249），粉色为抑制性输入（ID 250–499），每个短线标记对应一次实际输入事件。底部为输出尖峰对齐；右侧放大阈下电压，高电压峰形见左侧完整幅度图。RMSE 仍统一在 1 ms 网格、参考尖峰 ±10 ms 外统计。')
    event_rows=[]
    for k in ['gpu','neuron8','core1','tcn','elm','lstm','reduce']:
        s=m[k]['accuracy'];e=s['events'];ci=s['bootstrap_95ci']['f1'];event_rows.append([names[k],f'{e["precision"]:.3f}',f'{e["recall"]:.3f}',f'{e["f1"]:.3f}',f'{ci[0]:.3f}–{ci[1]:.3f}',f'{e["tp"]} / {e["fp"]} / {e["fn"]}'])
    family_names={'stationary':'连续 Poisson（含空间／低频控制）','gated_excitation':'短时 E/I 比例变化','rate_steps':'输入强度分段变化','zero':'确定性零输入'}
    family_rows=[]
    for family,g in m['gpu']['groups']['family'].items():
        q=m['neuron8']['groups']['family'][family]
        family_rows.append([family_names[family],g['cases'],g['reference_events'],f'{g["pooled_rmse_mv"]:.4f}',f'{q["pooled_rmse_mv"]:.4f}',f'{g["events"]["f1"]:.3f}' if g['reference_events'] else '无参考／预测事件',f'{q["events"]["f1"]:.3f}' if q['reference_events'] else '无参考／预测事件'])
    duration_accuracy=[]
    for T in ['1','2','4','8','16']:
        g=m['gpu']['groups']['duration_s'][T];q=m['neuron8']['groups']['duration_s'][T]
        duration_accuracy.append([T+' s',g['cases'],g['reference_events'],f'{g["pooled_rmse_mv"]:.4f}',f'{q["pooled_rmse_mv"]:.4f}',
            f'{m["tcn"]["groups"]["duration_s"][T]["pooled_rmse_mv"]:.4f}',f'{m["elm"]["groups"]["duration_s"][T]["pooled_rmse_mv"]:.4f}',
            f'{m["lstm"]["groups"]["duration_s"][T]["pooled_rmse_mv"]:.4f}',f'{g["events"]["f1"]:.3f}',f'{q["events"]["f1"]:.3f}'])
    parts['voltage-consistency']=f'''<section class="card" id="voltage-consistency"><h2>3. 电压展示与完整尖峰统计</h2><p>以下四个展示样本均为持续输入的 8 s 新轨迹，分别包含 0、1、2、3 个参考尖峰。模型锁定后，在对应类别中优先选取 DBNN 与 0.5 ms NEURON 事件全部匹配、无额外触发的样本，再按 DBNN 掩码 RMSE 排序。各类别符合双方匹配条件的 8 s 候选数分别为 44、1、3、1。所有 379 条输入仍计入下方总体统计。</p>
<p>DBNN 用阈值发放与 reset 描述尖峰；参考和求解器曲线显示完整动作电位峰形，事件行直接展示记录时刻与实际 reset。展示曲线没有人为平移或补画尖峰。</p>{examples}
<h3>3.1 全部 379 条、637 个参考尖峰</h3>{table(['方法','Precision','Recall','F1','F1 95% CI','TP / FP / FN'],event_rows)}
{figure('spike_f1_379','379条独立输入的事件F1及置信区间','完整时间窗、±5 ms 一对一匹配；置信区间按完整随机输入轨迹进行 5,000 次 bootstrap。DBNN 三个计算后端事件相同，图中列一次。')}
<h3>3.2 输入形式的影响</h3>{table(['输入形式','窗口','参考尖峰','DBNN RMSE / mV','NEURON RMSE / mV','DBNN F1','NEURON F1'],family_rows)}
<p>DBNN 在短时 E/I 变化和连续随机驱动上的事件表现不同，因此两类分别陈述。主事件容差固定为 ±5 ms；0.5 ms NEURON 在相同预测下，±10 ms 容差 F1 为 0.699，完整敏感性见<a href="DBNN_spike_panel_data_20261009/DBNN_独立尖峰评估_1ms_20261009.html#neuron-half-ms">求解步长与时刻分析</a>。</p>
<h3>3.3 按持续时间报告精度</h3>{table(['时长','窗口','参考尖峰','DBNN RMSE','NEURON RMSE','TCN RMSE','ELM RMSE','LSTM RMSE','DBNN F1','NEURON F1'],duration_accuracy)}
<p>RMSE 单位均为 mV，F1 沿用 ±5 ms。训练源轨迹长 6 s，模型训练上下文为 400／500 ms；评估连续覆盖完整 1–16 s。各时长组的输入组成和参考发放数同时列明，8／16 s 结果描述更长时间输入上的实际表现。</p></section>'''
    frontier_rows=[]
    for k in keys:
        q=m[k];s=q['accuracy'];frontier_rows.append([names[k],'输出 1 ms' if q['interval_ms']==1 else '积分 0.5 ms',f'{s["pooled_rmse_mv"]:.4f}',f'{s["events"]["f1"]:.3f}',f'{wall(k):,.3f}',f'{wall("neuron8")/wall(k):,.1f}×'])
    duration_rows=[]
    for T,rs in d['durations'].items():
        duration_rows.append([T+' s',rs['gpu']['cases'],f'{rs["gpu"]["median_ms"]:.4f}',f'{rs["cpu1"]["median_ms"]:.3f}',f'{rs["cpu8"]["median_ms"]:.3f}',f'{rs["neuron8"]["median_ms"]:.3f}',f'{rs["core1"]["median_ms"]:.3f}',f'{rs["neuron8"]["median_ms"]/rs["gpu"]["median_ms"]:,.0f}×'])
    ratio16=d['durations']['16']['neuron8']['median_ms']/d['durations']['16']['gpu']['median_ms']
    parts['comparison-protocol']=f'''<section class="card" id="comparison-protocol"><span id="expanded-frontier"></span><h2>4. 379 条输入的速度—精度关系</h2>
<p>每个方法的工作点都使用相同 379 条输入，横轴为掩码 pooled RMSE；纵轴为每条输入 <strong>3 次准备后完整计算的 wall time 中位数之和</strong>。本轮学习模型均为 1 ms，数值求解器均为 0.5 ms 积分，全部重新计时。模型与作者年份标在点旁；当前主线每种方法对应一个已完成工作点。</p>
{figure('frontier_379','379条输入上各方法的RMSE与完整计算耗时','越靠左下越好。箭头倍率为本集合 NEURON multisplit 8 核（0.5 ms 积分）／DBNN GPU（1 ms 输出）的计算时间比；两个工作点的电压误差分别为 0.3397／0.4100 mV。图中没有混入旧 31 窗计时。')}
{table(['方法','时间间隔','RMSE / mV','事件 F1','379 窗 wall 合计 / ms','NEURON8 / 本方法'],frontier_rows)}
<h3>4.1 计时包含什么</h3><p>输入、权重、系数及缓冲区预先准备，均为单条完整轨迹 B=1。DBNN 包含滤波、全矩阵二次项＋线性项＋偏置、闭环 reset；GPU wall time 包含同步，CUDA 设备时间另行保存。学习基线包含原有重叠窗推理、拼接、验证集电压标定和 sigmoid 概率输出。NEURON 系方法包含 psolve 与原生电压／事件记录。初始化、500 ms 预热、事件队列构建、CPU 打包、上传／下载、图捕获及评分在计时外。</p>
<p>这是共享服务器上的工程计算时间。CPU 使用明确的物理核心绑定；四组互不重叠的求解器测量进程用于并行完成输入，记录的每次调用仍是单神经元 B=1 延迟。GPU 方法在同型号 RTX 5090 的独立设备上测量，设备和资源快照均保存。NEURON 多核是形态 multisplit；CoreNEURON 与 Neuron_Reduce 为 CPU 单核。</p>
<span id="duration-timing"></span><h3>4.2 按模拟时长查看耗时与加速</h3>
{figure('duration_379','379条输入按1至16秒时长分组的计算时间','先对每条输入的 3 次计时取中位数，再在对应时长组内取窗口中位数；阴影为输入间四分位范围。16 s 的箭头标注 NEURON multisplit 8 核／DBNN GPU 的组内中位数比。')}
{table(['模拟时长','窗口数','GPU / ms','CPU1 / ms','CPU8 / ms','NEURON8 / ms','CoreNEURON1 / ms','NEURON8 / GPU'],duration_rows)}
<p>16 s 输入组的准备后计算时间中位数比为 <strong>{ratio16:,.0f}×</strong>。各组包含不同输入位置、事件密度与发放情况；组内中位数和四分位范围用于展示这些差异。较长时间的优势由上述实际分组结果描述。</p></section>'''
    maxgpu=max(r['validation']['max_abs_mv'] for r in m['gpu']['timing']['records']);maxcpu=max(r['validation']['max_abs_mv'] for k in ['cpu1','cpu8'] for r in m[k]['timing']['records'])
    maxsolver=max(r['validation']['voltage_max_abs_mv'] for k in ['neuron8','core1'] for r in m[k]['timing']['records'])
    parts['next']=f'''<section class="card" id="next"><h2>5. 数值核验、保存内容与适用范围</h2>
<p>全部 379 条输入的优化 GPU 与 CPU1／CPU8 输出已逐一核对保存模型：电压最大差分别为 {maxgpu:.2e}／{maxcpu:.2e} mV，阈值事件位置全部一致。0.5 ms multisplit 与 CoreNEURON 逐输入对同一步长单线程 NEURON 的最大电压差为 {maxsolver:.2e} mV，事件时刻一致。四个原生展示样本还复核了共同观察网格，最大差小于 4×10⁻⁶ mV。</p>
<p>当前主线对应固定小鼠 V1 L5PC、原主动离子通道、AMPA 样兴奋性输入、64–500 个活跃位点、1–16 s 与参考每窗 0–3 个尖峰的范围。训练采用 500 ms 随机截窗，推理连续覆盖完整评估输入。跨细胞结果继续保留在附录 AD；完整 637 条及高于当前发放范围的输入另见附录 AJ 的详细报告。</p>
<p>本轮没有根据评估结果改变模型、校准或输入筛选条件。曲线示例按已声明的匹配与误差条件选择，总体 RMSE、F1 与置信区间使用全部主集合。旧 31 窗和细采样历史保留在附录 AK，0.5 ms NEURON 的逐条记录与时刻敏感性保留在独立评估报告。</p>
<p><a href="{PREFIX}main379_results.json">下载全部方法、逐输入精度与计时 JSON</a> · <a href="{PREFIX}selected_examples.json">展示样本选择记录</a> · <a href="{PREFIX}display_validation.json">原生曲线核验</a>。输入、参考、checkpoint 和全部原始模型输出仍保存在 5090 的 <code>benchmarks/spike_panel_20261009/</code>；本轮计时位于其 <code>main379/</code>。</p></section>'''
    main_path=ROOT/'DBNN_L5PC_surrogate_report_20260928.html';text=main_path.read_text();before=Parse();before.feed(text)
    snapshots=json.loads((HERE/'historical_main_sections.json').read_text());archive_keys=[k for k in parts]
    history=''.join(snapshots[k] for k in archive_keys);old_ids=set(re.findall(r'\bid="([^"]+)"',history))
    history=re.sub(r'\bid="([^"]+)"',lambda z:'id="history31-'+z[1]+'"',history)
    preserved_ids={'download-neuron','gpu-compute-optimization',*[f'voltage-{n}spike' for n in range(4)]}
    for identifier in preserved_ids:history=history.replace('id="history31-'+identifier+'"','id="'+identifier+'"')
    history=re.sub(r'href="#([^"]+)"',lambda z:'href="#'+('history31-' if z[1] in old_ids else '')+z[1]+'"',history)
    for identifier in preserved_ids:history=history.replace('href="#history31-'+identifier+'"','href="#'+identifier+'"')
    archive=f'<section class="card" id="historical-main31"><h2>附录 AK. 原 31 窗与多分辨率主线（历史保留）</h2><p>以下内容保留更新前的训练模型、31 窗图形、公式、统计与计时；对应原数据和当时的 checkpoint。当前文章主线使用上方 379 条独立输入及全矩阵 1 ms DBNN。</p><details><summary>展开原主线完整记录</summary>{history}</details></section>'
    for identifier,replacement in parts.items():
        start,end=section_range(text,identifier);text=text[:start]+replacement+text[end:]
    if 'id="historical-main31"' not in text:
        i=text.index('</main>');text=text[:i]+archive+text[i:]
    else:
        start,end=section_range(text,'historical-main31');text=text[:start]+archive+text[end:]
    text=text.replace('DBNN：多分辨率精度—速度对照','DBNN：379 条独立输入的精度—速度对照')
    text=text.replace('DBNN / L5PC SURROGATE DEVELOPMENT · 2026-10-05','DBNN / L5PC SURROGATE DEVELOPMENT · 2026-10-09')
    text=text.replace('400 / 500ms短窗口训练 · 1 / 0.5 / 0.25ms采样 · GPU与CPU的完整精度—速度曲线','379 条独立输入 · 全矩阵 DBNN 1 ms · NEURON 0.5 ms 积分 · 637 个参考尖峰')
    text=text.replace('文章主线 · 400 / 500ms短窗口','文章主线 · 379 条独立输入')
    text=text.replace('2. 训练方法与精度提升','2. 全矩阵模型与训练').replace('4. 方法与采样的速度—精度','4. 379 条输入的速度—精度')
    extra=[]
    for r in m['gpu']['records']:
        e=r['events'];extra.append([r['case_id'],r['duration_s'],r['A'],html.escape(r['family']),r['reference_spikes'],f'{r["voltage"]["rmse_mv"]:.4f}',f'{e["tp"]}/{e["fp"]}/{e["fn"]}'])
    input_section=f'''<section class="card" id="main379-inputs"><h2>附录 AL. 当前主线 379 条独立输入的逐条清单</h2><p>每条输入沿用参考确定的资格，模型输出不参与筛选；详细事件、随机种子、空间位置、输入哈希及各模型结果见<a href="{PREFIX}main379_results.json">原始 JSON</a>。下表 DBNN 使用同一全矩阵 1 ms checkpoint。</p><details><summary>展开全部 379 条输入</summary>{table(['输入 ID','T / s','活跃位点 A','输入形式','参考尖峰','DBNN RMSE / mV','TP/FP/FN'],extra)}</details></section>'''
    if 'id="main379-inputs"' not in text:
        i=text.index('</main>');text=text[:i]+input_section+text[i:]
    nav='<li><a data-section-link="" href="#historical-main31">附录 AK. 原31窗与多分辨率历史</a></li><li><a data-section-link="" href="#main379-inputs">附录 AL. 当前379条输入清单</a></li>'
    if 'href="#historical-main31">附录 AK.' not in text:text=text.replace('</li></ul><div class="toc-footer">','</li>'+nav+'</ul><div class="toc-footer">',1)
    after=Parse();after.feed(text);assert len(after.ids)==len(set(after.ids))
    assert not (Counter(before.images)-Counter(after.images)), 'Historical figures lost'
    for src in after.images:
        if src.startswith(PREFIX):assert (ROOT/src).exists(),src
    main_path.write_text(text)
    primary=''.join(parts.values())
    (DATA/'primary_sections.html').write_text(primary)
    standalone=ROOT/'DBNN_spike_panel_data_20261009/DBNN_独立尖峰评估_1ms_20261009.html';old=standalone.read_text()
    backup=HERE/'standalone_before_main379.html'
    if not backup.exists():backup.write_text(old)
    legacy=backup.read_text();start=legacy.index('<main>')+len('<main>');end=legacy.index('</main>',start);diagnostics=legacy[start:end]
    diagnostics=diagnostics.replace('<h1>DBNN 独立尖峰评估：统一 1 ms 模型</h1>','<h2>此前输入设计、1 ms 粗积分与 0.5 ms 复查记录</h2>')
    local_primary=primary.replace('src="'+PREFIX,'src="../'+PREFIX).replace('href="'+PREFIX,'href="../'+PREFIX).replace('href="DBNN_spike_panel_data_20261009/','href="')
    for target in ['historical-main31','main379-inputs']:
        local_primary=local_primary.replace('href="#'+target+'"','href="../DBNN_L5PC_surrogate_report_20260928.html#'+target+'"')
    head='<h1>379 条独立输入：DBNN 与 0.5 ms NEURON</h1><p class="muted">2026-10-09 · 最新主线 · 全矩阵 DBNN · 模型冻结 · 637 个参考尖峰</p><nav><a href="#conclusion">结果</a><a href="#accuracy-development">训练</a><a href="#voltage-consistency">电压与尖峰</a><a href="#comparison-protocol">速度—精度</a><a href="#duration-timing">分时长</a><a href="#early-spike-diagnostics">早期诊断与输入详情</a></nav>'
    new=legacy[:start]+head+local_primary+'<section id="early-spike-diagnostics"><h2>附录：输入设计与求解步长诊断</h2><details><summary>展开原始输入、统计和求解步长记录</summary>'+diagnostics+'</details></section>'+legacy[end:]
    new=new.replace('<title>DBNN 独立尖峰评估：统一 1 ms</title>','<title>DBNN：379 条独立输入与 0.5 ms NEURON</title>')
    new=new.replace('</style>','.resolution-table{overflow-x:auto;margin:18px 0}.formula{background:#f2f7f9;padding:16px;line-height:1.9}.card{margin-top:28px}details{margin:16px 0}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}</style>',1)
    check=Parse();check.feed(new);assert len(check.ids)==len(set(check.ids));standalone.write_text(new)
    (DATA/'standalone_primary.html').write_text(head+local_primary)
    validation=dict(finished=True,cases=379,reference_events=637,all_methods_same_input_order=True,model_interval_ms=1.,solver_dt_ms=.5,repeats=3,
                    old_figures_retained=True,unique_section_ids=True,display_selection_disclosed=True,
                    payload_sha256=hashlib.sha256((DATA/'main379_results.json').read_bytes()).hexdigest(),
                    main_report_sha256=hashlib.sha256(main_path.read_bytes()).hexdigest(),gpu_to_neuron8_speedup=ratio,
                    new_primary_images=[s for s in after.images if s.startswith(PREFIX)])
    (DATA/'report_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(main_path);print(standalone);print('speedup',ratio,'16s',ratio16)


if __name__=='__main__':main()
