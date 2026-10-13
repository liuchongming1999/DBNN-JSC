# -*- coding: utf-8 -*-
"""Detailed report and static scientific figures for the unified 1-ms panel."""
import argparse
import html
from pathlib import Path
import shutil
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_panel import read,write,sha


LABELS={'dbnn':'DBNN · full matrix','neuron8':'NEURON · multisplit 8 cores','core1':'CoreNEURON · CPU 1 core',
        'reduce1':'Neuron_Reduce','tcn_author_default':'TCN · author recipe','elm_author_default':'Branch-ELM · author recipe',
        'lstm_common150':'LSTM','tcn_common150':'TCN · common 150-ms burn','elm_common150':'Branch-ELM · common 150-ms burn','neuron1':'NEURON · CPU 1 core'}
CHINESE={'dbnn':'DBNN（全矩阵）','neuron8':'NEURON multisplit，8 核','core1':'CoreNEURON CPU，1 核','reduce1':'Neuron_Reduce',
         'tcn_author_default':'TCN（作者默认训练）','elm_author_default':'Branch-ELM（作者默认训练）','lstm_common150':'LSTM',
         'tcn_common150':'TCN（共同 150 ms 缓冲对照）','elm_common150':'Branch-ELM（共同 150 ms 缓冲对照）','neuron1':'NEURON CPU，1 核'}
COLORS={'dbnn':'#008577','neuron8':'#d58a28','core1':'#b85372','reduce1':'#aa7134','tcn_author_default':'#ce5038',
        'elm_author_default':'#58677b','lstm_common150':'#258bad','tcn_common150':'#df8069','elm_common150':'#8593a5','neuron1':'#dcaa6a'}
MAIN=['dbnn','neuron8','core1','reduce1','tcn_author_default','elm_author_default','lstm_common150']


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();work=a.work.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    d=read(work/'comparison.json');assert d['finished'];models=d['models'];base=models['dbnn'];low=base['low_density'];all_=base['all_generated']
    audit_path=work/'dt_audit/report_fragment.html'
    audit_fragment=audit_path.read_text() if audit_path.exists() else ''
    half_path=work/'methods/neuron_dt0.5/report_fragment.html'
    half_fragment=half_path.read_text() if half_path.exists() else ''
    if half_fragment:
        shutil.copy2(work/'methods/neuron_dt0.5/complete.json',out/'neuron_dt0.5_complete.json')
    if audit_fragment:
        shutil.copy2(work/'dt_audit/solver_dt_audit.json',out/'solver_dt_audit.json')
        for ext in ['png','svg']:
            shutil.copy2(work/f'dt_audit/figures/solver_dt_audit.{ext}',fig/f'solver_dt_audit.{ext}')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263d4b',
                         'text.color':'#263d4b','axes.edgecolor':'#a7b8c2','xtick.color':'#516b7a','ytick.color':'#263d4b','figure.facecolor':'white','savefig.facecolor':'white'})
    f,ax=plt.subplots(figsize=(12,5.7));y=np.arange(len(MAIN))[::-1]
    for yy,k in zip(y,MAIN):
        s=models[k]['low_density'];v=s['events']['f1'];ci=s['bootstrap_95ci']['f1'] or [v,v]
        ax.barh(yy,v,height=.52,color=COLORS[k],alpha=.95)
        ax.errorbar(v,yy,xerr=np.array([[max(0,v-ci[0])],[max(0,ci[1]-v)]]),fmt='none',ecolor='#344e5d',capsize=4,lw=1.5)
        ax.text(min(.98,max(v,ci[1])+.025),yy,f'{v:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]',va='center',fontsize=10)
    ax.set_yticks(y,[LABELS[k] for k in MAIN],fontweight='bold');ax.set_xlim(0,1.18)
    ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('Event F1 · 95% confidence interval over independent trajectories')
    ax.grid(axis='x',color='#e4ecef',lw=.8);ax.set_axisbelow(True)
    f.suptitle('Spike prediction on independent inputs',x=.04,ha='left',fontweight='bold',fontsize=17)
    f.text(.04,.90,f'{low["cases"]} low-spike windows · {low["reference_events"]} reference events · 1-ms models · 0.1-ms reference · ±5-ms matching',fontsize=10,color='#617b8a')
    f.subplots_adjust(left=.29,right=.98,top=.82,bottom=.14)
    f.savefig(fig/'event_comparison_1ms.png',dpi=190);f.savefig(fig/'event_comparison_1ms.svg');plt.close(f)
    families=['stationary','rate_steps','gated_excitation'];names=['Stationary Poisson','Rate steps','Brief E/I change'];groups=base['groups']['family']
    f,axs=plt.subplots(1,2,figsize=(12,4.5));vals=[];cis=[]
    for family in families:
        s=groups[family];vals.append(s['events']['f1']);cis.append(s['bootstrap_95ci']['f1'])
    yy=np.arange(3);axs[0].bar(yy,vals,color=['#5f899e','#79aab4','#008577'],width=.58)
    for j,(v,ci,fam) in enumerate(zip(vals,cis,families)):
        axs[0].errorbar(j,v,yerr=np.array([[max(0,v-ci[0])],[max(0,ci[1]-v)]]),fmt='none',color='#344e5d',capsize=4)
        axs[0].text(j,max(v,ci[1])+.045,f'{v:.3f}\nn={groups[fam]["reference_events"]} spikes',ha='center',va='bottom',fontsize=10)
    axs[0].set_xticks(yy,names);axs[0].set_ylim(0,1.14);axs[0].set_ylabel('DBNN event F1');axs[0].set_title('Input form matters',fontweight='bold')
    for k in ['dbnn','tcn_author_default','elm_author_default','lstm_common150']:
        s=models[k]['tolerance_sensitivity_ms'];xx=[1,2,5,10,20];vv=[s[str(t)]['f1'] for t in xx]
        axs[1].plot(xx,vv,'o-',color=COLORS[k],label=LABELS[k],lw=2)
    axs[1].axvline(5,color='#8799a4',ls='--',lw=1);axs[1].set_ylim(0,1.02);axs[1].set_xticks([1,5,10,20]);axs[1].set_xlabel('Matching tolerance (ms)');axs[1].set_ylabel('Event F1')
    axs[1].set_title('Fixed predictions, different timing tolerance',fontweight='bold');axs[1].legend(fontsize=8,frameon=False,loc='lower right')
    for ax in axs:ax.grid(axis='y',color='#e6edef');ax.set_axisbelow(True)
    f.tight_layout();f.savefig(fig/'dbnn_input_and_timing.png',dpi=190);f.savefig(fig/'dbnn_input_and_timing.svg');plt.close(f)

    def fmt(v,n=3):return '—' if v is None else f'{v:.{n}f}'
    def row(k,pool='low_density'):
        s=models[k][pool];ev=s['events'];ci=s['bootstrap_95ci']['f1'];ci_text='—' if ci is None else f'{ci[0]:.3f}–{ci[1]:.3f}'
        return f'<tr><th>{CHINESE[k]}</th><td>{s["pooled_rmse_mv"]:.4f}</td><td>{fmt(ev["precision"] if s["predicted_events"] else None)}</td><td>{ev["recall"]:.3f}</td><td>{ev["f1"]:.3f}</td><td>{ci_text}</td><td>{ev["tp"]} / {ev["fp"]} / {ev["fn"]}</td></tr>'
    header='<tr><th>模型</th><th>RMSE / mV</th><th>Precision</th><th>Recall</th><th>F1</th><th>F1 95% CI</th><th>TP / FP / FN</th></tr>'
    def table(keys,pool='low_density'):return '<div class="scroll"><table><thead>'+header+'</thead><tbody>'+''.join(row(k,pool) for k in keys)+'</tbody></table></div>'
    distribution={str(n):sum(r['reference_spikes']==n for r in base['records']) for n in range(max(r['reference_spikes'] for r in base['records'])+1)}
    protocol_rows=[]
    for k in ['tcn_author_default','elm_author_default','lstm_common150','tcn_common150','elm_common150']:
        pp=models[k]['protocol'];cfg=pp['training_config']
        protocol_rows.append(f'<tr><th>{CHINESE[k]}</th><td>{cfg["window_ms"]}</td><td>{cfg["burn_in_loss_ms"]}</td><td>{pp["threshold"]:.9g}</td><td>{html.escape(pp["checkpoint_sha256"][:14])}…</td></tr>')
    family_rows=[]
    for fam in families:
        s=groups[fam];e=s['events'];ci=s['bootstrap_95ci']['f1'];name=dict(zip(families,['连续 Poisson','25/50 Hz 分段变化','40 ms E/I 比例变化']))[fam]
        family_rows.append(f'<tr><th>{name}</th><td>{s["cases"]}</td><td>{s["reference_events"]}</td><td>{s["pooled_rmse_mv"]:.4f}</td><td>{e["f1"]:.3f}</td><td>{ci[0]:.3f}–{ci[1]:.3f}</td></tr>')
    duration_rows=[]
    for T,s in base['groups']['duration_s'].items():
        duration_rows.append(f'<tr><th>{T} s</th><td>{s["cases"]}</td><td>{s["reference_events"]}</td><td>{s["pooled_rmse_mv"]:.4f}</td><td>{s["events"]["f1"]:.3f}</td></tr>')
    new_ci=low['bootstrap_95ci']['f1'];tol=base['tolerance_sensitivity_ms'];par=d['reference_parity']
    content=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DBNN 独立尖峰评估：统一 1 ms</title>
<style>body{{margin:0;background:#f5f8fa;color:#243d4c;font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1120px;margin:32px auto;background:white;padding:32px 40px;border-radius:14px}}h1{{font-size:29px;line-height:1.4;margin:0}}h2{{font-size:22px;margin-top:34px}}h3{{font-size:18px}}a{{color:#087f77}}.muted{{font-size:14px;color:#66808f}}.lead{{background:#eaf5f1;border-left:4px solid #008577;padding:15px 20px;border-radius:6px}}nav a{{margin-right:16px}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{padding:9px 11px;border-bottom:1px solid #dce7eb;text-align:right}}th:first-child{{text-align:left;font-weight:500}}thead{{background:#f2f7f9}}.scroll{{overflow-x:auto}}img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:14px;color:#657e8c}}code{{font-size:13px;overflow-wrap:anywhere}}@media(max-width:700px){{main{{margin:0;padding:24px 17px}}h1{{font-size:24px}}}}</style></head><body><main>
<h1>DBNN 独立尖峰评估：统一 1 ms 模型</h1><p class="muted">2026-10-09 · 在 5090 完成 · 固定小鼠 V1 L5PC、形态和主动通道 · 新生成输入、冻结模型与阈值</p>
<nav><a href="#result">结果</a><a href="#inputs">输入设计</a><a href="#metrics">统计方法</a><a href="#strata">分层结果</a><a href="#controls">训练流程对照</a><a href="#files">原始文件</a></nav>
<p class="lead">新评估包含 <strong>{d['input_cases']} 条输入</strong>，共 <strong>{all_['reference_events']} 个参考尖峰</strong>。按此前“每窗最多 3 个尖峰、最小间隔 50 ms”的范围，主统计使用 <strong>{low['cases']} 条输入、{low['reference_events']} 个尖峰</strong>。全矩阵 DBNN 的 F1 为 <strong>{low['events']['f1']:.3f}</strong>，按独立输入轨迹计算的 95% 置信区间为 <strong>{new_ci[0]:.3f}–{new_ci[1]:.3f}</strong>。</p>
<section id="result"><h2>1. 初始 1 ms 设置的比较（保留对照）</h2><p>参考模型统一使用 NEURON dt = 0.1 ms；此表学习模型的输入／输出间隔为 1 ms，求解器积分步长为 1 ms。学习模型沿用已保存的 1 ms checkpoint，阈值和电压标定也直接沿用原验证集结果。DBNN 使用最新全二次矩阵模型，对角项可训练；本轮没有新增训练或零对角实验。</p>
{table(MAIN)}<figure><img src="figures/event_comparison_1ms.png" alt="统一1ms模型的事件F1及按独立输入轨迹计算的95%置信区间"><figcaption>统一主统计：{low['cases']} 窗、{low['reference_events']} 个参考尖峰，完整窗口评分、±5 ms 一对一匹配。置信区间按输入轨迹整体重采样。</figcaption></figure>
<p>原有 31 窗中的 F1 = 0.90 基于 10 次尖峰计数，其中同一原始输入的一个尖峰在 4、8、16 s 窗口重复计入。新输入全部重新采样；新结果更充分地描述了这一细胞在当前输入集合中的事件表现。零输入对照单独固定，不作为新增独立随机样本重复重采样。</p><p>这里 NEURON／CoreNEURON 使用积分步长 dt = 1 ms，输出间隔也为 1 ms；学习模型的 1 ms 则是输入／输出时间分辨率。主集合中 NEURON 实际检测到 17 次事件，但在 ±5 ms 内没有匹配到参考事件（TP = 0、FP = 17、FN = 637），所以 F1 = 0；完整集合检测到 18 次事件。此前将 F1 = 0 表述为“未检测到尖峰”已纠正。较小积分步长的复查见下方。</p></section>
{half_fragment}
{audit_fragment}
<section id="inputs"><h2>2. 怎样增加尖峰而保持输入多样性</h2><p>先用 28 条独立 pilot 输入，仅根据 NEURON 的放电数和间隔选择输入方案。40 ms 的短时 E/I 比例变化在四条 pilot 中产生 2–3 个相隔数秒的尖峰；更长窗口容易超出此前设定的低尖峰范围。因此正式输入使用 40 ms，pilot 与正式输入随机种子完全不同，也未参考 DBNN 预测。</p>
<div class="scroll"><table><thead><tr><th>输入集合</th><th>数量</th><th>具体形式</th></tr></thead><tbody>
<tr><th>自然连续 Poisson</th><td>512</td><td>500 位点；E/I 均为 50 Hz；T=1/2/4/8/16 s，分别 32/32/64/256/128 个新随机种子</td></tr>
<tr><th>短时 E/I 比例变化</th><td>48</td><td>T=4/8/16 s；背景 E/I 25 Hz；三个相隔至少 1 s 的 40 ms 窗口内 E=50 Hz、I=10 Hz</td></tr>
<tr><th>输入强度分段变化</th><td>32</td><td>T=2/4/8/16 s；每隔 1 s 在 25 与 50 Hz 间切换，E/I 同步变化</td></tr>
<tr><th>位点／空间位置对照</th><td>24</td><td>A=64/128/256；随机、分层、近端、远端，各两个新随机种子；8 s、50 Hz</td></tr>
<tr><th>低强度对照</th><td>16</td><td>500 位点、8 s，10/25 Hz 各八个新随机种子</td></tr><tr><th>零输入对照</th><td>5</td><td>T=1/2/4/8/16 s，作为确定性控制</td></tr></tbody></table></div>
<p>仍使用同一组 500 个实际突触点位（250 E、250 I），不改变突触权重、动力学、形态或主动离子通道。事件按既有流程在 1 ms 网格独立采样。空间近端／远端在这次按胞体到突触的实际路径距离划分，完整位置表已保存。</p>
<p>完整集合的参考放电数分布：<code>{html.escape(json.dumps(distribution,ensure_ascii=False))}</code>。主统计的资格仅由参考轨迹确定；超过每窗 3 个或间隔小于 50 ms 的轨迹也保留，下面给出完整集合的结果。</p>
<h3>不筛选参考放电数的完整 637 条输入</h3>{table(MAIN,'all_generated')}</section>
<section id="metrics"><h2>3. 怎样评分与计算置信区间</h2><p>胞体真实尖峰由参考 NEURON 的 −20 mV 向上穿越检测器记录。DBNN 事件由其固定阈值和 reset 产生；TCN、Branch-ELM、LSTM 使用其固定验证阈值的概率峰值。统一采用 ±5 ms 一对一匹配：每个参考或预测事件最多配对一次，累计 TP、FP、FN 后计算 precision、recall 与 F1。</p>
<p>电压误差仍统一剔除参考尖峰前后 ±10 ms，在共同 1 ms 观察网格汇总。零参考尖峰窗口也计入额外触发，避免只考察有尖峰的窗口。DBNN 在主集合的额外事件率为 {low['false_positives_per_minute']:.3f} 次/分钟。</p>
<p>置信区间使用 5,000 次轨迹级 bootstrap，每次抽取完整输入轨迹，并重新累计事件计数。相同轨迹中的多个尖峰一起抽取；五个确定性零输入对照各固定计入一次。区间描述固定细胞、固定模型在这一输入分布上的变化。</p>
<p>新的 NEURON 程序重放原有含尖峰输入 case_0062，电压最大差为 {par['voltage_max_gap_mv']:.1f} mV，尖峰时刻最大差为 {par['event_time_max_gap_ms']:.1f} ms。CoreNEURON 与 multisplit 输出另外逐条核对同一步长的 NEURON 单线程结果。</p></section>
<section id="strata"><h2>4. 输入形式与时间精度</h2><div class="scroll"><table><thead><tr><th>输入类型</th><th>窗口数</th><th>参考尖峰</th><th>DBNN RMSE / mV</th><th>F1</th><th>95% CI</th></tr></thead><tbody>{''.join(family_rows)}</tbody></table></div>
<p>短时 E/I 比例变化中，尖峰更集中在明确的刺激窗口，DBNN 的事件表现较高；连续随机驱动中，接近阈值的自然涨落带来更明显的时刻与事件差异。以上是本次固定模型的分层观测。</p>
<figure><img src="figures/dbnn_input_and_timing.png" alt="DBNN按输入类型分层的F1和固定预测下容差敏感性"><figcaption>左：各输入类型的主统计。右：预测保持不变，分别用 1/2/5/10/20 ms 容差评分；主指标始终为 5 ms。</figcaption></figure>
<p>DBNN 在 5/10/20 ms 容差下的 F1 分别为 {tol['5']['f1']:.3f}、{tol['10']['f1']:.3f}、{tol['20']['f1']:.3f}。这组补充评分帮助区分严格时刻匹配和更宽时间范围内的事件对应，不用于重新选模型或阈值。</p>
<div class="scroll"><table><thead><tr><th>模拟时长</th><th>主统计窗口</th><th>参考尖峰</th><th>DBNN RMSE / mV</th><th>F1</th></tr></thead><tbody>{''.join(duration_rows)}</tbody></table></div></section>
<section id="controls"><h2>5. 已有训练流程对照</h2><p>主表采用已经完成作者默认训练流程的 TCN、Branch-ELM，并保留之前明确要求的共同 150 ms 训练缓冲对照。两套 checkpoint 均在本次输入生成前保存，使用相同源训练／验证数据，未因新输入上的表现重新训练、选阈值或选择其中一版。</p>
{table(['tcn_author_default','tcn_common150','elm_author_default','elm_common150','lstm_common150'])}
<div class="scroll"><table><thead><tr><th>模型流程</th><th>训练窗 / ms</th><th>无损失缓冲 / ms</th><th>固定事件阈值</th><th>checkpoint SHA 前缀</th></tr></thead><tbody>{''.join(protocol_rows)}</tbody></table></div>
<p>所有学习模型的适配程序都先重放旧输入并核对已保存电压与事件；其 checkpoint、标定、事件阈值和来源记录可在原始 JSON 中核查。NEURON／CoreNEURON 此次按输入分片并行完成单次精度检查，计时处于并发评估环境；本报告聚焦精度和事件统计，论文用时比较继续采用明确独立计时流程。</p></section>
<section id="files"><h2>6. 文件与复现</h2><p><a href="comparison.json">完整逐输入比较 JSON</a> · <a href="manifest.json">全部 637 条输入清单</a> · <a href="site_metadata.json">500 个物理位置</a> · <a href="pilot_design_decision.json">仅按参考模型选择输入方案</a> · <a href="reference_parity.json">参考程序复核</a> · <a href="model_definition.json">全矩阵 DBNN 定义</a></p>
<p>输入 raster、全部参考电压、每个模型的电压／概率／预测事件和原始 NEURON 输出均保存在 5090：<code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/spike_panel_20261009/</code>。各条输入的随机种子、ID、事件数据哈希和时长均在清单中；各模型输出可用于后续分析，不需要重新训练。</p><p class="muted">本次扩充单独构成事件统计证据；原有 31 窗继续保留为覆盖性和计时面板。论文的 F1 小节可使用本次独立输入集，并分别列明自然随机驱动与结构化刺激结果。</p></section></main></body></html>'''
    if audit_fragment:
        content=content.replace('<a href="#inputs">输入设计</a>','<a href="#solver-step-audit">求解步长复查</a><a href="#inputs">输入设计</a>')
    if half_fragment:
        content=content.replace('<a href="#solver-step-audit">','<a href="#neuron-half-ms">NEURON 0.5 ms</a><a href="#solver-step-audit">',1)
    primary_path=work/'main379/standalone_primary.html'
    if primary_path.exists():
        start=content.index('<main>')+len('<main>');end=content.index('</main>',start)
        diagnostics=content[start:end].replace('<h1>DBNN 独立尖峰评估：统一 1 ms 模型</h1>','<h2>此前输入设计、1 ms 粗积分与 0.5 ms 复查记录</h2>')
        content=content[:start]+primary_path.read_text()+'<section id="early-spike-diagnostics"><h2>附录：输入设计与求解步长诊断</h2><details><summary>展开原始输入、统计和求解步长记录</summary>'+diagnostics+'</details></section>'+content[end:]
        content=content.replace('<title>DBNN 独立尖峰评估：统一 1 ms</title>','<title>DBNN：379 条独立输入与 0.5 ms NEURON</title>')
        content=content.replace('</style>','.resolution-table{overflow-x:auto;margin:18px 0}.formula{background:#f2f7f9;padding:16px;line-height:1.9}.card{margin-top:28px}details{margin:16px 0}summary{cursor:pointer;font-weight:600}</style>',1)
    path=out/'DBNN_独立尖峰评估_1ms_20261009.html';path.write_text(content)
    for source,name in [(work/'comparison.json','comparison.json'),(work/'confirmatory/manifest.json','manifest.json'),
                        (work/'site_metadata.json','site_metadata.json'),(work/'pilot_design_decision.json','pilot_design_decision.json'),
                        (work/'reference_parity.json','reference_parity.json'),(work/'model_definition.json','model_definition.json')]:
        if source.exists():shutil.copy2(source,out/name)
    write(out/'report_validation.json',dict(finished=True,report=str(path),input_cases=d['input_cases'],main_cases=low['cases'],
            reference_events_all=all_['reference_events'],reference_events_main=low['reference_events'],methods=len(models),all_comparison_dt_ms=1.,
            comparison_sha256=sha(work/'comparison.json'),additional_neuron_solver_dt_ms=.5 if half_fragment else None,
            additional_neuron_primary_cases=379 if half_fragment else None,
            output_interval_ms=1.,figures=['event_comparison_1ms.png','dbnn_input_and_timing.png']))
    print(path,flush=True)


if __name__=='__main__':main()
