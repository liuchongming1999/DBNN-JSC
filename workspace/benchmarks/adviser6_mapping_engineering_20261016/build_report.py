"""Standalone evidence report; main manuscript and old reports untouched."""
from pathlib import Path
import argparse,json,datetime,html,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();W=a.work;O=a.out;O.mkdir(parents=True,exist_ok=True)
 mapping=read(W/'morphology_mapping.json');audit=read(W/'protocol_audit.json');eng=read(W/'engineering/complete.json');rb=read(W/'engineering_rollback/complete.json');progress=read(W/'aligned/elm_1ms/progress.json',{});done=read(W/'aligned/elm_1ms/complete.json');ev=read(W/'elm_evaluation/complete.json')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9cabb7','axes.labelcolor':'#243c4b','text.color':'#243c4b','savefig.facecolor':'white'})
 imgs=[]
 if eng:
  fig,axes=plt.subplots(1,2 if rb else 1,figsize=(12 if rb else 7,4.9),squeeze=False);ax=axes[0,0];keys=list(eng['summary']);v=[eng['summary'][k]['wall_ms'] for k in keys]
  labels=['Dense, sequential','+ active-site packing','+ parallel filter','+ fused drive','+ event reset','+ CUDA Graph']
  ax.barh(np.arange(6),v,color=['#a5b3bd']*4+['#6794b0','#008b7c'],height=.65);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,max(v)*1.2);ax.set_xlabel('Sum of per-input median times (ms)');ax.set_title('Cumulative implementation changes',loc='left',weight='bold');ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
  for j,x in enumerate(v):ax.text(x+max(v)*.018,j,f'{x:.1f}',va='center')
  if rb:
   ax=axes[0,1];vals=list(rb['summary'].values());u=[x['wall_ms'] for x in vals];labels=['No active-site packing','Sequential filter','Unfused drive','Sequential reset','Complete implementation','No CUDA Graph']
   ax.barh(range(6),u,color=['#7897ad']*4+['#008b7c','#7897ad'],height=.65);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,max(u)*1.2);ax.set_xlabel('Sum of per-input median times (ms)');ax.set_title('Remove one component from full version',loc='left',weight='bold');ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
   for j,x in enumerate(u):ax.text(x+max(u)*.018,j,f'{x:.1f}',va='center')
  fig.tight_layout(w_pad=3);fig.savefig(O/'engineering_ablation.png',dpi=180);fig.savefig(O/'engineering_ablation.svg');plt.close(fig);imgs.append('<figure><img src="engineering_ablation.png"><figcaption>379条输入；每条预热5次、计时30次，累加各输入的中位数。左图的逐步收益依赖前序优化；右图每次仅关闭完整实现中的一项。图中时间均为预处理后完整推理的wall time。</figcaption></figure>')
 p=progress;steps=p.get('steps',0);rate=p.get('run_elapsed_s',p.get('elapsed_s',0))/max(p.get('run_updates',steps),1);hours=(350000-steps)*rate/3600
 now=datetime.datetime.now().astimezone();eta=now+datetime.timedelta(hours=hours)
 status='完整训练与独立评估已完成' if ev else ('完整训练已完成，正在独立评估' if done else f'训练进行中：{steps:,} / 350,000 次更新（{100*steps/350000:.2f}%）')
 timing=''; parity='';conclusion=''
 if eng:
  s=eng['summary'];base=s['G0_dense_serial']['wall_ms'];fast=s['G5_cuda_graph']['wall_ms'];parity=f'全部6种实现均通过379/379条输出一致性检查；最大电压差为{max(v["max_abs_mv"] for v in s.values()):.3g} mV，所有尖峰事件逐条完全相同。共同RMSE约{s["G5_cuda_graph"]["rmse_mv"]:.6f} mV，F1={s["G5_cuda_graph"]["events"]["f1"]:.6f}。'
  timing='<table><thead><tr><th>逐步实现</th><th>总wall time / ms</th><th>总device time / ms</th><th>输出一致性</th></tr></thead><tbody>'+''.join(f'<tr><td>{k}</td><td>{v["wall_ms"]:.3f}</td><td>{v["device_ms"]:.3f}</td><td>{v["parity_passed"]}/{v["cases"]}</td></tr>' for k,v in s.items())+'</tbody></table>'
  conclusion=f'完整工程实现使同一个DBNN的总计算时间从{base:.2f} ms降到{fast:.2f} ms，为{base/fast:.2f}倍加速。这是实现方式之间的加速，不能直接解释为DBNN相对NEURON或其他学习模型的加速。单独加入并行滤波时，左图总时间有所上升，应结合右图判断其在最终组合中的作用，不能假定每个步骤都有独立正收益。'
 if rb:
  timing+='<h3>从完整实现每次关闭一项</h3><table><thead><tr><th>实现</th><th>wall time / ms</th><th>相对完整实现</th><th>输出一致性</th></tr></thead><tbody>'+''.join(f'<tr><td>{k}</td><td>{v["wall_ms"]:.3f}</td><td>{v["wall_ms"]/rb["summary"]["R4_complete_graph"]["wall_ms"]:.2f}×</td><td>{v["parity_passed"]}/379</td></tr>' for k,v in rb['summary'].items())+'</tbody></table>'
 elmtable='本次完整预算尚未完成，暂不报告新模型的测试精度或速度。训练完成后，程序先在原32条验证轨迹上锁定事件阈值，再对原379条测试输入计算RMSE、F1和计时；测试结果不参与模型或映射选择。'
 if ev:
  elmtable='<table><tr><th>模型</th><th>RMSE / mV</th><th>F1</th><th>prepared time / ms</th></tr>'+''.join(f'<tr><td>{k}</td><td>{v["rmse_mv"]:.6f}</td><td>{v["events"]["f1"]:.6f}</td><td>{v.get("prepared_wall_ms","见工程消融")}</td></tr>' for k,v in ev['summary'].items())+'</table>'
 text=f'''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>导师第六条：Branch-ELM映射修正与DBNN工程消融</title><style>body{{margin:0;background:#f3f6f8;color:#243c4b;font:16px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1120px;margin:36px auto;padding:34px 44px;background:white;border-radius:12px}}h1{{font-size:28px;line-height:1.4}}h2{{margin-top:38px;border-bottom:1px solid #dce4e9;font-size:23px}}h3{{font-size:18px}}.callout{{padding:16px 20px;border-left:4px solid #008b7c;background:#edf8f5}}.muted{{color:#6f8392;font-size:14px}}table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #dce4e9;text-align:left}}th{{background:#edf3f6}}img{{width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:14px;color:#647b8a}}code{{overflow-wrap:anywhere}}.small{{font-size:14px}}@media(max-width:700px){{main{{margin:0;padding:20px}}table{{display:block;overflow:auto}}}}</style><main>
<h1>导师第六条：Branch-ELM输入映射与DBNN工程实现</h1><p class="muted">更新于 {now.strftime('%Y-%m-%d %H:%M:%S %Z')}。本文件独立于论文和此前的实验报告。</p>
<div class="callout"><b>{status}</b><br>DBNN检查点、模型结构、训练和主文稿均保持不变。工程消融{'已完成两种设计' if rb else '已完成逐步设计' if eng else '进行中'}；Branch-ELM只更换形态映射，按默认流程重新训练一次。</div>
<h2>1. 固定的实验边界</h2><p>本次使用原117条拟合轨迹、32条验证轨迹，以及原379条主线测试输入；训练/验证与测试的输入哈希没有重叠。测试仍以0.0125 ms NEURON解为共同参考，在1 ms输出网格上计算去除参考尖峰±10 ms后的汇总RMSE；尖峰按±5 ms逐一匹配，包含统一的1e−6 ms边界浮点容差。379条输入对应680个参考尖峰。未筛选637条及压力测试仍保留在原独立档案中。</p>
<h2>2. Branch-ELM：修正什么，保留什么</h2><p>旧映射将随机的500个位点编号先做E/I交错，再取重叠编号窗口。随机编号不代表树突邻近关系，45个虚拟分支中最后3个为空。新映射直接使用细胞形态中的电缆路径距离：第一个锚点取离胞体最近的位点；接下来优先覆盖尚未覆盖的位点，再按距离分散锚点；每个分支取锚点附近最近的100个位点。所有选择只使用形态，不使用电压或测试结果，500个位点全部得到覆盖。</p>
<table><tr><th>项目</th><th>旧映射</th><th>新映射</th></tr><tr><td>分支数 × 每分支输入槽位</td><td>45 × 100</td><td>45 × 100</td></tr><tr><td>非空分支 / 有效槽位</td><td>42 / 3800</td><td>45 / 4500</td></tr><tr><td>分支内位点对的平均路径距离</td><td>{audit['old_geometry']['mean_pair_distance_um']:.1f} μm</td><td>{audit['corrected_geometry']['mean_pair_distance_um']:.1f} μm</td></tr><tr><td>训练参数量</td><td>8002</td><td>8002</td></tr></table>
<p>本次修正同时恢复了原来空置的分支。因此如果精度改善，能说明这一形态映射适配更合适，但不能把改善单独归因于空间次序。模型公式、记忆维度20、时间常数范围1–150 ms、随机种子930291、批量8、Adam、逐更新余弦学习率、损失与模型选择标准均保留。训练35轮，每轮10000次更新；窗口500 ms，采用作者公开代码默认的全窗损失。推理保留原150 ms重叠。使用原作者执行路径，不额外调参或优化训练流程。</p>
<p>默认设置对照审计通过：除映射及其记录文字外，新旧有效配置相同。这是固定种子下的单次修正对照。</p>
<p class="callout">{status}。{'' if done else f'当前运行速度粗估还需{hours:.1f}小时，约{eta.strftime("%m月%d日 %H:%M")}结束训练；实际耗时受共享服务器负载和验证开销影响。'}</p><p>{elmtable}</p>
<h2>3. DBNN：相同模型，改变工程实现</h2><p>{conclusion}</p>{''.join(imgs)}{timing}<p>{parity}</p>
<p>逐步实现依次加入有效位点压缩、并行前缀滤波、融合双线性/线性/偏置计算、事件式reset及CUDA Graph。额外对照每次仅关闭完整实现的一项。所有版本保留完整训练矩阵和非零对角、同一阈值、同一输入顺序与FP32 IEEE精度；没有重新训练，也没有更改reset规则。</p>
<p>计时边界是预处理后单轨迹完整推理，包含CPU发射与GPU同步；位点整理、数据传输、模型准备和图捕获在计时之外。device time另用CUDA事件记录，不能与wall time混为一谈。所有379条输入全部保留，不能依据输出一致性筛掉个别困难输入。两组计时在同一张5090上完成，Branch-ELM在另一张5090训练；服务器为共享环境。</p>
<h2>4. 留存与后续处理</h2><p>训练完成后自动执行验证集阈值标定、379条主线测试、新旧Branch-ELM配对计时，并保存逐输入电压、概率、尖峰、指标和日志。本文报告中的新ELM结果只在这些步骤全部完成后显示。当前不据部分训练结果改动论文。</p>
<p class="small">远程实验目录：<code>{html.escape(str(W))}</code><br>DBNN检查点SHA-256：<code>{audit['dbnn_checkpoint_sha256']}</code><br>映射文件、形态距离矩阵、默认配置差异审计、训练日志、逐输入消融输出及计时均已保留。公开实现：<a href="https://github.com/AaronSpieler/elmneuron/tree/52e68a6d39523ac6613a586699b116e8e606dda3">作者固定版本代码</a>。</p></main></html>'''
 (O/'DBNN_映射修正与工程消融_20261016.html').write_text(text)
 print(O/'DBNN_映射修正与工程消融_20261016.html')
if __name__=='__main__':main()
