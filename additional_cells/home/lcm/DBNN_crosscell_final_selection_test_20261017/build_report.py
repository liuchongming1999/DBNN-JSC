"""Independent evaluation dashboard; no manuscript edits."""
from pathlib import Path
import fcntl,html,time
from common8 import read,write,sha
p=Path(__file__).resolve().parent;guard=open(p/'report.lock','w');fcntl.flock(guard,fcntl.LOCK_EX)
pr=read(p/'locked_protocol.json');state=read(p/'campaign_state.json') if (p/'campaign_state.json').exists() else {'status':'waiting_training','training_completed':0}
ml=read(p/'models_locked.json') if (p/'models_locked.json').exists() else None
labels={'granule':'小脑颗粒细胞','golgi':'小脑Golgi细胞','basket':'小脑篮状细胞','hay_l5':'大鼠L5','human_l23':'人L2/3','ca1':'海马CA1'}
def f(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def ci(x):return '—' if x is None else f'{x[0]:.3f}–{x[1]:.3f}'
rows=[];summary=[];details=[]
for i,j in enumerate(pr['jobs']):
 path=Path(j['base'])/'evaluation/complete.json';name=labels[j['cell']]+' / '+j['profile'].replace('_','+').upper()
 if not path.exists():
  sp=path.with_name('status.json');n=read(sp)['completed'] if sp.exists() else 0
  rows.append(f'<tr><td>{name}</td><td colspan="9">待全部模型冻结后评估；当前本组 {n}/182</td></tr>');continue
 c=read(path);assert c['all_inputs_retained'] and len(c['records'])==182 and c['test_manifest_sha256']==j['test_manifest_sha256']
 e=c['events'];v=c['voltage'];m=ml['models'][str(i)];assert c['checkpoint_sha256']==m['checkpoint_sha256']
 vals=[name,182,e['tp']+e['fn'],e['tp']+e['fp'],f(v['pooled_rmse_mv'],4),f(e['f1']),f"{e['tp']}/{e['fp']}/{e['fn']}",ci(c['bootstrap']['f1_ci95']),c['undefined_voltage_inputs'],m['diagonal_nonzero']]
 rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>')
 summary.append(dict(cell=j['cell'],profile=j['profile'],inputs=182,reference_spikes=e['tp']+e['fn'],predicted_spikes=e['tp']+e['fp'],voltage=v,events=e,bootstrap=c['bootstrap'],undefined_voltage_inputs=c['undefined_voltage_inputs'],selected_candidate=m['selected_candidate'],diagonal_nonzero=m['diagonal_nonzero'],diagonal_max_abs=m['diagonal_max_abs'],prepared_wall_ms=c['prepared_wall_ms'],checkpoint_sha256=c['checkpoint_sha256'],test_manifest_sha256=c['test_manifest_sha256'],evaluation_sha256=sha(path),groups=c['groups']))
 sections=[]
 for key,groups in c['groups'].items():
  tr=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [label,g['inputs'],g['events']['tp']+g['events']['fn'],f(g['voltage']['pooled_rmse_mv'],4),f(g['events']['f1']),f"{g['events']['tp']}/{g['events']['fp']}/{g['events']['fn']}"])+'</tr>' for label,g in groups.items())
  sections.append(f'<h4>{key}</h4><table><tr><th>条件</th><th>输入数</th><th>参考尖峰</th><th>RMSE/mV</th><th>F1</th><th>TP/FP/FN</th></tr>{tr}</table>')
 details.append(f'<details><summary>{name}：分组结果与计时</summary><p>模型候选 {html.escape(m["selected_candidate"])}；非零对角项 {m["diagonal_nonzero"]}/500；RMSE 95%区间 {ci(c["bootstrap"]["rmse_ci95_mv"])} mV。182条预处理后GPU总时间 {f(c["prepared_wall_ms"],2)} ms，3次中位数；共享服务器计时，不能与旧31窗或其他步长的求解器总时间直接相除。</p>{"".join(sections)}</details>')
verified=read(p/'verification.json') if (p/'verification.json').exists() else None
stamp=time.strftime('%Y-%m-%d %H:%M:%S %z')
body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>跨细胞最终选择模型固定面板复评</title><style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:35px auto;padding:0 24px;color:#203547}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #d4dfe8}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f8}}.note{{background:#edf6f4;padding:16px}}summary{{cursor:pointer;font-weight:600;padding:12px;background:#f1f5f9}}code{{overflow-wrap:anywhere}}p{{max-width:1100px}}</style><h1>跨细胞最终选择模型：固定面板复评</h1><p>更新于 {stamp}；状态 {html.escape(state['status'])}；训练完成 {state.get('training_completed',0)}/12；完整测试结果 {len(summary)}/12；最终校验 {'通过' if verified and verified['passed'] else '待完成'}。</p><p class="note">本报告独立保存。本评估流程保持主线小鼠L5PC模型、阈值和主图不变；核验通过后，另行更新论文跨细胞正文和附录。各细胞与突触类型分别训练模型，检验的是跨细胞适用性，并非同一套参数直接迁移。</p><h2>固定协议</h2><p>沿用此前已锁定的新种子面板：每个条件182条，共2184条。该面板已用于此前训练方案和此前跨细胞候选的评估，本次保持相同输入和参考解进行复评；它与本轮训练/验证输入及种子不重叠，但不是一批从未查看过的测试输出。12组模型全部完成后，只按原验证规则选定并冻结所有权重、阈值和checkpoint哈希，再开始任何本轮测试，不用测试结果调参或挑选模型。</p><p>每组包含1/2/4/8/16s ×64/128/256/500个位点 ×10/25/50Hz ×平稳、兴奋性门控、速率变化三类输入，共180条，另加1s和16s两个零输入对照。随机、分层、近端、远端空间模式按固定循环分配；AMPA和AMPA+NMDA使用配对输入。完整保留全部输入，不按尖峰数、burst、误差或预测表现筛选。训练阶段沿用旧训练清单及既有短前缀，测试阶段不采用这些筛选规则。</p><p>训练流程与旧版一致，从同一份各细胞电压拟合checkpoint重放尖峰训练，对角项从0初始化，学习率为其他参数0.1倍。完整预算重放后补存训练后候选；每个候选分别保留或屏蔽对角项，使用共用阈值网格和同一reset校准规则，再按原验证分数选定。全部12个模型、reset参数和哈希在本轮测试前冻结。原始零对角候选仍允许胜出。</p><p>DBNN采样1ms，参考积分步长与每种细胞原先固定的配置一致。电压RMSE排除参考尖峰±10ms；F1采用±5ms一对一匹配，浮点边界容差10⁻⁶ms。按完整输入轨迹进行10000次bootstrap。若mask覆盖整条轨迹，该条电压误差记为不可定义，事件仍计入，不删除输入。</p><h2>完整测试统计</h2><table><tr><th>细胞/突触</th><th>输入数</th><th>参考尖峰</th><th>预测尖峰</th><th>RMSE/mV</th><th>F1</th><th>TP/FP/FN</th><th>F1 95%区间</th><th>无有效电压样本</th><th>非零对角项</th></tr>{''.join(rows)}</table><h2>分组结果</h2>{''.join(details)}<h2>记录与校验</h2><p>保留逐输入参考与预测电压、无reset驱动、事件、输入哈希、模型哈希、逐输入指标、计时重复及分组统计。每条推理需与FFT驱动及FP64顺序reset核验电压和事件一致，CUDA图计时也需相同核验。数值不等价时仅使用预定FP64-reset回退，保持原权重、阈值和输入不变，并保留实际回退计时。</p><p>协议SHA-256：<code>{sha(p/'locked_protocol.json')}</code>；远程目录：<code>{p}</code>。完整结果须在覆盖182×12输入、重算逐输入指标与bootstrap、验证全部哈希后才标记完成。</p></html>'''
(p/'DBNN_跨细胞最终模型固定面板评估_20261017.html').write_text(body)
write(p/'summary.json',dict(complete=bool(verified and verified['passed']),status=state['status'],completed_conditions=len(summary),total_conditions=12,rows=summary,publication_update=False))
print(f'REPORT {len(summary)}/12 {state["status"]}')
