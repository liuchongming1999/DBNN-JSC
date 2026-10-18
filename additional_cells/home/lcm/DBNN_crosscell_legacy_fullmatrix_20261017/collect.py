"""Read-only validation evidence collection, independent of publication assets."""
from pathlib import Path
import json,math,html,hashlib,time,fcntl
import torch
from common8 import read,write,sha
p=Path(__file__).resolve().parent
guard=open(p/'collection.lock','w');fcntl.flock(guard,fcntl.LOCK_EX)
lock=read(p/'locked_protocol.json');state=read(p/'campaign_state.json');rows=[];checks=[]
for i,j in enumerate(lock['jobs']):
 out=p/'results'/j['cell']/j['profile']/'dt1.0ms';s=out/'final_selection.json';row=dict(cell=j['cell'],profile=j['profile'],status=state['jobs'].get(str(i),{}).get('status','queued'),fit=j['fit'],validation=j['validation'])
 old=read(Path(j['old_output'])/'final_selection.json');v=old['selected']['validation'];row['old']=dict(rmse=v['pooled_rmse_mv'],f1=v['events']['f1'],score=v['score'])
 if s.exists():
  d=read(s);v=d['selected']['validation'];pr=d['protocol'];cp=out/'best_checkpoint.pt'
  assert d['finished'] and d['updates']==d['expected_updates']==j['expected_updates'] and not d['test_evaluated']
  assert sha(cp)==d['checkpoint_sha256']
  assert sha(j['manifest'])==pr['input_manifest_sha256']==j['manifest_sha256']
  assert read(j['manifest'])['cases']==read(out/'training_manifest.json')['cases']
  assert sha(j['starting_checkpoint'])==pr['original_checkpoint_sha256']==j['starting_checkpoint_sha256']
  for k in ['fit_cases','validation_cases','epochs','refinement_epochs','trials','seed','batch_size','lr','voltage_loss','event_loss','reset','calibration','validation_selection','voltage_rmse_limit_mv']:
   assert pr[k]==old['protocol'][k],(i,k)
  st=torch.load(cp,map_location='cpu',weights_only=False)['model_state'];diag=st['bilinear.weight'][0].diagonal();assert all(z.isfinite().all() for z in st.values())
  for f in (out/'training').glob('*/*_history.json'):
   for rec in read(f):assert math.isclose(rec['learning_rates'][1]/rec['learning_rates'][0],.1,rel_tol=1e-12)
  a=torch.load(j['starting_checkpoint'],map_location='cpu',weights_only=False)['model_state'];b=torch.load(out/'starting_checkpoint.pt',map_location='cpu',weights_only=False)['model_state'];assert all(torch.equal(z,b[k]) for k,z in a.items()) and a['bilinear.weight'][0].diagonal().count_nonzero()==0
  replay=read(out/'baseline_replay.json');assert abs(replay['rmse_difference_mv'])<2e-4
  row.update(status='complete',new=dict(rmse=v['pooled_rmse_mv'],f1=v['events']['f1'],score=v['score']),candidate=d['selected']['name'],diagonal_max_abs=float(diag.abs().max()),diagonal_nonzero=int(diag.count_nonzero()),updates=d['updates'],elapsed_s=d['elapsed_s'],checkpoint=str(cp),checkpoint_sha256=d['checkpoint_sha256'],baseline_boundary_counts_equal=replay['boundary_counts_equal'])
  checks.append(dict(cell=j['cell'],profile=j['profile'],passed=True,matching_original_protocol=True,complete_budget=True,initial_checkpoint_exact=True,zero_initial_diagonal=True))
 elif (out/'progress.json').exists():
  z=read(out/'progress.json');row['progress']=dict(trial=z['trial'],stage=z['stage'],epoch=z['epoch'],elapsed_s=z['elapsed_s'])
 rows.append(row)
n=sum(r['status']=='complete' for r in rows);result=dict(updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),complete=n==12,completed=n,total=12,state=state['status'],validation_only=True,test_evaluated=False,rows=rows)
write(p/'validation_snapshot.json',result);write(p/'verification.json',dict(complete=n==12,completed=n,checks=checks,passed_for_completed=True))
labels={'granule':'小脑颗粒细胞','golgi':'小脑Golgi细胞','basket':'小脑篮状细胞','hay_l5':'大鼠L5','human_l23':'人L2/3','ca1':'海马CA1'}
def f(x):return '—' if x is None else f'{x:.4f}'
trs=[]
for r in rows:
 v=r.get('new',{});vals=[labels[r['cell']],r['profile'].replace('_','+').upper(),r['fit'],r['validation'],f(r['old']['rmse']),f(v.get('rmse')),f(r['old']['f1']),f(v.get('f1')),r['status'],r.get('candidate','—'),f(r.get('diagonal_max_abs'))]
 trs.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>')
body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>跨细胞旧流程全矩阵对照</title><style>body{{font:16px/1.7 system-ui;color:#203547;max-width:1300px;margin:35px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{padding:9px;text-align:right;border-bottom:1px solid #d3dee8}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef4f8}}p{{max-width:1000px}}.note{{background:#eef4f8;padding:14px}}</style><h1>跨细胞：旧训练流程，仅释放对角项</h1><p class="note">完成{n}/12组。更新时间：{result['updated']}。仅验证集结果，尚未进行新的独立测试；主线模型、阈值、正文和主图保持不变。</p><p>从旧版尖峰训练原来使用的同一份各细胞电压拟合checkpoint出发，重放原流程。上游单突触物理初始化与电压拟合阶段不重跑，旧、新对照共享相同起点。唯一模型改动为释放W的对角参数，初始值严格为0。对角项使用独立AdamW参数组，学习率为其他参数的0.1倍，余弦调度保持该比例。</p><p>其余设置沿用旧版：原训练/验证清单及已有短前缀、整段源轨迹拟合、seed310049、batch8、尖峰损失权重0/0.2/1三组、每组1000+400轮、主学习率0.0003/0.00006、原电压与导数损失、teacher reset与验证校准、验证选择目标及电压误差限制。不进行新的按输出筛选，不用测试集挑版本。保留既有训练清单的本轮结果，并不等于已在无筛选的新测试上验证。</p><table><tr><th>细胞</th><th>突触</th><th>Fit</th><th>Validation</th><th>旧RMSE</th><th>新RMSE</th><th>旧F1</th><th>新F1</th><th>状态</th><th>获选候选</th><th>max|Wii|</th></tr>{''.join(trs)}</table><p>RMSE单位mV。原有checkpoint与仅reset重新校准的候选按旧规则保留；若这些候选胜出，可能仍为零对角，不能据此声称非零对角训练收益。旧F1来自归档，新匹配保留已经批准的±5ms+1e−6ms浮点容差，baseline重放单独记录是否影响计数。</p><p>每个已完成模型均检查：初始checkpoint及训练清单哈希、旧协议字段一致、完整更新预算、参数有限、全程学习率比、实际导出对角项与checkpoint哈希。模型及各候选保留在服务器目录 /home/lcm/DBNN_crosscell_legacy_fullmatrix_20261017。</p></html>'''
(p/'DBNN_跨细胞旧流程全矩阵对照_20261017.html').write_text(body)
print(json.dumps({'completed':n,'state':state['status'],'finished_rows':[r for r in rows if r['status']=='complete']},ensure_ascii=False))
