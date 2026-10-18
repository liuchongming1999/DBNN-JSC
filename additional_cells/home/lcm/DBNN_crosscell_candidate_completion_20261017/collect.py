"""Verify and report the finite, validation-only candidate completion."""
from pathlib import Path
import math,json,html,hashlib,tarfile,time
import numpy as np
import torch
from common8 import read,write,sha,aggregate,event_aggregate

p=Path(__file__).resolve().parent
lock=read(p/'locked_protocol.json')
old=Path(lock['training_reference'])
state=read(p/'campaign_state.json')
assert state['status']=='complete' and not state['errors'],state
torch.set_num_threads(1)

def rank(c):
    v=c['validation']
    return (not v['voltage_eligible'],v['score'])

def compact(c):
    v=c['validation']
    return dict(name=c['name'],source=c['source'],reset=c['reset'],
                rmse=v['pooled_rmse_mv'],f1=v['events']['f1'],score=v['score'],
                eligible=v['voltage_eligible'],events=v['events'])

def verify_metrics(c,limit):
    v=c['validation'];cases=v['cases']
    a=aggregate([x['voltage'] for x in cases]);e=event_aggregate([x['events'] for x in cases])
    assert abs(a['pooled_rmse_mv']-v['pooled_rmse_mv'])<1e-10
    assert all(e[k]==v['events'][k] for k in ['tp','fp','fn','f1'])
    normalized=float(np.mean([x['voltage']['sse']/x['voltage']['n']/max(x['voltage']['reference_ss']/x['voltage']['n'],.04) for x in cases]))
    expected=a['pooled_rmse_mv']**2+.02*normalized+.25*(1-e['f1'])
    assert abs(expected-v['score'])<1e-10
    assert v['voltage_eligible']==(a['pooled_rmse_mv']<=limit+1e-10)
    assert sha(c['checkpoint'])==c['checkpoint_sha256']

rows=[];checks=[];candidate_metrics=0
for job in lock['jobs']:
    folder=p/'results'/job['cell']/job['profile']/'dt1.0ms'
    reference=old/'results'/job['cell']/job['profile']/'dt1.0ms'
    current=read(folder/'final_selection.json');previous=read(reference/'final_selection.json')
    assert current['updates']==job['expected_updates']==current['expected_updates']
    for key in ['epochs','refinement_epochs','trials','seed','batch_size','lr','optimizer',
                'diagonal_lr_multiplier','input_manifest_sha256','original_checkpoint_sha256',
                'voltage_loss','event_loss','reset','voltage_rmse_limit_mv']:
        assert current['protocol'][key]==previous['protocol'][key],(job['cell'],job['profile'],key)
    assert current['raw_candidate_count']==38 and current['expanded_candidate_count']==76
    assert not current['test_evaluated'] and current['training_path_preserved']
    assert sha(job['starting_checkpoint'])==job['starting_checkpoint_sha256']
    assert sha(job['manifest'])==job['manifest_sha256']
    path=read(folder/'training_path_selection.json')
    assert path['selected']['name']==previous['selected']['name']
    assert len(path['candidates'])==len(previous['candidates'])==20
    max_state_gap=0.;max_history_gap=0.
    for a,b in zip(path['candidates'],previous['candidates']):
        assert a['name']==b['name'] and a['reset']==b['reset']
        x=torch.load(a['checkpoint'],map_location='cpu',weights_only=False)['model_state']
        y=torch.load(b['checkpoint'],map_location='cpu',weights_only=False)['model_state']
        assert x.keys()==y.keys()
        for key in x:
            gap=float((x[key]-y[key]).abs().max())
            max_state_gap=max(gap,max_state_gap)
            assert gap<1e-7,(job['cell'],job['profile'],a['name'],key,gap)
    for hf in sorted((folder/'training').glob('*/*_history.json')):
        a=read(hf);b=read(reference/hf.relative_to(folder))
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert x['epoch']==y['epoch'] and x['updates']==y['updates'] and x['learning_rates']==y['learning_rates']
            gap=abs(x['validation']['pooled_rmse_mv']-y['validation']['pooled_rmse_mv'])
            max_history_gap=max(gap,max_history_gap)
            assert gap<1e-6,(job['cell'],job['profile'],hf.name,x['epoch'],gap)
            assert all(x['validation']['events'][k]==y['validation']['events'][k] for k in ['tp','fp','fn'])
    pairs=read(folder/'paired_validation.json')['pairs'];assert len(pairs)==38
    for pair in pairs:
        for c in [pair['on'],pair['off']]:
            verify_metrics(c,current['protocol']['voltage_rmse_limit_mv']);candidate_metrics+=1
        on=torch.load(pair['on']['checkpoint'],map_location='cpu',weights_only=False)['model_state']
        off=torch.load(pair['off']['checkpoint'],map_location='cpu',weights_only=False)['model_state']
        for key in on:
            if key=='bilinear.weight':
                diff=on[key]-off[key];diff[0].diagonal().zero_()
                assert torch.count_nonzero(diff)==0
                assert torch.count_nonzero(off[key][0].diagonal())==0
            elif key not in ['v_th','reset_amp','tau_reset']:
                assert torch.equal(on[key],off[key]),key
        prefix=len(pair['shared_thresholds'])*9+2
        grids=[read(Path(pair[mode]['checkpoint']).with_name('reset_grid.json')) for mode in ['on','off']]
        assert [x['params'] for x in grids[0][:prefix]]==[x['params'] for x in grids[1][:prefix]]
    chosen=min(current['candidates'],key=rank)
    assert chosen['name']==current['selected']['name']
    assert sha(folder/'best_checkpoint.pt')==current['checkpoint_sha256']
    cp=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
    assert float(cp['bilinear.weight'][0].diagonal().abs().max())==current['diagonal_max_abs']
    on=min([x['on'] for x in pairs],key=rank);off=min([x['off'] for x in pairs],key=rank)
    nonzero_pair=min([x for x in pairs if x['diagonal_nonzero_count']>0],key=lambda x:rank(x['on']))
    rows.append(dict(cell=job['cell'],profile=job['profile'],fit=job['fit'],validation=job['validation'],
        previous=compact(previous['selected']),selected=compact(current['selected']),best_on=compact(on),best_off=compact(off),
        best_trained_nonzero=compact(nonzero_pair['on']),paired_off_of_best_trained_nonzero=compact(nonzero_pair['off']),
        best_trained_nonzero_direct_drive_rms_mv=nonzero_pair['direct_diagonal_drive_rms_mv'],
        diagonal_max_abs=current['diagonal_max_abs'],nonzero_diagonal_count=int(torch.count_nonzero(cp['bilinear.weight'][0].diagonal())),
        selected_kind=('trained_nonzero_retained' if torch.count_nonzero(cp['bilinear.weight'][0].diagonal()) else
                       'trained_diagonal_masked' if any(x['raw_candidate']==current['selected']['source']['raw_candidate'] and x['diagonal_nonzero_count']>0 for x in pairs) else
                       'original_zero_snapshot'),
        score_change=current['selected']['validation']['score']-previous['selected']['validation']['score'],
        updates=current['updates'],checkpoint=str(folder/'best_checkpoint.pt'),checkpoint_sha256=current['checkpoint_sha256'],
        selection_changed=current['selected']['source']['raw_candidate']!=previous['selected']['name'],
        pairs=[dict(raw_candidate=x['raw_candidate'],source=x['source'],diagonal_nonzero_count=x['diagonal_nonzero_count'],
            direct_diagonal_drive_rms_mv=x['direct_diagonal_drive_rms_mv'],on=compact(x['on']),off=compact(x['off'])) for x in pairs]))
    checks.append(dict(cell=job['cell'],profile=job['profile'],passed=True,full_budget_updates=current['updates'],
        raw_candidates=38,calibrated_variants=76,max_original_state_gap=max_state_gap,
        max_original_history_rmse_gap_mv=max_history_gap,stage_selection_unchanged=True,
        paired_switch_only_diagonal=True,paired_calibration_grid_equal=True))

# Prior frozen test models remain immutable; no test voltage file is opened.
frozen=read(Path('/home/lcm/DBNN_crosscell_legacy_test_20261017/models_locked.json'))
write(p/'frozen_models_manifest_copy.json',frozen)
assert len(frozen['models'])==12
for item in frozen['models'].values():
    assert sha(item['checkpoint'])==item['checkpoint_sha256']
    original_selection=read(Path(item['source_checkpoint']).with_name('final_selection.json'))
    assert sha(item['source_checkpoint'])==original_selection['checkpoint_sha256']

summary=dict(complete=True,validation_only=True,test_evaluated=False,main_model_changed=False,manuscript_changed=False,
    completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),conditions=12,raw_candidates=456,calibrated_variants=912,
    selected_nonzero_diagonal=sum(x['nonzero_diagonal_count']>0 for x in rows),
    selected_zero_diagonal=sum(x['nonzero_diagonal_count']==0 for x in rows),
    selection_kinds={kind:sum(x['selected_kind']==kind for x in rows) for kind in
                     ['trained_nonzero_retained','trained_diagonal_masked','original_zero_snapshot']},rows=rows)
write(p/'validation_summary.json',summary)
write(p/'verification.json',dict(passed=True,conditions=checks,candidate_metrics_recomputed=candidate_metrics,
    fixed_test_models_preserved=True,test_evaluated=False,full_budget=True))

esc=lambda x:html.escape(str(x))
names={'granule':'颗粒细胞','golgi':'Golgi','basket':'篮状细胞','hay_l5':'大鼠 L5b','human_l23':'人 L2/3','ca1':'CA1'}
body=[]
for x in rows:
    a,b=x['previous'],x['selected']
    mode='非零对角' if x['nonzero_diagonal_count'] else '零对角'
    if x['selected_kind']=='trained_diagonal_masked':mode+='（训练后屏蔽）'
    elif x['selected_kind']=='original_zero_snapshot':mode+='（原始快照）'
    body.append(f'<tr><td>{names[x["cell"]]}</td><td>{esc(x["profile"])}</td><td>{a["rmse"]:.6f} / {a["f1"]:.4f}</td><td>{b["rmse"]:.6f} / {b["f1"]:.4f}</td><td>{mode}</td><td>{a["score"]:.6f} → {b["score"]:.6f}</td></tr>')
details=[]
for x in rows:
    b=x['selected'];on=x['best_on'];off=x['best_off']
    details.append(f'<details><summary>{names[x["cell"]]} · {esc(x["profile"])}：{esc(b["name"])}</summary><p>来源：{esc(b["source"])}。reset：{esc(b["reset"])}。</p><p>保留对角候选中的最佳结果：RMSE {on["rmse"]:.6f} mV，F1 {on["f1"]:.4f}，J {on["score"]:.8f}；屏蔽对角候选中的最佳结果：RMSE {off["rmse"]:.6f} mV，F1 {off["f1"]:.4f}，J {off["score"]:.8f}。</p><p>最终矩阵非零对角元素 {x["nonzero_diagonal_count"]} 个，绝对值最大 {x["diagonal_max_abs"]:.8g}。完整候选比较及逐输入指标见 JSON 记录。</p></details>')
    nz=x['best_trained_nonzero'];paired=x['paired_off_of_best_trained_nonzero']
    extra=f'<p>实际具有非零对角项的候选中，最佳结果为 RMSE {nz["rmse"]:.6f} mV，F1 {nz["f1"]:.4f}，J {nz["score"]:.8f}；将同一快照的对角屏蔽并重新校准后为 RMSE {paired["rmse"]:.6f} mV，F1 {paired["f1"]:.4f}，J {paired["score"]:.8f}。这个配对比较用于区分对角项本身与其他参数更新的影响。</p>'
    details[-1]=details[-1].replace('</details>',extra+'</details>')
document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跨细胞候选补齐与验证选择</title><style>body{font-family:system-ui,sans-serif;color:#20364a;background:#f7fafc;max-width:1140px;margin:40px auto;padding:0 24px;line-height:1.75}h1{font-size:28px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:white}td,th{padding:12px;border-bottom:1px solid #dce5ed;text-align:left}th{background:#e7eff5}details{background:white;padding:16px;margin:12px 0}summary{cursor:pointer;font-weight:600}.notice{border-left:4px solid #178d80;background:#edf8f5;padding:16px}code{overflow-wrap:anywhere}</style><h1>跨细胞候选补齐与验证选择</h1>'''
document+=f'<p class="notice">12组完整训练与验证比较已完成，共456个原始候选、912个对角开关变体。最终选择非零对角 {summary["selected_nonzero_diagonal"]} 组，零对角 {summary["selected_zero_diagonal"]} 组。本报告仅报告验证集选择；原2184条测试结果、已冻结模型、主线模型与论文保持不变。</p>'
document+='<h2>同一规则下的选择结果</h2><p>单元格依次为电压 RMSE（mV）/ 尖峰 F1。原选择来自上一次完整训练；新选择补齐正更新快照并统一校准reset。J越小越好，先满足原电压误差门槛，再按原J分数排序。</p><table><thead><tr><th>细胞</th><th>突触</th><th>原选择</th><th>新选择</th><th>实际对角</th><th>J</th></tr></thead><tbody>'+''.join(body)+'</tbody></table>'
document+='<h2>候选与reset的比较方式</h2><p>每个事件损失权重均完整训练1000+400轮，数据、种子、优化器、学习率及预算不变。新增保留每阶段至少经过一次更新的最佳联合分数快照、最佳电压快照和最后一次更新快照。含第0轮的原候选与EMA仍保留。新增候选不参与第二阶段初始化，因此原训练路径不变。</p><p>完整训练结束后，每个候选分别保留或屏蔽对角项，其他参数相同。每对共用两条轨迹给出的阈值网格并集，以同一源模型reset为锚点，保留同一候选reset作为热启动，使用原振幅、时间常数与局部细化规则。两种结构均重新校准reset，再按同一验证规则比较。零对角候选没有被强制排除。</p><p>J = pooled RMSE² + 0.02 × 平均逐输入归一化均方误差 + 0.25 × (1 − F1)。归一化方差下限为0.04 mV²；原电压门槛为源模型验证RMSE + max(0.05 mV, 25%源RMSE)。电压统计排除参考尖峰±10 ms，事件按±5 ms一对一匹配，边界容差1e−6 ms。</p>'
document+='<h2>如何理解结果</h2><p>保留对角项的模型类别包含零对角模型，但某个训练快照的非零对角项不一定改善验证表现。这里通过同一快照的对角开关区分对角项的直接作用，并通过补齐候选检查原来的快照预筛是否遗漏了较好的模型。最终选择仍是验证集结果，不能直接当作新的测试泛化结果；较小的J差值也不等同于统计显著优势。</p>'
document+='<h2>逐组选择记录</h2>'+''.join(details)
document+='<h2>核验与原始记录</h2><p>全部完整预算、原训练路径与历史候选权重、912个候选的逐输入汇总指标、成对权重差异、共用reset网格和最终导出均已核验。原训练路径权重差要求小于1e−7，验证历史RMSE差小于1e−6 mV，事件计数一致。</p><p><a href="validation_summary.json">完整验证结果</a> · <a href="verification.json">核验记录</a> · <a href="locked_protocol.json">锁定协议</a> · <a href="实验说明.txt">实验说明</a></p></html>'
(p/'DBNN_跨细胞候选补齐与验证选择_20261017.html').write_text(document)
with tarfile.open(p/'validation_records.tar.gz','w:gz') as tar:
    for pattern in ['results/*/*/dt1.0ms/*.json','results/*/*/dt1.0ms/best_checkpoint.pt','results/*/*/dt1.0ms/training/*/*_history.json','*.py','*.cpp','*.diff','*.txt','locked_protocol.json','preflight.json','smoke_verification.json','verification.json','validation_summary.json']:
        for f in p.glob(pattern):tar.add(f,arcname=str(f.relative_to(p)))
write(p/'archive_manifest.json',dict(path='validation_records.tar.gz',bytes=(p/'validation_records.tar.gz').stat().st_size,sha256=sha(p/'validation_records.tar.gz')))
print(json.dumps(dict(passed=True,conditions=12,selected_nonzero_diagonal=summary['selected_nonzero_diagonal'],selected_zero_diagonal=summary['selected_zero_diagonal'])))
