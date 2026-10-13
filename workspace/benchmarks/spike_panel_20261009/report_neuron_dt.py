# -*- coding: utf-8 -*-
"""Add the completed primary-subset NEURON 0.5-ms evaluation to the report."""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
from html.parser import HTMLParser


class Parse(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='img':self.images.append(a.get('src',''))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--main-report',type=Path)
    a=p.parse_args();data=a.data
    new=json.loads((data/'neuron_dt0.5_complete.json').read_text());old=json.loads((data/'comparison.json').read_text())
    assert new['finished'];s=new['low_density'];assert s['cases']==379 and s['reference_events']==637
    old_rows=[r for r in old['models']['dbnn']['records'] if r['low_density']]
    assert [r['input_sha256'] for r in old_rows]==[r['input_sha256'] for r in new['records']]
    digest=hashlib.sha256((data/'comparison.json').read_bytes()).hexdigest()
    assert digest==new['protocol']['original_comparison_sha256']
    models=[('NEURON，积分 0.5 ms',s),('NEURON，积分 1 ms（粗步长对照）',old['models']['neuron1']['low_density']),
            ('DBNN，输入／输出 1 ms',old['models']['dbnn']['low_density'])]
    def tr(name,v):
        e=v['events'];ci=v['bootstrap_95ci']['f1']
        return f'<tr><th>{name}</th><td>{v["pooled_rmse_mv"]:.4f}</td><td>{e["precision"]:.3f}</td><td>{e["recall"]:.3f}</td><td>{e["f1"]:.3f}</td><td>{ci[0]:.3f}–{ci[1]:.3f}</td><td>{e["tp"]} / {e["fp"]} / {e["fn"]}</td></tr>'
    family_names={'stationary':'连续 Poisson','gated_excitation':'短时 E/I 比例变化','rate_steps':'输入强度分段变化','zero':'零输入对照'}
    family_rows=[]
    for k,v in new['groups']['family'].items():
        e=v['events'];f1=f'{e["f1"]:.3f}' if v['reference_events'] else '—'
        family_rows.append(f'<tr><th>{family_names[k]}</th><td>{v["cases"]}</td><td>{v["reference_events"]}</td><td>{v["pooled_rmse_mv"]:.4f}</td><td>{f1}</td><td>{e["tp"]} / {e["fp"]} / {e["fn"]}</td></tr>')
    durations=[]
    for k,v in new['groups']['duration_s'].items():
        durations.append(f'<tr><th>{k} s</th><td>{v["cases"]}</td><td>{v["reference_events"]}</td><td>{v["pooled_rmse_mv"]:.4f}</td><td>{v["events"]["f1"]:.3f}</td></tr>')
    tolerance=[]
    for t,v in new['tolerance_sensitivity_ms'].items():
        tolerance.append(f'<tr><th>±{t} ms</th><td>{v["precision"]:.3f}</td><td>{v["recall"]:.3f}</td><td>{v["f1"]:.3f}</td><td>{v["tp"]} / {v["fp"]} / {v["fn"]}</td></tr>')
    median=float(np.median([r['voltage']['rmse_mv'] for r in new['records']]))
    largest=sorted(new['records'],key=lambda r:r['voltage']['rmse_mv'],reverse=True)[:5]
    e=s['events'];ci=s['bootstrap_95ci']['f1']
    fragment=f'''<section id="neuron-half-ms"><h2>新增：NEURON 0.5 ms 积分的完整主统计</h2>
<p>已在 5090 完成原主集合 <strong>379 条输入、637 个参考尖峰</strong>的复跑。内部积分步长为 <strong>0.5 ms</strong>，输入事件保持原样，电压输出仍为 <strong>1 ms</strong>；参考仍用 0.1 ms 积分。每条输入使用独立新进程构建同一小鼠 V1 L5PC、500 位点、突触和主动离子通道。输入资格由此前参考轨迹确定，本轮未重新筛选。</p>
<p class="lead">NEURON 0.5 ms 积分：掩码 pooled RMSE <strong>{s['pooled_rmse_mv']:.4f} mV</strong>；检测到 <strong>{s['predicted_events']} 个事件</strong>，其中 <strong>{e['tp']} 个在 ±5 ms 内匹配</strong>。Precision = {e['precision']:.3f}，Recall = {e['recall']:.3f}，F1 = <strong>{e['f1']:.3f}</strong>（95% CI {ci[0]:.3f}–{ci[1]:.3f}）。</p>
<div class="scroll"><table><thead><tr><th>设置</th><th>掩码 RMSE / mV</th><th>Precision</th><th>Recall</th><th>F1</th><th>F1 95% CI</th><th>TP / FP / FN</th></tr></thead><tbody>{''.join(tr(n,v) for n,v in models)}</tbody></table></div>
<p>这三行使用完全相同的输入、参考、观察网格与评分：电压剔除参考尖峰前后 ±10 ms；事件使用完整窗口、±5 ms 一对一匹配。5,000 次轨迹级 bootstrap 沿用原随机种子和确定性零输入固定计入方式。DBNN checkpoint、输出与阈值保持原结果。</p>
<p>0.5 ms 的逐轨迹 RMSE 中位数为 {median:.4f} mV，平均逐轨迹 VE 为 {s['mean_case_ve']:.4f}，轨迹内方差加权 VE 为 {s['within_trace_weighted_ve']:.4f}。配对事件的绝对时间误差中位数为 {e['median_abs_error_ms']:.3f} ms（统计范围为 ±5 ms 内的匹配）。</p>
<h3>按输入形式分层</h3><div class="scroll"><table><thead><tr><th>输入</th><th>窗口</th><th>参考尖峰</th><th>掩码 RMSE / mV</th><th>F1</th><th>TP / FP / FN</th></tr></thead><tbody>{''.join(family_rows)}</tbody></table></div>
<h3>按模拟时长分层</h3><div class="scroll"><table><thead><tr><th>时长</th><th>窗口</th><th>参考尖峰</th><th>掩码 RMSE / mV</th><th>F1</th></tr></thead><tbody>{''.join(durations)}</tbody></table></div>
<h3>固定预测的匹配容差检查</h3><div class="scroll"><table><thead><tr><th>匹配容差</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th></tr></thead><tbody>{''.join(tolerance)}</tbody></table></div>
<p>主 F1 仍使用 ±5 ms；其他容差只帮助了解事件时间偏移，预测和检测阈值保持一致。每个窗口的电压、事件、误差和分层统计均已保存。</p>
<p>本轮仅复跑 NEURON 单线程求解，16 个独立输入进程并发用于缩短评估时间。新增结果覆盖上述 379 条主集合；原完整 637 条的 1 ms 积分数据及 CoreNEURON／multisplit 结果保留。并发评估中的计时未作为新的论文速度结果。</p>
<p><a href="neuron_dt0.5_complete.json">下载完整逐输入结果及统计 JSON</a>。远程原始电压与事件：<code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/spike_panel_20261009/methods/neuron_dt0.5/raw/</code>。</p></section>'''
    (data/'neuron_dt0.5_fragment.html').write_text(fragment)
    report=data/'DBNN_独立尖峰评估_1ms_20261009.html';text=report.read_text()
    if '<section id="neuron-half-ms">' in text:
        start=text.index('<section id="neuron-half-ms">');end=text.index('</section>',start)+len('</section>')
        text=text[:start]+fragment+text[end:]
    else:text=text.replace('<section id="solver-step-audit">',fragment+'\n<section id="solver-step-audit">')
    if 'href="#neuron-half-ms"' not in text:text=text.replace('<a href="#solver-step-audit">','<a href="#neuron-half-ms">NEURON 0.5 ms</a><a href="#solver-step-audit">',1)
    check=Parse();check.feed(text);assert len(check.ids)==len(set(check.ids))
    report.write_text(text)
    if a.main_report:
        text=a.main_report.read_text();before=Parse();before.feed(text)
        notice=f'''<div id="neuron-half-ms-notice"><h3>379 条主集合的 NEURON 0.5 ms 积分复跑</h3><p>已完成相同 379 条输入、637 个参考尖峰；输出仍为 1 ms。NEURON 的掩码 RMSE = {s['pooled_rmse_mv']:.4f} mV，F1 = {e['f1']:.3f}（95% CI {ci[0]:.3f}–{ci[1]:.3f}），TP／FP／FN = {e['tp']}／{e['fp']}／{e['fn']}。上方 1 ms 积分表保留为粗步长对照，完整新结果见<a href="DBNN_spike_panel_data_20261009/DBNN_独立尖峰评估_1ms_20261009.html#neuron-half-ms">0.5 ms 主统计</a>。</p></div>'''
        if 'id="neuron-half-ms-notice"' in text:
            start=text.index('<div id="neuron-half-ms-notice">');end=text.index('</div>',start)+len('</div>')
            text=text[:start]+notice+text[end:]
        else:
            start=text.index('<section class="card" id="independent-spike-1ms-20261009">');end=text.index('</section>',start)
            text=text[:end]+notice+text[end:]
        after=Parse();after.feed(text);assert before.images==after.images and len(after.ids)==len(set(after.ids))
        a.main_report.write_text(text)
    validation=dict(finished=True,cases=379,reference_events=637,solver_dt_ms=.5,output_interval_ms=1.,
                    original_comparison_sha256_unchanged=digest,input_order_and_hashes_match=True,
                    new_result_sha256=hashlib.sha256((data/'neuron_dt0.5_complete.json').read_bytes()).hexdigest(),
                    largest_voltage_errors=[dict(case_id=r['case_id'],rmse_mv=r['voltage']['rmse_mv']) for r in largest])
    (data/'neuron_dt0.5_report_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(s,indent=2))


if __name__=='__main__':main()
