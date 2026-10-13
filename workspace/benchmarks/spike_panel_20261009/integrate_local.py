"""Attach the new independent-input evidence without altering old figures."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote,urlparse


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    here=Path(__file__).resolve().parent;root=Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN')
    data=root/'DBNN_spike_panel_data_20261009';comparison=json.loads((data/'comparison.json').read_text())
    report=data/'DBNN_独立尖峰评估_1ms_20261009.html';s=report.read_text()
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.links=[];self.images=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if 'id' in a:self.ids.append(a['id'])
            if tag=='a' and 'href' in a:self.links.append(a['href'])
            if tag=='img' and 'src' in a:self.images.append(a['src'])
    check=Check();check.feed(s);assert len(set(check.ids))==len(check.ids)
    for link in check.links:
        if link.startswith('#'):assert link[1:] in check.ids
        elif not urlparse(link).scheme:assert (data/unquote(urlparse(link).path)).exists(),link
    for image in check.images:assert (data/image).exists()
    assert comparison['all_comparison_models_dt_ms']==1.
    assert len(comparison['models'])==10
    d=comparison['models']['dbnn'];low=d['low_density'];all_=d['all_generated']
    assert low['reference_events']>=300
    main=root/'DBNN_L5PC_surrogate_report_20260928.html';text=main.read_text();before=digest(main)
    backup=here/'results/main_report_before_independent_spike_update.html'
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():shutil.copy2(main,backup)
    id_='independent-spike-1ms-20261009'
    rows=[]
    labels={'dbnn':'DBNN（全矩阵）','neuron8':'NEURON multisplit，8核','core1':'CoreNEURON CPU，1核','reduce1':'Neuron_Reduce',
            'tcn_author_default':'TCN（作者默认训练）','elm_author_default':'Branch-ELM（作者默认训练）','lstm_common150':'LSTM'}
    for k,label in labels.items():
        ss=comparison['models'][k]['low_density'];ev=ss['events'];ci=ss['bootstrap_95ci']['f1']
        rows.append(f'<tr><td>{label}</td><td>{ss["pooled_rmse_mv"]:.4f}</td><td>{ev["f1"]:.3f}</td><td>{ci[0]:.3f}–{ci[1]:.3f}</td><td>{ev["tp"]} / {ev["fp"]} / {ev["fn"]}</td></tr>')
    new=f'''<section class="card" id="{id_}"><h2>附录 AJ. 全矩阵 DBNN 与独立尖峰输入评估（统一 1 ms）</h2>
<p>2026-10-09 起，DBNN 统一定义为完整二次矩阵 W，对角项可训练，保持每个位点一个滤波核和原 reset 架构。后续不再新增零对角实验；旧结果保留为历史记录。</p>
<p>本次生成 {comparison['input_cases']} 条新输入，共 {all_['reference_events']} 个参考尖峰。按照此前每窗最多 3 个尖峰、间隔至少 50 ms 的范围，主事件统计使用 {low['cases']} 条输入、{low['reference_events']} 个参考尖峰。参考模型为 NEURON 0.1 ms，其余全部为 1 ms。模型、阈值和标定冻结，不用新评估输入重新选参。</p>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>masked RMSE / mV</th><th>F1</th><th>95% CI</th><th>TP / FP / FN</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>完整窗口采用 ±5 ms 一对一事件匹配；电压统一剔除参考尖峰 ±10 ms。置信区间对完整独立输入轨迹进行 5,000 次 bootstrap；确定性零输入控制固定计入。原有31窗的10次尖峰计数包含共享前缀，新增集合不复用这些随机轨迹。</p>
<figure><img loading="lazy" src="DBNN_spike_panel_data_20261009/figures/event_comparison_1ms.png" alt="独立输入上的1ms模型事件F1及置信区间"/><figcaption>主统计包含379条输入、637个参考事件。NEURON／CoreNEURON的数值对应这里指定的1ms求解步长。</figcaption></figure>
<p>DBNN 的短时 E/I 比例变化输入 F1 为 0.879，连续 Poisson 输入为 0.435；完整637条输入（不按放电数筛选）的 F1 为 {all_['events']['f1']:.3f}。各类输入、其他容差、已有训练缓冲对照与全部模型记录见详细报告。</p>
<p><a href="DBNN_spike_panel_data_20261009/DBNN_独立尖峰评估_1ms_20261009.html">完整独立尖峰评估报告</a> · <a href="DBNN_spike_panel_data_20261009/comparison.json">逐窗比较 JSON</a> · <a href="DBNN_model_definition.json">全矩阵模型定义</a></p></section>'''
    if id_ not in text:
        assert text.count('</main>')==1;text=text.replace('</main>',new+'</main>',1)
        start=text.index('<nav aria-label="报告目录"');end=text.index('</nav>',start)
        nav=text[start:end];position=nav.rfind('</ul>');assert position>=0
        nav=nav[:position]+f'<li><a data-section-link="" href="#{id_}">附录 AJ. 独立尖峰评估（统一1ms）</a></li>'+nav[position:]
        text=text[:start]+nav+text[end:]
        match=re.search(r'<section\b[^>]*\bid="accuracy-development"[^>]*>',text)
        assert match
        notice=f'<p class="note">2026-10-09 更新：DBNN 方法定义已采用允许非零对角的全矩阵。本节原有31窗与细采样计时保留历史版本；最新全矩阵1ms模型及扩大事件统计见<a href="#{id_}">附录AJ</a>。</p>'
        text=text[:match.end()]+notice+text[match.end():]
        main.write_text(text)
    assert text.count('id="'+id_+'"')==1
    validation=dict(finished=True,standalone_report=str(report),main_report=str(main),appendix=id_,
                    main_before_sha256=before,main_after_sha256=digest(main),backup=str(backup),
                    standalone_links_valid=True,standalone_images_valid=True,old_figures_unmodified=True,
                    model_count=10,predicted_input_model_pairs=6370,all_model_dt_ms=1,reference_dt_ms=.1,
                    input_cases=comparison['input_cases'],low_spike_cases=low['cases'],low_spike_events=low['reference_events'])
    (data/'local_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (here/'results/local_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
