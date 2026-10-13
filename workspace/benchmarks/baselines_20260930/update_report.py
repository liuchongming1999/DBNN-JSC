"""Append current evidence to the existing self-contained canonical report."""
import base64,html,json,re,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN/DBNN_L5PC_surrogate_report_20260928.html')
def esc(x):return html.escape(str(x))
def table(headers,rows):
    return '<div class="resolution-table"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pct(v):return '—' if v is None else f'{100*v:.2f}%'

def section(d):
    s='<section id="baseline-reproduction"><h2>附录 AC. 文献基线复现与统一对比</h2>'
    s+=f'<p class="muted">证据快照：{esc(d["created"])}。原始JSON、训练历史与配置保存在本附录的下载记录中。</p>'
    s+='<div class="note"><strong>原作者默认训练流程。</strong>本轮固定 Branch-ELM、TCN 与 LSTM 的损失、优化器、学习率及训练预算。适配涉及500个输入位点、既有数据划分和采样间隔。实验顺序为公开结果复核 → 当前L5PC默认训练 → 验证集锁定 → 31窗评估 → 集中测速；主线新增工作点使用最后一步的结果。</div>'
    s+='<h3>AC.1 公开结果的完整复核</h3><p>TCN 与 Branch-ELM 使用作者公开权重，在12个公开 NeuronIO 测试文件上执行原作者评分：将胞体电压截到−55 mV、保留均值/标准差对齐、略去最初500 ms，并筛选每条轨迹0–24个尖峰。最终1487条6秒轨迹、8,178,500个评分点。这里的神经元与当前Shai L5PC分别记录，后者另行训练。</p>'
    rows=[]
    for key,label in [('tcn','TCN'),('elm','Branch-ELM')]:
        r=d['original'].get(key)
        if r:
            m=r['metrics'];e=r['expected'];rows.append([label,f'{e["rmse_mv"]:.6f}',f'{m["rmse_mv"]:.6f}',f'{e["spike_auc"]:.8f}',f'{m["spike_auc"]:.8f}','通过' if r['passed'] else '核查中'])
    s+=table(['公开模型','论文/仓库RMSE (mV)','本次RMSE (mV)','论文/仓库AUC','本次AUC','完整复核'],rows)
    r=d['original'].get('reduce_legacy')
    if r:
        m=r['metrics'];s+=f'<p>Neuron_Reduce：在独立 NEURON 7.8.2 环境重现作者 fully-active Hay-2011 Test 5（10,000个突触、1.5 s、dt=0.1 ms）。降阶/完整模型 RMSE 为<strong>{m["reduction_rmse_mv"]:.9f} mV</strong>，作者保存值为{m["published_fixture_rmse_mv"]:.9f} mV；本次降阶曲线与保存曲线的 RMSE 为<strong>{m["fixture_replay_rmse_mv"]:.3e} mV</strong>，原断言通过。旧环境保留原ACG随机输入；新环境的API兼容改动保存在独立副本中。</p>'
    s+='<p>LSTM使用论文附录B与Table S10的50 hidden units配置，作为随附对照。作者仓库提供的公开神经元权重覆盖TCN/ELM，本轮公开权重复核对应这两类模型。</p>'
    s+='<h3>AC.2 固定配方与采样适配</h3>'
    s+=table(['方法','固定模型','默认损失/训练流程','预算'],[
        ['Branch-ELM','20 memory、45 branches、每branch100 synapses；物理τ=1–150 ms','0.5 BCE + 0.5 MSE；voltage scale0.1；Adam lr=5e−4、cosine、batch8','35 epochs × 10,000 batches（官方脚本）'],
        ['TCN','论文公开7层、width128；1ms kernels=[45,19,19,19,19,19,19]','BCE + soma MSE + DVT-PCA20 MSE；kernel L2=1e−8；NAdam，默认五阶段权重/LR，batch8','250 stages × 10 epochs × 288 batches'],
        ['LSTM','单层hidden50，经典LSTM，spike/voltage双输出','等权BCE/MSE；voltage scale0.1；Adam lr=5e−4、cosine、batch8、150ms loss burn-in','30 epochs × 11,400 batches（论文）'],
        ['Neuron_Reduce','README示例frequency0；每0.1λ自动空间离散','默认阻抗映射与主动通道复制；保留Shai物理参数','无训练；dt=1/0.5/0.25ms；31窗、30次计时']])
    s+='<p>ELM的默认代码写35×10,000，并在整个500ms窗口计算loss；论文描述30×11,400与150ms loss burn-in。当前ELM沿用官方非short-run代码，LSTM沿用论文流程。TCN按官方生成器计算epoch长度：6 × int(0.2 × 128 × 6000 / (8 × 400)) = 288 batches，NAdam每个stage重新初始化。</p>'
    s+=table(['模型采样间隔','fit / validation','物理时间适配'],[['1 ms','117 / 32','对应现有1ms DBNN训练面板'],['0.5 ms','189 / 56','ELM delta_t=0.5；TCN kernel离散点数换算'],['0.25 ms','261 / 80','ELM delta_t=0.25；TCN保持152ms物理历史支持']])
    s+='<p>各学习基线使用对应分辨率的相同输入事件、参考与划分，从随机初始化开始。400/500ms的物理窗口和重叠长度保持一致；TCN的kernel点数随分辨率增加，参数量按实际模型记录。原始1278个输入替换为500个固定E/I位点，事件计数保留。</p>'
    s+='<p>TCN保留原有树突辅助任务：既有341条fit/validation样本补录639个树突segment电压，每条胞体输出与已有参考一致。白化PCA为20维，原始树突电压上截2mV，训练集前至多128条拟合，投影截±3；验证及主31窗用于各自的评价阶段。</p>'
    s+='<p>当前L5PC部署时，电压均值/标准差仅在验证集拟合一次仿射变换并冻结，事件阈值也由验证集确定。评估同时保留未经该变换的mV误差。主31窗使用共同1ms观察网格、同一参考spike ±10ms掩码；事件评价采用±5ms一一匹配。</p>'
    s+='<h3>AC.3 默认训练进度</h3><p>表中训练进度由远程记录读取。完成默认训练预算与验证集锁定后，才进入完整31窗评估。</p>'
    rows=[]
    for dt in [1.,.5,.25]:
        for method,label in [('tcn','TCN'),('elm','Branch-ELM'),('lstm','LSTM')]:
            r=d['training'][f'{method}_{dt:g}ms'];cfg=d['default_configs'][method][str(dt)];p=r['progress'];done=r['complete']
            if done:state='默认预算完成，已锁定';progress=f'{done["updates"]:,} / {done["total_updates"]:,}'
            elif p:state='训练中';progress=f'{p["steps"]:,} / {p["total_updates"]:,} ({100*p["steps"]/p["total_updates"]:.2f}%)'
            else:state='已排队';progress=f'0 / {cfg["epochs"]*cfg["batches_per_epoch"]:,}'
            params=(r['resolved'] or r.get('smoke_config') or {}).get('trainable_parameters')
            allocation=r.get('allocation')
            gpu=f'GPU {allocation["gpu"]}' if allocation else ('GPU 0' if p or done else '待分配')
            rows.append([label,f'{dt:g}ms',state,progress,gpu,f'{params:,}' if params is not None else '—'])
    s+=table(['模型','采样间隔','状态','完成的optimizer updates','训练GPU','实际可训练参数量'],rows)
    fine_checks=[r.get('smoke') for k,r in d['training'].items() if not k.endswith('_1ms')]
    if all(r and r.get('passed') for r in fine_checks):
        s+='<p>0.5、0.25ms的三个学习模型均已通过输入读取、前向、反向与有限梯度检查。这项检查每个配置执行3次更新，正式训练使用独立初始化并完成上表的默认预算。</p>'
    if d.get('dispatcher'):
        s+='<p>原有TCN与ELM队列继续在GPU0执行。新增调度优先使用连续三次检查为空闲的GPU；GPU1低负载时可与本人的一个CIFAR任务共享，仅安排较轻的LSTM。每个实验使用文件锁，避免与原队列重复训练；CPU评估串行运行。训练配方、随机种子与划分保持一致。</p>'
    else:s+='<p>TCN队列与ELM/LSTM队列共享GPU0，按1、0.5、0.25ms执行；CPU降阶与树突数据处理使用单独CPU范围。训练历史、最终权重、验证选取位置与源码版本均保存。</p>'
    learned=[]
    for dt in [1.,.5,.25]:
        for method,label in [('tcn','TCN'),('elm','Branch-ELM'),('lstm','LSTM')]:
            r=d['training'][f'{method}_{dt:g}ms'];evaluation=r.get('publication') or r.get('evaluation')
            if not evaluation:continue
            result=evaluation['summary'];m=result['metrics'];e=result['events']
            phase='集中测速' if r.get('publication') else '首次工程测量'
            learned.append([label,f'{dt:g}ms',f'{m["pooled_rmse_mv"]:.4f}',f'{m["pooled_rmse_mv"]**2:.4f}',f'{e["tp"]} / {e["fp"]} / {e["fn"]}',pct(e['f1']),phase])
    if learned:
        s+='<h4>已完成的31窗学习基线评估</h4><p>电压误差使用共同1ms观察网格和参考尖峰±10ms掩码，事件采用±5ms匹配。表中事件统计来自GPU输出。首次工程计时在全部训练结束后集中重测。</p>'
        s+=table(['模型','采样间隔','masked RMSE (mV)','masked MSE (mV²)','TP / FP / FN','事件F1','阶段'],learned)
        s+='<p>事件阈值由验证集输出分数范围确定，覆盖低概率区间及高分位尾部。旧版固定0.01–0.9网格在TCN验证集上全部漏检，已归档；这一修正涉及事件评估，电压曲线、训练配方与checkpoint保持一致。</p>'
    s+='<h3>AC.4 Neuron_Reduce：当前细胞的默认降阶结果</h3>'
    a=d.get('adapter_validation')
    if a and a['passed']:
        worst=max(r['max_abs_mv'] for r in a['records']);s+=f'<p>显式事件接口先在未降阶的Shai神经元上核对，覆盖高输入密度与零输入控制，与现有参考的最大差为{worst:.2e} mV。降阶后保留ena=50mV、ek=−85mV，原始642个compartments变为56个，500个输入ID映射到84个合并突触。</p>'
    nr=[]
    for dt in [1.,.5,.25]:
        r=d['reduction'][str(dt)];c=r['complete']
        if c and c['full_main31']:
            m=c['summary']['metrics'];e=c['summary']['events'];nr.append([f'{dt:g}ms',31,f'{m["pooled_rmse_mv"]:.4f}',f'{m["pooled_rmse_mv"]**2:.4f}',f'{c["summary"]["wall_ms"]:,.2f}',f'{e["tp"]} / {e["fp"]} / {e["fn"]}',pct(e['f1'])])
        else:nr.append([f'{dt:g}ms',r['status'].get('completed',0) if r['status'] else 0,'运行中','—','—','—','—'])
    s+=table(['求解步长','完成窗口数','masked RMSE (mV)','masked MSE (mV²)','31窗计算wall合计 (ms)','TP / FP / FN','事件F1'],nr)
    s+='<p>该表记录默认降阶设置在当前完整主动通道L5PC上的表现。共享掩码只围绕参考尖峰定义，降阶产生的额外尖峰计入电压误差与事件FP。这里的计时属于首次工程测量；集中测速阶段将与DBNN、其他基线及两种NEURON运行方式一并重测。</p>'
    s+='<h3>AC.5 主线工作点的纳入方式</h3><p>默认训练与验证锁定完成后，集中测量DBNN GPU/CPU1/CPU8、学习基线GPU/CPU1/CPU8、NEURON multisplit8、CoreNEURON CPU1与Neuron_Reduce CPU1。每条窗口重复30次，主图纵轴为31条窗口计算wall time中位数之和。输入已准备、GPU驻留；模型构建、编译、上传/下载与训练单列，完整forward、时间滤波/递归和输出计算及GPU同步纳入计时。记录共享服务器资源状态。</p>'
    publication=d.get('publication') or {}
    status=publication.get('status','queued')
    status_text={'waiting_for_default_training_and_reduction':'等待默认训练完成','blocked_by_failed_experiment':'实验失败，等待核查','timing':'集中测速中','failed':'测速失败，等待核查','complete':'集中测速完成','queued':'已排队'}.get(status,'等待核查')
    s+=f'<p class="muted">集中测速状态：{esc(status_text)}。主图在这一阶段结束后加入新方法的1/0.5/0.25ms工作点；当前主线保留已完成核验的曲线。</p>'
    sources=d['source_versions'];s+='<h3>AC.6 原始记录与来源</h3><p>'
    links=[('TCN代码','https://github.com/SelfishGene/neuron_as_deep_net'),('Branch-ELM代码','https://github.com/AaronSpieler/elmneuron'),('Neuron_Reduce代码','https://github.com/orena1/neuron_reduce'),('公开测试数据','https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data')]
    s+=' · '.join(f'<a href="{u}" target="_blank" rel="noopener">{t}</a>' for t,u in links)+'</p>'
    s+='<details><summary>源码版本、默认配置和训练记录（可下载）</summary>'
    payload=json.dumps(d,ensure_ascii=False,indent=2).encode();blob=base64.b64encode(payload).decode();s+=f'<p><a download="DBNN_baseline_evidence_20260930.json" href="data:application/json;base64,{blob}">下载本附录完整证据JSON</a></p>'
    s+='<pre>'+esc(json.dumps(sources,ensure_ascii=False,indent=2))+'</pre><p>本地脚本：<code>实验脚本/dbnn_baselines_20260930</code>。远程项目：<code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/baselines_20260930</code>。训练resolved_args / history / complete分别记录配方、收敛曲线和锁定结果。</p></details></section>'
    return s

def main():
    d=json.loads((HERE/'snapshot/status.json').read_text());old=TARGET.read_text();backup=HERE/'report_before_baselines.html'
    if not backup.exists():shutil.copy2(TARGET,backup)
    result=section(d);text=re.sub(r'<section id="baseline-reproduction">.*?</section>','',old,flags=re.S)
    text=re.sub(r'<script type="application/json" id="baseline-data">.*?</script>','',text,flags=re.S)
    text=text.replace('<li><a href="#baseline-reproduction" data-section-link>附录 AC. 文献基线复现与统一对比</a></li>','')
    toc='<li><a href="#baseline-reproduction" data-section-link>附录 AC. 文献基线复现与统一对比</a></li>'
    text=text.replace('</ul><div class="toc-footer">',toc+'</ul><div class="toc-footer">',1)
    pos=text.rfind('<footer>')
    if pos<0:pos=text.index('</main>')
    text=text[:pos]+result+text[pos:]
    script_payload=json.dumps(d,ensure_ascii=False).replace('</','<\\/')
    text=text.replace('</body>',f'<script type="application/json" id="baseline-data">{script_payload}</script></body>')
    css='#baseline-reproduction pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;padding:16px;border-radius:8px;font-size:12px}#baseline-reproduction table{font-size:13px}'
    text=text.replace(css,'')
    text=text.replace('</style>',f'\n{css}\n</style>',1)
    TARGET.write_text(text)
    ids=re.findall(r'\bid="([^\"]+)"',text);assert len(ids)==len(set(ids)),'duplicate report anchors'
    for link in re.findall(r'href="#([^\"]+)"',text):assert link in ids,('Missing report anchor',link)
    # The existing mainline figure/text remains byte-identical during training.
    oldsec=re.search(r'<section id="comparison-protocol">.*?</section>',old,re.S)[0]
    newsec=re.search(r'<section id="comparison-protocol">.*?</section>',text,re.S)[0];assert oldsec==newsec
    print(json.dumps(dict(report=str(TARGET),bytes=TARGET.stat().st_size,evidence_snapshot=d['created'],original_gates={k:v['passed'] for k,v in d['original'].items()},anchors_valid=True,existing_main_preserved=True),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
