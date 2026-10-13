"""Publish completed cross-region results into Appendix AD of the existing report."""
import argparse
import base64
import html
import json
import re
from collections import Counter
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError:
    plt = None

HERE = Path(__file__).resolve().parent
FIG = HERE/'figures'
TARGET = Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN/DBNN_L5PC_surrogate_report_20260928.html')
ORDER = ['granule','basket','golgi','hay_l5','human_l23','ca1']
ZH = {'granule':'小脑颗粒细胞','basket':'小脑篮状细胞','golgi':'小脑Golgi细胞',
      'hay_l5':'大鼠皮层L5b','human_l23':'人皮层L2/3','ca1':'海马CA1锥体细胞'}
EN = {'granule':'Granule · cerebellum','basket':'Basket · cerebellum','golgi':'Golgi · cerebellum',
      'hay_l5':'L5b · rat cortex','human_l23':'L2/3 · human cortex','ca1':'CA1 · hippocampus'}
PROFILE = {'ampa':'AMPA','ampa_nmda':'AMPA + NMDA'}
COL = {'gpu':'#008675','multisplit':'#ce8730','core':'#b75170'}
MARK = {1.:'o',.5:'D',.25:'^',.1:'s',.05:'v'}


def esc(value): return html.escape(str(value))
def pct(value): return f'{100*value:.2f}%'
def table(headers, rows):
    return '<div class="resolution-table"><table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def image(stem, caption):
    blob=(FIG/(stem+'.png')).read_bytes()
    return '<figure><img loading="lazy" width="'+str(int.from_bytes(blob[16:20],'big'))+'" height="'+str(int.from_bytes(blob[20:24],'big'))+'" src="data:image/png;base64,'+base64.b64encode(blob).decode()+'" alt="'+esc(caption)+'"><figcaption>'+esc(caption)+'</figcaption></figure>'
def save_figure(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/(stem+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(FIG/(stem+'.svg'),bbox_inches='tight')
    plt.close(fig)


def plots(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(12.1,5.4))
    offsets={1.:-.14,.5:0.,.25:.14}
    for dt in [1.,.5,.25]:
        values=[data['models'][name]['profiles']['ampa']['gpu']['summary'][str(dt)]['metrics']['pooled_rmse_mv'] for name in ORDER]
        ax.scatter([i+offsets[dt] for i in range(6)],values,s=88,marker=MARK[dt],color=COL['gpu'],edgecolor='white',label=f'AMPA · {dt:g} ms',zorder=4)
    nmda=[data['models'][name]['profiles']['ampa_nmda']['gpu']['summary']['1.0']['metrics']['pooled_rmse_mv'] for name in ORDER]
    ax.scatter([i+.29 for i in range(6)],nmda,s=88,marker='o',color='#537dac',edgecolor='white',label='AMPA + NMDA · 1 ms',zorder=4)
    ax.set_xticks(range(6),['Granule\ncerebellum','Basket\ncerebellum','Golgi\ncerebellum','L5b\nrat cortex','L2/3\nhuman cortex','CA1\nhippocampus'])
    ax.set(ylabel='Masked pooled RMSE (mV)',ylim=(0,.49),xlim=(-.5,5.6))
    ax.grid(axis='y',color='#e1e7ed');ax.set_axisbelow(True)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False,fontsize=9)
    fig.suptitle('DBNN across brain regions and cell types',fontsize=16,weight='bold',color='#20364a')
    ax.set_title('31 reference-selected windows per condition · full active channels · reference dt = 0.025 ms',fontsize=10,color='#667888',pad=15)
    fig.tight_layout(rect=[0,.08,1,.94]);save_figure(fig,'DBNN_multineuron_accuracy_20261004')
    for profile in ['ampa','ampa_nmda']:
        fig,axes=plt.subplots(2,3,figsize=(13,9.1))
        for ax,name in zip(axes.flat,ORDER):
            model=data['models'][name];condition=model['profiles'][profile]
            all_points=[]
            g=condition['gpu']
            points=[(float(dt),s['metrics']['pooled_rmse_mv'],s['wall_ms']) for dt,s in g['summary'].items()]
            for method in ['gpu','multisplit','core']:
                current=points if method=='gpu' else [(s['runtime']['dt_ms'],s['metrics']['pooled_rmse_mv'],s['wall_ms']) for s in condition['solvers'].values() if s['runtime']['engine']==method]
                current=sorted(current,key=lambda v:-v[0])
                if not current:continue
                all_points.extend(current)
                ax.plot([max(1e-7,v[1]) for v in current],[v[2] for v in current],color=COL[method],lw=1.8,ls='-' if method=='gpu' else '--')
                for dt,rmse,wall in current:
                    ax.scatter(max(1e-7,rmse),wall,marker=MARK[dt],s=58,color=COL[method],edgecolor='white',linewidth=.8,zorder=4)
            xs=[max(1e-7,v[1]) for v in all_points];ys=[v[2] for v in all_points]
            ax.set(xscale='log',yscale='log',xlim=(min(xs)/1.5,max(xs)*1.7),ylim=(min(ys)/2,max(ys)*2),xlabel='Masked pooled RMSE (mV)',ylabel='31-window compute wall sum (ms)')
            ax.grid(which='major',color='#e7ecf1');ax.set_axisbelow(True)
            ax.set_title(EN[name],weight='bold',color='#253d4b',pad=10)
        handles=[Line2D([0],[0],color=COL[m],lw=2,label=t) for m,t in [('gpu','DBNN GPU'),('multisplit','NEURON multisplit 8 cores'),('core','CoreNEURON CPU 1 core')]]
        handles += [Line2D([0],[0],color='#5f707e',marker=MARK[dt],ls='none',label=f'{dt:g} ms') for dt in [1.,.5,.25,.1,.05]]
        fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,fontsize=9.2)
        fig.suptitle('Cross-cell accuracy–runtime trade-off · '+PROFILE[profile],fontsize=16,weight='bold',color='#20364a')
        fig.text(.5,.940,'31 paired windows per cell · reference spikes ±10 ms excluded · lower-left is better',ha='center',fontsize=10,color='#637886')
        fig.text(.5,.005,'Prepared complete compute · shared server · GPU 30 repeats, solvers 5 repeats · only valid solver configurations',ha='center',fontsize=9,color='#71808d')
        fig.tight_layout(rect=[0,.15,1,.92]);save_figure(fig,'DBNN_multineuron_frontier_'+profile+'_20261004')


def section(data):
    models=data['models']
    valid=sum(len(q['solvers']) for r in models.values() for q in r['profiles'].values())
    invalid=sum(len(q['invalid']) for r in models.values() for q in r['profiles'].values())
    s='<section id="multineuron-generalization"><h2>附录 AD. 不同脑区与细胞的完整推广结果</h2>'
    s+=f'<p class="muted">核验时间：{esc(data["collected"])}。六个细胞的计划训练和31窗评估已完成：24个DBNN工作点、{valid}组有效求解器配置；另有{invalid}个MultiSplit配置保留无效记录。所有结果来自已保存的checkpoint、逐窗输出和计时JSON。</p>'
    s+='<div class="note"><strong>当前结果的适用范围。</strong>颗粒细胞AMPA和CA1 AMPA具有较好的电压拟合与尖峰检测；人L2/3的AMPA+NMDA在31窗内匹配了全部11个参考尖峰。Golgi两种条件均漏掉全部参考尖峰，篮状细胞的尖峰检测也偏弱。六种细胞的成功与困难情形均在下表保留。</div>'
    s+='<h3>AD.1 脑区、细胞与原模型</h3>'
    rows=[]
    for name in ORDER:
        r=models[name];spec=r['spec'];op=r['operating_point'];c=r['coverage'] or {}
        rows.append([ZH[name],spec['author'],spec['modeldb'],c.get('site_count',500),c.get('unique_electrical_segments','—'),spec['celsius'],op.get('reference_dt_ms',.025),op.get('holding_current_na','—')])
    s+=table(['细胞','原模型作者','ModelDB','输入位点','覆盖电学段','温度 / °C','参考dt / ms','保持电流 / nA'],rows)
    s+='<p>各细胞保留原模型形态、nseg与完整主动离子通道，在树突长度上分层布置250个兴奋和250个抑制位置。位置索引与电学段分别计数；多个位置可能落在同一电学段。存在自发放电的模型使用固定保持电流建立低放电工作区间，同一细胞的DBNN和两种求解器共享输入、突触参数及保持电流。</p>'
    s+='<p>原模型源码：'+' · '.join(f'<a href="https://modeldb.science/{models[n]["spec"]["modeldb"]}" target="_blank" rel="noopener">{esc(ZH[n])} / {models[n]["spec"]["modeldb"]}</a>' for n in ORDER)+'。</p>'
    s+='<h3>AD.2 电压拟合的总览</h3>'+image('DBNN_multineuron_accuracy_20261004','每种条件在自身预定的31个参考窗口上评分。AMPA包含1／0.5／0.25 ms，AMPA+NMDA完成1 ms；RMSE在共同1 ms观察网格上计算并剔除参考尖峰±10 ms。')
    rows=[]
    for name in ORDER:
        for profile in ['ampa','ampa_nmda']:
            for dt,z in models[name]['profiles'][profile]['gpu']['summary'].items():
                m=z['metrics'];e=z['events']
                rows.append([ZH[name],PROFILE[profile],f'{float(dt):g} ms',f'{m["pooled_rmse_mv"]:.4f}',f'{m["pooled_rmse_mv"]**2:.5f}',e['tp']+e['fn'],e['tp']+e['fp'],f'{e["tp"]}/{e["fp"]}/{e["fn"]}',pct(e['precision']),pct(e['recall']),pct(e['f1']),f'{z["wall_ms"]:.3f}'])
    s+=table(['细胞','突触条件','采样间隔','RMSE / mV','MSE / mV²','参考尖峰','预测尖峰','TP/FP/FN','Precision','Recall','F1','GPU计算wall / ms'],rows)
    s+='<p>每行先合计该31窗的TP、FP、FN，再算F1 = 2TP / (2TP + FP + FN)。参考事件取NEURON胞体电压向上越过−20 mV的时刻，预测事件取DBNN reset触发时刻；同一窗口内按时间顺序、±5 ms容差一一匹配。无尖峰窗口中的额外预测计入FP。±10 ms掩码仅作用于电压误差。</p>'
    s+='<h3>AD.3 尖峰数量与输入覆盖</h3>'
    rows=[]
    for name in ORDER:
        for profile in ['ampa','ampa_nmda']:
            q=models[name]['profiles'][profile];records=q['gpu']['records'];counts=Counter(r['variants']['1.0']['events']['tp']+r['variants']['1.0']['events']['fn'] for r in records)
            manifest=q['manifest']['cases'];durations=Counter(float(r['duration_ms'])/1000 for r in manifest)
            event=q['gpu']['summary']['1.0']['events']
            rows.append([ZH[name],PROFILE[profile],*[counts[i] for i in range(4)],sum(i*n for i,n in counts.items()),', '.join(f'{t:g}s:{n}' for t,n in sorted(durations.items())),f'{event["median_abs_error_ms"]:.3f}' if event['median_abs_error_ms'] is not None else '无匹配事件'])
    s+=table(['细胞','条件','0 spike窗','1 spike窗','2 spike窗','3 spike窗','参考尖峰合计','各时长窗口数','1 ms匹配时间误差中位数 / ms'],rows)
    s+='<p>AMPA与AMPA+NMDA分别依据参考结果选择31窗，合计372个条件窗口、219个参考尖峰。候选输入覆盖1、2、4、8、16 s，64／128／256／500活跃位点，10／25／50 Hz，以及随机、分层、近端和远端位置。每窗限制0–3个参考尖峰，最小ISI≥20 ms；DBNN预测误差不参与窗口筛选。</p>'
    s+='<h3>AD.4 各细胞的精度—速度关系</h3>'
    for profile in ['ampa','ampa_nmda']:
        s+=image('DBNN_multineuron_frontier_'+profile+'_20261004',PROFILE[profile]+'：每个面板内的方法使用同一31窗。各点表示实际采样间隔／求解步长；DBNN GPU每窗30次，求解器每窗5次，均先取中位数再对31窗求和。')
    s+='<details><summary>展开全部有效求解器工作点与计时范围</summary>'
    rows=[]
    for name in ORDER:
        for profile,q in models[name]['profiles'].items():
            for key,z in q['solvers'].items():
                m=z['metrics'];e=z['events'];rt=z['runtime'];label='NEURON multisplit 8核' if rt['engine']=='multisplit' else 'CoreNEURON CPU 1核'
                rows.append([ZH[name],PROFILE[profile],label,f'{rt["dt_ms"]:g} ms',len(z['records']),f'{m["pooled_rmse_mv"]:.4f}',f'{z["wall_ms"]:,.2f}',f'{e["tp"]}/{e["fp"]}/{e["fn"]}',pct(e['f1'])])
    s+=table(['细胞','条件','方法','求解步长','窗口数','RMSE / mV','计算wall / ms','TP/FP/FN','尖峰F1'],rows)
    s+='<p>完整计算计入滤波／驱动／reset或数值求解调用及其同步。形态构建、编译、模型加载、事件准备、外部输入打包、上传／下载、预热和外部插值在计时外完成；求解器内部的必要数据迁移保留。GPU为RTX 5090，CPU为AMD EPYC 9554；所有记录来自共享服务器。有效配置的重复次数、CPU亲和性、输入哈希和资源状态随下载JSON保存。</p></details>'
    s+='<h3>AD.5 训练、参考与突触条件</h3>'
    s+='<p>每个细胞独立从弱单点响应初始化，沿用一个双指数核、零对角二次矩阵、线性项、偏置和硬阈值reset。AMPA按1 → 0.5 → 0.25 ms继续训练；AMPA+NMDA独立训练1 ms。采用log-τ、加权masked MSE + 0.1倍物理时间导数MSE，AdamW、cosine、梯度裁剪和EMA；主阶段1000轮，验证集校准reset后重建目标，再以0.2倍学习率细化400轮。保存点及reset参数由对应验证集锁定。</p>'
    s+=table(['条件','兴奋性输入','抑制性输入','完成的DBNN间隔'],[
        ['AMPA','Exp2Syn：E=0 mV，τrise=0.5 ms、τdecay=5 ms','Exp2Syn：E=−75 mV，τrise=1 ms、τdecay=10 ms','1／0.5／0.25 ms'],
        ['AMPA + NMDA','同上AMPA，加Mg阻塞双指数NMDA，τrise=2 ms、τdecay=80 ms；峰值电导比0.5','同上','1 ms']])
    s+='<p>ExpSyn／Exp2Syn描述电导随时间的波形，AMPA／NMDA描述受体类型。先前的ExpSyn用于单指数输入的二次效应初筛；完整训练评估采用上表的双指数AMPA及AMPA+NMDA条件。NMDA的Mg阻塞因子为1 / [1 + Mg·exp(−0.062V) / 3.57]，Mg=1。</p>'
    s+='<details><summary>参考步长核验、选定权重与VE诊断</summary><p>六种细胞均使用0.025 ms参考。篮状细胞的步长重放记录及新增皮层／海马模型的预定收敛检查随证据保存；参考步长在训练前确定。L5PC主线继续使用自身0.1 ms参考。</p>'
    rows=[]
    for name in ORDER:
        for profile,q in models[name]['profiles'].items():
            for dt,z in q['training'].items():
                sel=z['selected'];v=sel['validation'];m=q['gpu']['summary'][dt]['metrics']
                rows.append([ZH[name],PROFILE[profile],f'{float(dt):g} ms',sel['name'],sel['epoch'],f'{v["pooled_rmse_mv"]:.4f}',pct(m['mean_case_ve']),pct(m['within_trace_weighted_ve']),z['checkpoint_sha256']])
    s+=table(['细胞','条件','间隔','验证选定候选','阶段内轮数','验证RMSE','测试平均VE','测试方差加权VE','checkpoint SHA-256'],rows)+'</details>'
    s+='<h3>AD.6 已完成范围与无效配置</h3>'
    s+=f'<p>六个细胞各完成AMPA三档和AMPA+NMDA一档训练、验证锁定与31窗GPU评估；{valid}组求解器配置完成各31窗、每窗5次测速。以下{invalid}个MultiSplit配置触发最小事件延迟限制，未生成有效工作点，图中仅绘制有效结果。</p>'
    rows=[]
    for name in ORDER:
        for profile,q in models[name]['profiles'].items():
            for key,z in q['invalid'].items():rows.append([ZH[name],PROFILE[profile],f'{z["runtime"]["dt_ms"]:g} ms','最小可用事件延迟为0或小于固定求解步长',z.get('case_id','—')])
    s+=table(['细胞','条件','MultiSplit步长','错误原因','停止窗口'],rows)
    blob=base64.b64encode(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()).decode()
    s+='<h3>AD.7 原始结果与复查入口</h3><p><a download="DBNN_multineuron_final_20261004.json" href="data:application/json;base64,'+blob+'">下载六种细胞的完整结果、训练选择与来源哈希JSON</a>。逐窗电压、驱动与reset事件保存在各项目的results／细胞／条件／gpu目录；求解器输出保存在对应solvers目录，训练权重保存在dt*ms／best_checkpoint.pt。</p>'
    s+='<p>远程项目：<code>/home/lcm/DBNN_multineuron_20261002</code>与<code>/home/lcm/DBNN_noncerebellar_20261002</code>；原始参考和输入分别保存在<code>/data/lcm/DBNN_multineuron_20261002</code>、<code>/data/lcm/DBNN_noncerebellar_20261002</code>。</p></section>'
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true');p.add_argument('--merge-only',action='store_true');a=p.parse_args()
    data=json.loads((HERE/'snapshot/multineuron_final.json').read_text())
    rendered=HERE/'snapshot/multineuron_final_section.html'
    if not a.merge_only:
        assert plt is not None,'Render with the remote scientific runtime.'
        plots(data);rendered.write_text(section(data))
    if a.render_only:
        print(json.dumps(dict(rendered=str(rendered),models=list(data['models'])),ensure_ascii=False));return
    body=rendered.read_text();assert data['collected'] in body
    text=TARGET.read_text();old=re.search(r'<section id="multineuron-generalization">.*?</section>',text,re.S)[0]
    # Preserve the existing initial quadratic-screen evidence as a folded record.
    if '<h3>AD.5 二次效应' in old:
        pilot=old[old.index('<h3>AD.5 二次效应'):old.index('<h3>AD.6 ',old.index('<h3>AD.5 二次效应'))]
        pilot=re.sub(r'<h3>.*?</h3>','',pilot,count=1)
        body=body.replace('<h3>AD.7 原始结果', '<details><summary>早期小脑三种细胞的二次效应初筛记录</summary>'+pilot+'</details><h3>AD.7 原始结果',1)
    text=text.replace(old,'',1)
    insertion=text.rfind('<footer>');assert insertion>0
    text=text[:insertion]+body+text[insertion:]
    toc_pattern=r'<li><a href="#multineuron-generalization" data-section-link>.*?</a></li>'
    text=re.sub(toc_pattern,'',text)
    toc='<li><a href="#multineuron-generalization" data-section-link>附录 AD. 不同脑区与细胞的完整结果</a></li>'
    text=text.replace('</ul><div class="toc-footer">',toc+'</ul><div class="toc-footer">',1)
    ids=re.findall(r'\bid="([^"]+)"',text);assert len(ids)==len(set(ids))
    assert all(anchor in ids for anchor in re.findall(r'href="#([^"]+)"',text))
    TARGET.write_text(text)
    print(json.dumps(dict(report=str(TARGET),collected=data['collected'],models=len(data['models']),anchors_valid=True),ensure_ascii=False))


if __name__=='__main__':main()
