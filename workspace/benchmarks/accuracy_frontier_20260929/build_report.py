"""Add accuracy-oriented locked configurations to the existing main frontier."""
import base64,copy,hashlib,html,json,re,shutil,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;RES=HERE/'results';FIG=HERE/'figures';S=HERE.parent
TARGET=Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN/DBNN_L5PC_surrogate_report_20260928.html')
sys.path.insert(0,str(S/'dbnn_training_20260929'))
import analyze_selected as legacy
import update_main_report as voltage_plot
from common import read,write,sha
HAS_B6=(RES/'gpu_single_kernel/complete.json').exists()
GPU_RES=RES/('gpu_single_kernel' if HAS_B6 else 'gpu')
SELECTION_PATH=GPU_RES/'primary_selection.json' if HAS_B6 else RES/'primary_selection.json'
NAMES=['original_1ms','previous_0p5ms','balanced_1ms','fresh_warm_0p5ms','fresh_scratch_0p5ms']
IDS=['B0','B1','B2','B3','B4']
LABELS=['B0 · 原 1 ms 模型','B1 · 此前 0.5 ms 适配','B2 · 1 ms 加强训练','B3 · B0 起点，0.5 ms 新数据微调','B4 · 0.5 ms 从头训练']
SHORT=['B0: previous 1 ms','B1: previous 0.5 ms','B2: balanced 1 ms','B3: B0 to 0.5 ms','B4: random init 0.5 ms']
COLORS=['#78999b','#92adb9','#176fa0','#149070','#95919a']
MARKERS=['o','o','P','D','X']
if HAS_B6:
    NAMES+=['b2_warm_0p5ms'];IDS+=['B6']
    LABELS+=['B6 · B2 起点，0.5 ms 新数据微调'];SHORT+=['B6: B2 to 0.5 ms']
    COLORS+=['#a36634'];MARKERS+=['^']
ACTIVE_LABEL='、'.join(IDS)
FRESH_LABEL='、'.join(IDS[3:])

def section(s,i):return re.search(r'<section id="'+re.escape(i)+r'">.*?</section>',s,re.S)[0]
def replace(s,i,new):return s.replace(section(s,i),new,1)
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def table(h,rows):
    e=lambda x:html.escape(str(x))
    return '<div style="overflow-x:auto"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def save(fig,name):
    FIG.mkdir(exist_ok=True);fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def image(name,caption):
    b=(FIG/(name+'.png')).read_bytes();w=int.from_bytes(b[16:20],'big');h=int.from_bytes(b[20:24],'big')
    return f'<figure><img loading="lazy" width="{w}" height="{h}" src="data:image/png;base64,{base64.b64encode(b).decode()}" alt="{html.escape(caption,quote=True)}"><figcaption>{caption}</figcaption></figure>'

def promoted(src,payload):
    name=src['runtime']['primary_config'];d=copy.deepcopy(payload['current_main']);rows={r['case_id']:r for r in src['records']};arrays={}
    for c in d['cases']:
        cid=c['case_id'];r=rows[cid]['variants'][name];pred=np.load(GPU_RES/name/(cid+'.npz'));dt=r['dt_ms']
        ref=np.load(S/'dbnn_dt_20260929/results/test/references'/(cid+'_reference.npz'))
        refv=ref['voltage'][::2].astype(float);v=pred['voltage'][::round(1/dt)].astype(float)
        c['metrics']['dbnn']=legacy.metrics(refv,v,legacy.mask_for(len(refv),c['reference_events_ms']))
        c['events']['dbnn']=dict(times_ms=pred['reset_events_ms'].tolist(),**legacy.match_events(c['reference_events_ms'],pred['reset_events_ms']))
        c['timing_ms']['dbnn']=r['wall']['median_ms'];a={'reference':refv,'dbnn':v}
        c.pop('old_checkpoint_metrics',None)
        for solver in ['neuron_0.25','neuron_0.5','neuron_1']:
            a[solver]=np.load(S/'dbnn_neuron_single_20260928/expanded_20260929/results/multisplit_highcore_results'/f'nt8_dt{solver[7:]}_{cid}.npz')['voltage'].ravel()
            c['timing_ms'][solver]=next(z for z in payload['compute_fair']['cases'] if z['case_id']==cid)['timing_ms'][solver]
        arrays[cid]=a
    d['summary']={m:legacy.aggregate([c['metrics'][m] for c in d['cases']]) for m in d['methods']}
    d['events']={m:legacy.event_sum([c['events'][m] for c in d['cases']]) for m in d['methods']}
    d['total_time_ms']={m:sum(c['timing_ms'][m] for c in d['cases']) for m in d['methods']}
    d['groups']={str(n):dict(count=sum(c['spike_class']==n for c in d['cases']),voltage={m:legacy.aggregate([c['metrics'][m] for c in d['cases'] if c['spike_class']==n]) for m in d['methods']},events=legacy.event_sum([c['events']['dbnn'] for c in d['cases'] if c['spike_class']==n])) for n in range(4)}
    d['examples']=[]
    for n in range(4):
        pool=[c for c in d['cases'] if c['spike_class']==n and c['input_events']>0]
        if n==0:pool=[c for c in pool if c['case_id'].startswith('case_')]
        best=min(pool,key=lambda c:(-c['events']['dbnn']['f1'],-c['metrics']['dbnn']['ve'],c['metrics']['dbnn']['rmse_mv']))
        e=best['events']['dbnn'];center=e['pairs_ms'][0][0] if e['pairs_ms'] else (best['reference_events_ms'][0] if n else best['duration_s']*500)
        d['examples'].append(dict(case_id=best['case_id'],spike_class=n,zoom_center_ms=center,available_cases=len(pool)))
    d['selection']=read(SELECTION_PATH);d['configuration']=name
    d['tolerance']={str(t):{m:legacy.event_sum([legacy.match_events(c['reference_events_ms'],c['events'][m]['times_ms'],t) for c in d['cases']]) for m in d['methods']} for t in [1.,2.,5.,10.]}
    d['previous_checkpoint_same_cases']=payload['current_main']['summary']['dbnn']
    d['source_sha256']={str(p):sha(p) for p in sorted((GPU_RES/name).glob('*.npz'))}
    d['example_rule']='Checkpoint selected and locked on 56 validation traces before main31. Within each reference-spike class choose best event F1, masked VE, then masked RMSE. Zero-spike examples require nonzero newly sampled inputs. All31 statistics retained.'
    d['timer']='Prepared complete GPU compute wall; no packing or transfers. NEURON timing uses solver_ms.'
    return d,arrays

def plots(src,payload):
    old=payload['compute_fair'];summary=src['summary'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,key,xlabel in zip(axes,['pooled_rmse_mv','mean_case_ve'],['Masked pooled RMSE (mV)','1 - mean per-window VE']):
        for n,label,col,mk in zip(NAMES,SHORT,COLORS,MARKERS):
            v=summary[n];x=v['metrics'][key];x=1-x if key=='mean_case_ve' else x
            ax.scatter(x,v['wall_ms'],s=95 if mk=='*' else 48,c=col,marker=mk,label='GPU '+label,zorder=5,edgecolors='white',linewidths=.5)
        for m,c,mk,label in [('cpu1','#8876ad','s','DBNN CPU 1 core (B0)'),('cpu8','#56739d','D','DBNN CPU 8 cores (B0)')]:
            x=old['summary'][m][key];x=1-x if key=='mean_case_ve' else x
            ax.scatter(x,old['totals_ms'][m],color=c,marker=mk,s=55,label=label)
        for prefix,col,mk,label in [('neuron','#c48532','^','NEURON multisplit 8 cores'),('core','#b65373','v','CoreNEURON CPU 1 core')]:
            xs=[];ys=[]
            for dt in ['0.25','0.5','1']:
                v=old['summary'][prefix+'_'+dt][key];xs.append(1-v if key=='mean_case_ve' else v);ys.append(old['totals_ms'][prefix+'_'+dt])
                if prefix=='neuron':ax.annotate(dt+' ms',(xs[-1],ys[-1]),xytext=(4,-16),textcoords='offset points',fontsize=8)
            ax.plot(xs,ys,color=col,marker=mk,ms=6,lw=1,label=label)
        ax.set(xlabel=xlabel,ylabel='Sum of 31 median compute wall times (ms)',yscale='log');ax.grid(alpha=.16)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,fontsize=8,bbox_to_anchor=(.5,.008))
    fig.suptitle('Prepared complete computation | lower-left is better | 31 windows | shared server',fontsize=11)
    fig.tight_layout(rect=[0,.19,1,.93]);save(fig,'accuracy_frontier')
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.5))
    for ax,key,xlabel in zip(axes,['pooled_rmse_mv','mean_case_ve'],['Masked pooled RMSE (mV)','1 - mean per-window VE']):
        for i,(n,col,mk) in enumerate(zip(NAMES,COLORS,MARKERS)):
            v=summary[n];x=v['metrics'][key];x=1-x if key=='mean_case_ve' else x;y=v['wall_ms']
            ax.scatter(x,y,s=140 if mk=='*' else 70,color=col,marker=mk)
            offsets=[(8,10),(-22,-18),(8,-18),(8,10),(8,10),(-22,12)]
            ax.annotate(IDS[i],(x,y),xytext=offsets[i],textcoords='offset points',fontsize=10,color=col)
        ax.set(xlabel=xlabel,ylabel='GPU compute wall sum (ms)');ax.margins(x=.22,y=.25);ax.grid(alpha=.15)
    fig.suptitle('DBNN detail | same-batch interleaved GPU timers | 30 repeats per window',fontsize=11)
    fig.legend(handles=handles[:len(NAMES)],labels=labels[:len(NAMES)],ncol=3,loc='lower center',frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,.12,1,.94]);save(fig,'accuracy_zoom')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for i,n in enumerate(NAMES[2:],2):
        hist=read(RES/n/'primary_history.json')+[{**r,'epoch':r['epoch']+read(RES/n/'protocol.json')['epochs']} for r in read(RES/n/'refine_history.json')]
        axes[0].plot([r['epoch'] for r in hist],[r['validation']['pooled_rmse_mv'] for r in hist],color=COLORS[i],label=SHORT[i])
        axes[1].plot([r['epoch'] for r in hist],[r['validation']['score'] for r in hist],color=COLORS[i],label=SHORT[i])
    axes[0].set(ylabel='Validation masked RMSE (mV)',yscale='log');axes[1].set(ylabel='Validation objective J',yscale='log')
    for ax in axes:ax.set_xlabel('Training epochs (including refinement)');ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8,frameon=False)
    fig.suptitle('Validation curves before final reset calibration | B2: 32 windows; '+', '.join(IDS[3:])+': 56 windows',fontsize=11)
    fig.tight_layout(rect=[0,.12,1,.94]);save(fig,'accuracy_training')

def new_voltage(d):
    label=LABELS[NAMES.index(d['configuration'])];cases={c['case_id']:c for c in d['cases']}
    s=('<section id="voltage-consistency"><h2>2. 电压拟合与放电匹配</h2>'
       f'<p>下面使用独立验证集选中的 <strong>{label}</strong>，在 {FRESH_LABEL} 中按同一 56 条验证输入的预设目标选择，随后锁定。所有候选保持原始单时间核、二次整合和 reset 结构。原 1 ms 模型作为 B0 保留在误差—时间图和本节历史对照中。</p>'
       '<p>所有电压指标统一在 1 ms 观察位置、参考尖峰前后各 10 ms 之外计算；0.5 ms 模型保留其原生半毫秒 reset 时刻进行事件匹配。电压不做幅度或时间调整。</p>'
       '<p>各参考 spike 类别内按事件 F1、掩码 VE、RMSE 依次选取匹配较好的例子；0-spike 示例限定非零新采样输入。所有 31 条输入均进入统计，3-spike 类仅有一条。</p>')
    for ex in d['examples']:
        c=cases[ex['case_id']];n=ex['spike_class'];m=c['metrics']['dbnn'];e=c['events']['dbnn']
        s+=f'<span id="voltage-{n}spike"></span>'+image(f'selected_voltage_{n}',f'{n}-spike 示例 {c["case_id"]}：RMSE {m["rmse_mv"]:.3f} mV，VE {pct(m["ve"])}；TP/FP/FN={e["tp"]}/{e["fp"]}/{e["fn"]}。橙色为共同的参考尖峰评分掩码；DBNN 显示原始电压及实际 reset 时刻。')
    rows=[]
    for n in range(4):
        g=d['groups'][str(n)];m=g['voltage']['dbnn'];e=g['events'];rows.append([n,g['count'],f'{m["equal_trajectory_rmse_mv"]:.4f}',pct(m['mean_case_ve']),f'{e["tp"]}/{e["fp"]}/{e["fn"]}',pct(e['precision']),pct(e['recall'])])
    s+='<h3>全部 31 个窗口的统计</h3>'+table(['参考 spikes','窗口数','等权 RMSE / mV','平均 VE','TP/FP/FN','Precision','Recall'],rows)
    m=d['summary']['dbnn'];e=d['events']['dbnn']
    s+=(f'<p>总体 pooled RMSE <strong>{m["pooled_rmse_mv"]:.4f} mV</strong>，平均 VE <strong>{pct(m["mean_case_ve"])}</strong>，方差加权 VE {pct(m["within_trace_ve"])}；保留 {m["points"]:,} / 203,031 个电压点。事件 precision / recall / F1 = {pct(e["precision"])} / {pct(e["recall"])} / {pct(e["f1"])}（±5 ms 匹配）。</p>'
        '<p>RMSE 按保留点数汇总；平均 VE 对 31 个窗口等权；方差加权 VE 的权重为各窗口保留点数乘参考方差。NEURON 事件取胞体上穿 −20 mV，DBNN 事件取实际 reset；图中保留所有额外 reset 和未匹配参考事件。</p></section>')
    return s

def main():
    src=read(GPU_RES/'complete.json');assert src['finished'] and len(src['records'])==31
    src['summary']={n:src['summary'][n] for n in NAMES}
    src['runtime']['variants']={n:src['runtime']['variants'][n] for n in NAMES}
    for r in src['records']:r['variants']={n:r['variants'][n] for n in NAMES}
    assert all(v['kernels_per_site']==1 for v in src['runtime']['variants'].values())
    assert src['runtime']['primary_config'] in NAMES
    api=read(RES/'api_validation.json');assert api['finished'] and all(r['passed'] for r in api['records'])
    for r in src['records']:
        for n in NAMES:assert r['variants'][n]['numerical']['passed'] and len(r['variants'][n]['wall']['runs_ms'])==30
    before=TARGET.read_text();back=HERE/'report_before_accuracy.html'
    if not back.exists():back.write_text(before)
    report=back.read_text();match=re.search(r'<script type="application/json" id="neuron-data">(.*?)</script>',report,re.S);payload=json.loads(match[1])
    baseline=payload['compute_fair']['summary']['gpu']
    for key in ['pooled_rmse_mv','mean_case_ve']:
        assert abs(src['summary']['original_1ms']['metrics'][key]-baseline[key])<1e-5
    d,arrays=promoted(src,payload);plots(src,payload)
    voltage_plot.FIG=FIG;voltage_plot.NAMES['dbnn']='DBNN '+SHORT[NAMES.index(d['configuration'])]
    voltage_plot.plots(d,arrays)
    primary=d['configuration'];label=LABELS[NAMES.index(primary)];win=src['summary'][primary];b0=src['summary']['original_1ms'];old=payload['compute_fair']
    # Archive the previous representative curves once, without duplicate anchors.
    archived=section(report,'voltage-consistency');body=archived[archived.index('>')+1:-len('</section>')]
    body=re.sub(r'\bid="([^"]+)"',lambda m:'id="b0-history-'+m[1]+'"',body)
    archive='<section id="accuracy-history"><h2>附录 Z. B0 电压对照与精度探索记录</h2><p>保留新增精度配置之前的 1 ms 基准结果，主图中的 CPU 点仍对应这一 B0 checkpoint。</p><details class="history-fold"><summary>展开 B0 的 0 / 1 / 2 / 3-spike 曲线与事件统计</summary>'+body+'</details></section>'
    report=report.replace('</main>',archive+'</main>',1);report=replace(report,'voltage-consistency',new_voltage(d))
    cs=section(report,'comparison-protocol');left=cs.index('<h3>3.1');right=cs.index('<span id="duration-timing">')
    head='<h3>3.1 多配置 DBNN 的速度—精度比较</h3>'
    rows=[]
    for n,label0 in zip(NAMES,LABELS):
        q=src['summary'][n];m=q['metrics'];e=q['events'];rows.append([label0,f'{m["pooled_rmse_mv"]:.4f}',pct(m['mean_case_ve']),pct(m['within_trace_weighted_ve']),pct(e['f1']),f'{q["wall_ms"]:.3f}',f'{q["device_ms"]:.3f}'])
    for m,label0 in [('cpu1','DBNN CPU 1 核 · B0'),('cpu8','DBNN CPU 8 核 · B0')]+[(p+'_'+dt,lab+' · dt='+dt+' ms') for p,lab in [('neuron','NEURON multisplit 8 核'),('core','CoreNEURON CPU 1 核')] for dt in ['0.25','0.5','1']]:
        mm=old['summary'][m];rows.append([label0,f'{mm["pooled_rmse_mv"]:.4f}',pct(mm['mean_case_ve']),'—',pct(old['events'][m]['f1']),f'{old["totals_ms"][m]:.3f}','—'])
    head+=table(['配置','RMSE / mV','平均 VE','方差加权 VE','放电 F1','完整计算 wall / ms','CUDA 设备 / ms'],rows)
    head+=f'<p>GPU {len(NAMES)} 个点在同一轮、同一张 RTX 5090 上交错计时，每窗 30 次，输入已驻留显存；均包含完整滤波、二次／线性／偏置整合和硬阈值 reset，模型结构相同。CPU 的两个点对应 B0，沿用已验证的历史计时。NEURON／CoreNEURON 对照沿用原始 solver_ms。资源均为共享服务器条件，时间为 31 窗各自中位数之和。</p>'
    head+='<span id="expanded-frontier"></span>'+image('accuracy_frontier','误差—计算时间主图：保持原 DBNN 模型结构，比较训练和分辨率配置，保留 CPU 与 NEURON 对照。越左下越好。')
    head+=image('accuracy_zoom',f'DBNN GPU 区域放大，{ACTIVE_LABEL} 与上表对应；每个点都用该配置自己的 checkpoint、真实误差与完整推理时间。')
    head+='<p class="muted">全部 31 窗参与两幅图，含两个零输入窗口和部分共享前缀。零输入时本组 DBNN 输出为常数，两个零输入窗口的 VE 约为 0；这会影响等权平均 VE。保留这一统一统计口径，另列方差加权 VE。</p>'
    cs=cs[:left]+head+cs[right:];cs=cs.replace('<h3>3.2 按模拟时长拆分计算耗时</h3>','<h3>3.2 基准 B0：按模拟时长拆分计算耗时</h3>')
    cs=cs.replace('<h3>3.3 同一输入上的计算加速倍数</h3>','<h3>3.3 基准 B0：同一输入上的计算加速倍数</h3>')
    report=replace(report,'comparison-protocol',cs)
    text='<section id="accuracy-development"><h2>3.7 本轮精度配置与训练结果</h2><p>保持原始 DBNN 模型结构，通过训练和采样分辨率提高精度。B2 从 B0 加强 1 ms 训练；B3 从 B0 使用新 0.5 ms 数据继续训练；B4 从随机权重训练。</p>'
    if HAS_B6:
        text+='<p><strong>B6 从 B2 继续训练：</strong>继承 B2 的权重及 reset 参数，使用与 B3 相同的 189 / 56 条拟合／验证输入、初始学习率 0.001 和 1000+400 轮预算。该路线在此前 B0–B4 结果之后补充，B6 的保存点和 reset 仍由验证集选择并在其 31 窗评估前锁定。</p>'
    else:
        text+='<p class="note"><strong>B6 正在补充：</strong>从 B2 checkpoint 出发，保持 B3 的数据、学习率和训练流程，继续训练 0.5 ms 模型。完成并验证后再加入主图。</p>'
    text+='<p>B5 双时间核方案已退出当前候选范围；已有实验记录保留备查，后续仅使用原模型结构。</p>'
    rows=[]
    for n,label0 in zip(NAMES[2:],LABELS[2:]):
        q=read(RES/n/'final_selection.json');p=q['protocol'];v=q['selected']['validation'];init='随机初始化' if p['random_initialization'] else ('B2' if n=='b2_warm_0p5ms' else 'B0');rows.append([label0,init,f'{p["fit_cases"]} / {p["validation_cases"]}',f'{p["epochs"]} + 400',q['selected']['name'],f'{v["pooled_rmse_mv"]:.4f}',pct(v['mean_case_ve']),pct(v['events']['f1']),f'{v["score"]:.5f}'])
    text+=table(['配置','初始化','拟合 / 验证条数','训练轮数','验证所选保存点','验证 RMSE','平均 VE','F1','选择目标 J'],rows)
    text+=('<p>新增 96 条 6 s 输入：64 / 128 / 256 / 500 个位点，10 / 25 / 50 Hz，各 8 次独立实现，含随机、分层、近端和远端位置；72 条拟合、24 条验证。输入为 0.5 ms 时间格上的泊松计数，保留同一时间格内多个事件。NEURON 仍使用完整主动通道、dt=0.1 ms、500 ms 预热，并记录 0.5 ms 目标。与原 117 / 32 条数据合并后为 189 / 56 条。</p>'
        '<p>训练更重视小幅亚阈值响应：每窗 MSE 权重为 0.3+0.7/max(参考方差, 0.04 mV²)，在拟合集上归一化，并加入 0.1 倍物理时间差分误差。先拟合驱动项，在验证集校准 reset，再重建驱动目标训练 400 轮。比较最佳验证保存点、最佳电压保存点和末期 EMA。</p>'
        '<div class="formula">J = pooled RMSE² + 0.02 mV² · meanᵢ[MSEᵢ / max(Varᵢ(vref), 0.04 mV²)] + 0.05 mV² · (1−F1<sub>±5ms</sub>)</div>'
        f'<p>{FRESH_LABEL} 共用 56 条验证输入，按 J 选中 <strong>{label}</strong>。每个候选的权重及 reset 参数在其 31 窗评估之前锁定。B2 的验证集为原 32 条，单独按其验证结果选取。</p>'
        '<h3>统一使用原模型结构</h3><div class="formula">kᵢ(t) = ωᵢ(1−exp(−t/τrise,ᵢ))exp(−t/τdecay,ᵢ)<br>yᵢ = kᵢ ∗ xᵢ；u = Σᵢ yᵢ + yᵀWy + b；v[n] = u[n] + r[n]<br>r[n+1] = exp(−Δt/τreset)r[n] − a·1[v[n]≥θ]</div>'
        '<p>各位点保持一组双指数滤波参数，使用同一二次矩阵、线性项、偏置及事件 reset。0.5 ms 配置只改变时间采样间隔，时间常数仍以物理毫秒存储。</p>')
    text+=image('accuracy_training','完整验证曲线：纵轴为最终 reset 校准前的验证指标；表格使用各保存点最终校准后的结果。不同验证集的曲线按标签区分。')
    maximum=max(r['variants'][n]['numerical']['max_abs_mv'] for r in src['records'] for n in NAMES)
    text+=f'<p>部署一致性：{len(NAMES)} 配置 × 31 窗共 {31*len(NAMES)} 次独立 FFT＋矩阵乘法＋CPU 事件递推核对全部通过，最大全曲线差 {maximum:.2e} mV，reset 时刻全部一致。训练任务结束后再进行交错计时；CPU 打包、H2D、D2H 均未进入主图 GPU 计算时间。</p>'
    text+=f'<p>推理接口此前另通过 {len(api["records"])} 组检查，覆盖原生 0.5 ms 输入、整数毫秒输入、更换 CUDA Graph 输入及零输入。物理时间常数只转换一次；最大电压差 {api["maximum_error_mv"]:.2e} mV。当前部署 checkpoint 使用上面的逐窗独立核对。</p>'
    text+='<details><summary>公共 0.5 ms 观察网格的补充核对</summary>'
    text+=table(['配置','RMSE / mV','平均 VE','方差加权 VE'],[[label0,f'{src["summary"][n]["fine_grid_metrics"]["pooled_rmse_mv"]:.4f}',pct(src['summary'][n]['fine_grid_metrics']['mean_case_ve']),pct(src['summary'][n]['fine_grid_metrics']['within_trace_weighted_ve'])] for n,label0 in zip(NAMES,LABELS)])
    text+='<p>0.5 ms 模型使用原生输出，1 ms 模型在半毫秒位置做线性插值；参考来自 NEURON 0.1 ms 求解轨迹。评分仍剔除同一物理 ±10 ms 尖峰窗口。</p></details>'
    selected_file=TARGET.parent/'DBNN_accuracy_selected_20260929.pt'
    source_file=RES/primary/'best_checkpoint.pt'
    assert sha(source_file)==src['runtime']['variants'][primary]['checkpoint_sha256']
    shutil.copy2(source_file,selected_file)
    text+=f'<p><a href="{selected_file.name}" download>下载本轮验证选中的主线 checkpoint：{label}</a>。原 B0 checkpoint 文件保持原样。</p>'
    text+='<details><summary>Checkpoint 与全部逐窗数值</summary>'
    text+=table(['配置','checkpoint SHA-256','下载'],[[LABELS[i],src['runtime']['variants'][n]['checkpoint_sha256'],'保留在 accuracy_frontier_20260929 结果目录'] for i,n in enumerate(NAMES)])
    rr=[]
    for r in src['records']:
        for n,label0 in zip(NAMES,LABELS):
            q=r['variants'][n];rr.append([r['case_id'],label0,r['duration_s'],r['A'],f'{q["metrics"]["rmse_mv"]:.4f}',pct(q['metrics']['ve']),f'{q["events"]["tp"]}/{q["events"]["fp"]}/{q["events"]["fn"]}',f'{q["wall"]["median_ms"]:.4f}'])
    text+=table(['窗口','配置','T / s','A','RMSE','VE','TP/FP/FN','计算 wall / ms'],rr)+'</details></section>'
    ds=section(report,'dt-resolution')
    dsnew=ds.replace('<h2>3.6 DBNN 从 1 ms 提高到 0.5 ms：精度与速度</h2>',
        '<h2>附录 Z.1 前期分辨率对照：从 B0 的 1 ms 到 0.5 ms</h2><p class="note">本节保留此前 200 轮适配研究，对应主图 B0 / B1；最新训练配置见第 3.6 节。</p>')
    dsnew=dsnew.replace('当前主线 checkpoint 与前面各节的锁定结果保留，本节单列分辨率探索。','原 1 ms checkpoint 以 B0 保留，本节单列此前分辨率探索。')
    report=report.replace(ds,text,1)
    report=report.replace(archive,archive+dsnew,1)
    newcon=(f'<section id="conclusion"><span id="backup-neuron-main" class="expanded-anchor"></span><h2>主线：DBNN 的电压精度与计算速度</h2>'
        f'<p>主图现有 <strong>{len(NAMES)} 个相同模型结构的 DBNN GPU 配置</strong>，同时保留基准 CPU 和 NEURON／CoreNEURON 对照。本轮按 56 条独立验证输入选中的配置为 <strong>{label}</strong>；在固定 31 窗上，参考尖峰 ±10 ms 外 pooled RMSE 为 <strong>{win["metrics"]["pooled_rmse_mv"]:.4f} mV</strong>，平均 VE 为 <strong>{pct(win["metrics"]["mean_case_ve"])}</strong>，方差加权 VE 为 {pct(win["metrics"]["within_trace_weighted_ve"])}。</p>'
        f'<p>该配置的 31 窗完整 GPU 计算 wall 中位数之和为 <strong>{win["wall_ms"]:.3f} ms</strong>，CUDA 设备时间为 {win["device_ms"]:.3f} ms。事件 precision / recall / F1 = {pct(win["events"]["precision"])} / {pct(win["events"]["recall"])} / {pct(win["events"]["f1"])}。原 1 ms 基准 B0 在本轮同批重测为 {b0["wall_ms"]:.3f} ms、RMSE {b0["metrics"]["pooled_rmse_mv"]:.4f} mV。</p>'
        '<p>固定同一个完整主动 L5PC、相同 31 窗输入与参考电压，计时使用准备后的完整计算，排除 CPU 打包和传输。电压与事件例子见第 2 节，速度—精度主图见第 3.1 节，新训练配置见第 3.7 节；历史研究单独保留。</p></section>')
    report=replace(report,'conclusion',newcon)
    setup=section(report,'neuron-comparison')
    setup=setup.replace('验证选定后锁定的新 checkpoint；每次 1 条输入，原生输出间隔 1 ms','各配置独立验证选择后锁定；每次 1 条输入，原生输出间隔 1 或 0.5 ms')
    setup=setup.replace('同一 checkpoint，单条输入；1/2/4/8 个物理核心','基准 B0 checkpoint，单条输入；1/2/4/8 个物理核心')
    setup=setup.replace('拟合与验证均采用相同的 1 ms 量化事件回放目标。扩展训练共 117 条拟合与 32 条验证输入；具体选择和各候选见','B0 / B2 使用原 117 条拟合、32 条验证输入及 1 ms 事件。新 0.5 ms 配置扩展为 189 / 56 条，见第 3.7 节。B0 的历史选择见')
    setup=setup.replace('</section>','<p><strong>本轮更新：</strong>主线代表曲线采用第 3.7 节验证选择的新配置，旧 1 ms checkpoint 标为 B0；GPU 各配置在同批重新测量。新增训练输入与 31 窗主线输入的事件哈希互不相同。</p></section>')
    report=replace(report,'neuron-comparison',setup)
    ts=section(report,'training-selection');ts=ts.replace('<h2>附录 A.', '<h2>附录 A.',1).replace('</h2>','</h2><p class="note">本节是 B0 及此前 checkpoint 的训练依据；最新精度配置和选择见第 3.7 节。</p>',1);report=replace(report,'training-selection',ts)
    report=replace(report,'next',f'<section id="next"><h2>4. 当前比较范围</h2><p>主图展示 {len(NAMES)} 个保持原模型结构、经过独立实现核对的 GPU 配置，全部在同一 31 窗输入上使用相同误差定义。{FRESH_LABEL} 的主线代表配置由共同的 56 条验证输入选定，各保存点在其 31 窗评估前锁定。GPU、CPU 和 NEURON 的时间均按各自完整计算调用记录；GPU 各点已交错重测，CPU 与求解器保留原始已验证测量批次。</p><p>输入覆盖 1–16 s、64–500 个活跃位点和 0–3 次参考尖峰，其中部分窗口共享前缀。全部窗口及未匹配事件保留在统计中；展示曲线按第 2 节公开规则选择。后续研究保持单时间核、二次整合和 reset 结构。</p></section>')
    next_link=re.search(r'<li><a href="#next" data-section-link>.*?</a></li>',report)[0]
    report=report.replace(next_link,'<li><a href="#accuracy-development" data-section-link>3.7 精度配置与新训练</a></li>'+next_link,1)
    marker='<li><a href="#timing-legacy" data-section-link>附录 Y. 旧端到端计时与线程诊断</a></li>'
    report=report.replace(marker,marker+'<li><a href="#accuracy-history" data-section-link>附录 Z. B0 电压对照</a></li>',1)
    report=re.sub(r'<li><a href="#dt-resolution" data-section-link>.*?</a></li>','',report,count=1)
    marker='<li><a href="#accuracy-history" data-section-link>附录 Z. B0 电压对照</a></li>'
    report=report.replace(marker,marker+'<li><a href="#dt-resolution" data-section-link>附录 Z.1 前期 0.5 ms 适配</a></li>',1)
    report=report.replace('第 3.7 节','第 3.6 节').replace('3.7 本轮精度','3.6 本轮精度').replace('3.7 精度配置','3.6 精度配置')
    payload['accuracy_frontier']=dict(**src,promoted=d,primary_selection=read(SELECTION_PATH),training={n:read(RES/n/'final_selection.json') for n in NAMES[2:]},api_validation=api,architecture_constraint='Original single temporal kernel per site, same quadratic integration and reset; B5 excluded at user request.',active_ids=IDS)
    payload['baseline_main_B0']=payload['current_main']
    payload['current_main']=d
    payload['active_compute_result']='accuracy_frontier'
    m=re.search(r'(<script type="application/json" id="neuron-data">)(.*?)(</script>)',report,re.S)
    report=report[:m.start()]+m[1]+json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+m[3]+report[m.end():]
    report=re.sub(r'<footer>.*?</footer>',f'<footer>更新于 {datetime.now():%Y-%m-%d %H:%M} · 保持原模型结构的 {len(NAMES)} 配置误差—计算时间比较；全部电压评分使用统一参考尖峰 ±10 ms 掩码。</footer>',report,count=1,flags=re.S)
    footer=re.search(r'<footer>.*?</footer>',report,re.S)[0]
    report=report.replace(footer,'',1).replace('</main>',footer+'</main>',1)
    ids=re.findall(r'\bid="([^"]+)"',report);assert len(ids)==len(set(ids));assert all(x in ids for x in re.findall(r'href="#([^"]+)"',report))
    write(RES/'analysis.json',dict(summary=src['summary'],primary=primary,promoted=d,numerical_max_abs_mv=maximum))
    tmp=TARGET.with_suffix('.tmp');tmp.write_text(report);tmp.replace(TARGET)
    write(RES/'artifact_qa.json',dict(report=str(TARGET),sha256=sha(TARGET),anchors_valid=True,browser_verified=False,models=len(NAMES),main_windows=31,numerical_checks=31*len(NAMES)))
    print(json.dumps(dict(report=str(TARGET),primary=primary,summary=src['summary']),ensure_ascii=False))

if __name__=='__main__':main()
