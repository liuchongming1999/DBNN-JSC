"""Evidence-only appendix and standalone report for the multineuron campaign."""
import argparse,base64,html,io,json,re,shutil
from pathlib import Path
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt=None

NAMES={'granule':'颗粒细胞','golgi':'Golgi 细胞','basket':'篮细胞'}
ENGLISH={'granule':'Granule','golgi':'Golgi','basket':'Basket'}
MODELDB={'granule':265584,'golgi':266806,'basket':2018018}
PROFILES={'article_expsyn':'文中 ExpSyn','ampa':'AMPA','ampa_nmda':'AMPA + NMDA'}
def esc(x):return html.escape(str(x))
def table(headers,rows):
    return '<div class="mn-table"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def figure(fig,alt):
    bio=io.BytesIO();fig.savefig(bio,format='png',dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
    return '<figure><img alt="'+esc(alt)+'" src="data:image/png;base64,'+base64.b64encode(bio.getvalue()).decode()+'"></figure>'
def fmt(v,n=3):return '待完成' if v is None else f'{v:.{n}f}'
def sampling_text(x):
    if x['complete']:return '采样完成'
    p=x['progress'];return f"参考 {p['completed']}/{p['total']}" if p else '待采样'

def pilot_plot(d):
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=False)
    colors=['#9AA7AF','#00897B','#D08B43'];labels=['Linear','Quadratic','Cubic']
    for ax,(profile,label) in zip(axes,PROFILES.items()):
        available=[]
        for k in NAMES:
            r=d['models'][k]['screens'][profile]
            if r['complete']:available.append((k,r['result']['summary']))
        for j,name in enumerate(['linear','quadratic','cubic']):
            ax.bar(np.arange(len(available))+(j-1)*.23,[v[name]['mean_heldout_rmse_mv'] for k,v in available],.21,color=colors[j],label=labels[j])
        ax.set_xticks(range(len(available)),[ENGLISH[k] for k,v in available]);ax.set_title({'article_expsyn':'Article ExpSyn','ampa':'AMPA','ampa_nmda':'AMPA + NMDA'}[profile],fontsize=12)
        ymax=max([v[name]['mean_heldout_rmse_mv'] for k,v in available for name in ['linear','quadratic','cubic']],default=1.)
        ax.set_ylim(0,max(ymax*1.18,1e-8));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if not available:ax.text(.5,.5,'Reference pilot in progress',ha='center',va='center',transform=ax.transAxes,color='#607080');ax.set_xticks([])
    axes[0].set_ylabel('Mean held-out RMSE (mV)');axes[-1].legend(frameon=False,fontsize=9)
    fig.suptitle('Paired-input adequacy pilot · 8 pairs · 4 × 4 strengths · per-panel y scales',fontsize=12,y=1.03)
    fig.tight_layout()
    return figure(fig,'预定突触对初筛：线性、二次和三次近似的留出误差。仅画完成的条件。')

def frontier_plot(model,r,profile):
    gpu=r['gpu'].get(profile);solvers=r['solvers'].get(profile,{})
    if not gpu:return ''
    fig,axes=plt.subplots(1,2,figsize=(8,6.8),sharey=True)
    styles=[('gpu','DBNN GPU','#00897B'),('multisplit','NEURON multisplit · 8 cores','#D28B36'),('core','CoreNEURON · 1 core','#B95271')]
    for mode,label,color in styles:
        if mode=='gpu':points=[(float(dt),x['metrics']['pooled_rmse_mv'],x['wall_ms']) for dt,x in gpu['summary'].items()]
        else:points=[(x['runtime']['dt_ms'],x['metrics']['pooled_rmse_mv'],x['wall_ms']) for key,x in solvers.items() if key.startswith(mode+'_') and x.get('finished') and len(x['records'])==31]
        points=sorted(points,reverse=True)
        for ax,power in zip(axes,[1,2]):
            if points:
                ax.plot([x[1]**power for x in points],[x[2] for x in points],color=color,lw=2,label=label,zorder=3)
                for dt,err,t in points:
                    ax.scatter(err**power,t,s=55,color=color,marker={1.:'o',.5:'D',.25:'^',.1:'s',.05:'v'}[dt],edgecolor='white',lw=.8,zorder=4)
            ax.set_yscale('log');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True)
    axes[0].set_xlabel('Masked pooled RMSE (mV)');axes[1].set_xlabel('Masked pooled MSE (mV²)');axes[0].set_ylabel('Sum of 31 median compute wall times (ms)')
    axes[1].legend(frameon=False,loc='upper right',fontsize=8)
    fig.suptitle(f'{ENGLISH[model]} · {PROFILES[profile]} · lower-left is better',fontsize=12)
    return figure(fig,f'{NAMES[model]} {PROFILES[profile]} 的精度与计算耗时。只纳入完整31窗结果。')

def section(d):
    models=d['models'];s='<section id="multineuron-generalization"><h2>附录 AD. 多神经元推广与 AMPA / NMDA 敏感性</h2>'
    s+=f'<p class="muted">实测记录时间：{esc(d["collected"])}。训练、完整测试与测速分别记录；下表只将已完成的测量作为结果。</p>'
    s+='<h3>AD.1 模型选择与研究范围</h3><p>依据 Ma et al. (2026) <em>A hierarchical organization of synaptic integration complexity across cerebellar neuron types</em>，优先测试二次效应较合适的篮细胞、Golgi 细胞，并加入以近线性整合为主的颗粒细胞。每个模型独立拟合500位点的DBNN，使用同一架构、训练配方和推理后端。</p>'
    rows=[]
    for k in NAMES:
        r=models[k];op=r['operating_point'] or {};spec=op.get('spec',{});c=r['coverage'] or {}
        variant=spec.get('variant','') or ('Basket / morphology/01bc.ASC' if k=='basket' else '')
        rows.append([NAMES[k],f'<a href="https://modeldb.science/{MODELDB[k]}" target="_blank" rel="noopener">{MODELDB[k]}</a>',esc(spec.get('article_label',''))+' / '+esc(variant),
                     str(c.get('site_count','待检查'))+' / '+str(c.get('unique_electrical_segments','待检查')),esc(spec.get('author',''))])
    s+=table(['神经元','ModelDB','论文表1 / 实现','输入位点 / 覆盖的电学段','原模型来源'],rows)
    s+='<p>原始形态、nseg和主动离子通道参数保留，统一温度32°C。500个位点是按树突长度分层布置的250E + 250I空间输入网格；多个位置可落在同一电学段。颗粒细胞原模型只有4个树突电学段，密集位点主要提供输入位置索引。篮细胞选用表1的BC₂；图4展示的BC₁来自另一实现。</p>'
    s+='<h3>AD.2 完整主动通道与低放电工作区间</h3><p>当前实验的输出范围为每个窗口0–3次尖峰、最小ISI≥20ms。原模型存在自发放电时，使用固定胞体保持电流建立安静基线；各比较方法共享这一电流、暖启动状态和输入。原始无输入自发放电单独保存。</p>'
    rows=[]
    for k,r in models.items():
        op=r['operating_point'] or {};rows.append([NAMES[k],fmt(op.get('native_firing_hz'),2),fmt(op.get('holding_current_na'),3),fmt(op.get('rest_mv'),3),fmt(op.get('weight_us'),8)])
    s+=table(['神经元','原生无输入放电 (Hz)','保持电流 (nA)','安静电位 (mV)','基准E事件强度 (µS)'],rows)
    s+=table(['突触条件','兴奋性输入','抑制性输入','用途'],[
        ['文中 ExpSyn','E=0mV，τ=4ms','E=−75mV，τ=5ms','成对输入的二次充分性初筛'],
        ['AMPA','Exp2Syn：E=0mV，τ₁=0.5ms、τ₂=5ms','Exp2Syn：E=−75mV，τ₁=1ms、τ₂=10ms','主要推广：DBNN 1 / 0.5 / 0.25ms'],
        ['AMPA + NMDA','同上AMPA + Mg阻塞的双指数NMDA；τ₁=2ms、τ₂=80ms、Mg=1；峰值电导比0.5','同上抑制性输入','首批敏感性测试：DBNN 1ms']])
    s+='<p><strong>ExpSyn是电导的时间波形实现，AMPA / NMDA是受体类型。</strong>对时间tₖ的一次输入，ExpSyn使用 g(t)=w·exp[−(t−tₖ)/τ]（t≥tₖ），电流 i(t)=g(t)[V(t)−E]；多次输入的电导相加。它表示瞬时上升、单指数衰减。原文E=0mV、τ=4ms的兴奋性ExpSyn可视为简化的快速兴奋性突触。我们标注的AMPA条件改用Exp2Syn，以两个时间常数描述有限上升和衰减，并归一化事件峰值电导。</p>'
    s+='<p>两种快速兴奋性实现均通过[V−E]保留驱动力的电压依赖。AMPA+NMDA进一步加入较慢的NMDA电流，其电导还乘以随膜电压变化的Mg阻塞因子；这会引入更强的电压依赖和较长记忆。当前NMDA/AMPA峰值比0.5是本轮固定实验条件。参见<a href="https://www.neuronsimulator.org/en/latest/progref/modelspec/programmatic/mechanisms/mech.html#expsyn" target="_blank" rel="noopener">NEURON ExpSyn / Exp2Syn 官方说明</a>。</p>'
    s+='<p>NMDA条件使用统一的确定性受体模型，Mg阻塞函数为 1 / [1 + Mg·exp(−0.062V) / 3.57]。其作用是控制检验电压依赖与较长时间尺度对同一DBNN架构的影响。各细胞原作者的短时突触可塑性参数另行保留在vendor代码中。</p>'
    s+='<h3>AD.3 独立数据、参考与选择规则</h3><p>参考使用NEURON dt=0.025ms，保存0.25ms电压；先暖启动2s。1ms面板采集96条6s源轨迹，0.5ms和0.25ms各增加48条独立源轨迹。每个源种子预先指定fit / validation；训练前仅依据参考，在6、4、2、1s中选取最长的0–3尖峰、最小ISI≥20ms前缀，每个源保留一个窗口。批处理补齐部分的电压及导数损失均屏蔽。持续放电源保留作诊断。</p>'
    s+='<p>独立测试先生成180条参考候选，覆盖T=1、2、4、8、16s，A=64、128、256、500，Poisson 10 / 25 / 50Hz，以及随机、分层、近端、远端输入。背景输入持续存在，并加入有限强度瞬态和1–3事件倍数。仅依据参考尖峰数与预定覆盖规则选择31窗；测试候选的DBNN误差不参与选择。</p>'
    s+='<p>主误差在共同1ms观察网格上计算，所有方法剔除同一参考尖峰±10ms。主指标为pooled RMSE / MSE；事件统计为±5ms一一匹配的precision、recall、F1。VE保留作诊断。AMPA与NMDA分别选择31窗，汇总误差对应各自样本范围；对NMDA影响的配对判断使用相同输入的共同样本。</p>'
    audit=models['basket'].get('reference_step_audit')
    if audit:
        s+='<details><summary>参考步长：为何现有小脑数据仍保留0.025ms？</summary><p>在篮细胞AMPA锁定输入中，预定选择0 / 1 / 2 / 3尖峰各一条最短轨迹，使用单核NEURON重放相同事件。以下是四条复核轨迹的合并统计；完整31窗的求解器统计见AD.7。</p>'
        s+=table(['重放dt','masked pooled RMSE (mV)','TP / FP / FN','事件F1'],[[f'{float(dt):g}ms',fmt(q['metrics']['pooled_rmse_mv'],6),f"{q['events']['tp']} / {q['events']['fp']} / {q['events']['fn']}",fmt(q['events']['f1'],3)] for dt,q in audit['summary'].items()])
        s+='<p>0.025ms重放与已保存参考吻合；0.1ms在无尖峰样本上误差很小，但部分有尖峰轨迹出现时间偏移及漏配。现有小脑面板继续保留0.025ms。新皮层 / 海马模型先检查0.1ms：预定六组相同输入与0.025ms比较，要求至少三组符合低尖峰范围、含有尖峰覆盖、每组masked RMSE≤0.03mV且全部尖峰在±5ms内一一匹配；通过后才采集0.1ms参考，否则采用0.025ms。参考步长在DBNN训练及独立测试之前锁定，数据条件散列记录实际值。</p></details>'
    s+='<h3>AD.4 当前训练与推理配方</h3><p>每个位点保留一个双指数核。每个神经元从自身弱单点响应初始化τ与ω，二次权重从零开始。损失为加权masked电压MSE + 0.1×加权物理时间导数MSE；窗口权重 w ∝ 0.3 + 0.7 / max(Var(V), 0.04)，仅按fit窗口归一化。</p>'
    s+='<p>采用log-τ、AdamW（lr=0.001、wd=10⁻⁶）、cosine、梯度裁剪1和EMA=0.995，主训练1000 epochs。验证集校准阈值/reset幅度/时间常数，再更新teacher reset并以0.2倍学习率细化400 epochs。1→0.5→0.25ms逐级从前一档锁定checkpoint继续拟合新增数据，最终checkpoint只由验证集决定。</p>'
    s+='<p>推理使用当前完整优化路径：active IDs压缩 → Triton双指数前缀扫描 → 二次项、线性项与bias融合 → GPU事件reset → 缓存工作区与CUDA Graph。FP32 IEEE；时间常数统一以物理毫秒保存，在后端换算一次。</p>'
    s+='<h3>AD.5 二次效应的初步数值检验</h3><p>初筛使用8组预定近/远端EE突触对和4×4强度网格；12个强度组合拟合，4个对角组合留出。表中平均值覆盖完成的8组完整电压轨迹。这里明确给出突触、保持电流、强度范围与留出规则；后续DBNN和多项式近似可在同一批留出曲线上比较。</p>'
    rows=[]
    for k,r in models.items():
        for profile in PROFILES:
            q=r['screens'][profile]
            if q['complete']:
                z=q['result']['summary'];rows.append([NAMES[k],PROFILES[profile],*[fmt(z[p]['mean_heldout_rmse_mv'],4) for p in ['linear','quadratic','cubic']],f"{100*z['quadratic']['mean_heldout_ve']:.2f}%"])
            else:
                progress=q['progress'];rows.append([NAMES[k],PROFILES[profile],f"初筛 {progress['completed'] if progress else 0}/8",'—','—','—'])
    s+=table(['神经元','条件','线性 RMSE','二次 RMSE','三次 RMSE','二次平均 VE'],rows)+pilot_plot(d)
    s+='<p class="muted">原文图4为选定模型、突触对和即时积分条件的展示。本初筛记录8对/4×4网格及完整120ms轨迹。各条件依据参考调节成对刺激强度，表中AMPA / NMDA各自误差用于筛选适合的工作区间；同强度的受体影响需由配对输入检验。完整DBNN准确率以下方31窗评估为准。</p>'
    s+='<h3>AD.6 实际采样和训练状态</h3>'
    rows=[]
    for k,r in models.items():
        st=r['state'];rows.append([NAMES[k],f"GPU {st.get('gpu','待分配')}",esc(st.get('phase','待启动')),esc(st.get('status','待启动'))+(' / 进程存活' if st.get('process_alive') else ' / 当前进程已退出')])
    s+=table(['神经元','训练设备','当前步骤','控制任务状态'],rows)
    for k,r in models.items():
        q=r.get('parallel_nmda_state')
        if q:s+=f'<p>并行NMDA任务：{NAMES[k]}，GPU {q["gpu"]}，步骤 {esc(q.get("phase","准备"))}，状态 {esc(q["status"])}。该任务复用相同训练/推理代码。独立CoreNEURON单核面板使用CPU56–60并行，multisplit使用CPU48–55资源锁，重复次数和计时口径保持一致。</p>'
    rows=[]
    for k,r in models.items():
        for profile,dts in [('ampa',[1.,.5,.25]),('ampa_nmda',[1.])]:
            for dt in dts:
                t=r['training'][profile][str(dt)];sampling=r['sampling'][profile][f'train_{dt}ms']
                if t['complete']:
                    v=t['selection']['selected']['validation'];protocol=t['selection']['protocol']
                    state='已锁定：'+str(protocol['fit_cases'])+' fit / '+str(protocol['validation_cases'])+' validation';score=fmt(v['pooled_rmse_mv'],4)
                else:
                    progress=t['progress'];state=f"{progress['stage']} epoch {progress['epoch']}" if progress else sampling_text(sampling);score=fmt(progress['validation']['pooled_rmse_mv'],4) if progress else '—'
                rows.append([NAMES[k],PROFILES[profile],f'{dt:g}ms',esc(state),score])
    s+=table(['神经元','条件','DBNN间隔','进度 / 参考筛选范围','验证 RMSE (mV)'],rows)
    solver_progress=[]
    for k,r in models.items():
        for profile,zs in r.get('solver_progress',{}).items():
            for key,z in zs.items():
                if r['solvers'][profile].get(key,{}).get('finished'):continue
                latest=z.get('latest',{});solver_progress.append([NAMES[k],PROFILES[profile],esc(key),f"{z['completed']}/{z['total']}",fmt(latest.get('median_ms'),1)])
    if solver_progress:s+=table(['神经元','条件','测速配置','已测窗口','最近窗口计算中位数 (ms)'],solver_progress)
    s+='<p class="muted">验证误差用于checkpoint选择；完整31窗评估在每档checkpoint锁定后进行。篮细胞首轮结果用于发现训练窗口长度覆盖不足，因而目前的同一31窗属于开发期重复评估；后续正式投稿可另锁定新种子的评估组。</p><h3>AD.7 完整31窗与精度—速度关系</h3>'
    rows=[]
    for k,r in models.items():
        for profile in ['ampa','ampa_nmda']:
            g=r['gpu'][profile]
            if not g:rows.append([NAMES[k],PROFILES[profile],'DBNN GPU','待完整31窗','—','—','—','—']);continue
            for dt,z in g['summary'].items():
                m=z['metrics'];e=z['events'];rows.append([NAMES[k],PROFILES[profile],'DBNN GPU',f'{float(dt):g}ms',fmt(m['pooled_rmse_mv'],4),fmt(m['pooled_rmse_mv']**2,5),fmt(z['wall_ms'],3),f"{e['precision']:.3f} / {e['recall']:.3f} / {e['f1']:.3f}"])
            for key,z in r['solvers'][profile].items():
                if not z.get('finished'):continue
                m=z['metrics'];e=z['events'];rows.append([NAMES[k],PROFILES[profile],esc(key.split('_')[0]),f"{z['runtime']['dt_ms']:g}ms",fmt(m['pooled_rmse_mv'],4),fmt(m['pooled_rmse_mv']**2,5),fmt(z['wall_ms'],3),f"{e['precision']:.3f} / {e['recall']:.3f} / {e['f1']:.3f}"])
            for engine,label in [('multisplit','NEURON multisplit 8核'),('core','CoreNEURON CPU 1核')]:
                if not any(key.startswith(engine+'_') and z.get('finished') for key,z in r['solvers'][profile].items()):
                    rows.append([NAMES[k],PROFILES[profile],label,'待完整31窗','—','—','—','—'])
            s+=frontier_plot(k,r,profile)
            from complete_plots import complete_plots
            s+=complete_plots(k,r,profile,Path(__file__).resolve().parent,Path('/data/lcm/DBNN_multineuron_20261002'),figure)
    s+=table(['神经元','条件','方法','间隔 / 阶段','RMSE (mV)','MSE (mV²)','31窗wall合计 (ms)','precision / recall / F1'],rows)
    invalid=[]
    for k,r in models.items():
        for profile,zs in r['solvers'].items():
            for key,z in zs.items():
                if z.get('invalid'):
                    q=z['invalid'];invalid.append([NAMES[k],PROFILES[profile],esc(key),esc(q['case_id']),esc(q['error'])])
    if invalid:
        s+='<details><summary>求解器无有效结果的步长与运行条件</summary>'
        s+=table(['神经元','条件','配置','触发样本','原因'],invalid)+'</details>'
    archive=[]
    for k,r in models.items():
        old=r.get('archived_6s_only_result')
        if old:
            for dt,z in old['summary'].items():
                e=z['events'];archive.append([NAMES[k],f'{float(dt):g}ms',fmt(z['metrics']['pooled_rmse_mv'],4),fmt(z['wall_ms'],3),f"{e['precision']:.3f} / {e['recall']:.3f} / {e['f1']:.3f}"])
    if archive:
        s+='<details><summary>开发记录：仅筛选完整6s窗口的首轮结果</summary><p>首轮训练集将0–3尖峰限制在完整6s内，事件率覆盖窄于1–2s评估窗口。已保留该轮checkpoint、选择记录与全部31窗输出。当前训练以参考选择更短前缀，架构、物理条件和fit / validation源种子保持一致，重新从各神经元的弱单点响应初始化。</p>'
        s+=table(['神经元','DBNN间隔','RMSE (mV)','31窗wall合计 (ms)','precision / recall / F1'],archive)+'</details>'
    s+='<p>电压示例优先使用8s窗口，再按事件匹配和masked RMSE选择展示。汇总误差包括全部31窗。分时长耗时图显示每组的中位数和Q1–Q3。</p><p>比较NEURON multisplit 8核和CoreNEURON CPU 1核。求解器扫描0.05、0.1、0.25、0.5、1ms；数值不稳定和mindelay等运行约束分别保留原始原因。纵轴为各窗口完整计算wall time中位数之和：DBNN包括滤波、完整耦合与reset及完成同步；GPU输入已驻留，CPU打包、上传、下载单列。求解器包括继续求解调用及调用内部的状态传输，外部事件准备与输出插值单列。当前工程重复次数为DBNN30次、求解器5次。</p>'
    s+='<h3>AMPA / NMDA 的共同输入检验</h3><p>在两种条件均完成后，取各自31窗的共同输入ID，核对事件序列、位点和时长的散列。在相同1ms观察网格上，共同剔除两种参考尖峰窗口的并集；各条件仍使用自身验证集选出的DBNN。这样可比较固定输入下加入NMDA后模型的逼近误差。</p>'
    paired=[]
    for k,r in models.items():
        q=r.get('paired_ampa_nmda')
        if q and q['cases']:
            paired.append([NAMES[k],q['cases'],fmt(q['summary']['ampa']['pooled_rmse_mv'],4),fmt(q['summary']['ampa_nmda']['pooled_rmse_mv'],4)])
        else:paired.append([NAMES[k],'待两种条件完整评估','—','—'])
    s+=table(['神经元','共同输入窗口数','AMPA RMSE (mV)','AMPA + NMDA RMSE (mV)'],paired)
    ve_rows=[]
    for k,r in models.items():
        q=r.get('paired_ampa_nmda')
        if q and q['cases']:
            for profile,z in q['summary'].items():
                ve_rows.append([NAMES[k],PROFILES[profile],f"{100*z['mean_case_ve']:.2f}%",f"{100*z['within_trace_weighted_ve']:.2f}%"])
    if ve_rows:
        s+='<details><summary>共同输入的VE诊断</summary><p>RMSE以mV表示绝对误差；VE使用去除误差均值后的误差方差与该条件参考方差的比值。NMDA会改变参考电压的波动尺度，因此同时保留这两类指标及原始方差记录。</p>'
        s+=table(['神经元','条件','平均窗口VE','窗口内方差加权VE'],ve_rows)+'</details>'
    s+='<h3>AD.8 数值兼容与原始记录</h3>'
    rows=[]
    for k,r in models.items():
        for engine in ['core','multisplit']:
            q=r['adapter_checks'].get(engine);rows.append([NAMES[k],engine,fmt(q.get('max_abs_mv'),8) if q else '待检查','通过' if q and q.get('passed') else '待检查'])
    s+=table(['神经元','运行方式','与NEURON最大电压差 (mV)','短程转移检查'],rows)
    s+='<p>兼容修改保存在隔离机制副本中：NEURON/CoreNEURON的声明与局部变量作用域修正，原始vendor目录和通道数值表达式保留。source_audit.json记录修改与散列；每个新形状的GPU后端还要通过FFT完整计算+顺序reset的数值oracle后，才纳入测速。</p>'
    s+='<h3>AD.9 非小脑模型：文献来源、选择理由与运行状态</h3>'
    s+=table(['相关方法','作者原实验使用的神经元','与本轮推广的对应'],[
        ['<a href="https://github.com/SelfishGene/neuron_as_deep_net" target="_blank" rel="noopener">TCN</a>','Hay et al. (2011) 大鼠皮层L5锥体细胞，AMPA / NMDA输入','新增官方Hay L5模型；TCN发布数据使用biophys5b等变体，当前采用官方biophys3，模型家族相同、具体参数版本明确区分'],
        ['<a href="https://github.com/AaronSpieler/elmneuron" target="_blank" rel="noopener">Branch-ELM / LSTM（NeuronIO）</a>','共享上述TCN生成的L5单神经元输入 / 电压数据','增加另一个算法并不等于增加一个神经元类型；本轮独立训练DBNN'],
        ['<a href="https://www.nature.com/articles/s41467-019-13932-6" target="_blank" rel="noopener">Neuron_Reduce</a>','主要L5模型，另测试皮层兴奋 / 抑制细胞、海马抑制性细胞、丘脑、小脑Golgi、人类L2/3等形态和更大的模型集合','新增人类Eyal (2016) L2/3；该方法提供较丰富的跨形态测试先例'],
        ['<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6226578/" target="_blank" rel="noopener">hLN / hGLM</a>','皮层L2/3、颗粒细胞、海马CA1','支持纳入CA1；Migliore / Poirazi候选目前按各自来源测试，hLN中CA1的确切代码版本仍需逐一核对'],
        ['<a href="https://papers.nips.cc/paper_files/paper/2025/file/8ec3ae4d5ad40d1a53627fadde960d4c-Paper-Conference.pdf" target="_blank" rel="noopener">NOBLE</a>','人类皮层PVALB与VIP细胞；主要为胞体阶跃电流输入','可作为后续抑制性皮层细胞候选；准确模型条目、作者数据及多位点突触任务适配需继续核对']])
    extension=d.get('noncerebellar')
    if extension:
        rows=[]
        reasons={'hay_l5':'具主动树突 / Ca尖峰的复杂大鼠皮层L5树；连接TCN等研究的模型家族','human_l23':'增加物种与皮层层次；所选2016 Fig.3c模型的胞体 / 轴突主动、树突被动','ca1':'海马脑区；保留树突Na / K / Ih，可检验不同树突电流组成'}
        for k,r in extension['models'].items():
            spec=r['spec'];state=r.get('state') or {};ref=r.get('reference_convergence');checks=r.get('adapter_checks',{})
            passed=all(q and q.get('passed') for q in checks.values()) and len(checks)==3
            rows.append([esc(spec['label']),f'<a href="https://modeldb.science/{spec["modeldb"]}" target="_blank" rel="noopener">{spec["modeldb"]}</a>',reasons[k],'三种求解方式检查通过' if passed else '兼容性检查中',f'{ref["chosen_reference_dt_ms"]:g}ms' if ref else '0.1ms优先，收敛检查中',esc(state.get('phase','待启动'))+' / '+esc(state.get('resource_status',state.get('status','待启动')))])
        s+=table(['新模型','ModelDB','覆盖价值','下载 / 兼容状态','实际参考dt','当前步骤'],rows)
        s+='<p>三个模型的源码、机制和适配器已保存在5090，有限任务已启动。统一500位点、AMPA三档DBNN间隔与AMPA+NMDA 1ms敏感性；每个细胞独立初始化 / 拟合，沿用AD.4配方。形态、nseg、原始通道和源模型温度分别保留。测试覆盖T=1–16s、不同位点集合及10 / 25 / 50Hz，按参考预定规则选择31窗。仅增加DBNN、NEURON multisplit 8核和CoreNEURON CPU 1核结果。</p>'
        s+='<p>另已下载<a href="https://modeldb.science/20212" target="_blank" rel="noopener">Poirazi et al. (2003) CA1 / 20212</a>（复杂树突非线性）与<a href="https://modeldb.science/2488" target="_blank" rel="noopener">Mainen &amp; Sejnowski (1996) / 2488</a>（多种皮层形态）作后续候选。当前人类模型来自Eyal (2016) / 195667。Eyal (2018) / 238347的元数据可访问，实际源码需进一步获取。</p>'
    s+='<h3>AD.10 剩余实验与预计时间</h3>'
    b=d.get('baseline_elm_025') or {}
    s+=table(['任务','实际状态','按当前速度估计'],[
        ['原L5PC外部对比',f'Branch-ELM 0.25ms：{b.get("steps",0):,}/{b.get("total_updates",350000):,}更新；其余训练已结束','约1–2小时完成剩余训练，另留最终评估和测速时间'],
        ['现有三种小脑细胞的准确率','AMPA三档均完成；篮细胞NMDA完成；颗粒 / Golgi的NMDA参考与评估继续运行','预计今日余下几小时陆续得到完整准确率'],
        ['现有小脑NEURON / CoreNEURON测速','独立CoreNEURON单核任务并行；每条件31窗、每窗5次计时重复','约18–30小时；较小dt及较长轨迹占主要耗时'],
        ['新增皮层 / 海马三模型','刚启动参考收敛、采样、独立训练和测速；CPU64–71 / GPU3排队执行','先按12–24小时预留，首批参考采样完成后更新；总体约24–36小时量级']])
    s+='<p class="muted">以上从本页实测记录时间起估算。CoreNEURON 0.05ms当前最近1s窗口约需54s（颗粒）和66s（Golgi）计算；每个窗口还要执行一次预热和五次正式重复。共享服务器负载、最终参考步长与有效测试窗口数会改变结束时间。新增任务的正式准确率在各自完整31窗评估后填入。</p>'
    s+='<p class="muted">'+esc(d['resource'])+'</p>'
    payload=base64.b64encode(json.dumps(d,ensure_ascii=False,indent=2).encode()).decode()
    s+=f'<details><summary>下载本附录的证据与配置</summary><p><a download="DBNN_multineuron_evidence_20261002.json" href="data:application/json;base64,{payload}">完整JSON记录</a></p><p>远程项目：<code>/home/lcm/DBNN_multineuron_20261002</code><br>原始参考：<code>/data/lcm/DBNN_multineuron_20261002</code><br>本地脚本：<code>实验脚本/dbnn_multineuron_20261002</code></p><p><a href="https://doi.org/10.1007/s11571-026-10556-7" target="_blank" rel="noopener">Ma et al. (2026)</a>；上方各ModelDB条目提供原模型来源。原始Article.pdf在DBNN目录保留。</p></details></section>'
    return s

CSS='''#multineuron-generalization{scroll-margin-top:20px}#multineuron-generalization .mn-table{overflow-x:auto;margin:18px 0}#multineuron-generalization table{border-collapse:collapse;width:100%;font-size:13px}#multineuron-generalization th,#multineuron-generalization td{padding:10px 12px;text-align:left;border-bottom:1px solid #dee6e9}#multineuron-generalization th{background:#eef5f6;white-space:nowrap}#multineuron-generalization img{max-width:100%;height:auto}#multineuron-generalization code{overflow-wrap:anywhere}#multineuron-generalization figure{margin:24px 0}'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--canonical',type=Path);p.add_argument('--rendered',type=Path,help='Reuse a remotely rendered scientific plot; local merge requires no plotting package.');a=p.parse_args()
    d=json.loads((a.work/'snapshot.json').read_text())
    if a.rendered:
        body=re.search(r'<section id="multineuron-generalization">.*?</section>',a.rendered.read_text(),re.S).group()
        assert d['collected'] in body,'Rendered report and evidence snapshot differ'
    else:
        if plt is None:raise RuntimeError('Use the remote scientific plotting environment or --rendered')
        body=section(d)
    standalone='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DBNN 多神经元推广实验</title><style>body{font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#243d4b;background:#f6f8fa;margin:0}main{max-width:1120px;margin:36px auto;padding:30px;background:white;border-radius:16px}h2{font-size:28px}h3{margin-top:32px}a{color:#008678}.muted{color:#607781;font-size:14px}details{padding:16px;background:#f5f8fa}'+CSS+'</style></head><body><main>'+body+'</main></body></html>'
    (a.work/'DBNN_multineuron_report_20261002.html').write_text(standalone)
    if a.canonical:
        old=a.canonical.read_text();backup=a.work/'report_before_multineuron.html'
        if not backup.exists():shutil.copy2(a.canonical,backup)
        text=re.sub(r'<!-- MULTINEURON_20261002_START -->.*?<!-- MULTINEURON_20261002_END -->','',old,flags=re.S)
        toc='<li><a href="#multineuron-generalization" data-section-link>附录 AD. 多神经元推广与NMDA敏感性</a></li>'
        text=text.replace(toc,'');text=text.replace('</ul><div class="toc-footer">',toc+'</ul><div class="toc-footer">',1)
        pos=text.rfind('<footer>');pos=text.index('</main>') if pos<0 else pos
        text=text[:pos]+'<!-- MULTINEURON_20261002_START -->'+body+'<!-- MULTINEURON_20261002_END -->'+text[pos:]
        text=text.replace(CSS,'');text=text.replace('</style>',CSS+'</style>',1)
        ids=re.findall(r'\bid="([^\"]+)"',text);assert len(ids)==len(set(ids))
        assert all(x in ids for x in re.findall(r'href="#([^\"]+)"',text))
        for anchor in ['comparison-protocol','baseline-reproduction']:
            pat='<section id="'+anchor+'">.*?</section>'
            assert re.search(pat,old,re.S).group()==re.search(pat,text,re.S).group(),('Existing result changed',anchor)
        tmp=a.canonical.with_suffix('.tmp');tmp.write_text(text);tmp.replace(a.canonical)
    print(json.dumps(dict(snapshot=d['collected'],report=str(a.work/'DBNN_multineuron_report_20261002.html'),canonical=str(a.canonical) if a.canonical else None),ensure_ascii=False))

if __name__=='__main__':main()
