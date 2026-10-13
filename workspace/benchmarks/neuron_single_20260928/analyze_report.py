"""Build evidence-led NEURON-first report sections, preserving prior artifacts."""
from pathlib import Path
import base64,hashlib,html,json,re,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H/'results';FIG=H/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})
COL={'neuron':'#63778a','multisplit':'#ce8d34','core':'#3b77bd','cvode':'#a15eaa','dbnn':'#078568','original':'#79a4d5'}

def load(p):return json.loads(Path(p).read_text())
def med(r,key='solver_call'):return r['timing'][key]['median_ms']
def fm(x):return f'{x:,.3f}' if abs(x)<10 else f'{x:,.2f}'
def pic(name,alt):return '<figure><img loading="lazy" src="data:image/png;base64,'+base64.b64encode((FIG/(name+'.png')).read_bytes()).decode()+'" alt="'+html.escape(alt)+'"></figure>'
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def table(head,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def vol(name):
    if name.endswith('_multisplit') and (R/(name+'_checked.npz')).exists():name+='_checked'
    return np.load(R/(name+'.npz'),allow_pickle=True)['voltage'][0]
def err(x,y):return dict(rmse_mv=float(np.sqrt(np.mean((x-y)**2))),max_abs_mv=float(np.max(np.abs(x-y))))
def match_spikes(pred,target,tol=5.):
    i=j=k=0
    while i<len(pred) and j<len(target):
        if abs(pred[i]-target[j])<=tol:k+=1;i+=1;j+=1
        elif pred[i]<target[j]:i+=1
        else:j+=1
    return dict(predicted=len(pred),neuron=len(target),matched=k,precision=k/len(pred) if len(pred) else None,recall=k/len(target) if len(target) else None)

def build():
    records={p.stem:load(p) for p in R.glob('*.json')}
    good={k:v for k,v in records.items() if v.get('finished')}
    gpu=load(H/'dbnn_gpu.json');assert gpu.get('finished')
    cpu={n:load(H/f'dbnn_cpu{n}.json') for n in [1,2,4,8]}
    assert all(r.get('finished') for r in cpu.values())
    pool=load(H/'throughput.json');assert pool.get('finished')
    def gr(A,T,B=1,rate='sampled'):return next(r for r in gpu['cases'] if (r['A'],r['T'],r['B'],r['rate'])==(A,T,B,rate))
    def cr(A,n=1):return next(r for r in cpu[n]['cases'] if r['A']==A)
    def nr(A,T,engine):
        name=f'main_A{A}_T{T}_{engine}'
        if name+'_checked' in good:name+='_checked'
        return good[name]
    engines=['neuron','multisplit','core','cvode']
    if 'main_A500_T6001_multisplit_busy' in good:engines.insert(2,'multisplit_busy')
    labs={'neuron':'NEURON 1T','multisplit':'NEURON multisplit','multisplit_busy':'multisplit + busy wait','core':'CoreNEURON CPU','core_single_final':'CoreNEURON CPU 1T','cvode':'CVode (atol 1e-4)'}
    evidence=dict(created=time.strftime('%FT%T%z'),main=[],fidelity=[],neuron_errors=[],records=records,gpu=gpu,cpu=cpu,throughput=pool)
    rows=[]
    for T in [1001,6001]:
        for A in [32,128,500]:
            g=gr(A,T);r0=nr(A,T,'neuron');ref=vol(f'main_A{A}_T{T}_neuron')[0]
            candidates=[(e,nr(A,T,e)) for e in engines if e!='cvode']
            if A==500 and T==6001 and 'core_single_final' in good:candidates.append(('core_single_final',good['core_single_final']))
            best_e,best_r=min(candidates,key=lambda it:med(it[1]))
            gt=g['timing']['optimized_FP32']['median_ms'];ge=g['transfer']['dense_gather_h2d_forward_d2h']['median_ms']
            go=min(v['median_ms'] for k,v in g['timing'].items() if k.startswith('original'))
            vo=np.load(R/g['voltage_file'])['voltage'][0];accuracy=err(vo,ref)
            e=dict(A=A,T=T,neuron_best=best_e,neuron_best_ms=med(best_r),neuron_threads=best_r['config']['threads'],gpu_core_ms=gt,gpu_e2e_ms=ge,
                speed_ratio_core=med(best_r)/gt,speed_ratio_e2e=med(best_r)/ge,original_gpu_ms=go,vs_original=go/gt,accuracy=accuracy)
            evidence['main'].append(e)
            rows.append([f'{A} / {(T-1)/1000:g} s',fm(med(r0)),f'{fm(med(best_r))}<br><small>{labs[best_e]} · {best_r["config"]["threads"]}T</small>',fm(gt),fm(ge),f'{med(best_r)/ge:,.0f}×',fm(accuracy['rmse_mv'])])
            for engine in engines[1:]:
                v=vol(f'main_A{A}_T{T}_{engine}')[0]
                er=err(v,ref);er.update(A=A,T=T,engine=engine);evidence['neuron_errors'].append(er)
    # NEURON thread study: each data point is a single cell/trajectory.
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.5))
    for names,label,c in [([f'pilot_neuron_n{n}_recheck' for n in [1,2,4,8]],'NEURON, no split',COL['neuron']),
                          ([f'pilot_core_n{n}' for n in [1,2,4,8]],'CoreNEURON CPU',COL['core']),
                          ([f'pilot_ms_n{n}_record_checked' for n in [2,4,8]],'Multisplit, wait',COL['multisplit']),
                          ([f'pilot_ms_n{n}_busy1_initialized' for n in [2,4,8]],'Multisplit, busy wait','#b85930')]:
        rr=[good[n] for n in names if n in good]
        if rr:axs[0].plot([r['config']['threads'] for r in rr],[med(r) for r in rr],'o-',label=label,color=c)
    axs[0].set_xticks([1,2,4,8]);axs[0].set_xlabel('Allocated physical CPU cores / requested threads');axs[0].set_ylabel('Single-cell solver wall time (ms)');axs[0].set_title('Thread screening: 500 sites, 0.3 s');axs[0].legend(fontsize=8,frameon=False)
    for engine in engines:
        vals=[med(nr(A,6001,engine)) for A in [32,128,500]]
        axs[1].plot([32,128,500],vals,'o-',label=labs[engine],color=COL.get(engine,'#b85930'))
    axs[1].set_xscale('log',base=2);axs[1].set_xticks([32,128,500],['32','128','500']);axs[1].set_yscale('log');axs[1].set_xlabel('Active input sites');axs[1].set_ylabel('Solver wall time (ms, log scale)');axs[1].set_title('Confirmation: one cell, 6 s');axs[1].legend(fontsize=8,frameon=False)
    save(fig,'neuron_modes')
    # Fresh original implementation versus latest pipeline; show transfers.
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.7))
    cases=evidence['main'];xx=np.arange(len(cases));w=.25
    for off,key,label,c in [(-w,'original_gpu_ms','Original PyTorch (faster F/R)',COL['original']),(0,'gpu_core_ms','Latest DBNN: GPU core',COL['dbnn']),(w,'gpu_e2e_ms','Latest: gather + transfers + core','#d5993e')]:
        axs[0].bar(xx+off,[r[key] for r in cases],w,label=label,color=c)
    axs[0].set_xticks(xx,[f'A{r["A"]}\n{(r["T"]-1)/1000:g}s' for r in cases]);axs[0].set_yscale('log');axs[0].set_ylim(.1,2.1);axs[0].set_ylabel('Wall time (ms)');axs[0].legend(fontsize=8,frameon=False,loc='upper left');axs[0].set_title('All five optimizations enabled')
    for off,label,c,vals in [(-w,'NEURON 1 CPU thread',COL['neuron'],[med(nr(A,6001,'neuron')) for A in [32,128,500]]),
        (0,'Original PyTorch, 1 CPU thread',COL['original'],[min(v['median_ms'] for k,v in cr(A)['timing'].items() if k.startswith('original')) for A in [32,128,500]]),
        (w,'DBNN native CPU, 1 thread',COL['dbnn'],[cr(A)['timing']['optimized_CPU']['median_ms'] for A in [32,128,500]])]:
        axs[1].bar(np.arange(3)+off,vals,w,label=label,color=c)
    axs[1].set_xticks([0,1,2],['A32','A128','A500']);axs[1].set_yscale('log');axs[1].set_ylim(.4,1e5);axs[1].set_ylabel('Wall time (ms)');axs[1].set_title('Same hardware: 1 CPU core, 6 s');axs[1].legend(fontsize=8,frameon=False,loc='upper left')
    save(fig,'latest_dbnn_latency')
    # Error/time frontier uses finer physical integration, not the learned model.
    fine=vol('dt_0.0125')[0];front=[]
    for dt in [1.,.5,.2,.1,.05,.025,.0125]:
        name=f'dt_{dt:g}' if dt!=.1 else 'main_A500_T1001_neuron';r=good[name];v=vol(name)[0]
        front.append(dict(label=f'dt={dt:g} ms',kind='fixed',wall_ms=med(r),**err(v,fine)))
    for atol in [1e-3,1e-4,1e-5]:
        name=f'cv_{atol:g}' if atol!=1e-4 else 'main_A500_T1001_cvode';r=good[name];v=vol(name)[0]
        front.append(dict(label=f'atol={atol:g}',kind='cvode',wall_ms=med(r),**err(v,fine)))
    evidence['solver_frontier']=front
    density=[]
    for rate in ['quiet','low','sampled','high']:
        name=f'rate_{rate}_neuron' if rate!='sampled' else 'main_A500_T1001_neuron'
        r=good[name];g=gr(500,1001,rate=rate);p=np.load(R/g['voltage_file'])['voltage'][0]
        e=err(p,vol(name)[0]);sp=match_spikes(np.flatnonzero(p>=-57.5),r['spikes'][0][0])
        density.append(dict(rate=rate,input_events=r['input_events'],dbnn_ms=g['timing']['optimized_FP32']['median_ms'],**e,**sp))
    evidence['input_density_accuracy']=density
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.7))
    for typ,label,col in [('fixed','NEURON fixed step',COL['neuron']),('cvode','CVode',COL['cvode'])]:
        rr=[r for r in front if r['kind']==typ and r['rmse_mv']>0];axs[0].scatter([r['wall_ms'] for r in rr],[r['rmse_mv'] for r in rr],label=label,color=col)
        for i,r in enumerate(rr):
            offset=(5,4 if i%2 else -13)
            if typ=='cvode':offset=[(-85,-35),(-40,55),(12,18)][i]
            elif r['label']=='dt=0.025 ms':offset=(-78,20)
            axs[0].annotate(r['label'],(r['wall_ms'],r['rmse_mv']),xytext=offset,textcoords='offset points',fontsize=7,
                            arrowprops=dict(arrowstyle='-',color='#9eabb4',lw=.7) if typ=='cvode' or r['label']=='dt=0.025 ms' else None)
    axs[0].set_xscale('log');axs[0].set_yscale('log');axs[0].set_xlim(120,48000);axs[0].set_ylim(.0005,1);axs[0].set_xlabel('Solver wall time (ms)');axs[0].set_ylabel('RMSE vs dt=0.0125 ms (mV)');axs[0].set_title('Numerical accuracy / latency, 1 s');axs[0].legend(fontsize=8,frameon=False)
    for e in ['neuron','multisplit','cvode']:
        rr=[good.get(f'rate_{rate}_{e}_checked',good[f'rate_{rate}_{e}']) if rate!='sampled' else nr(500,1001,e) for rate in ['quiet','low','sampled','high']]
        axs[1].plot(np.arange(4),[med(r) for r in rr],'o-',label=labs[e],color=COL[e])
    axs[1].set_xticks(range(4),['Quiet','10% events','Sampled','Extra events']);axs[1].set_yscale('log');axs[1].set_ylabel('Solver wall time (ms)');axs[1].set_title('Event density, 500 sites, 1 s');axs[1].legend(fontsize=8,frameon=False)
    save(fig,'solver_accuracy_density')
    # Three inputs independently replayed in NEURON, current optimized B=3 path.
    r=good['fidelity_3x6s'];target=vol('fidelity_3x6s');g=gr(500,6001,3);pred=np.load(R/g['voltage_file'])['voltage'];th=-57.5
    fig,axs=plt.subplots(3,2,figsize=(11.5,8.3),gridspec_kw={'width_ratios':[1.65,1]})
    for b in range(3):
        e=err(pred[b],target[b]);sp=match_spikes(np.flatnonzero(pred[b]>=th),r['spikes'][0][b]);e.update(sample=b,**sp);evidence['fidelity'].append(e)
        t=np.arange(6001)/1000;axs[b,0].plot(t,target[b],color=COL['neuron'],lw=.8,label='NEURON');axs[b,0].plot(t,pred[b],color=COL['dbnn'],lw=.8,alpha=.85,label='Latest DBNN')
        axs[b,0].set_title(f'Input {b}: voltage RMSE {e["rmse_mv"]:.3f} mV',fontsize=10);axs[b,0].set_ylabel('Soma voltage (mV)')
        center=int(np.argmax(abs(pred[b]-target[b])));lo=max(0,center-100);hi=min(6001,center+101)
        axs[b,1].plot(t[lo:hi],target[b,lo:hi],color=COL['neuron'],lw=1.2);axs[b,1].plot(t[lo:hi],pred[b,lo:hi],color=COL['dbnn'],lw=1.2)
        axs[b,1].set_title('Window around maximum error',fontsize=10)
    axs[0,0].legend(frameon=False,fontsize=8);axs[-1,0].set_xlabel('Physical time (s)');axs[-1,1].set_xlabel('Physical time (s)');save(fig,'matched_voltage')
    # Independent-trial throughput, separate from single trajectory acceleration.
    pooled=[]
    serial=good['batch_serial_8x1s'];ref=vol('batch_serial_8x1s')
    for row in pool['cases']:
        mismatch=0.
        for f in row['worker_files']:
            wr=good[Path(f).stem];v=vol(Path(f).stem)
            mismatch=max(mismatch,float(np.max(abs(v-ref[wr['config']['samples']]))))
        row['voltage_max_diff_vs_serial']=mismatch;pooled.append(row)
    fig,axs=plt.subplots(1,2,figsize=(11.2,4.4));pp=[r['processes'] for r in pooled];tt=[r['timing']['median_ms'] for r in pooled]
    axs[0].plot(pp,tt,'o-',color=COL['neuron']);axs[0].set_xticks(pp);axs[0].set_xlabel('Persistent NEURON worker processes (1 core each)');axs[0].set_ylabel('Eight-trial batch wall time (ms)');axs[0].set_title('Same cell model, eight independent inputs')
    bg=gr(500,1001,8);be=bg['transfer']['dense_gather_h2d_forward_d2h']['median_ms'];bo=min(v['median_ms'] for k,v in bg['timing'].items() if k.startswith('original'))
    labels=['NEURON\n1 worker','NEURON\n8 workers','Original\nPyTorch GPU','Latest GPU\ncore','Latest GPU\ngather + I/O'];vals=[tt[0],tt[-1],bo,bg['timing']['optimized_FP32']['median_ms'],be]
    axs[1].bar(range(5),vals,color=[COL['neuron'],COL['core'],COL['original'],COL['dbnn'],'#d5993e']);axs[1].set_xticks(range(5),labels,fontsize=8);axs[1].set_yscale('log');axs[1].set_ylabel('Batch wall time (ms, log scale)');axs[1].set_title('CPU / GPU scopes explicitly separated');save(fig,'trial_throughput')

    # Report sections.
    largest=next(r for r in cases if r['A']==500 and r['T']==6001)
    c1=cr(500);c8=cr(500,8);max_backend=max(r['accuracy_vs_original']['optimized_FP32']['max_abs_mv'] for r in gpu['cases'])
    cpurows=[[n,fm(cr(500,n)['timing']['optimized_CPU']['median_ms']),fm(min(v['median_ms'] for k,v in cr(500,n)['timing'].items() if k.startswith('original')))] for n in [1,2,4,8]]
    tcrows=[]
    for A in [32,128,500]:
        g=gr(A,6001)
        if 'precision_strict' in g['timing']:
            tcrows.append([A,fm(g['timing']['optimized_FP32']['median_ms']),fm(g['timing']['precision_strict']['median_ms']),f"{g['accuracy_vs_original']['precision_strict']['max_abs_mv']:.2e}",g['accuracy_vs_original']['precision_strict']['reset_bin_difference']])
    intro=f'''<section id="conclusion"><h2>本轮重点：与详细生物物理仿真器 NEURON 比较</h2>
<p>在同一个 Shai L5PC 神经元、相同突触位点和同一批离散事件输入上，补测了 NEURON 固定步长、单细胞 multisplit、CoreNEURON CPU、CVode 变步长，以及独立输入的多进程吞吐量。DBNN 的主结果使用我们当前的完整优化链路；原始 F / R 两种 PyTorch 实现仍保留为软件基线。</p>
<div class="stats"><div><span>500 位点 · 单条 6 s · 最新 GPU core</span><b>{fm(largest['gpu_core_ms'])} ms</b><span>融合驱动项 + 前缀扫描 + 事件 reset + 缓存 / Graph</span></div><div><span>同一条件 · 输入整理 / 传输 / 取回均计入</span><b>{fm(largest['gpu_e2e_ms'])} ms</b><span>同 dt=0.1 ms 的已测最快 NEURON 系路径：{fm(largest['neuron_best_ms'])} ms</span></div><div><span>11 种输入 / batch 情况 · 优化后端数值检查</span><b>{max_backend:.2e} mV</b><span>相对原 R 实现最大差异；reset 时间格差为 0</span></div></div>
<p>GPU 加速比包含算法与硬件两部分；下文同时给出 <strong>1 个 CPU 核上的对照</strong>。代理模型的电压误差、动作电位事件匹配与后端浮点误差分别报告。原第二章的 GPU 规模扩展和逐层消融已移至后面的备用资料。</p>
<p class="muted">资源状态：绑定了物理 CPU 核；服务器仍有其他任务，GPU 6 也有其他进程。本轮结果按共享资源测量标注，计时分布和配置随报告附出。</p></section>'''
    caps=[['普通 NEURON + nthread','1 / 2 / 4 / 8 线程','单细胞未拆分时，主要积分工作仍集中在一个线程；本轮实测各线程配置。'],
          ['NEURON multisplit','2 / 4 / 8 线程；busy wait 兼容性检查','把一个多区室细胞的计算树拆给多个线程；保留相同离子通道、突触与固定步长。'],
          ['CoreNEURON CPU','1 / 2 / 4 / 8 线程','更适合向量化的数据布局 / 计算引擎；单个未拆分细胞并不会自动分给所有线程。'],
          ['CVode','atol 10⁻³ / 10⁻⁴ / 10⁻⁵','自适应时间步长；在相同 1 ms 输出时刻评价误差，并检查输入事件密度的影响。'],
          ['独立输入并行','1 / 2 / 4 / 8 个进程','每个进程复用一个相同细胞模型；报告 8 条独立输入的批延迟和吞吐量。'],
          ['CoreNEURON GPU','当前环境未测','已安装的 NEURON 9.0.2 为 CORENRN_ENABLE_GPU=OFF；本机没有所需 NVIDIA HPC SDK 编译器，需另建 GPU 版本。'],
          ['MPI / 跨节点','机制支持，当前未测','可分配独立任务，也可做模型并行；本轮聚焦一台服务器上的一个细胞及其独立输入。']]
    sec2=f'''<section id="neuron-main"><h2>2. NEURON 支持哪些比较方式，实际有多快？</h2>
<p><strong>NEURON 可以使用多线程，单个多区室神经元也能通过 multisplit 并行。</strong>“申请了几个线程”和“一个细胞的计算是否真正拆开”需要分别核对。以下方式已按相同模型和输入测试；尚未运行的能力单独标出。</p>
{table(['路径','本轮配置 / 状态','单细胞场景中的含义'],caps)}
<p class="muted">依据：<a href="https://www.neuronsimulator.org/en/latest/progref/modelspec/programmatic/network/parcon.html#ParallelContext.Threads">NEURON Threads / multisplit 官方文档</a>、<a href="https://www.neuronsimulator.org/en/latest/coreneuron/installation.html">CoreNEURON 安装要求</a>。当前官方接口不支持将 multisplit 模型直接交给 CoreNEURON，二者分别比较。busy wait 在当前 9.0.2 构建中触发内部 worker 断言，保存了错误日志；正式计时使用正常等待策略。</p>
{pic('neuron_modes','NEURON 各线程策略以及 6 秒模拟不同活跃位点的速度')}
<p>线程筛选使用 0.3 s 输入，每项预热 1 次、测量 3 次。主网格使用测试输入 0，重复用于测量运行时间分布。将筛选得到的 multisplit 和 CoreNEURON 配置固定，再独立测量 1 s 与 6 s；CVode 主表固定 atol=10⁻⁴。下表的“已测最快”取本表内普通 NEURON、multisplit 和 CoreNEURON 的实测中位数。</p>
{table(['活跃位点 / 时长','NEURON 1T<br>solver / ms','同 dt=0.1 ms 的已测最快<br>solver / ms','最新 DBNN GPU<br>core / ms','最新 DBNN GPU<br>整理 + I/O + core / ms','NEURON solver /<br>DBNN 整理 + I/O + core','代理电压<br>RMSE / mV'],rows)}
<p>上表输出范围一致：整个有效时间窗的胞体电压，每 1 ms 一个点。NEURON solver 含记录；CoreNEURON 含在线状态交换及采样。GPU 完整边界从 CPU 上的带符号稠密 raster 开始，包含活跃列 gather、H2D、全部 DBNN 计算及 D2H。NEURON 的 500 ms 初始稳态准备、事件排队和模型构建另列原始记录。core 与传输边界分组测量，微秒级的小差异应结合原始计时分布阅读。</p>
{pic('latest_dbnn_latency','最新 DBNN 相对原始 PyTorch 的完整耗时，以及同 CPU 单线程的 NEURON 对照')}
<h3>我们的主路径具体启用了什么？</h3>
{table(['阶段','本轮主结果实际执行','准备 / 计时边界'],[
['位点集合','已知 active IDs 压缩输入与 WAA；未用 N&gt;500 的人工扩展权重替代生物模型','索引和压缩权重可复用；每次输入 gather 在完整耗时中计入'],
['时间滤波','Triton 双指数仿射前缀扫描；逐形状比较 64 / 128 / 256 / 512 点块','当前状态为零初始、整窗推理；最终计时与块长度筛选分开'],
['驱动项','当前优化版二次型融合 kernel（FP32）：跨时间复用、上三角权重及线性项 / 偏置融合','直接计算每时刻二次型；不物化整个 YW；新时长显式采用融合配置并检查完整输出'],
['reset','GPU 内逐事件寻找下一次越阈值位置，区间按指数恢复状态','单条与多条输入均走事件 kernel；全部 reset 计算包含在 core'],
['调度','系数 / 权重 / workspace 缓存，CUDA Graph 重放，按形状选扫描块','编译 / capture / 配置筛选独立列 prepare；Graph 已检查更换输入能更新结果']])}
<p>CPU 对照使用本机原生递推、BLAS 二次项和原生 reset。相同 500 位点、6 s 下的 CPU 扩展如下；GPU 专属前缀 kernel 与 Graph 只出现在 GPU 列。</p>
{table(['CPU 核 / 线程','DBNN 优化 CPU / ms','原始 PyTorch 较快 F/R / ms'],cpurows)}
<p>另外重测 CoreNEURON 的单线程 500 位点 / 6 s：{fm(med(good['core_single_final']))} ms，可直接与同条件的 DBNN 单线程 {fm(c1['timing']['optimized_CPU']['median_ms'])} ms 对照。</p>
<p>TF32 作为显式可选精度路径，以下是本轮 6 s 单条输入的完整 GPU core 计时。strict profile 使用此前逐形状验证过的组合；全部输出重新与原 R 核对。本章主表统一列 FP32，Tensor Core 细节保留在备用资料。</p>
{table(['活跃位点','完整 FP32 core / ms','strict profile core / ms','最大差 / mV','reset 格差'],tcrows)}</section>'''
    fidelity_rows=[[e['sample'],fm(e['rmse_mv']),fm(e['max_abs_mv']),e['neuron'],e['predicted'],e['matched'],f'{e["precision"]:.3f}' if e['precision'] is not None else '—',f'{e["recall"]:.3f}' if e['recall'] is not None else '—'] for e in evidence['fidelity']]
    num_rows=[[r['label'],fm(r['wall_ms']),f'{r["rmse_mv"]:.6f}',f'{r["max_abs_mv"]:.6f}'] for r in front]
    sec3=f'''<section id="neuron-accuracy"><h2>3. 速度对应什么精度和适用范围？</h2>
<p>先校验 NEURON 不同求解方式的数值误差，再评价 DBNN 对 NEURON 的代理误差。固定步长对照保持相同离子通道和突触模型；变步长对照通过 atol 控制局部误差。较粗步长能降低时间开销，误差变化同时展示。</p>
{pic('solver_accuracy_density','NEURON 步长与误差权衡，以及安静和高事件密度输入的求解速度')}
<p>无输入时，CVode 的自适应步长能明显降低积分工作；本轮 500 位点安静输入由固定步长的 {fm(med(good['rate_quiet_neuron']))} ms 降至 {fm(med(good['rate_quiet_cvode']))} ms。事件密集时则需要频繁处理动力学变化，图中分别给出了相应耗时。</p>
{table(['输入 / 1 s','输入事件数','DBNN GPU core / ms','电压 RMSE / mV','NEURON / DBNN 事件','±5 ms 匹配'],[[e['rate'],e['input_events'],fm(e['dbnn_ms']),fm(e['rmse_mv']),f"{e['neuron']} / {e['predicted']}",e['matched']] for e in density])}
<p>高频条件是在原输入上额外加入独立事件；当前权重在该条件下的电压 RMSE 为 {density[-1]['rmse_mv']:.3f} mV。进一步训练需要覆盖更宽的事件密度，并同时优化动作电位时序。安静、低频与原采样条件的结果一并保留。</p>
<details class="history-fold"><summary>展开：步长 / 容差、耗时与细步长参考的完整数值</summary><div class="history-body">{table(['求解配置','solver / ms','RMSE / mV','最大差 / mV'],num_rows)}<p>参考为 dt=0.0125 ms 的同一 1 s 输入；这是本轮的数值收敛参照。0.025 与 0.0125 ms 的差异也列出。</p></div></details>
<h3>冻结 checkpoint：同输入回放的电压与事件</h3>
<p>以下为重建测试集的第 0 / 1 / 2 条输入，每条 6 s、500 位点。NEURON 直接回放与 DBNN 相同的量化事件；图中保留动作电位峰值，右侧放大最大误差附近。速度优化使用同一个 checkpoint，训练结果没有被后端修改。</p>
{pic('matched_voltage','三条独立输入的 NEURON 与优化 DBNN 电压，以及最大误差局部')}
{table(['输入','RMSE / mV','最大差 / mV','NEURON 事件','DBNN reset','±5 ms 匹配','precision','recall'],fidelity_rows)}
<p>NEURON 事件使用 −20 mV 上穿；DBNN 使用 checkpoint 的 −57.5 mV reset 阈值，按 ±5 ms 一一匹配。表中的电压 RMSE 为全时段原始电压、包含动作电位峰值。平均电压误差与事件召回率共同界定当前模型的使用范围。</p>
<p>模型复现进展沿用此前记录：重建 110 条轨迹（100 训练 / 10 测试）；参考 checkpoint 已完成推理复现和评估；从头训练目前完成 1 epoch 通路验证，完整收敛复现待继续。原连续事件数据上的 10 条测试评估，以及本次量化事件回放的评估，分别列在正文和备用资料中。</p>
<p>后端数值检查：11 个输入 / batch 配置全部有限值，相对原 R 最大差 {max_backend:.2e} mV，reset 时间格差 0。生物拟合精度见上表；两者分别衡量代理模型与计算实现。</p></section>'''
    sec4=f'''<section id="neuron-throughput"><h2>4. 同一神经元的多条独立输入：吞吐量比较</h2>
<p>一个详细神经元模型反复接受不同输入，是稠密采样 / 参数探索时的另一种常见工作负载。这里固定 8 条输入、每条 1 s，NEURON 用 1 / 2 / 4 / 8 个持久进程，每个进程固定一个物理核、复用模型和已预热的 SaveState；DBNN 使用 B=8 的完整优化路径。</p>
{pic('trial_throughput','同一神经元八条独立输入的多进程和 GPU 吞吐量')}
{table(['NEURON 进程数','8 条完整 trial / ms','轨迹 / s','相对串行','与串行电压最大差 / mV'],[[r['processes'],fm(r['timing']['median_ms']),fm(8000/r['timing']['median_ms']),f'{tt[0]/r["timing"]["median_ms"]:.2f}×',f'{r["voltage_max_diff_vs_serial"]:.2e}'] for r in pooled])}
<p>NEURON 这一项先缓存 500 ms 无输入后的状态；每条输入计入 SaveState 恢复、事件排队、1 s 求解和输出拷贝。初次生物稳态准备、进程启动和模型创建属于一次准备成本。这样重复模拟无需反复积分 500 ms（<a href="https://www.neuronsimulator.org/en/9.0.1/progref/simctrl/savstate.html">SaveState 官方接口</a>）。采用同步起点与跨进程单调时钟实测批延迟。所有多进程输出均逐点核对串行输出。</p>
<p>同一批输入，DBNN GPU core 为 {fm(bg['timing']['optimized_FP32']['median_ms'])} ms；从 CPU 稠密输入整理到输出回 CPU 为 {fm(be)} ms。原始 F/R 的整批和逐条调用都已计时，图中取最快的 {fm(bo)} ms。多进程提升描述的是独立 trial 吞吐量；单条轨迹的线程并行见第二节。</p></section>'''
    nrn_error=max(x['max_abs_mv'] for x in evidence['neuron_errors'] if x['engine']=='core')
    sec5=f'''<section id="neuron-protocol"><h2>5. 测量边界、复现检查与尚待补齐的对照</h2>
{table(['项目','本轮固定的条件 / 记录'],[
['生物模型','Shai L5PC，196 个 section、642 个 nseg；CoreNEURON 的包含端点节点统计为 839。500 个注册突触位置，250 兴奋 / 250 抑制；只激活指定的位点子集。'],
['物理时间','主固定步长 dt=0.1 ms、500 ms 无输入准备、有效 1 / 6 s；胞体电压按严格物理时间每 1 ms 对齐。psolve 终点若尚未执行记录事件，直接读取同一时刻的终端状态。'],
['输入','同一量化 raster，事件从 500 ms 开始；E/I 权重及时间常数固定。每个匹配对照保存输入 hash 和 active IDs hash。'],
['NEURON / Core 数值','重新编译相同的 12 个活跃机制供两引擎使用；未使用的旧 epsp.mod 不参与模型。已核对冻结原机制版本；CoreNEURON 与对应 NEURON 主网格最大电压差 '+f'{nrn_error:.2e} mV。'],
['输出记录','普通 NEURON / CVode 每 1 ms 记录；CoreNEURON 和最终 multisplit 对照按内部 dt 记录，再按时间戳降采样。这样可同时避开 Core 自定义间隔记录限制和多线程自定义记录的时序差异。所有模式均检查输出时刻；额外整理成本单独记录。'],
['重复与选型','NEURON 预热 1 次 + 3 次测量；三条精度轨迹为单次重放。GPU 每形状先筛选扫描块，再独立随机交错测量 40 次，保留 median / mean / std / p95；CPU 每项 15 次。'],
['资源','CPU affinity 固定为物理核 48 起的连续核；GPU 测试宿主线程固定 CPU 80，CUDA_VISIBLE_DEVICES=6。各 CPU 基准依次执行，服务器其他任务保留。'],
['精度','NEURON 使用双精度计算；DBNN 主路径为 FP32，TF32 显式另列。数值后端一致性用完整输出和 reset 时间格核对。'],
['端到端','一次准备记录覆盖已启动进程中的模型构建、优化对象构建、编译 / capture 和调优；Python 启动与磁盘读取不在主计时内。稳态 core、整理 / 传输边界和重置 trial 总时间分别记录。']])}
<p>Docker 可以固定依赖版本，并通过 cpuset / 线程环境变量限制资源。CPU quota 限制的是可用 CPU 时间，可能引入节流；taskset / cpuset 也不会自动排除其他进程。正式速度基准仍需预留物理 CPU 核与空闲 GPU，并在预热后配对测量。</p>
<p>本轮完成当前安装环境可运行的 CPU 求解器、线程 / 进程方式和最新 GPU 代理对照。CoreNEURON GPU 和跨节点 MPI 列为下一步独立实验，报告中没有代入未经测量的速度值。</p>
<p class="muted">官方依据：<a href="https://www.neuronsimulator.org/en/9.0.0/coreneuron/running-a-simulation.html">CoreNEURON 运行与状态交换</a>；<a href="https://www.neuronsimulator.org/en/9.0.0/progref/modelspec/programmatic/network/netcon.html">NetCon.event(t) 的交付时间语义</a>；<a href="https://github.com/neuronsimulator/nrn/discussions/2090">CoreNEURON 自定义记录间隔的限制与替代方式</a>。本轮通过实测确认逐 dt 记录后降采样可恢复完整轨迹。</p>
<details class="history-fold"><summary>下载本轮全部配置、计时分布及分析数据</summary><div class="history-body"><p>新实验目录：<code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/neuron_single_20260928/</code>。各 worker 保存 JSON、NPZ 和日志；输入与输出均可追溯。</p><button type="button" id="download-neuron">下载 NEURON 主对照数据 JSON</button></div></details></section>'''
    nextsec='''<section id="next"><h2>6. 下一步：精度目标、在线场景与正式速度基准</h2><p>首先对齐原始数据生成协议，完成充分训练，并以电压 RMSE、动作电位时序和事件召回率共同筛选 checkpoint；再将训练好的权重接入当前优化后端。核函数加速已经能够与模型训练解耦。</p><p>应用上继续验证沿树突的密集位置采样、未见输入位点的近邻近似与位置分辨率误差。当前生物模型验证覆盖注册的 500 个位点；超过该范围的旧规模实验保留在附录，作为计算规模证据。</p><p>计算上加入持续在线的滤波状态与 reset 状态迁移、滑动窗口及更短延迟目标。随后在预留 CPU / GPU 的环境进行独占复测，并构建 CoreNEURON GPU 版本作同设备对照；同时记录一次准备成本和可摊销的调用次数。</p></section>'''
    sections=dict(conclusion=intro,**{'neuron-main':sec2,'neuron-accuracy':sec3,'neuron-throughput':sec4,'neuron-protocol':sec5,'next':nextsec})
    (H/'analysis.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    (H/'new_sections.json').write_text(json.dumps(sections,ensure_ascii=False))
    return sections,evidence

def reorder(body,sections,evidence):
    # Preserve each old section and its scripts/MathML byte for byte except titles.
    starts=list(re.finditer(r'<section\b[^>]*id="([^"]+)"[^>]*>',body))
    old={}
    for m in starts:
        end=body.index('</section>',m.end())+len('</section>');old[m.group(1)]=body[m.start():end]
    main=re.search(r'<main[^>]*>',body);endmain=body.index('</main>',main.end())
    head=body[main.end():starts[0].start()]
    head=re.sub(r'<h1>.*?</h1>','<h1>单神经元的快速代理：<br>完整优化 DBNN 与 NEURON 的对照</h1>',head,flags=re.S)
    head=re.sub(r'<p>.*?</p>','<p>与导师讨论用的阶段报告 · 单轨迹延迟 / 多输入吞吐量 / 数值与生物精度 · 完整优化链路已纳入主结果</p>',head,flags=re.S)
    title_map={'current-selection':'备用 A. 当前后端选型与淘汰记录','gpu':'备用 A.1 GPU：位点数、活跃度和数据传输（原第二章）','original-audit':'备用 A.2 逐层核对：从原 F / R 替换一个模块','neuron':'备用 B. 早期 NEURON 单线程基准（历史计时）','backend-ablation':'备用 C. 后端探索与历史对照','drive-analysis':'备用 C.1 初版完整驱动项（已由优化版替代）','quadratic-analysis':'备用 C.2 初版纯二次项与 GEMM','quad-specialized':'备用 D. 当前优化版二次型融合 kernel（FP32）','tf32-five-layers':'备用 E. TF32 五层调优与精度选项','fidelity':'备用 F. 数据重建、训练记录与原测试集评估','raw':'原始记录与下载'}
    for key,title in title_map.items():old[key]=re.sub(r'<h2>.*?</h2>',f'<h2>{title}</h2>',old[key],count=1,flags=re.S)
    prior=old['conclusion'].replace('id="conclusion"','id="prior-conclusion"',1)
    prior=re.sub(r'<h2>.*?</h2>','<h2>备用 G. 前几轮结果摘要（原始记录）</h2>',prior,count=1,flags=re.S)
    prior=prior.replace('</h2>','</h2><p class="muted">以下保留历史测量及当时的资源状态；本轮 NEURON 主对照见第二至第五节。</p>',1)
    previous_next=old['next'].replace('id="next"','id="prior-next"',1)
    previous_next=re.sub(r'<h2>.*?</h2>','<h2>备用 H. 先前讨论的优化方向</h2>',previous_next,count=1,flags=re.S)
    order=[sections['conclusion'],old['design'],sections['neuron-main'],sections['neuron-accuracy'],sections['neuron-throughput'],sections['neuron-protocol'],sections['next']]
    backup=['current-selection','gpu','original-audit','neuron','backend-ablation','drive-analysis','quadratic-analysis','quad-specialized','tf32-five-layers','fidelity']
    order += [old[k] for k in backup]+[prior,previous_next,old['raw']]
    footer=f'<footer>更新于 {time.strftime("%Y-%m-%d %H:%M")} · 同一份持续更新的单文件报告；图像与数据内嵌。NEURON 主对照已重新按物理时间对齐。</footer>'
    result=body[:main.end()]+head+'\n'.join(order)+footer+body[endmain:]
    nav='<div class="toc-body"><div class="toc-label">NEURON 主对照</div><ul>'
    links=[('report-top','报告首页'),('conclusion','本轮结果'),('design','1. 模型与五层优化'),('neuron-main','2. NEURON 方式与速度'),('neuron-accuracy','3. 精度与事件密度'),('neuron-throughput','4. 多输入吞吐量'),('neuron-protocol','5. 协议与待补对照'),('next','6. 下一步')]
    nav+=''.join(f'<li><a href="#{k}" data-section-link>{label}</a></li>' for k,label in links)+'</ul><div class="toc-label">备用资料</div><ul>'
    nav+=''.join(f'<li><a href="#{k}" data-section-link>{html.escape(title_map[k])}</a></li>' for k in backup)
    nav+='<li><a href="#prior-conclusion" data-section-link>G. 历史摘要</a></li><li><a href="#prior-next" data-section-link>H. 先前方向</a></li><li><a href="#raw" data-section-link>原始记录与下载</a></li></ul><div class="toc-footer"><a href="#report-top">回到顶部</a></div></div>'
    result=re.sub(r'<div class="toc-body">.*?</div>\s*</details>',nav+'</details>',result,count=1,flags=re.S)
    payload=json.dumps(evidence,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    extra='<script type="application/json" id="neuron-data">'+payload+'</script><script>document.getElementById("download-neuron").onclick=()=>{const b=new Blob([document.getElementById("neuron-data").textContent],{type:"application/json"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="DBNN_NEURON_single_cell_20260928.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script>'
    result=result.replace('</body>',extra+'</body>')
    return result

if __name__=='__main__':
    sections,evidence=build()
    # Source stays the approved report until the whole new dataset passes checks.
    p=Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN/DBNN_L5PC_surrogate_report_20260928.html')
    before=H/'report_before_neuron_focus.html'
    if not before.exists():before.write_text(p.read_text())
    body=before.read_text();assert 'id="neuron-main"' not in body
    new=reorder(body,sections,evidence);(H/'updated_report.html').write_text(new)
    print('BUILT',H/'updated_report.html',len(new.encode()),'bytes')
