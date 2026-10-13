"""Generate the self-contained advisor report and exportable scientific plots."""
from pathlib import Path
import base64,html,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from original_layer_guide import build_layer_guide, LAYER_CSS, LAYER_JS
from baseline_report import build_baseline_section
from quadratic_report import build_quadratic_section
from drive_report import build_drive_section
from model_formulas import build_model_formulas, MODEL_CSS
from quad_optimized_report import build_optimized_quad_section
from tf32_report import build_tf32_section
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
def load(name):return json.loads((HERE/name).read_text())
S=load('results_event.json');C=load('confirmation.json');V=load('validation.json');F=load('fidelity.json')
N=load('neuron_results.json')
F['same_raster_neuron'].sort(key=lambda r:r['case'])
X=load('transfer_pairs.json');xs={r['index']:r for r in X['cases']}
cs={r['index']:r for r in C['cases']};ss={r['index']:r for r in S['cases']};ns={r['index']:r for r in N['cases']}
if set(cs)!=set(range(14)) or set(ns)!=set(range(14)):raise RuntimeError('All 14 paired scenarios must finish before publishing the final report.')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white'})
blue='#3168c9';green='#11866b';orange='#ce7c27';grey='#6b7787';red='#b5474a'
def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(FIG/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def pic(name,alt):
    return '<img src="data:image/png;base64,'+base64.b64encode((FIG/f'{name}.png').read_bytes()).decode()+'" alt="'+html.escape(alt)+'">'
def ms(c,name='optimized'):return c['timings'][name]['median_ms']
def fmt(x):return f'{x:,.3f}' if x<100 else f'{x:,.1f}'
def sname(c):return f"N={c['N']:,}, A={c['A']}, B={c['B']}, T={(c['T']-1)/1000:g}s"

# Candidate-site scaling, same A and T. Includes a packed PyTorch control.
fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,ids,A in zip(axs,[[6,7,8],[9,10,11]],[32,128]):
    rows=[cs[i] for i in ids];x=[r['N'] for r in rows]
    for label,col,key in [('Original PyTorch',blue,'best_original'),('Packed PyTorch control',orange,'packed_pytorch_best'),('Optimized kernels',green,None)]:
        y=[ms(r,r[key]) if key else ms(r) for r in rows]
        ax.plot(x,y,'o-',color=col,label=label)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(x,[str(t) for t in x]);ax.set_xlabel('candidate sites N')
    ax.set_title(f'{A} active sites, 1 s, batch 1')
axs[0].set_ylabel('GPU core-forward wall time (ms)');axs[0].legend(frameon=False,fontsize=9)
fig.tight_layout();save(fig,'candidate_scaling')

fig,axs=plt.subplots(1,2,figsize=(11,4.4))
rows=[cs[i] for i in [3,1,2,0]];a=[r['A'] for r in rows]
for label,col,key in [('Original PyTorch',blue,'best_original'),('Packed PyTorch',orange,'packed_pytorch_best'),('Optimized',green,None)]:
    axs[0].plot(a,[ms(r,r[key]) if key else ms(r) for r in rows],'o-',label=label,color=col)
axs[0].set_xscale('log',base=2);axs[0].set_xticks(a,[str(t) for t in a]);axs[0].set_yscale('log')
axs[0].set_xlabel('active sites A (N=500, T=6 s, B=1)');axs[0].set_ylabel('GPU wall time (ms)');axs[0].legend(frameon=False,fontsize=9)
rows=[cs[i] for i in [0,1,4,5,8,11]];xx=np.arange(len(rows));w=.36
axs[1].bar(xx-w/2,[xs[r['index']]['timings']['core']['median_ms'] for r in rows],w,color=green,label='Core forward (paired session)')
axs[1].bar(xx+w/2,[xs[r['index']]['timings']['packed_h2d_forward_d2h']['median_ms'] for r in rows],w,color=orange,label='Packed H2D + forward + D2H')
axs[1].set_xticks(xx,[f"{r['N']}/{r['A']}\nB{r['B']}, {(r['T']-1)/1000:g}s" for r in rows],fontsize=8)
axs[1].set_yscale('log');axs[1].set_ylabel('wall time (ms)');axs[1].set_xlabel('N/A, batch and simulated duration')
axs[1].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'active_and_transfer')

# Fair same-thread CPU comparison with NEURON. GPU is a separate figure.
ids=[0,1,2,3,8,11];rows=[ns[i] for i in ids];x=np.arange(len(rows));w=.25
fig,ax=plt.subplots(figsize=(11,4.9))
for offset,label,col,vals in [(-w,'NEURON, CPU 1 thread',grey,[r['core']['median_ms'] for r in rows]),
    (0,'Original PyTorch, CPU 1 thread',blue,[min(v['median_ms'] for k,v in r['cpu_one_thread'].items() if k.startswith('pytorch')) for r in rows]),
    (w,'Packed native, CPU 1 thread',green,[r['cpu_one_thread']['packed_native_cpu_1t']['median_ms'] for r in rows])]:
    ax.bar(x+offset,vals,w,label=label,color=col)
ax.set_yscale('log');ax.set_xticks(x,[f"N{r['N']}/A{r['A']}\n{(r['T']-1)/1000:g}s" for r in rows]);ax.set_ylabel('core-forward wall time (ms, log scale)')
ax.legend(frameon=False,ncol=3,fontsize=9,loc='lower center',bbox_to_anchor=(.5,1.02));ax.set_title('Single-thread CPU: surrogate algorithm and sparse support',pad=40);fig.tight_layout();save(fig,'cpu_neuron')

# Each baseline has its own panel and explicit hardware scope.
ids=sorted(cs);yy=np.arange(len(ids));fig,axs=plt.subplots(1,2,figsize=(11,7.2),sharey=True)
vals=[[cs[i]['speedup'] for i in ids],[ns[i]['core']['median_ms']/ms(cs[i]) for i in ids]]
for ax,v,col,title in zip(axs,vals,[blue,green],['Original PyTorch GPU / optimized GPU','NEURON CPU 1T / optimized GPU']):
    ax.barh(yy,v,color=col,height=.65);ax.set_xscale('log');ax.set_xlim(.7,max(v)*2.2)
    ax.axvline(1,color=grey,lw=.8);ax.set_xlabel('speed ratio (log scale)');ax.set_title(title,fontsize=11)
    for y,z in zip(yy,v):ax.text(z*1.04,y,f'{z:,.1f}x' if z<100 else f'{z:,.0f}x',va='center',fontsize=8)
axs[0].set_yticks(yy,[f"#{i} N{cs[i]['N']}/A{cs[i]['A']}, B{cs[i]['B']}, {(cs[i]['T']-1)/1000:g}s" for i in ids],fontsize=8)
axs[0].invert_yaxis();fig.tight_layout();save(fig,'speedup_dual_baseline')

# Backend exploration, equal graph mode for each column.
cols=['packed_cuda_cublas_graph','packed_triton_cublas_graph','packed_cute_cublas_graph','packed_fft_cublas_graph','packed_fft_cutlass_graph','packed_cuda_direct_graph']
labels=['CUDA/CUB\n+ cuBLAS','Triton scan\n+ cuBLAS','CuTe DSL\n+ cuBLAS','FFT\n+ cuBLAS','FFT\n+ CUTLASS SIMT','CUDA scan\n+ fused quadratic']
ids=[0,1,2,4,5,8,11];mat=np.array([[ss[i]['methods'][k]['median_ms'] for k in cols] for i in ids])
rel=mat/mat.min(axis=1,keepdims=True)
fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(np.log2(rel),cmap='Blues',aspect='auto',vmin=0,vmax=max(1,np.log2(rel.max())))
for i in range(len(ids)):
    for j in range(len(cols)):
        ax.text(j,i,f'{mat[i,j]:.3f} ms',ha='center',va='center',color='white' if np.log2(rel[i,j])>np.log2(rel.max())*.55 else '#182230',fontsize=9)
ax.set_xticks(np.arange(len(cols)),labels,fontsize=8);ax.set_yticks(np.arange(len(ids)),[f"N{cs[i]['N']}/A{cs[i]['A']}, B{cs[i]['B']}" for i in ids],fontsize=9)
ax.grid(False);ax.set_title('Backend exploration: 30 interleaved measurements / configuration')
fig.colorbar(im,ax=ax,label='log2(latency / fastest graph in row)',shrink=.8);fig.tight_layout();save(fig,'backend_matrix')

fig,ax=plt.subplots(figsize=(9,4.3));keys=list(V['ablation']);xx=np.arange(len(keys));w=.26
for of,label,col,key in [(-w,'New sequential CUDA reset',grey,'sequential_reset'),(0,'New event CUDA reset',green,'event_reset'),(w,'CUTLASS FastF32 + event reset',orange,'cutlass_tensor_event')]:
    vals=[V['ablation'][k][key]['median_ms'] for k in keys];ax.bar(xx+of,vals,w,label=label,color=col)
ax.set_xticks(xx,[k.replace('N500_','').replace('_',', ') for k in keys]);ax.set_ylabel('GPU wall time (ms)');ax.set_title('Internal backend ablation: both CUDA reset implementations are new')
ax.legend(frameon=False,fontsize=9);fig.tight_layout();save(fig,'reset_ablation')

traces=np.load(HERE/'fit_traces.npz');yp=traces['prediction'];yt=traces['target']
rs=np.sqrt(((yp-yt)**2).mean(1));mid=int(np.argmin(np.abs(rs-np.median(rs))));t=np.arange(6001)/1000
peak=int(np.argmax(np.abs(yp[mid]-yt[mid])));lo=max(0,peak-50);hi=min(6001,peak+51)
fig,axs=plt.subplots(2,1,figsize=(11,5.8),gridspec_kw={'height_ratios':[1.25,1]})
for ax,start,end in [(axs[0],0,6001),(axs[1],lo,hi)]:
    ax.plot(t[start:end],yt[mid,start:end],color=grey,lw=1,label='NEURON target')
    ax.plot(t[start:end],yp[mid,start:end],color=green,lw=1,label='Reference DBNN')
    ax.set_ylabel('soma voltage (mV)');ax.set_xlabel('time (s)')
axs[0].set_title(f'Median-RMSE held-out trace: sample {mid}, RMSE {rs[mid]:.3f} mV');axs[0].legend(frameon=False,ncol=2)
axs[1].set_title('Zoom at the largest voltage discrepancy in the selected trace');fig.tight_layout();save(fig,'fit_trace')

fig,axs=plt.subplots(1,2,figsize=(11,4.3))
axs[0].bar(np.arange(10),rs,color=blue);axs[0].axhline(rs.mean(),color=green,ls='--',label=f'mean {rs.mean():.3f} mV');axs[0].legend(frameon=False)
axs[0].set_xlabel('held-out sample');axs[0].set_ylabel('sample RMSE vs NEURON (mV)');axs[0].set_title('Surrogate approximation fidelity')
eq=V['equivalence'];names=[k for k,v in eq.items() if 'max_abs_mv' in v]
axs[1].bar(np.arange(len(names)),[eq[k]['max_abs_mv'] for k in names],color=green)
axs[1].set_xticks(np.arange(len(names)),[k.replace('_','\n') for k in names],fontsize=8);axs[1].set_ylabel('max |optimized - PyTorch| (mV)');axs[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axs[1].set_title('Backend equivalence: same frozen checkpoint')
fig.tight_layout();save(fig,'fidelity_equivalence')

LR=json.loads((ROOT/'results/low_rank_screen_20260928.json').read_text());ranks=sorted(int(k) for k in LR['ranks'])
fig,ax=plt.subplots(figsize=(8,3.8))
for key,label,col in [('voltage_rmse_mv','RMSE',blue),('max_abs_mv','Maximum error',red)]:
    ax.plot(ranks,[LR['ranks'][str(k)][key] for k in ranks],'o-',color=col,label=label)
ax.set_yscale('log');ax.set_xlabel('retained rank');ax.set_ylabel('error vs dense DBNN (mV)');ax.legend(frameon=False);fig.tight_layout();save(fig,'lowrank')

def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+str(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
gpurows=[];cpurows=[]
for i,c in sorted(cs.items()):
    n=ns[i];orig=c['best_original'];pg=c['packed_pytorch_best']
    gpurows.append([i,sname(c),fmt(ms(c,orig)),fmt(ms(c,pg)),fmt(ms(c)),fmt(c['timings']['optimized']['p95_ms']),f"{c['speedup']:.2f}×",f"{c['kernel_speedup_over_packed']:.2f}×"])
    origcpu=min(v['median_ms'] for k,v in n['cpu_one_thread'].items() if k.startswith('pytorch'));newcpu=n['cpu_one_thread']['packed_native_cpu_1t']['median_ms'];neuron=n['core']['median_ms']
    cpurows.append([i,sname(c),fmt(neuron),fmt(origcpu),fmt(newcpu),f'{origcpu/newcpu:.1f}×',f'{neuron/newcpu:,.0f}×',f'{neuron/ms(c):,.0f}×'])

keys=['reference','scratch_1ep_cpu','scratch_1ep_gpu','finetune_1ep','finetune_4ep_partial']
titles=['旧参考 checkpoint','从头训练 1 epoch / CPU','从头训练 1 epoch / GPU','参考初始化微调 1 epoch','4 epoch 微调中的最佳保存点']
fitrows=[[name,f"{V['checkpoints'][key]['mean_sample_rmse_mv']:.4f}",f"{V['checkpoints'][key]['pooled_rmse_mv']:.4f}"] for key,name in zip(keys,titles)]
spikes=F['test_samples'];true=sum(len(s['neuron_spikes_ms']) for s in spikes);pred=sum(len(s['dbnn_reset_spikes_ms']) for s in spikes)
match=sum(s['matches']['5'] for s in spikes);prec=match/pred if pred else 0;recall=match/true if true else 0
details=[]
for i,c in sorted(cs.items()):
    rr=[]
    for name,v in sorted(ss[i]['methods'].items()):
        rr.append([html.escape(name),fmt(v['median_ms']),fmt(v['p95_ms']),f"{v['rmse_mv']:.2e}",f"{v['max_abs_mv']:.2e}",v['spike_mismatch_bins'],v['n']])
    details.append(f'<details><summary>配置 {i}：{sname(c)} · 独立复测路径 {html.escape(c["backend"])}</summary>'+table(['实现','median ms','p95 ms','RMSE mV','max mV','reset 不同格数','重复数'],rr)+'</details>')
headline=cs[8];sparse=ns[1];sparsecpu=sparse['cpu_one_thread']['packed_native_cpu_1t']['median_ms'];sparseold=min(v['median_ms'] for k,v in sparse['cpu_one_thread'].items() if k.startswith('pytorch'))
quadratic_section, quadratic_data = build_quadratic_section(HERE, table, pic, save)
drive_section, drive_data = build_drive_section(HERE, table, pic, save, quadratic_data)
baseline_section, baseline_data = build_baseline_section(HERE, table, pic, save)
specialized_section, specialized_data = build_optimized_quad_section(HERE, table, pic, save)
tf32_section, tf32_data = build_tf32_section(HERE, table, pic, save)
assert baseline_data['baseline_sha256'] == S['baseline_sha256']
assert baseline_data['checkpoint_sha256'] == S['checkpoint_sha256']
raw_json=json.dumps({'confirmation':C,'neuron':N,'validation':V,'fidelity':F,'transfer_pairs':X,'quadratic_isolated':quadratic_data,'drive_forward':drive_data,'original_baseline_audit':baseline_data,'quadratic_specialized':specialized_data,'tf32_five_layers':tf32_data},ensure_ascii=False).replace('</', '<\\/')
layer_guide, layer_popover = build_layer_guide(cs, baseline_data, specialized_data, tf32_data)
model_formulas = build_model_formulas(table)

body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>DBNN 稀疏位点加速实验报告 · 2026-09-28</title>
<style>
:root{{--ink:#183044;--muted:#587080;--line:#d4e0e5;--accent:#087b68;--blue:#3168c9;--bg:#f2f6f7}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1160px;margin:auto;padding:30px 24px 70px}}header{{padding:34px;background:#123d4b;color:white;border-radius:14px}}
h1{{font-size:30px;line-height:1.3;margin:8px 0 12px}}h2{{font-size:23px;margin:0 0 14px}}h3{{font-size:17px;margin:24px 0 8px}}
section{{padding:26px;margin-top:22px;background:white;border:1px solid var(--line);border-radius:12px}}p{{margin:9px 0}}.eyebrow{{font-size:12px;letter-spacing:1px}}
nav{{display:flex;gap:16px;flex-wrap:wrap;margin-top:22px}}a{{color:var(--accent)}}nav a{{font-size:14px}}.muted,figcaption{{color:var(--muted);font-size:13px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;padding:18px 0}}.stats b{{display:block;font-size:27px;color:var(--accent)}}
.note{{background:#edf6f3;border-left:4px solid var(--accent);padding:12px 16px;margin:16px 0}}.warm{{background:#fff5e8;border-color:#b57519}}
figure{{margin:20px 0}}img{{max-width:100%;height:auto;display:block}}figcaption{{margin-top:8px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#edf3f5}}td:first-child{{font-weight:600}}
code,pre{{font-family:ui-monospace,SFMono-Regular,monospace}}code{{font-size:12px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f5;padding:14px;border-radius:8px;font-size:12px}}
details{{border-bottom:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;font-weight:500}}button{{font:inherit;background:#087b68;color:white;border:0;border-radius:6px;padding:9px 15px;cursor:pointer}}
.formula{{text-align:center;background:#f2f7f6;padding:16px;line-height:2;font-family:Georgia,serif;font-size:17px}}footer{{font-size:12px;color:var(--muted);margin-top:24px}}
@media(max-width:700px){{main{{padding:14px 10px 40px}}header,section{{padding:20px 16px}}h1{{font-size:25px}}.two,.stats{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}header{{background:white;color:var(--ink);padding:0}}section{{border:0;padding:12px 0;break-inside:auto}}figure{{break-inside:avoid}}nav,button{{display:none}}details table{{display:table}}}}
{LAYER_CSS}
{MODEL_CSS}
</style></head><body><main>
<header><div class="eyebrow">DBNN / L5PC SURROGATE DEVELOPMENT · 2026-09-28</div><h1>密集候选位点，稀疏输入：<br>从活跃位点压缩到 GPU 事件驱动计算</h1>
<p>与导师讨论用的阶段报告 · 原始 PyTorch / NEURON 双基线 · 14 种工作负载 · CUDA、CuTe DSL、CUTLASS 与 Triton 实测</p></header>
<nav><a href="#conclusion">主要结果</a><a href="#design">五层优化</a><a href="#gpu">GPU 对比</a><a href="#original-audit">原代码逐层配对</a><a href="#neuron">CPU / NEURON</a><a href="#drive-analysis">驱动项内部消融</a><a href="#quadratic-analysis">二次项内部消融</a><a href="#quad-specialized">IEEE 二次型 kernel</a><a href="#tf32-five-layers">TF32 五层调优</a><a href="#fidelity">拟合与误差</a><a href="#next">后续推进</a><a href="#raw">原始记录</a></nav>

<section id="conclusion"><h2>当前进展与主要结果</h2>
<p>已经实现整窗活跃位点压缩、CUDA/CUB 与 CuTe DSL 的并行前缀扫描、融合二次项、缓存 FFT、CUDA Graph，以及 GPU 内部的事件驱动 reset。原始 <code>source/DBNN.py</code> 保持原样。所有后端使用同一个冻结 checkpoint；新增算法先核对电压与 reset 时序，再做速度比较。</p>
<div class="stats"><div><span>8,000 候选 / 32 活跃 / 1 s / B=1</span><b>{headline['speedup']:.1f}×</b><span class="muted">相对同 GPU 原始 PyTorch 的较快实现</span></div>
<div><span>500 候选 / 32 活跃 / 6 s / CPU 单线程</span><b>{sparseold/sparsecpu:.1f}×</b><span class="muted">原始 PyTorch → 稀疏 C++ CPU</span></div>
<div><span>10 条完整测试轨迹</span><b>≈ 1.1 × 10⁻⁵ mV</b><span class="muted">FP32 优化路径与原始实现的最大差异；reset 格数差 0</span></div></div>
<div class="note warm">资源口径：GPU 3 上仍有其他训练任务，本轮没有暂停它们。CPU 线程数与 affinity 已固定，GPU 方法采用随机交错计时；探索阶段每点 30 次，独立复测 50 次。这里保留完整计时分布，适合比较当前实现，独占资源复测仍是正式发表速度结果前的下一步。</div>
<p>主基线直接使用你的原始 F / R 代码，并增加相同活跃集合上的压缩 PyTorch 对照。Batch=8 时，原始 F/R 的整批调用和逐条调用都会计时，取较快路径作为整体比较基线。</p>
<p>本次修订将五层说明逐项对齐原计算方式，补测了“原 forward 只换一个模块”的 4 组配对实验。
原 R 已有缓存和分块 cumsum，原 B=1 reset 已按事件推进，均明确列入基线。后端之间的比较集中到第 4 节内部消融。</p>
<div class="note"><strong>上一轮 IEEE 二次型专用 kernel：已完成两轮复测。</strong>
采用 IEEE FP32 的跨时间块复用与上三角收缩，在 500 活跃位点、B=8、6 s 时，完整驱动项为 324.30 μs，同轮 cuBLAS＋归约为 636.49 μs；
相同滤波 / reset 下完整 DBNN 的 Graph wall time 为 0.700 ms，对照为 0.973 ms。
另外单独放回原 F/R 进行了配对，五层说明中的二次项已更新。<a href="#quad-specialized">查看第 4.3 节的新图、原代码对照及全部 9 种形状</a>。
此前 14 配置主表保留原始测量，新结果独立列出；新测量使用共享 GPU 6。</div>
<div class="note"><strong>最新进展：TF32 五层调优完成。</strong>
新增 {tf32_data['summary']['quad_candidates']} 个二次项候选配置、{tf32_data['summary']['filter_candidates']} 个滤波候选配置，覆盖 13 种形状，并完成两轮完整推理复测。
Tensor Core 既用于二次型，也新增了短窗矩阵滤波；位点收集、reset 和调度分别核查。
提供沿用 10⁻³ mV 门限的 profile，以及显式选择的 0.01 mV 近似 profile；两者均要求已测 reset 时间点一致。
<a href="#tf32-five-layers">查看第 4.4 节速度、精度和五层结论</a>。</div></section>

<section id="design"><h2>1. 从应用场景出发的计算设计</h2>
{model_formulas}
{layer_guide}
<div class="note">500 位点实验使用原始 checkpoint 和重建测试输入。N=2,000 / 8,000 使用原参数的确定性扩展做计算规模实验，尚未在对应密集形态位点上重新训练。因此大 N 的结论是计算扩展性；其生物拟合精度留待密集采样验证。</div></section>

<section id="gpu"><h2>2. GPU：位点数、活跃度和数据传输</h2>
<figure>{pic('candidate_scaling','固定活跃位点时，候选位点数量与 GPU 耗时关系')}<figcaption>固定 A 与时间长度时，压缩后计算主要由 A 决定。蓝线为全维原始 PyTorch；橙线为同样活跃集上的原始 PyTorch；绿线为选定 kernel 的独立复测。图中包含 GPU 共享调度波动。</figcaption></figure>
<figure>{pic('active_and_transfer','活跃位点数量影响，以及包含主机传输后的耗时')}<figcaption>右图是额外的配对计时：core 和“主机端 packed raster 上传 → 前向 → 结果下载”随机交错各 100 次；不包含原始事件构造、位点选择或首次权重准备。它与主表的 PyTorch 对照在不同测量时段完成；共享 GPU 的调度波动使两批绝对耗时有所不同。</figcaption></figure>
{table(['#','配置','原 PyTorch ms','压缩 PyTorch ms','优化 ms','优化 p95 ms','总提升','kernel 提升'],gpurows)}
<p class="muted">总提升＝全维原始 PyTorch / 优化；kernel 提升＝压缩 PyTorch / 优化。除明确列出的 p95，主表采用 50 次独立复测的 median；后端选择来自此前 30 次探索。所有主表优化方法使用 FP32 路径，最大误差门槛 10⁻³ mV，且 reset 时间格完全一致。</p>
<details><summary>上传与下载配对计时：100 次 / 方法</summary>{table(['#','配置','配对 core median ms','含传输 median ms','含传输 p95 ms'],[[i,sname(cs[i]),fmt(x['timings']['core']['median_ms']),fmt(x['timings']['packed_h2d_forward_d2h']['median_ms']),fmt(x['timings']['packed_h2d_forward_d2h']['p95_ms'])] for i,x in sorted(xs.items())])}</details></section>

{baseline_section}

<section id="neuron"><h2>3. NEURON 与同线程 CPU 基线</h2>
<p>NEURON 固定 dt=0.1 ms，输出为 1 ms 的胞体电压；回放与 DBNN 完全相同的 1 ms 量化事件。细胞构建、500 ms warm-up、事件排队和输出复制分别记录，core 只计有效模拟窗口。候选表中未使用的突触已移除，避免额外的无输入突触拖慢 NEURON。</p>
<figure>{pic('cpu_neuron','NEURON 与原始及优化 DBNN 的单线程 CPU 比较')}<figcaption>三组均为同一服务器 CPU 单线程。NEURON 的详细多区室模拟与 DBNN 代理模型计算量不同，因此速度需要和第 5 节拟合精度一起阅读。</figcaption></figure>
<figure>{pic('speedup_dual_baseline','所有 14 种条件相对原始 PyTorch 和 NEURON 的速度比')}<figcaption>完整的双基线速度图。左边是同 GPU 的实现加速；右边比较单线程 NEURON 与 GPU 代理，包含算法和硬件的共同差异。NEURON batch=8 是串行时间；N&gt;500 为合成模型的计算扩展性测试。</figcaption></figure>
{table(['#','配置','NEURON 1T ms','PyTorch 1T ms','优化 CPU 1T ms','CPU→CPU 提升','NEURON/优化CPU','NEURON/优化GPU'],cpurows)}
<p class="muted">最后一列包含 CPU→GPU 的硬件差异；第六列为同线程算法实现收益。NEURON 每配置 1 次预热、3 次正式测量；0.3 s 测试保留已有 5 次测量。Batch=8 的 NEURON 是 8 条轨迹顺序运行的实测，不代表已优化的多核 NEURON 吞吐上限。</p>
<p class="muted">CPU 配置 #11 的优化路径呈现明显双峰：7 次测量中，3 次约 0.82–0.84 ms，4 次约 12.19–12.38 ms。主表保留全部样本的中位数；这个点优先安排独占条件复测，并排查调度与缓存影响。</p></section>

<section><h2>4. 内部消融：选择新增实现中的后端</h2>
<p>本节比较本轮新增后端的不同组合，用于决定采用哪条实现路径。
相对原代码的整体加速见第 2 节，原代码只替换一个模块的收益见第 2.1 节。</p>
<figure>{pic('backend_matrix','CUDA Triton CuTe DSL CUTLASS 多后端探索耗时热图')}<figcaption>这些都是实际实现与测量，并非仅列候选框架。CuTe DSL 4.2.0、CUTLASS C++ 3.9.2 已在当前驱动和 5090 上跑通。CUTLASS 矩阵核仍需与扫描、归约及 reset 组合；其整体收益取决于矩阵形状。</figcaption></figure>
<figure>{pic('reset_ablation','两个新增 CUDA reset 实现之间的内部消融与 CUTLASS TensorOp 对照')}<figcaption>灰色的顺序 CUDA 核与绿色的事件 CUDA 核均为本轮新写的实现。
此内部消融固定 CUDA/CUB 扫描、输入形状与 Graph 策略；A=32、B=8 的 4.21×对应这两个新实现的完整前向之比。
原代码是 B=1 的 Python 事件循环、B&gt;1 的 Python/PyTorch 时间格循环，直接配对结果见第 2.1 节。
第三组再替换为 CUTLASS FastF32 Tensor Core 驱动项。Triton block=64/128/256/512 的数据保留于 validation.json。</figcaption></figure>
<p>后端选择按 A、T 和 batch 做校准。新补的原代码配对实验显示，B=1 原 R 对滤波替换较敏感，B&gt;1 原整批调用受逐时间格 reset 主导；原 B=1 事件分支逐条调用已作为更快对照。
FastF32 在 10 条测试轨迹上的最大实现误差为 {V['equivalence']['cuda_cutlass_tensor']['max_abs_mv']:.2e} mV，reset 时间格无变化，其数值性质与 FP32 主表分开记录。</p></section>

{drive_section}

{quadratic_section}

{specialized_section}

{tf32_section}

<section id="fidelity"><h2>5. 两种精度：代理拟合与后端一致性</h2>
<p>原始训练 NPZ 未包含在提供的 DBNN 文件夹中；本轮按原脚本的形态、位点随机种子与突触参数重建数据，明确采用 500 ms warm-up。已验证新采样脚本能够逐点重复自己的输出；旧数据文件与完整收敛训练的复现仍待补齐。</p>
<p>重建数据共 110 条轨迹，100 条训练 / 10 条测试；训练端按 80/20 划分 fit / validation。参考 checkpoint 在测试集上的<strong>逐样本 RMSE 平均值为 {rs.mean():.4f} mV</strong>，全部采样点合并 RMSE 为 {V['checkpoints']['reference']['pooled_rmse_mv']:.4f} mV。以下所有训练记录均重新按这两个统一口径计算，修正了旧报告混用聚合方式的问题。</p>
<figure>{pic('fit_trace','测试集 RMSE 中位代表轨迹，含最大误差处放大图')}<figcaption>代表轨迹按测试 RMSE 接近中位数选取（sample {mid}）；放大窗口对准该轨迹最大误差处。低平均 RMSE 与动作电位峰值波形精度需要分别评估。</figcaption></figure>
<figure>{pic('fidelity_equivalence','10 个测试样本拟合误差，以及不同优化后端的数值一致性')}<figcaption>左：模型相对 NEURON 的误差。右：计算后端相对原始 PyTorch 的差异。两者具有不同含义，后端一致性不替代生物拟合验证。</figcaption></figure>
{table(['checkpoint','平均逐轨迹 RMSE / mV','合并全部时间格 RMSE / mV'],fitrows)}
<p>从头训练目前只完成 1 epoch 的通路验证，完整收敛复现尚待继续。微调到第 4 epoch 的实验，最佳保存点仍与第 1 epoch 相同，因此两行指标相同。</p>
<p>补充的事件检查：10 条测试轨迹共有 NEURON 事件 {true} 次，DBNN reset 事件 {pred} 次；按 ±5 ms 一一匹配得到 {match} 次，precision={prec:.3f}、recall={recall:.3f}。NEURON 的事件阈值是 −20 mV，DBNN 使用 checkpoint 中的 {F['model_reset_threshold_mv']:.3f} mV reset 阈值，比较对象是各自定义的事件时刻。</p>
<div class="note">因此当前 checkpoint 在这批重建输入上的结果应同时报告：电压 RMSE、动作电位事件精度和峰值波形差异。后续训练应把事件时序与亚阈值电压分别作为评估目标，先建立一致的数据与状态初始化口径，再判断代理模型达到的精度。</div>
<p>另用高频触发、无触发与阈值附近的人工 drive 检查事件驱动 reset：最大电压差 {V['adversarial_reset']['max_abs_mv']:.2e} mV，reset 时间格差 {V['adversarial_reset']['spike_mismatch_bins']}。</p>
<details><summary>同一量化事件回放下的 NEURON / DBNN 附加检查（原 500 位点范围）</summary>{table(['配置 #','A','T steps','Batch','RMSE / mV','max / mV'],[[r['case'],r['A'],r['T'],r['B'],fmt(r['rmse_mv']),fmt(r['max_abs_mv'])] for r in F['same_raster_neuron']])}<p class="muted">输入包括稀疏子集，与原始 50 Hz 全位点训练分布可能不同；此表用于检查该分布下代理精度。</p></details>
</section>

<section id="next"><h2>6. 下一阶段：把位点密度与稀疏计算真正连接起来</h2>
{table(['优先级','推进方向','需要完成的验证'],[
['P0','密集形态位点采样与位置映射','在同一细胞的真实树突候选点建立位置表；按分支、路径距离匹配输入。保留未参与训练的空间位点，测试位置误差与电压 / 事件精度的关系。'],
['P0','独占资源复测','取得单 GPU 使用窗口并隔离主机核；冻结后端选择，重复核心表，记录功耗、频率、CPU affinity 与占用。'],
['P1','在线状态管理','为跨窗口输入维护 aᵢ、cᵢ 与 reset 状态 r。动态活跃集必须包含尚有残留状态的位点；支持新位点加入和旧状态衰减。'],
['P1','两级分块滤波与相邻阶段融合','4.4 节已实现短窗 Tensor Core 矩阵滤波和块间 FP32 状态传播。继续比较纯 FP32 多级扫描，减少局部响应工作区的读写，测试输出合成与二次项之间的片上复用；按完整误差和 wall time 选择。'],
['P1','稀疏事件入口','直接接受 (site ID, event time) 数据；分别测 CPU raster 构建、GPU scatter 与直接事件 kernel，减少稠密 N×T 输入扫描与传输。'],
['P1','CUTLASS / CuTe DSL 继续适配','测试更适合 SM120 和瘦矩阵的 tile、Tensor Core 调度、融合 epilogue；以完整电压推理时间选择实现。'],
['P1','二次型专用 kernel 的扩展与 profiling','4.3 节已实现跨时间复用 W、三角收缩和融合归约，9 种形状通过两轮复测；继续扩展 batch=2/4/16 和在线短窗口，调整寄存器 / tile 并减少部分和归约调度。Nsight 硬件计数器当前无访问权限。'],
['P1','原 R 的 W 缓存命中','系数和扫描幂缓存已确认复用；W 缓存的版本键在清零对角线前取得，导致后续 lookup 未命中。本次原基线保留原行为，下一步单独修正键更新时机并测量收益。'],
['P1','torch.compile 与自动后端','新增入口已按 A/T/B 在 9 种形状选择通过复测的 IEEE 二次型，未知形状使用 cuBLAS；继续比较编译后的原始 PyTorch，并加入 reset 触发频率的校准。高频 reset 会增加事件核的重复搜索，需要与逐步递推做速度比较。'],
['P1','完整端到端与冷启动','在双方计入事件准备、位点映射、输入构造、传输和输出转换；首次编译、权重加载与活跃集变化另列。当前已测 core 和预打包输入的传输口径。'],
['P2','近似模式误差预算','对 TF32/BF16/FP16、状态尾部截断和低秩模型分别约束电压误差、最大误差与 spike/reset 时刻；不能只检查谱能量。'],
['P2','紧凑三角存储与块稀疏近似','4.3 节已采用保持同一二次型的上三角计算，当前 U 仍为二维权重张量；可继续减少权重存储与准备成本。低秩或截断块稀疏属于近似模式，需独立测精度—速度曲线。'],
['P2','多核与详细模拟器基线','加入 NEURON 多细胞并行、可兼容情况下的 CoreNEURON；与代理模型做相同核数的批量吞吐比较。'],
['P2','CPU 原生部署 / Rust','已经完成 C++/OpenMP 原型；Rust 可承接部署、位点索引和并行服务。进一步优先测 BLAS、SIMD、内存布局，收益由相同任务实测决定。'],
['P2','完整训练与泛化','从头收敛训练；覆盖输入频率、兴奋/抑制比例、簇状输入和新空间位点。随后建立精度—速度曲线。']])}
<details><summary>前一轮低秩结果保留：速度尚未实测</summary><figure>{pic('lowrank','二次权重低秩近似的电压误差')}<figcaption>秩 384 的实现差异 RMSE≈0.0064 mV、max≈0.0315 mV；更低秩可能造成约 8 mV 的 reset 差异。该近似没有加入本轮 FP32 主表。</figcaption></figure></details></section>

<section id="raw"><h2>7. 测量协议与可追溯记录</h2>
<p>运行于 5090 服务器的 RTX 5090（主基准使用物理 GPU 3，原代码逐层配对与 4.3 节新二次型使用 GPU 6，4.1 / 4.2 节内部消融使用 GPU 7），AMD EPYC 9554；PyTorch 2.8.0+cu128、CUDA Toolkit 12.8、驱动 570.133.07、Triton 3.4.0、CuTe DSL 4.2.0、CUTLASS C++ 3.9.2、NEURON 9.0.2。本轮使用项目 .venv；没有通过 Docker 建立硬件独占。</p>
<p>主指标为每次推理前后 synchronize 的 <code>time.perf_counter()</code> wall time，原始记录同时保留 mean、std、p95、每次耗时以及 GPU 利用率快照。探索还记录 CUDA event elapsed time。CPU DBNN 主表与 NEURON 都设单线程；附录记录了 CPU 8 线程结果。GPU 主机任务绑定 CPU 48–55，NEURON 绑定 CPU 56；affinity 限制执行位置，不保证其他进程不使用这些核。</p>
<p>通常同时报告三项：单次延迟（wall time，回答“等多久”）、吞吐量 B / wall time（回答“每秒多少条轨迹”）、模拟时间倍率 B × 模拟时长 / wall time（回答“每秒处理多少生物时间”）。本轮属于整窗离线推理；CUDA event 用于辅助分析设备时间，最终应用还要单列数据准备、传输和冷启动成本。</p>
<p>Docker 可以固定环境、cpuset 和设备可见性；<code>--gpus device=...</code> 并不独占 GPU，<code>--cpus</code> 是 CPU quota，可能引入节流。相关定义参见 <a href="https://docs.docker.com/engine/containers/resource_constraints/">Docker resource constraints</a>；异步 GPU 计时见 <a href="https://docs.pytorch.org/docs/stable/notes/cuda.html">PyTorch CUDA semantics</a>；CuTe DSL 见 <a href="https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html">NVIDIA 官方文档</a>。</p>
<pre>项目：/home/lcm/DBNN_L5PC_surrogate_20260928
脚本与原始结果：benchmarks/sparse_20260928/
冻结基线 SHA256：{S['baseline_sha256']}
checkpoint SHA256：{S['checkpoint_sha256']}

results_event.json       14 配置、多后端、30 次探索
confirmation.json        独立 50 次复测、强 PyTorch 对照、传输计时
transfer_pairs.json      core 与含传输口径随机交错，各 100 次
neuron_results.json      同事件回放与单线程 CPU / NEURON
validation.json          10 轨迹误差、reset 消融、TensorOp、block sweep
fidelity.json            代理精度、事件匹配与稀疏回放精度
quadratic_isolated.json  二次模块独立 sweep，16 形状，80 次 wall / 40 次设备计时
quadratic_repeat.json    全部 8 种 6 s 形状复测，100 次 wall / 60 次设备计时
drive_forward.json       优化路径内替换整窗 drive 表达式，8 形状，内部消融
drive_forward_repeat.json  上述优化路径独立复测，100 次 wall / 60 次设备计时
original_baseline_audit.json  原 F/R 逐模块替换与原 reset 直接配对，4 形状、30 次
original_baseline_audit.py    补测脚本；不修改冻结源文件
original_baseline_variants.txt  从原 forward 逐语句替换的可审阅版本
quadratic_tiled.cu / .py  多时间点复用、三角收缩、CUDA / Triton 候选
quad_optimized.json      9 形状筛选及固定配置第一轮 50 次配对
quad_confirm.json        独立第二轮 60 次、原代码 30 次配对和边界 / 10 轨迹检查
quad_source_replace.json 原 F/R 只替换新二次型，保留原滤波、reset 和 R 分块
quad_backend_policy.json 已验证形状的 IEEE FP32 实现选择表
quad_optimized_backend.py make_optimized_prepared：推理入口，未知形状保留 cuBLAS
quad_kernel_metadata.json 编译器资源记录；硬件 profiler 返回 ERR_NVGPUCTRPERM
tf32_kernels.py          显式舍入 TF32 二次型、短窗 Toeplitz Tensor Core 滤波
tf32_quad_sweep.json     13 形状二次项系统初筛，保留编译失败、配置和误差
tf32_layer_sweep.json    滤波初筛、位点 gather / reset 的 TF32 开关控制
tf32_evaluation.json     固定组合第一轮 50 次、原代码配对、全部 10 轨迹检查
tf32_repeat.json         独立第二轮 60 次及完整误差检查
tf32_backend_policy.json strict / approximate_0p01mv 两种显式精度选择表
tf32_pipeline.py         make_precision_prepared；旧 IEEE 入口保留
tf32_api_validation.json 两种 profile 的 Graph 更新、未知形状回退、非整齐维度检查
figures/*.png, *.svg     可单独用于汇报的实验图</pre>
<p>全部 kernel 的 warm-up、Graph capture、CuTe/Triton 编译均在稳态计时之外；每个 workload 的准备成本另存。NEURON 的细胞构建、warm-up 与事件排队也在 core 之外，均留有原始耗时。</p>
{''.join(details)}
<button id="download" type="button">下载本轮数据摘要 JSON</button><button id="print" type="button" style="margin-left:10px;background:#426070">打印 / 保存 PDF</button>
</section><footer>更新于 {time.strftime('%Y-%m-%d %H:%M')} · 单文件报告，所有图内嵌，可离线打开。速度图采用真实原始计时；扩展到 N&gt;500 的模型属于计算规模实验。</footer>
</main>{layer_popover}<script type="application/json" id="data">{raw_json}</script>
<script>{LAYER_JS}</script>
<script>document.getElementById('download').onclick=()=>{{const b=new Blob([document.getElementById('data').textContent],{{type:'application/json'}});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download='DBNN_sparse_benchmark_20260928.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}};document.getElementById('print').onclick=()=>window.print();</script></body></html>'''
from report_navigation import apply_reading_guide
body=apply_reading_guide(body)
from report_neuron_focus import apply_neuron_focus
body=apply_neuron_focus(body)
output=ROOT/'DBNN_L5PC_surrogate_report_20260928.html';output.write_text(body)
print(output);print('bytes',output.stat().st_size)
