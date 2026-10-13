#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the advisor-facing HTML report for the expanded comparison."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RES=HERE/"results" if (HERE/"results").exists() else HERE
LINK_ROOT="results/" if RES != HERE else ""
FIG=HERE/"figures"; FIG.mkdir(exist_ok=True)
COL={"dbnn":"#087f69","neuron":"#b06f2b","core":"#3274a8"}

def save(fig,name):
    fig.tight_layout(); fig.savefig(FIG/f"{name}.png",dpi=190,bbox_inches="tight");fig.savefig(FIG/f"{name}.svg",bbox_inches="tight");plt.close(fig)

def table(headers,rows):
    return '<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'

def figures(d):
    ag=d["aggregates"]; db=d["budget"]["macro_rmse_mv"]
    fig,ax=plt.subplots(figsize=(8.8,4.6))
    for eng in ["neuron","core"]:
        rows=[x for x in ag if x["engine"]==eng];rows.sort(key=lambda x:x["dt_ms"])
        ax.plot([x["dt_ms"] for x in rows],[x["macro_rmse_mv"] for x in rows],marker="o",lw=2,color=COL[eng],label=eng.upper()+" CPU")
    ax.axhline(db,color=COL["dbnn"],ls="--",lw=1.6,label=f"DBNN macro RMSE = {db:.3f} mV")
    ax.set(xlabel="NEURON/CoreNEURON solver step dt (ms)",ylabel="Macro RMSE per trajectory (mV)",title="Expanded 31-case voltage error")
    ax.set_xticks([.25,.5,1.]);ax.grid(alpha=.18);ax.legend(frameon=False);save(fig,"expanded_rmse_frontier")
    # Total steady path time across the same 31 cases.
    labels=["DBNN\nGPU\ndense e2e"]; vals=[sum(x["dbnn_timing"]["dense_gather_e2e"]["median_ms"] for x in _cases(d))]; colors=[COL["dbnn"]]
    for eng in ["core","neuron"]:
        for dt in [.25,1.0]:
            a=next(x for x in ag if x["engine"]==eng and x["dt_ms"]==dt);labels.append(f"{eng.upper()}\nCPU\ndt={dt:g}");vals.append(a["total_steady_ms"]);colors.append(COL[eng])
    fig,ax=plt.subplots(figsize=(9.2,4.7));bars=ax.bar(labels,vals,color=colors,width=.62)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v*1.25,f"{v/1000:.3g} s",ha="center",fontsize=9)
    ax.set_yscale("log");ax.set_ylabel("Sum of steady solver/output wall times (ms; log scale)");ax.set_title("Same 31 trajectories: total steady comparison");ax.grid(axis="y",alpha=.18);save(fig,"expanded_speed_total")
    cats=["0","1","2","3"];series=[]
    for name,color,fun in [("DBNN",COL["dbnn"],lambda c:d["budget"]["by_spike_class"].get(c,{}).get("macro_rmse_mv",np.nan)),("NEURON dt=.25",COL["neuron"],lambda c:next(x for x in ag if x["engine"]=="neuron" and x["dt_ms"]==.25)["by_spike_class"].get(c,{}).get("macro_rmse_mv",np.nan)),("CoreNEURON dt=.25",COL["core"],lambda c:next(x for x in ag if x["engine"]=="core" and x["dt_ms"]==.25)["by_spike_class"].get(c,{}).get("macro_rmse_mv",np.nan))]:series.append((name,color,[fun(c) for c in cats]))
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(4);w=.24
    for j,(name,color,vals) in enumerate(series):ax.bar(x+(j-1)*w,vals,w,label=name,color=color)
    ax.set_xticks(x,cats);ax.set_xlabel("Reference spike-count class");ax.set_ylabel("Macro RMSE (mV)");ax.set_title("Error stratified by event complexity");ax.legend(frameon=False);ax.grid(axis="y",alpha=.18);save(fig,"expanded_rmse_by_spikes")
    # Case-level DBNN versus dt=.25 NEURON, color by reference event class.
    case_rows={c["case_id"]:c for c in d["cases"] if "dbnn_rmse_mv" in c}; nrows=[c for c in d["cases"] if c.get("engine")=="neuron" and c.get("dt_ms")==.25];
    nmap={x["case_id"]:x for x in nrows}; xvals=[];yvals=[];colors2=[]
    for k,c in case_rows.items():
        if k in nmap:xvals.append(c["dbnn_rmse_mv"]);yvals.append(nmap[k]["rmse_mv"]);colors2.append(["#6ca6a1","#d08a3e","#b84a4a","#7b4ea3"][int(c["spike_class"])])
    fig,ax=plt.subplots(figsize=(6.5,5.6));ax.scatter(xvals,yvals,c=colors2,s=44,alpha=.85,edgecolor="white",linewidth=.4);mx=max(xvals+yvals)*1.05;ax.plot([0,mx],[0,mx],"--",color="#777",lw=1);ax.set(xlabel="DBNN RMSE (mV)",ylabel="NEURON dt=0.25 ms RMSE (mV)",title="Per-case error: DBNN vs fine enough solver step");ax.grid(alpha=.18);save(fig,"expanded_case_scatter")

def _cases(d): return [c for c in d["cases"] if "dbnn_rmse_mv" in c]

def main():
    d=json.loads((RES/"analysis.json").read_text()); val=json.loads((RES/"dbnn_validation.json").read_text()); figures(d); ag=d["aggregates"]
    def img(name,alt):return f'<figure><img src="figures/{name}.png" alt="{alt}"><figcaption>{alt}</figcaption></figure>'
    rows=[]
    rows.append(["DBNN optimized GPU", "native 1 ms", f"{d['budget']['micro_rmse_mv']:.3f}",f"{d['budget']['macro_rmse_mv']:.3f}",f"{sum(c['dbnn_timing']['dense_gather_e2e']['median_ms'] for c in _cases(d)):.2f}",f"{d['budget']['spike_count_exact_fraction']*100:.1f}%"])
    for eng in ["neuron","core"]:
        for dt in [.25,.5,1.0]:
            a=next(x for x in ag if x["engine"]==eng and x["dt_ms"]==dt)
            rows.append([eng.upper()+" CPU",f"{dt:g}",f"{a['micro_rmse_mv']:.3f}",f"{a['macro_rmse_mv']:.3f}",f"{a['total_steady_ms']/1000:.2f} s",f"{a['spike_count_exact_fraction']*100:.1f}%"])
    primary=f'''<section id="result"><h2>主结果</h2><p>扩展集合包含 <strong>{d['budget']['case_count']} 条</strong>完整轨迹、<strong>{d['budget']['point_count']:,}</strong>个共同 1 ms 观察点，覆盖 1、2、4、8、16 s，active-site 数为 64、128、256、500，并包含不同位置集合与 10、25、50 Hz 输入强度。主集合按 NEURON 参考轨迹分为 25 条 0-spike、3 条 1-spike、2 条 2-spike 和 1 条 3-spike；一条 6-spike 压力轨迹单独保存，没有混入主统计。</p>{table(['方法','dt / 原生步长','micro RMSE / mV','macro RMSE / mV','31-case steady 总时间','参考放电计数完全一致'],rows)}<p>如果只用电压 RMSE，dt=1 ms 的 NEURON/CoreNEURON 也落在 DBNN 的总体 RMSE 范围内；但它们分别只有 {next(x for x in ag if x['engine']=='neuron' and x['dt_ms']==1.)['spike_count_exact_fraction']*100:.1f}% 的 case 保持放电计数。以“电压误差不超过 DBNN 且放电计数全部一致”为更严格的生物事件约束时，dt=0.25 ms 是本轮两种 CPU 求解器都满足的配置。它的总 steady 时间相对 DBNN dense-gather e2e 路径分别约为 <strong>{next(x for x in ag if x['engine']=='neuron' and x['dt_ms']==.25)['total_steady_speedup']:,.0f}×</strong>（NEURON）和 <strong>{next(x for x in ag if x['engine']=='core' and x['dt_ms']==.25)['total_steady_speedup']:,.0f}×</strong>（CoreNEURON）。</p>{img('expanded_speed_total','31 条轨迹总 steady 时间')}{img('expanded_rmse_frontier','不同 dt 的宏平均 RMSE')}{img('expanded_rmse_by_spikes','按参考放电次数分层的误差')}</section>'''
    protocol=f'''<section id="protocol"><h2>比较口径</h2><ol><li><strong>参考：</strong>同一形态、机制、500 个候选突触和同一整数毫秒输入 raster；NEURON 固定步长 dt=0.1 ms 求解，再输出到 1 ms 网格。</li><li><strong>输入：</strong>manifest 固定 active IDs 和二值事件；E/I 符号、突触权重和动力学参数与原采样协议一致。被选中的 inactive sites 在 DBNN 中以零输入压缩掉，NEURON 仍保留完整形态模型。</li><li><strong>误差：</strong>micro RMSE 将全部电压点合并；macro RMSE 先对每条完整轨迹求 RMSE，再对轨迹等权。这样长轨迹不会单独决定结论。</li><li><strong>计时：</strong>NEURON/CoreNEURON 主表是 solver + 1 ms 输出插值的 steady 时间；full 时间另计 500 ms 初始化、预热和事件排队。DBNN 主表是 active 输入的 dense gather + H2D + 全部优化推理 + D2H；CUDA Graph 捕获、权重准备和编译不计入稳态。</li><li><strong>资源：</strong>本轮在共享 5090 服务器上运行，CPU 绑定单核但未独占，GPU 使用 RTX 5090 的一个可见设备；这些 wall time 是当前资源状态下的工程测量。</li></ol>{img('expanded_case_scatter','每条轨迹的 DBNN 与 NEURON dt=0.25 ms 误差')}</section>'''
    kernel=f'''<section id="kernel"><h2>DBNN 路径与新形状验证</h2><p>DBNN 使用当前认可的完整优化路径：active-site packing、Triton 仿射前缀扫描、完整二次项/线性项/偏置融合、GPU event reset、缓存工作区和 CUDA Graph，FP32 IEEE。新时长和 active-site 形状使用默认 FP32 Triton tile；没有把未重新筛选的形状标成旧表中的专用最优配置。</p><p>对 31 个新形状，将融合路径与同一 active subset 上的 cuBLAS + reduction 路径逐 case 对照：最大绝对差为 <strong>{val['max_abs_mv']:.3g} mV</strong>，最大 RMSE 为 <strong>{val['rmse_mv_max']:.3g} mV</strong>，reset 计数不一致为 {val['reset_count_mismatch']}，有限性检查通过。原始 JSON：<a href="{LINK_ROOT}dbnn_validation.json">dbnn_validation.json</a>。</p></section>'''
    events=f'''<section id="events"><h2>放电事件与误差解释</h2><p>DBNN 的放电统计采用 checkpoint 的 reset 阈值 −57.5 mV；NEURON/CoreNEURON 的放电统计采用胞体 −20 mV 上穿事件。两者阈值定义不同，因此放电计数只作为各自轨迹事件的并列诊断，RMSE 仍在共同电压网格上计算。</p><p>扩展集合中 DBNN 的总体宏 RMSE 为 {d['budget']['macro_rmse_mv']:.3f} mV，但按参考事件类别，0/1/2/3-spike 的宏 RMSE 分别为 {', '.join(f"{d['budget']['by_spike_class'][str(k)]['macro_rmse_mv']:.3f}" for k in range(4))} mV。误差和事件计数都随尖峰复杂度升高而更敏感；因此最终速度结论使用 dt=0.25 ms 的严格事件计数约束，dt=1 ms 仅作为 RMSE-only 的探索结果。</p></section>'''
    appendix=f'''<section id="appendix"><h2>文件与可复核证据</h2><p>输入清单：<a href="{LINK_ROOT}selected_manifest.json">selected_manifest.json</a>；6-spike 压力样本：<a href="{LINK_ROOT}stress_cases.json">stress_cases.json</a>；完整汇总：<a href="{LINK_ROOT}analysis.json">analysis.json</a>。每条 solver 和 DBNN 的原始 JSON 保存在 <code>{LINK_ROOT}solver_results/</code> 和 <code>{LINK_ROOT}dbnn_results/</code>。</p><p>本报告没有把失败的 CoreNEURON 首轮 runner 文件纳入汇总；那一轮的问题是重复注册同一个 gid，修复后重新运行并通过 31/31 case。NEURON/CoreNEURON 在三个 dt 上的逐 case 输出最大差均为 0（在保存的 1 ms 网格上）。</p></section>'''
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DBNN 扩展单神经元同精度比较</title><style>:root{{--ink:#183044;--muted:#607789;--line:#d9e3e8;--soft:#f4f8f8;--green:#087f69}}*{{box-sizing:border-box}}body{{margin:0;background:#eef2f3;color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",Arial,sans-serif}}main{{max-width:1180px;margin:0 auto;padding:34px 28px 70px;background:white;min-height:100vh}}h1{{font-size:30px;line-height:1.25;margin:0 0 8px}}h2{{font-size:22px;margin:36px 0 10px;border-left:4px solid var(--green);padding-left:12px}}p{{max-width:1050px}}.meta{{color:var(--muted);font-size:13px;margin-bottom:24px}}nav{{position:sticky;top:0;background:rgba(255,255,255,.96);border:1px solid var(--line);padding:10px 14px;border-radius:8px;z-index:2}}nav a{{margin-right:16px;color:var(--green);text-decoration:none;font-size:13px}}table{{width:100%;border-collapse:collapse;margin:15px 0 22px;font-size:13px}}th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}thead th{{background:var(--soft);color:#4f6878}}figure{{margin:22px 0 28px;padding:13px;border:1px solid var(--line);border-radius:9px;background:#fbfdfd}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;color:var(--muted);margin-top:7px}}li{{margin:7px 0}}code{{background:#f0f4f5;padding:2px 5px;border-radius:4px}}strong{{color:#075f51}}@media(max-width:700px){{main{{padding:22px 14px}}h1{{font-size:24px}}nav a{{display:inline-block;margin-bottom:5px}}th,td{{padding:6px 4px;font-size:11px}}}}</style></head><body><main><h1>DBNN 与 NEURON / CoreNEURON CPU 的扩展同精度比较</h1><div class="meta">2026-09-29 · L5PC 单细胞 · 31 条主轨迹 · 共同比较网格 1 ms</div><nav><a href="#result">主结果</a><a href="#protocol">比较口径</a><a href="#kernel">DBNN 验证</a><a href="#events">事件分层</a><a href="#appendix">原始文件</a></nav>{primary}{protocol}{kernel}{events}{appendix}</main></body></html>'''
    (HERE/"DBNN_expanded_matched_error_report_20260929.html").write_text(html)
    print(HERE/"DBNN_expanded_matched_error_report_20260929.html")

if __name__=="__main__":main()
