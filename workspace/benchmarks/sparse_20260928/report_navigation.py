"""Add reading navigation and implementation status without changing evidence."""
import json
import re


CSS = r"""
/* DBNN_READING_GUIDE_START */
html{scroll-behavior:smooth;scroll-padding-top:24px}
.report-layout{display:grid;grid-template-columns:238px minmax(0,1160px);gap:8px;max-width:1460px;margin:auto;padding:0 16px}
.report-layout main{width:100%;min-width:0;margin:0;padding-left:20px;padding-right:20px}
.report-sidebar{position:sticky;top:22px;align-self:start;padding-top:8px;margin-top:22px;max-height:calc(100vh - 44px)}
.report-sidebar nav{display:block;margin:0}
.toc-panel{border:1px solid var(--line);border-radius:10px;background:#fff;padding:0;overflow:hidden}
.toc-panel>summary{padding:13px 15px;color:var(--ink);font-weight:650;font-size:15px;list-style:none}
.toc-panel>summary::-webkit-details-marker{display:none}
.toc-panel>summary::after{content:'−';float:right;color:var(--accent)}
.toc-panel:not([open])>summary::after{content:'＋'}
.toc-body{max-height:calc(100vh - 128px);overflow:auto;padding:0 9px 12px;overscroll-behavior:contain}
.toc-label{margin:14px 8px 4px;font-size:11px;color:var(--muted);letter-spacing:.04em}
.toc-body ul{list-style:none;padding:0;margin:0}.toc-body li{margin:1px 0}
.toc-body a{display:block;padding:5px 8px;border-radius:5px;font-size:12px;line-height:1.6;color:#426070;text-decoration:none;border-left:3px solid transparent}
.toc-body a:hover{background:#f0f6f5;color:var(--accent)}
.toc-body a[aria-current=location]{background:#e5f3ed;color:#075949;border-left-color:#087b68;font-weight:600}
.toc-body .toc-sub a{padding-left:20px;font-size:11.5px}
.toc-body small{display:inline-block;margin-left:5px;font-size:10px;color:#7b6650}
.toc-footer{display:flex;gap:10px;margin:12px 8px 0;padding-top:9px;border-top:1px solid var(--line)}
.toc-footer a{font-size:11px;padding:2px 0;border:0}
.skip-report{position:fixed;top:-70px;left:16px;z-index:2000;background:white;padding:10px;border:2px solid var(--accent)}
.skip-report:focus{top:8px}
.impl-tag{display:inline-block;vertical-align:middle;font-size:11px;line-height:1.6;padding:2px 7px;border:1px solid transparent;border-radius:5px;font-weight:600;white-space:nowrap}
.impl-tag.current{color:#086348;background:#e5f3eb;border-color:#bddccb}
.impl-tag.optional{color:#245bb0;background:#edf3ff;border-color:#ceddf4}
.impl-tag.retained{color:#405a6f;background:#edf2f5;border-color:#d5e0e6}
.impl-tag.retired{color:#875015;background:#fff3e5;border-color:#ead1ae}
.impl-tag.candidate{color:#695976;background:#f3eff7;border-color:#ddd3e5}
.section-status{display:flex;align-items:flex-start;gap:10px;margin:0 0 15px;color:var(--muted);font-size:13px}
.section-status p{margin:0}.section-status>.impl-tag{margin-top:2px;flex:none}
.reading-route{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
.reading-route a{padding:8px 12px;border:1px solid #bed8cd;background:#f0f8f4;border-radius:6px;text-decoration:none;font-size:13px}
.selection-summary{background:#f3f8f6;padding:14px 16px;border-radius:8px;margin:14px 0}
.selection-summary strong{color:#086348}.status-table{min-width:650px}.status-table td:first-child{width:17%;font-weight:500}
.status-table td:nth-child(2){width:23%}.status-table th{font-size:12px}
.history-fold{border:1px solid #e3d9c8;border-radius:8px;padding:0;margin:16px 0 0}
.history-fold>summary{padding:12px 15px;background:#fbf7f1;color:#785731;font-size:13px}
.history-body{padding:0 16px 16px}.history-body h3{margin-top:22px}
.legacy-caption{font-size:11px;display:inline-block;margin-right:6px;color:#875015}
.selection-evidence th,.selection-evidence td{font-size:12px}.selection-evidence{min-width:680px}
.report-layout section,.report-layout h3[id]{scroll-margin-top:24px}
.toc-panel summary:focus-visible,.toc-body a:focus-visible,.reading-route a:focus-visible{outline:3px solid #c68a25;outline-offset:2px}
@media(max-width:1249px){
 .report-layout{display:block;max-width:1160px;padding:0}
 .report-sidebar{z-index:50;top:0;margin:0;padding:8px 20px;background:#f2f6f7f2;max-height:none}
 .toc-panel>summary{padding:10px 14px}.toc-body{max-height:62vh}
 .report-layout main{padding-top:14px}.report-layout section,.report-layout h3[id]{scroll-margin-top:80px}
 html{scroll-padding-top:80px}
}
@media(max-width:700px){.report-sidebar{padding:6px 10px}.report-layout main{padding:10px 10px 40px}.section-status{flex-wrap:wrap;gap:6px}.history-body{padding:0 10px 10px}.reading-route{gap:6px}.reading-route a{font-size:12px;padding:7px 9px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{.report-layout{display:block;max-width:none;padding:0}.report-sidebar,.skip-report,.reading-route{display:none}.report-layout main{padding:0}.history-fold{border:0}.history-body{padding:0}.history-fold>summary{background:transparent;font-weight:600}.impl-tag{background:transparent!important;border-color:#aab7ba!important;color:#183044!important}.selection-evidence,.status-table{min-width:0}}
/* DBNN_READING_GUIDE_END */
"""


JS = r"""
(() => {
  const panel = document.getElementById('report-toc-panel');
  const nav = document.getElementById('report-navigation');
  const desktop = window.matchMedia('(min-width:1250px)');
  const links = [...nav.querySelectorAll('a[data-section-link]')];
  const entries = links.map(link => ({link, target:document.getElementById(link.hash.slice(1))})).filter(x=>x.target);
  const arrange = () => { panel.open = desktop.matches; };
  arrange(); desktop.addEventListener('change', arrange);
  const highlight = id => entries.forEach(({link,target}) => {
    if (target.id === id) link.setAttribute('aria-current','location');
    else link.removeAttribute('aria-current');
  });
  const reveal = target => {
    for (let p=target.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true;
    const own = target.querySelector(':scope > details.history-fold');
    if (own) own.open=true;
  };
  const fromHash = () => {
    const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if(target){reveal(target);highlight(target.id);}
  };
  nav.addEventListener('click', event => {
    const link=event.target.closest('a[href^="#"]');
    if(!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey)return;
    const target=document.getElementById(link.hash.slice(1)); if(!target)return;
    reveal(target);if(!desktop.matches)panel.open=false;
    requestAnimationFrame(()=>{target.scrollIntoView({block:'start'});highlight(target.id);});
  });
  let scheduled=false;
  const update=()=>{
    scheduled=false;let best=null,bestTop=-Infinity;
    for(const item of entries){const rect=item.target.getBoundingClientRect();
      if(item.target.getClientRects().length && rect.top<=(desktop.matches?70:115) && rect.top>bestTop){best=item;bestTop=rect.top;}}
    highlight(best?best.target.id:'report-top');
  };
  window.addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(update);}},{passive:true});
  window.addEventListener('hashchange',fromHash);fromHash();update();
  let printClosed=[];
  window.addEventListener('beforeprint',()=>{printClosed=[...document.querySelectorAll('details.history-fold:not([open])')];printClosed.forEach(x=>x.open=true);});
  window.addEventListener('afterprint',()=>{printClosed.forEach(x=>x.open=false);printClosed=[];});
})();
"""


def badge(kind, label):
    return f'<span class="impl-tag {kind}">{label}</span>'


def table(headers, rows, cls):
    return '<div class="table-wrap"><table class="'+cls+'"><thead><tr>'+''.join('<th scope="col">'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def navigation():
    groups = [
        ('先看当前结论', [('report-top','报告首页'),('current-selection','当前方案与淘汰记录'),('conclusion','主要结果与计时范围')]),
        ('模型与算法', [('design','1. 计算设计'),('model-stage-one','1.1 滤波与二次整合','sub'),('model-stage-two','1.2 reset 事件迭代','sub'),('model-mapping','1.3 稀疏位点与实现','sub'),('five-layer-guide','五层优化详解','sub')]),
        ('双基线与精度证据', [('gpu','2. GPU 规模与传输'),('original-audit','2.1 原代码逐层配对','sub'),('neuron','3. CPU / NEURON'),('fidelity','5. 拟合精度与实现误差')]),
        ('当前采用的实现', [('quad-specialized','4.3 优化版 FP32 融合实现'),('tf32-five-layers','4.4 TF32 精度与速度选择')]),
        ('早期探索 · 保留对照', [('backend-ablation','4. 多后端内部消融','history'),('drive-analysis','4.1 初版完整驱动项','history'),('quadratic-analysis','4.2 初版纯二次项 / GEMM','history')]),
        ('后续与原始数据', [('next','6. 下一阶段'),('raw','7. 协议与原始记录')]),
    ]
    content=[]
    for title,items in groups:
        content.append(f'<p class="toc-label">{title}</p><ul>')
        for item in items:
            anchor,label,*flags=item;cls=' class="toc-sub"' if flags==['sub'] else ''
            content.append(f'<li{cls}><a data-section-link href="#{anchor}">{label}</a></li>')
        content.append('</ul>')
    return '<aside class="report-sidebar"><nav id="report-navigation" aria-label="报告目录"><details id="report-toc-panel" class="toc-panel" open><summary>目录与阅读导引</summary><div class="toc-body">'+''.join(content)+'<div class="toc-footer"><a href="#report-top">返回顶部 ↑</a><a href="#current-selection">看当前方案</a></div></div></details></nav></aside>'


def current_selection(data):
    confirmed=data['quadratic_specialized']['confirmation']
    policy=data['quadratic_specialized']['policy']
    count=sum(v['enabled'] for v in policy['shapes'].values())
    rows=[
        [badge('current','当前推荐 · FP32'),'优化版二次型融合实现','跨时间块复用 W、按形状选全矩阵或三角收缩、融合二次项与线性项，最后加偏置。<br>已启用 '+str(count)+' 个独立复测形状。','<a href="#quad-specialized">4.3：完整模块 / 完整推理</a>'],
        [badge('optional','按精度选用'),'优化版 TF32 / TF32x3 组合','在新版融合实现上选择乘法精度；滤波也可使用 Tensor Core。按完整电压误差与 reset 门限选组合。','<a href="#tf32-five-layers">4.4：13 个形状与两种 profile</a>'],
        [badge('retained','回退与对照保留'),'cuBLAS＋完整归约','未调优尺寸使用的通用路径；继续作为性能参照。归约同时包含二次、线性和偏置。','<a href="#selection-fallback">查看回退规则</a>'],
        [badge('retired','已被替代'),'初版逐时间点单 kernel','每个时间点单独读取 W 的 direct_kernel，已由跨时间分块的专用实现取代主推荐位置；保留早期测量。','<a href="#drive-analysis">4.1：历史对照</a> / <a href="#quad-specialized">4.3：同轮新旧比较</a>'],
        [badge('candidate','本轮未采用'),'本次 CUTLASS SIMT / FastF32 配置；单线程 / warp 候选','均已实际探索；当前形状选择表采用新版 Triton 分块 / 块对路径。该状态针对本次实现与配置。','<a href="#backend-ablation">4：后端探索</a> / <a href="#quad-specialized">4.3：候选与获选配置</a>'],
        [badge('retired','内部旧对照'),'新增的逐时间格 CUDA reset','现行路径采用 GPU 事件式 reset。这里淘汰的是早期新增 CUDA 版本；用户原 reset 函数继续作为固定基线。','<a href="#backend-ablation">4：顺序 / 事件内部消融</a>'],
        [badge('retained','候选与对照保留'),'缓存 FFT、CUB / CuTe 扫描','保留形状相关的早期测量；当前精度策略选择调优 Triton scan 或 Tensor Core 分块滤波。','<a href="#tf32-five-layers">4.4：当前滤波选择</a>'],
        [badge('retained','固定基线'),'用户原始 DBNN_F / DBNN_R 与 NEURON','继续用于实现加速、代理拟合和详细模拟器的对照；实验记录和原始源文件保持可追溯。','<a href="#original-audit">2.1：原代码对照</a> / <a href="#neuron">3：NEURON</a>'],
    ]
    comparison=[]
    for c in confirmed['cases']:
        if (c['A'],c['B'],c['T']) not in [(32,1,6001),(500,1,6001),(500,8,6001),(2048,1,6001)]:continue
        m=c['module_device'];old=m['old_direct']['median_ms']*1000;cub=m['cublas']['median_ms']*1000;new=m['new_ieee']['median_ms']*1000
        comparison.append([f'A={c["A"]}, B={c["B"]}'+(' *' if c['A']>500 else ''),f'{old:.2f}',f'{cub:.2f}',f'{new:.2f}',f'{old/new:.2f}×',f'{cub/new:.2f}×'])
    return '''<section id="current-selection"><h2>现在用哪一版：当前方案与淘汰记录</h2>
<p>报告保留了多轮探索。先阅读这里的采用状态，再看第 4.3 / 4.4 节的新版结果；第 4.1 / 4.2 节集中保留初版算子对照。</p>
<div class="reading-route"><a href="#quad-specialized">① 优化版 FP32 融合实现 →</a><a href="#tf32-five-layers">② TF32 精度与速度 →</a><a href="#fidelity">③ 拟合与数值误差 →</a></div>
<div class="selection-summary"><strong>当前推荐：以优化版二次型融合实现为主，按形状与精度要求选具体配置。</strong><br>
该实现同时计算二次项、线性项与偏置。小形状可以单核输出；较大形状通常是“分块融合计算＋少量部分和归约”两个 kernel。统一称为“优化版融合实现”，与初版的单时间点直算核区分。</div>
'''+table(['采用状态','实现','具体含义','证据 / 跳转'],rows,'status-table')+'''
<h3>初版为什么退出主推荐：同轮完整驱动项比较</h3>
<p class="muted">下表来自 <code>quad_confirm.json</code> 的第二轮固定配置复测；均为 6 s 窗口、IEEE FP32，完整包含二次＋线性＋偏置及必要的部分和归约。单位 μs，设备摊销时间，60 次测量，共享 GPU 6。</p>
'''+table(['活跃位点 / batch','初版直算 / μs','cuBLAS＋归约 / μs','优化版融合 / μs','初版 / 优化版','cuBLAS / 优化版'],comparison,'selection-evidence')+'''
<p class="muted">* A=2,048 为参数扩展的计算规模测试。上述速度比只对应完整驱动项；完整 DBNN 的 wall time 见第 4.3 节。保留历史表格的原始数据，避免跨测量轮次拼接速度比。</p>
<h3 id="selection-fallback">当前入口与尺寸回退</h3>
<p>保持 IEEE FP32：使用 <code>make_optimized_prepared</code>。接受已测误差门限并启用 Tensor Core：使用 <code>make_precision_prepared</code>，由 <code>strict</code> 或显式的 <code>approximate_0p01mv</code> 选择组合。500 位点、B=8 的 strict 组合保留新版 IEEE 二次项，只在滤波中使用 TF32；放宽至 0.01 mV 的方案才同时采用 TF32 二次项。</p>
<p>“未调优尺寸”指配置表没有对应的 (A,T,B)，例如 (500,6001,3)。先查询此前的 FP32 选择表；仍未命中时使用 Triton scan＋cuBLAS＋事件 reset。测试脚本明确关闭 cuBLAS 的 TF32，以形成 IEEE 对照。精度门限对应已测 10 条输入轨迹的检查结果。</p>
</section>'''


def apply_precision_names(body):
    """Display precision separately from the algorithm; retain raw log keys."""
    if 'id="precision-naming-note"' in body:
        return body
    # Protect scripts, literal code and every HTML tag (including image data).
    parts=re.split(r'(<script\b[^>]*>.*?</script>|<code\b[^>]*>.*?</code>|<[^>]+>)',body,flags=re.S|re.I)
    for i in range(0,len(parts),2):
        parts[i]=parts[i].replace('IEEE FP32','FP32').replace('IEEE','FP32')
        parts[i]=parts[i].replace('4.3 优化版 FP32 融合实现','4.3 二次型融合 kernel（FP32）')
        parts[i]=parts[i].replace('① 优化版 FP32 融合实现 →','① 二次型融合 kernel（FP32）→')
        parts[i]=parts[i].replace('4.3 当前推荐：优化版二次型融合实现','4.3 当前推荐：优化版二次型融合 kernel（FP32）')
        parts[i]=parts[i].replace('优化版二次型融合实现','优化版二次型融合 kernel')
    body=''.join(parts)
    body=body.replace('<td>优化版二次型融合 kernel</td>','<td>优化版二次型融合 kernel（FP32）</td>')
    body=body.replace('<td>优化版 TF32 / TF32x3 组合</td>','<td>优化版融合 kernel（TF32 / TF32x3）</td>')
    body=body.replace('<th>固定的实现</th>','<th>配置 ID（与日志一致）</th>')
    body=re.sub(r'<td>((?:triton|pair)_\d+_(?:full|tri)_ieee|new_ieee|optimized_new_ieee)</td>',r'<td><code>\1</code></td>',body)
    note='''<details id="precision-naming-note"><summary>技术说明：计算精度与日志名称的对应</summary>
<p>方法名称统一为“优化版二次型融合 kernel”，括号注明 FP32、TF32 或 TF32x3。算法设计包括跨时间块复用、三角收缩以及二次项与线性项的融合计算。</p>
<p><code>input_precision='ieee'</code> 是 Triton 中采用常规 FP32 乘法精度的参数，名称来自 IEEE 754 浮点标准。日志中的 <code>new_ieee</code>、<code>optimized_new_ieee</code> 和配置 ID 保留原值，便于追溯实测结果；它们对应报告中的 FP32 版本。</p></details>
'''
    marker='<h3>已经可以使用的入口与下一步</h3>'
    if body.count(marker)!=1:raise ValueError('Precision naming insertion point is missing')
    return body.replace(marker,note+marker,1)


def apply_reading_guide(body):
    if 'DBNN_READING_GUIDE_START' in body:
        return apply_precision_names(body)
    match=re.search(r'<script type="application/json" id="data">(.*?)</script>',body,re.S)
    if not match:raise ValueError('Embedded benchmark evidence is missing')
    data=json.loads(match.group(1))
    visible,tail=body[:match.start()],body[match.start():]
    visible=visible.replace('</style>',CSS+'\n</style>',1)
    visible=visible.replace('<body><main>','<body><a class="skip-report" href="#report-top">跳至报告正文</a><div class="report-layout">'+navigation()+'<main>',1)
    visible=visible.replace('<header>','<header id="report-top">',1)
    visible=re.sub(r'<nav><a href="#conclusion">.*?</nav>','',visible,count=1,flags=re.S)
    visible=visible.replace('</main>','</main></div>',1)
    visible=visible.replace('<section id="design">',current_selection(data)+'\n<section id="design">',1)
    visible=visible.replace('<div class="layer-guide"','<div id="five-layer-guide" class="layer-guide"',1)
    visible=visible.replace('<section><h2>4. 内部消融：选择新增实现中的后端</h2>', '<section id="backend-ablation"><h2>4. 历史探索：新增实现中的后端对照</h2>',1)

    def edit_section(anchor, fn):
        nonlocal visible
        pattern=r'(<section id="'+re.escape(anchor)+r'">)(.*?)(</section>)'
        visible,n=re.subn(pattern,lambda m:m[1]+fn(m[2])+m[3],visible,count=1,flags=re.S)
        if n!=1:raise ValueError('Section not found: '+anchor)

    def status(content, kind, label, message):
        end=content.index('</h2>')+5
        return content[:end]+'<div class="section-status">'+badge(kind,label)+'<p>'+message+'</p></div>'+content[end:]

    for anchor in ('gpu','neuron'):
        edit_section(anchor,lambda c:status(c,'retained','首轮基线记录','保留原始 14 配置测量。后续融合 / TF32 的增量结果分别见 <a href="#quad-specialized">4.3</a> / <a href="#tf32-five-layers">4.4</a>；这里的“优化”列对应当轮选定后端。'))
    def original_audit(c):
        c=c.replace('只换单核 drive / ms（速度比）','只换初版单核 drive / ms（速度比） '+badge('retired','已被替代'))
        return status(c,'retained','原代码配对记录','此处保留首次逐模块替换证据；“单核 drive”指初版直算实现。新版融合实现放回原 F/R 的配对结果见 <a href="#quad-specialized">4.3</a>。')
    edit_section('original-audit',original_audit)

    def historical(c, kind):
        c=c.replace('内部消融：完整驱动项的不同实现','历史对照：初版完整驱动项实现').replace('内部消融：纯二次项与 GEMM','历史对照：初版纯二次项与 GEMM')
        c=c.replace('单 kernel 全融合','初版单 kernel 全融合').replace('当前直接融合路径','当轮初版直接融合路径').replace('当前直接核','初版直接核')
        for label in ('初版单 kernel 全融合','完整融合','完整融合¹','Direct fused kernel','One-kernel full drive'):
            for tag in ('th','td'):
                c=c.replace(f'<{tag}>{label}</{tag}>',f'<{tag}>{label} '+badge('retired','已被替代')+f'</{tag}>')
        c=c.replace('<figcaption>','<figcaption><span class="legacy-caption">［初版历史对照］</span>')
        if kind=='drive':
            c=re.sub(r'<p>\s*下一步优先优化每个 block.*?</p>','<p>当时提出的跨时间权重复用与分块融合，已在 <a href="#quad-specialized">第 4.3 节</a>实现并复测。当前按新版完整驱动项和完整前向结果选择实现；本节初版直算核保留作历史对照。</p>',c,count=1,flags=re.S)
        if kind=='quadratic':
            c=re.sub(r'<p>\s*此次新增的是独立量化.*?</p>','<p>本节记录第一轮独立量化。其后已完成跨时间分块融合、三角收缩及 TF32 系统调优，当前采用状态见 <a href="#current-selection">当前方案</a>。</p>',c,count=1,flags=re.S)
            c=c.replace('4.1 节完整 drive 与完整前向比较选择后端','4.1 节当轮完整 drive 与完整前向比较；当前选择见第 4.3 / 4.4 节')
        end=c.index('</h2>')+5
        title,content=c[:end],c[end:]
        message='本节记录早期实现。初版逐时间点直算 kernel 已退出当前主推荐；cuBLAS 和原 F/R 表达式继续保留作对照。新版结论见 <a href="#quad-specialized">4.3</a> / <a href="#tf32-five-layers">4.4</a>。'
        return status(title,'retained','历史记录',message)+'<details class="history-fold"><summary>展开当轮完整表格、图与误差记录</summary><div class="history-body">'+content+'</div></details>'

    edit_section('drive-analysis',lambda c:historical(c,'drive'))
    edit_section('quadratic-analysis',lambda c:historical(c,'quadratic'))
    edit_section('backend-ablation',lambda c:status(c.replace('本节比较本轮新增后端的不同组合，用于决定采用哪条实现路径。','本节保留首轮新增后端的组合探索，记录当时的选择依据。'),'retained','内部探索记录','顺序 CUDA reset 已由事件式 reset 替代；本次 CUTLASS 配置作为已测候选保留。不同后端的采用状态见 <a href="#current-selection">当前方案</a>。'))
    def specialized(c):
        c=c.replace('4.3 新增二次型专用 kernel：跨时间块复用与三角收缩','4.3 当前推荐：优化版二次型融合实现')
        c=c.replace('<th>上一版自写 CUDA / μs</th>','<th>初版直算 / μs '+badge('retired','已被替代')+'</th>')
        c=c.replace('<th>新 IEEE / μs</th>','<th>优化版 IEEE / μs '+badge('current','当前推荐')+'</th>')
        c=c.replace('此次发布的策略全部使用 IEEE 路径','本节保留 IEEE 选择表；后续 TF32 / TF32x3 组合已在第 4.4 节独立发布')
        return status(c,'current','当前 FP32 主路径','已替代初版逐时间点直算的主推荐位置。完整驱动项按形状使用分块融合 / 三角块对计算，必要时增加一次部分和归约。')
    edit_section('quad-specialized',specialized)
    def precision(c):
        c=c.replace('未知形状回到原 IEEE 策略','配置表未覆盖的 (A,T,B) 使用此前 FP32 选择规则，仍未命中时使用 Triton scan＋cuBLAS＋事件 reset（见<a href="#selection-fallback">尺寸回退说明</a>）')
        return status(c,'optional','当前可选精度方案','基于优化版融合实现继续选择 TF32 / TF32x3；每种组合都有完整电压与 reset 检查，使用范围和门限在下表列明。')
    edit_section('tf32-five-layers',precision)
    visible=visible.replace('上一轮 IEEE 二次型专用 kernel：已完成两轮复测。','当前 FP32 主路径：优化版二次型融合实现，已完成两轮复测。')
    visible=visible.replace('保留前一轮 cuBLAS＋归约 / 单时间点 CUDA 的原代码配对','历史配对：cuBLAS＋归约 / 初版单时间点 CUDA（后者已被替代）')
    visible=visible.replace('<footer>','<footer>本次阅读更新：增加目录、当前方案与历史实现标记。',1)
    tail=tail.replace('</body>','<script id="report-navigation-script">'+JS+'</script></body>',1)
    return apply_precision_names(visible+tail)


if __name__=='__main__':
    import sys
    from pathlib import Path
    source,destination=map(Path,sys.argv[1:])
    destination.write_text(apply_reading_guide(source.read_text()))
    print(destination)
