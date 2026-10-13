"""Five layers anchored in the frozen user implementation, not new backends."""
import html
from layer_explanations import LAYER_CSS, LAYER_JS


def small_table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        '<th scope="col">' + h + '</th>' for h in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + str(v) + '</td>' for v in row) + '</tr>'
        for row in rows
    ) + '</tbody></table></div>'


def build_layer_guide(cs, audit, specialized=None, precision=None):
    cases = audit['cases']
    assert len(cases) == 4 and audit.get('finished')
    def ms(c, key, scope='forward'):
        return c[scope][key]['median_ms']
    def change(a, b):
        return f'{a:.3f} → {b:.3f} ms（{a/b:.2f}×）'
    def label(c):
        return f'A={c["A"]}, B={c["B"]}, 6 s'
    def replacement_rows(stage, methods):
        rows = []
        for c in cases:
            for kind in ['F', 'R']:
                before = ms(c, f'original_{kind}')
                rows.append([label(c), kind, f'{before:.3f}'] + [
                    f'{ms(c, f"{kind}_only_{stage}{key}"):.3f}（{before/ms(c, f"{kind}_only_{stage}{key}"):.2f}×）'
                    for key in methods])
        return rows
    def one_change(c, kind, key):
        return change(ms(c, f'original_{kind}'), ms(c, f'{kind}_only_{key}'))
    first = cases[0]
    n = first['forward']['original_F']['n']
    support_rows = []
    for i in (1, 8):
        c = cs[i]
        for kind in ['F', 'R']:
            before = c['timings'][f'pytorch_{kind}']['median_ms']
            after = c['timings'][f'pytorch_packed_{kind}']['median_ms']
            support_rows.append([f'N={c["N"]:,} → A={c["A"]}, B=1, {(c["T"]-1)/1000:g} s',
                                 kind, change(before, after)])
    reset_rows = []
    for c in cases:
        before = ms(c, 'original_reset', 'reset_module')
        after = ms(c, 'event_cuda', 'reset_module')
        loop = (ms(c, 'original_reset_trace_loop', 'reset_module') if c['B'] > 1 else before)
        reset_rows.append([label(c), '/'.join(map(str,c['reset_events_per_trace'])),
                           f'{before:.3f}', f'{loop:.3f}', f'{after:.3f}',
                           f'{before/after:.2f}× / {loop/after:.2f}×'])
    schedule_rows = [[label(c), change(ms(c, 'optimized_eager'), ms(c, 'optimized_graph'))] for c in cases]
    bf = cs[8]['timings']['pytorch_F']['median_ms']
    af = cs[8]['timings']['pytorch_packed_F']['median_ms']
    layers = [
        dict(id='support', name='位点集合', tag='原 F→压缩 F；原 R→压缩 R',
             original='F / R 都接收完整 B×N×T 输入，并使用完整 N×N 权重；原 forward 没有 active IDs 筛选。',
             operation='只保留整窗活跃位点 A，同步提取 x<sub>A</sub>、滤波参数和 W<sub>AA</sub>。',
             measured=f'<strong>{bf/af:.2f}×</strong> · 原 F 的压缩对照<br><span class="muted">N=8,000 → A=32，B=1，1 s；{bf:.3f} → {af:.3f} ms</span>',
             sourcecode='source/DBNN.py · DBNN_F.forward / DBNN_R.forward',
             actions='<ol><li>从已知 active IDs 收集整窗有输入的通道。</li><li>同时收集 τ、ω 和 W 的对应行、列。</li>'
                     '<li>原 F 与压缩 F 配对，原 R 与压缩 R 配对；两边保持同一 forward 算法。</li></ol>',
             benefit='<p>整窗无输入、初始状态为零的位点，其滤波状态始终为零。滤波工作量由 O(BTN) 降至 O(BTA)，'
                     '二次项由 O(BTN²) 降至 O(BTA²)。当前仍保存完整 checkpoint 的 N×N 权重。</p>',
             evidence=small_table(['条件','同一原实现','完整前向耗时与速度比'],support_rows),
             reading='<p>这一步量化减少零状态计算的收益。后续滤波、驱动项和 reset 的新增配对实验，两边都使用相同 A 个位点，'
                     '因此每层收益从同一个压缩后的原始实现起算。</p>',
             scope='50 次同步 wall time；参数收集和输入打包属于准备阶段。活跃集搜索另计；N=8,000 是参数扩展的计算规模测试。',
             source='confirmation.json · cases #1/#8 · pytorch_F / pytorch_packed_F / pytorch_R / pytorch_packed_R'),
        dict(id='filter', name='时间滤波', tag=f'原代码仅替换滤波 · {n} 次',
             original='F：每次重建双指数核并做核 FFT。<br>R：每 256 步做 FP64 缩放 cumsum，再转回 FP32；系数、幂及逆幂已有缓存。',
             operation='换成 Triton 的双状态前缀 scan；保留原驱动项和原 reset。R 的 256 步驱动项循环继续保留。',
             measured=f'A=32, B=1：<br>F：{one_change(first,"F","filter")}<br>R：{one_change(first,"R","filter")}',
             sourcecode='source/DBNN.py · DBNN_F.create_kernels / forward；DBNN_R._state_scan_chunk / _get_scan_terms / forward',
             actions='<ol><li>双指数状态的每一步写成 s←a·s+x，用可结合的仿射变换做并行前缀扫描。</li>'
                     '<li>Triton 一次提供整窗滤波输出；F 继续调用原 bilinear，R 继续按原 256 步分块计算线性和二次项。</li>'
                     '<li>仅替换滤波的实验采用 eager 调用，保留原 reset 分支和其主机调度。</li></ol>',
             benefit='<p>相对 F，省去每次构造滤波核及核 FFT，改用 O(BTA) 的状态扫描。'
                     '相对 R，将原有的分块缩放、两次 cumsum、dtype 转换及状态组合放入 scan kernel，减少中间读写与主机调度。'
                     'R 原先已经按前缀和求状态；这里优化的是执行方式、融合及中间精度。</p>'
                     '<p>原 R 的扫描中间量为 FP64，新 scan 为 FP32。电压误差和 reset 事件逐项核对。</p>',
             evidence=small_table(['条件','原版','原 forward / ms','只换滤波 / ms（整体速度比）'],replacement_rows('filter',[''])),
             reading='<p>以上为原始完整 forward 与只换滤波后的完整 forward。数值包含原驱动项与 reset，'
                     '收益反映替换滤波带来的应用变化。缓存 FFT、CUB、CuTe 和 Triton 之间的选择移到第 4 节内部消融。</p>',
             scope='相同 packed 输入、checkpoint、FP32 输出；原 R 缓存预热且开启。初始状态为零。R 将整窗 scan 输出按原大小切回驱动项循环。',
             source='original_baseline_audit.json · forward.original_{F,R} / {F,R}_only_filter；original_baseline_variants.txt'),
        dict(id='quadratic', name='二次耦合', tag=f'含线性＋偏置 · {n} 次',
             original='F：整窗 F.bilinear＋线性求和。<br>R：每 256 步做 mm/@、逐元素乘、sum，再加线性项与偏置。',
             operation='在原调用位置替换完整 drive：cuBLAS＋融合归约，或单 CUDA 核计算二次、线性和偏置。',
             measured=f'A=32, B=1，cuBLAS＋归约：<br>F：{one_change(first,"F","drive_cublas")}<br>R：{one_change(first,"R","drive_cublas")}',
             sourcecode='source/DBNN.py · BilinearModified.forward；DBNN_F.forward；DBNN_R.forward 中的 chunk 循环',
             actions='<ol><li>固定原来的滤波与 reset，只改变 drive=yᵀWy+Σy+b 的实现。</li>'
                     '<li>cuBLAS 做 YW；一个归约核把逐元素乘、二次求和、线性求和和偏置合并。</li>'
                     '<li>单核版本直接计算完整 drive。R 仍逐个 256 步窗口调用，各版本都计入必要的布局转换。</li></ol>',
             benefit='<p>算术规模维持 O(BTA²)。cuBLAS 加归约减少求和与加法的 kernel 调用；单核融合还省去 YW 中间张量。'
                     '原 R 的 mm/@ 本身已走 PyTorch 的 GPU 矩阵乘法路径，因此主要考察额外融合和调度收益。</p>',
             evidence=small_table(['条件','原版','原 forward / ms','只换 cuBLAS＋归约 / ms（速度比）','只换单核 drive / ms（速度比）'],
                                  replacement_rows('drive',['_cublas','_direct'])),
             reading='<p>这组保留了 R 原有的时间分块。此前把 R 的表达式重排成一次整窗 GEMM 的实验，'
                     '在第 4.1 / 4.2 节标为“整窗表达式对照”，用于内部算子选择。</p>',
             scope='相同活跃集合，FP32、TF32 关闭；完整 forward 同步 wall time。新 drive 的权重与工作区预先准备，原 R 缓存照常开启。',
             source='original_baseline_audit.json · forward.original_{F,R} / {F,R}_only_drive_{cublas,direct}；original_baseline_variants.txt'),
        dict(id='reset', name='reset', tag=f'直接调用原 reset 函数 · {n} 次',
             original='无梯度且 B=1：Python 驱动的事件循环，nonzero / item 找最早事件。<br>B&gt;1：Python/PyTorch 逐时间格循环。',
             operation='保留按事件推进的数学过程，把每条轨迹的搜索、归约、状态更新放入一个 GPU block。',
             measured='A=500, B=8，共 11 次触发：<br>原 B=1 分支逐条调用 → 事件 CUDA：<br>'+change(ms(cases[3],'original_reset_trace_loop','reset_module'),ms(cases[3],'event_cuda','reset_module')),
             sourcecode='source/DBNN.py · _apply_reset_closed_loop；条件 (not torch.is_grad_enabled()) and bsz == 1',
             actions='<ol><li>对固定驱动曲线 u[n]，每条轨迹使用一个 256 线程 block，线程按步长 256 分担剩余时间点。</li>'
                     '<li>并行检查 u[n]+α<sup>n−s</sup>ρ，并归约得到最早阈值事件。</li>'
                     '<li>写出直到该事件的最终电压，更新下一时间点的 reset 状态；在 GPU 内继续下一轮。</li></ol>',
             benefit='<p>原 B=1 已有事件算法。新增收益来自合并 GPU 操作和消除每次事件回到 Python 的同步；'
                     'B&gt;1 则让每条轨迹也走事件路径。原 B&gt;1 的逐时间格循环同样计时，并增加逐条调用原 B=1 分支的对照。</p>'
                     '<p>当前事件核重复扫描后缀，工作量随事件数增长。每条轨迹的事件数一并记录。</p>',
             evidence=small_table(['条件','每条轨迹事件数','原整批 / ms','原 B=1 逐条调用 / ms','事件 CUDA / ms','相对整批 / 逐条速度比'],reset_rows),
             reading='<p>这里直接调用原 <code>_apply_reset_closed_loop</code>，三列使用完全相同的 u[n]。'
                     'B=1 时前两列是同一个调用。原完整 forward 中仅替换 reset 的结果见下表：</p>'+
                     small_table(['条件','原版','原 forward / ms','只换 reset / ms（整体速度比）'],replacement_rows('reset',['']))+
                     '<p>先前“顺序 CUDA → 事件 CUDA”的 4.21×来自我们新写的两个后端，保留在内部消融中。</p>',
             scope='模块计时包括原函数的输出分配与指数幂构造；事件核复用输出和指数幂。故收益覆盖 reset 的实现与准备复用。返回所有时间格电压，事件一致性逐项检查。',
             source='original_baseline_audit.json · reset_module / reset_events_per_trace / forward.*_only_reset'),
        dict(id='scheduling', name='调度', tag='原有缓存保留；Graph 单列内部消融',
             original='F：Python 发起算子，每次构造滤波核。<br>R：系数和扫描幂已有缓存，另设 W 缓存分支；分块循环与 reset 由 Python 调度。',
             operation='在原有缓存基础上增加 FFT 频响、输出工作区和 reset 幂缓存；全 GPU 路径支持 CUDA Graph replay。',
             measured='原代码 → 优化后的整体收益见第 2 节。<br><span class="muted">Graph 独立收益在新路径内测量，详见说明。</span>',
             sourcecode='source/DBNN.py · DBNN_R._cache_enabled_for_this_forward / _get_coefficients / _get_scan_terms / _get_quadratic_weight',
             actions='<ol><li>保留原 R 已有的缓存，新增固定 checkpoint 的 FFT 频率响应、reset 指数幂及输出工作区。</li>'
                     '<li>把事件循环搬入 GPU 后，固定形状和地址的完整路径可捕获为 CUDA Graph。</li>'
                     '<li>主结果用实际原 F/R 比较最终优化路径；Graph 开关单独用于选择新路径的执行方式。</li></ol>',
             benefit='<p>减少重复参数处理、临时张量分配和主机提交次数。原 B=1 的事件搜索包含数据相关的 nonzero / item，'
                     '需完成全 GPU reset 改写后才能采用当前 Graph 方案；调度与 reset 改写存在依赖。</p>',
             evidence='<p>内部消融：以下两端均为新增的 Triton＋cuBLAS＋事件 CUDA 路径：</p>'+
                     small_table(['条件','优化后 eager → Graph（速度比）'],schedule_rows),
             reading='<p>原代码与最终路径的整体配对结果见第 2 节和 2.1 节。上表用于决定新路径是否启用 Graph。</p>'
                     '<p>本次另核对了缓存命中：系数与扫描幂可复用；W 缓存的键取自清零对角线之前的参数版本号，'
                     '清零后版本号增加，连续三次调用均未命中已有 W 条目。配对实验保留了这一原行为；修正 W 缓存可单独作为后续消融。</p>',
             scope='编译、参数准备和 Graph capture 计入准备成本；这里比较准备后的稳态推理。新旧层次的收益存在交互，各单项从同一原版起算。',
             source='source/DBNN.py；original_baseline_audit.json · optimized_eager / optimized_graph；confirmation.json')
    ]
    if specialized is not None:
        latest = specialized['source_replacement']
        assert latest.get('finished')
        latest_rows = []
        for c in latest['cases']:
            for kind in ['F', 'R']:
                a = c['wall'][f'original_{kind}']['median_ms']
                b = c['wall'][f'new_{kind}']['median_ms']
                latest_rows.append([label(c), kind, f'{a:.3f}', f'{b:.3f}', f'{a/b:.2f}×'])
        c = latest['cases'][0]
        q = layers[2]
        prior_evidence = q['evidence']
        q.update(
            tag='最新专用二次型 · 原代码只换 drive · 30 次',
            operation='多个时间点复用权重 tile；合并 Wᵢⱼ 与 Wⱼᵢ，跳过下三角块；在块内同时归约二次、线性和偏置。',
            measured='A=500, B=1：<br>F：'+change(c['wall']['original_F']['median_ms'], c['wall']['new_F']['median_ms'])+
                     '<br>R：'+change(c['wall']['original_R']['median_ms'], c['wall']['new_R']['median_ms']),
            actions='<ol><li>将固定 W 预处理为上三角 U：非对角项为 Wᵢⱼ＋Wⱼᵢ，对角项保留。此重排在实数代数下保持 yᵀWy。</li>'
                    '<li>多个时间点共同加载一个权重 tile；将矩阵块乘法结果在寄存器中直接乘 y 并求和，省去整张 YW 的写回。</li>'
                    '<li>每个时间点只写少量部分和，再合并并加偏置；线性项也在同一流程计算。</li>'
                    '<li>原代码配对保留滤波、reset 和 R 的 256 步分块；新版整窗路径另测 scan＋事件 reset 下的增量收益。</li></ol>',
            benefit='<p>上三角块约减半主要乘加，多个时间点复用权重读取；与 cuBLAS＋归约相比还减少 YW 中间张量的写回和读取。'
                    'FP32 求和次序有所变化，逐项检查电压和 reset 时序。此轮采用 IEEE FP32。</p>'
                    '<p>R 的原始 chunk 较短，原滤波及 reset 的主机调度仍有成本，模块提速需要用完整 forward 检验。</p>',
            evidence=small_table(['条件','原版','原 forward / ms','只换新二次型 / ms','整体速度比'],latest_rows),
            reading='<p>F 在 B=1 时获得约 1.31×的完整前向收益；R 在这组保留原 256 步分块的配对中没有加速。'
                    'B=8 原整批调用仍受原 reset 循环主导。<a href="#quad-specialized">第 4.3 节</a>另列完整 drive 的设备时间、整窗优化路径和全部形状。</p>'
                    '<details><summary>保留前一轮 cuBLAS＋归约 / 单时间点 CUDA 的原代码配对</summary>'+prior_evidence+'</details>',
            scope='相同活跃集合和冻结 checkpoint；IEEE FP32；原 F/R eager 完整 forward，同步 wall time，30 次交错测量。新权重及工作区预先准备，必要的输入布局转换在计时内。',
            source='quad_source_replace.json / quad_source_replace.py；quad_confirm.json；quadratic_tiled.py / quadratic_tiled.cu')
    if precision is not None:
        tested = precision['repeat']['cases']
        c = next(c for c in tested if c['A']==500 and c['B']==1 and c['T']==6001)
        def device_us(scope, name):
            return c[scope][name]['median_ms']*1000
        head = '<h4>新增 TF32 核查（新后端内部对照）</h4>'
        notes = [
            '<p>已对相同 IDs 的输入 gather 打开 / 关闭 TF32 进行配对，输出逐位一致。该操作不包含矩阵乘法，TF32 不减少 FP32 张量的存储和复制量。'
            '稀疏位点压缩仍由减少 N→A 的计算规模提供收益。</p>',
            f'<p>新增短窗 Toeplitz 矩阵滤波，块内用 Tensor Core，块间状态保持 FP32。A=500、B=1、6 s：'
            f'原 scan128 {device_us("filter_device","previous_ieee"):.2f} μs；调优 scan {device_us("filter_device","tuned_ieee"):.2f} μs；'
            f'TF32 {device_us("filter_device","filter_tf32"):.2f} μs；TF32x3 {device_us("filter_device","filter_tf32x3"):.2f} μs。'
            '包含块内计算、块间扫描和最终合成；模型电压与 reset 另做完整检查。</p>',
            f'<p>新核加入普通 TF32 显式舍入、TF32x3 和不同 tile。A=500、B=1、6 s 的完整 drive：'
            f'IEEE {device_us("drive_device","ieee"):.2f} μs；TF32 {device_us("drive_device","tf32"):.2f} μs。'
            '此处是相同 Y/W 的内部模块比较；是否启用还需通过全部 10 条轨迹的完整电压门限。'
            '原 F/R 逐模块替换证据仍保留在上面的配对表。</p>',
            '<p>相同驱动曲线下打开 / 关闭 TF32，reset 输出逐位一致。当前搜索、比较、归约和衰减没有 Tensor Core 矩阵乘法，继续使用 FP32。'
            '当上游滤波或二次项采用近似精度时，全部 reset 时间点参与 profile 准入检查。</p>',
            '<p>新增 13 种形状的显式精度 profile，分别给出 eager、Graph wall 与设备时间。默认新 profile 沿用 10⁻³ mV 门限；'
            '0.01 mV 近似 profile 需显式选择。Graph 仍用于减少主机提交，TF32 不直接加速调度操作。</p>'
        ]
        for layer,note in zip(layers,notes):
            layer['reading'] += head+note+'<p><a href="#tf32-five-layers">查看第 4.4 节全部形状、误差及两轮计时</a>。</p>'
            layer['tag'] += '；TF32 核查见说明'
    rows, articles = [], []
    for number, layer in enumerate(layers, 1):
        lid = layer['id']
        rows.append(f'<tr><th scope="row"><button type="button" class="layer-trigger" data-layer="{lid}" '
                    f'aria-label="{layer["name"]}：原代码、改动与实测" aria-haspopup="dialog" '
                    f'aria-expanded="false" aria-controls="layer-popover"><span class="layer-index">0{number}</span>'
                    f'<strong>{layer["name"]}</strong><span class="layer-trigger-hint">悬停 / 点击查看 ↗</span></button></th>'
                    f'<td>{layer["original"]}</td><td>{layer["operation"]}</td><td>{layer["measured"]}'
                    f'<span class="layer-badge">{layer["tag"]}</span></td></tr>')
        content = (f'<h4>你的原代码实际怎样计算</h4><p>{layer["original"]}</p>'
                   f'<p class="layer-source">{html.escape(layer["sourcecode"])}</p>'
                   f'<h4>本轮具体改了什么</h4>{layer["actions"]}'
                   f'<h4>相对原代码节省什么</h4>{layer["benefit"]}'
                   f'<h4>配对实测</h4>{layer["evidence"]}{layer["reading"]}'
                   f'<h4>条件与计时范围</h4><p>{layer["scope"]}</p>'
                   f'<p class="layer-source"><b>数据位置</b><br>{html.escape(layer["source"])}</p>')
        articles.append(f'<article id="layer-detail-{lid}" data-title="{layer["name"]}"><h3>0{number} · {layer["name"]}</h3>'
                        f'<div class="layer-detail-content">{content}</div></article>')
    guide = '''<div class="layer-guide" id="layer-guide">
<div class="layer-guide-heading"><h3>五层优化：以你的原始代码为比较起点</h3><span>鼠标悬停预览 · 点击固定阅读</span></div>
<p>主对照固定为冻结的 <code>DBNN_F / DBNN_R</code>：保留原有缓存、分块和 reset 分支。
滤波与驱动项只替换对应模块；reset 直接调用原函数对比。新增后端之间的选择统一列入内部消融。</p>
<p class="muted" id="layer-help">左列名称可悬停或用 Tab 聚焦，点击 / Enter 固定说明；Esc 关闭。下方也可展开全部。</p>
<div class="table-wrap"><table class="layer-table" aria-describedby="layer-help"><thead><tr>
<th scope="col">层次 / 详细说明</th><th scope="col">原代码实际计算方式</th><th scope="col">本轮对应改动</th><th scope="col">相对原代码的实测</th>
</tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table></div>
<p class="layer-protocol">位点压缩使用既有 50 次配对记录；滤波、完整驱动项和 reset 使用新补的 30 次交错测量。
滤波、驱动项表内为完整 forward；reset 主表为单独模块，完整 forward 另列。单位和对照在每张表中标明。
各单项均从对应原版独立替换，层间速度比不相乘；多项合并的收益直接测完整路径。
服务器为共享 GPU，保留逐次耗时、分位数和占用快照，最终性能数字仍需独占窗口复测。</p>
<details class="layer-full"><summary>展开全部五层说明（原代码、改动、理论收益与实测）</summary>''' + ''.join(articles) + '''</details></div>'''
    popover = '''<aside class="layer-popover" id="layer-popover" role="dialog" aria-modal="false" aria-labelledby="layer-popover-title" hidden>
<div class="layer-popover-head"><div><div class="layer-popover-mode" id="layer-popover-mode">悬停预览 · 点击名称固定</div>
<h3 id="layer-popover-title"></h3></div><button type="button" class="layer-close" aria-label="关闭优化说明" title="关闭（Esc）">×</button></div>
<div id="layer-popover-body" class="layer-popover-body" tabindex="0" aria-label="优化说明内容，可滚动"></div></aside>'''
    return guide, popover
