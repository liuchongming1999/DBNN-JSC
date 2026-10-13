"""Two-stage DBNN equations, numbered consistently and rendered as offline MathML.

PPT slide 2 supplies k_i, v_i, a_i, c_i, beta_i, gamma_i, a_jk and v_0.
The frozen DBNN.py supplies the causal, next-grid-step reset convention.
"""
from html import escape


def tag(name, *parts, **attrs):
    attributes = ''.join(f' {k}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
    return f'<{name}{attributes}>' + ''.join(parts) + f'</{name}>'


def mi(value, normal=False):
    return tag('mi', escape(value), **({'mathvariant': 'normal'} if normal else {}))


def mo(value):
    return tag('mo', escape(value))


def mn(value):
    return tag('mn', str(value))


def row(*parts):
    return tag('mrow', *parts)


def sub(base, index):
    return tag('msub', base, index)


def sup(base, power):
    return tag('msup', base, power)


def frac(top, bottom):
    return tag('mfrac', top, bottom)


def par(value):
    return row(mo('('), value, mo(')'))


def at(base, index):
    return row(base, mo('['), index, mo(']'))


def exp(value):
    return sup(mi('e', True), value)


def total(index, upper, value):
    return row(tag('munderover', mo('∑'), index, upper), value)


def indicator(condition):
    return row(mi('𝟏', True), mo('{'), condition, mo('}'))


def lines(*values):
    return tag('mtable', *(tag('mtr', tag('mtd', value)) for value in values),
               columnalign='left', rowspacing='0.6em')


def cases(items):
    return row(mo('{'), tag('mtable', *(tag('mtr', tag('mtd', value), tag('mtd', condition))
                                       for value, condition in items),
                           columnalign='left left', columnspacing='1.4em', rowspacing='0.55em'))


def equation(number, label, value, description):
    rendered = tag('math', row(value), xmlns='http://www.w3.org/1998/Math/MathML',
                   display='block', **{'aria-label': description})
    return (f'<div class="model-equations" id="dbnn-eq-{number}"><div class="model-equation">'
            f'<span class="model-eq-label">{escape(label)}</span><div class="model-math-scroll">'
            + rendered + f'</div><a class="model-eq-number" href="#dbnn-eq-{number}" '
            f'aria-label="公式 1.{number}">(1.{number})</a></div></div>')


def eqref(number):
    return f'<a href="#dbnn-eq-{number}">式 (1.{number})</a>'


MODEL_CSS = '''
/* DBNN_MODEL_CSS_START */
.model-equations{background:#f2f7f6;border:1px solid #dce9e5;border-radius:8px;margin:12px 0;padding:8px 16px;scroll-margin-top:16px}
.model-equation{display:grid;grid-template-columns:76px minmax(0,1fr) 48px;align-items:center;gap:10px;padding:9px 0}
.model-eq-label{font:12px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#587080}
.model-eq-number{text-align:right;text-decoration:none;font:14px/1.5 Georgia,serif;color:#587080;white-space:nowrap}
.model-eq-number:focus-visible{outline:2px solid #087b68;outline-offset:4px}
.model-math-scroll{overflow-x:auto;padding:3px 4px 7px}
.model-math-scroll math{font-size:20px;min-width:max-content;margin:0 auto;color:#183044}
.model-source{font-size:13px;color:var(--muted)}
.model-step-list{padding-left:24px}.model-step-list li{margin:8px 0}
.model-flow{background:#edf6f3;border-left:4px solid #087b68;padding:12px 16px;margin:16px 0;font-size:16px;line-height:1.85}
.model-index-table th,.model-index-table td{font-size:13px}
.model-algorithm{border-top:2px solid #087b68;border-bottom:1px solid #d4e0e5;margin:18px 0;padding:12px 0}
.model-algorithm h4{margin:0 0 8px}
#design h3{margin-top:30px;scroll-margin-top:16px}#design h4{font-size:15px;margin:22px 0 8px}
#design .model-reference-table td:first-child{min-width:146px;font-weight:400}
@media(max-width:700px){.model-equations{padding:7px 10px}.model-equation{grid-template-columns:minmax(0,1fr) 48px;gap:5px}.model-math-scroll{grid-column:1/-1;grid-row:2}.model-math-scroll math{font-size:18px;margin:0}.model-eq-label{font-weight:600}.model-eq-number{grid-column:2;grid-row:1}}
@media print{.model-equations{break-inside:avoid}.model-math-scroll{overflow:visible}.model-math-scroll math{font-size:17px}.model-algorithm{break-inside:avoid}}
/* DBNN_MODEL_CSS_END */
'''


def build_model_formulas(table):
    i, j, k, n, m, t, N, T, ell, K = [mi(s) for s in ('i', 'j', 'k', 'n', 'm', 't', 'N', 'T', 'ℓ', 'K')]
    zero, one = mn(0), mn(1)
    plus, minus, eq, comma = mo('+'), mo('−'), mo('='), mo(',')
    gap = tag('mspace', width='1em')
    dt = row(mi('Δ'), t)
    tau_r = sub(mi('τ'), row(mi('r'), comma, i))
    tau_d = sub(mi('τ'), row(mi('d'), comma, i))
    tau_reset = sub(mi('τ'), mi('reset', True))
    omega = sub(mi('ω'), i)
    ki, xi, vi = [sub(mi(s), i) for s in ('k', 'x', 'v')]
    ai, ci, beta, gamma = [sub(mi(s), i) for s in ('a', 'c', 'β', 'γ')]
    v0 = sub(mi('v'), zero)
    ajk = sub(mi('a'), row(j, k))
    u_n, r_n, v_n, z_n = [at(mi(s), n) for s in ('u', 'r', 'v', 'z')]
    alpha, amp = mi('α'), sub(mi('A'), mi('r'))
    vth = sub(mi('v'), mi('th', True))
    nm1 = row(n, minus, one)
    ellm1, ellp1 = row(ell, minus, one), row(ell, plus, one)
    n_ell, n_prev = sub(n, ell), sub(n, ellm1)
    def candidate(iteration, time=n):
        return at(sup(mi('v'), par(iteration)), time)
    inv_rates = row(frac(one, tau_r), plus, frac(one, tau_d))
    kernel = row(omega, par(row(one, minus, exp(row(minus, frac(t, tau_r))))),
                 exp(row(minus, frac(t, tau_d))), indicator(row(t, mo('≥'), zero)))
    linear = total(row(i, eq, one), N, at(vi, n))
    quadratic = total(row(j, eq, one), N,
                      total(row(k, eq, one), row(j, minus, one),
                            row(ajk, at(sub(mi('v'), j), n), at(sub(mi('v'), k), n))))
    drive = row(u_n, eq, linear, plus, quadratic, plus, v0)
    vector = mi('𝐯', True)
    vvec = at(vector, n)
    weight_mapping = row(ajk, eq, sub(mi('W'), row(j, k)), plus, sub(mi('W'), row(k, j)),
                         comma, gap, j, mo('>'), k, comma, gap,
                         sub(mi('W'), row(j, j)), eq, zero)
    matrix_drive = row(u_n, eq, v0, plus, sup(mi('𝟏', True), mi('T', True)), vvec, plus,
                       sup(vvec, mi('T', True)), mi('W'), vvec)

    f_kernel = equation(1, '响应核', row(ki, par(t), eq, kernel), '因果双指数响应核')
    f_filter = equation(2, '卷积滤波', row(at(vi, n), eq, at(par(row(xi, mo('⊗'), ki)), n), eq,
                        total(row(m, eq, zero), n,
                              row(at(xi, m), ki, par(row(par(row(n, minus, m)), dt))))),
                        '位点贡献等于带符号脉冲输入与双指数核的离散因果卷积')
    r_coeff = equation(3, '衰减系数', row(beta, eq, exp(row(minus, frac(dt, tau_d))), comma, gap,
                       gamma, eq, exp(row(minus, dt, par(inv_rates)))),
                       '两个位点滤波状态的逐格衰减比例 beta 和 gamma')
    r_states = equation(4, '状态递推', lines(
        row(at(ai, n), eq, at(xi, n), plus, beta, at(ai, nm1)),
        row(at(ci, n), eq, at(xi, n), plus, gamma, at(ci, nm1)),
        row(at(ai, row(minus, one)), eq, at(ci, row(minus, one)), eq, zero)),
        '两个状态各自递推，初始值均为零')
    r_voltage = equation(5, '位点贡献', row(at(vi, n), eq, omega, par(row(at(ai, n), minus, at(ci, n)))),
                         '位点电压贡献为两个滤波状态之差乘以幅度')
    integration = equation(6, '共同整合', drive, '两个滤波路径共用的线性与双线性整合，输出 reset 前的驱动曲线 u')
    matrix = equation(7, '矩阵对应', lines(weight_mapping, matrix_drive),
                      'PPT 中位点对系数等于代码两个方向权重之和，胞体驱动可写为二次型')
    ode = equation(8, '连续状态', lines(
        row(xi, par(t), eq, total(row(m, eq, zero), row(T, minus, one),
                                 row(at(xi, m), mi('δ'), par(row(t, minus, m, dt))))),
        row(frac(row(mi('d', True), ai), row(mi('d', True), t)), eq,
            minus, frac(ai, tau_d), plus, xi, par(t)),
        row(frac(row(mi('d', True), ci), row(mi('d', True), t)), eq,
            minus, par(inv_rates), ci, plus, xi, par(t)),
        row(vi, par(t), eq, omega, par(row(ai, par(t), minus, ci, par(t))))),
        '带脉冲跳跃的连续时间状态方程，对应阶段一的精确离散递推')

    parameters = equation(9, 'reset 参数', row(alpha, eq, exp(row(minus, frac(dt, tau_reset))),
                          comma, gap, tau_reset, mo('>'), zero, comma, gap, amp, mo('≥'), zero),
                          'alpha 是每格保留比例，tau reset 才是物理时间常数')
    initialization = equation(10, '迭代初值', row(candidate(zero), eq, u_n, comma, gap,
                             sub(n, zero), eq, minus, one),
                             '第零轮候选曲线等于驱动曲线，前一个事件位置用负一作初始标记')
    S_ell = sub(mi('𝒮'), ell)
    selection = equation(11, '最早触发', lines(
        row(S_ell, eq, mo('{'), n, mo('∈'), mo('{'), zero, comma, mi('…'), comma,
            row(T, minus, one), mo('}'), mo(':'), n, mo('>'), n_prev, comma,
            candidate(ellm1), mo('≥'), vth, mo('}')),
        row(n_ell, eq, mi('min', True), S_ell, comma, gap, S_ell, mo('≠'), mo('∅'))),
        '每一轮只确认剩余曲线上最早达到阈值的时间格，空集合时停止')
    curve_update = equation(12, '整段修正', row(candidate(ell), eq, cases([
        (candidate(ellm1), row(zero, mo('≤'), n, mo('≤'), n_ell)),
        (row(candidate(ellm1), minus, amp, sup(alpha, row(n, minus, n_ell, minus, one))),
         row(n_ell, plus, one, mo('≤'), n, mo('<'), T)),
    ])), '保留已确认前缀，从本次触发的下一时间格开始对整段后缀叠加负向指数响应')
    final_output = equation(13, '最终结果', row(v_n, eq, candidate(K), comma, gap, z_n, eq,
                           total(row(ell, eq, one), K, indicator(row(n, eq, n_ell)))),
                           '确认 K 次事件后得到最终电压，z 在每个事件时间格为一，在其他格为零')
    history = equation(14, '累计 reset', row(r_n, eq, minus, amp,
                       total(row(m, eq, zero), nm1,
                             row(sup(alpha, row(n, minus, one, minus, m)), at(mi('z'), m))),
                       comma, gap, v_n, eq, u_n, plus, r_n),
                       '最终 reset 状态是历次已发生触发的指数历史之和，电压等于驱动加该状态')
    sequential = equation(15, '逐格等价式', lines(
        row(at(mi('r'), zero), eq, zero),
        row(v_n, eq, u_n, plus, r_n, comma, gap, z_n, eq, indicator(row(v_n, mo('≥'), vth))),
        row(at(mi('r'), row(n, plus, one)), eq, alpha, r_n, minus, amp, z_n)),
        '与曲线事件迭代等价的逐格递推，n 在所有行中仍表示物理时间格')
    s_ell = sub(mi('s'), ell)
    rho = sub(mi('ρ'), ell)
    gpu_start = equation(16, '后缀候选', lines(
        row(s_ell, eq, n_prev, plus, one, comma, gap, rho, eq, at(mi('r'), s_ell)),
        row(candidate(ellm1), eq, u_n, plus, sup(alpha, row(n, minus, s_ell)), rho,
            comma, gap, s_ell, mo('≤'), n, mo('<'), T)),
        '每轮用当前后缀起点与一个标量 reset 边界状态重建整段候选曲线')
    gpu_next = equation(17, '边界更新', row(sub(mi('s'), ellp1), eq, n_ell, plus, one,
                         comma, gap, sub(mi('ρ'), ellp1), eq,
                         sup(alpha, row(n_ell, minus, s_ell, plus, one)), rho, minus, amp),
                         '确认最早事件后按实际跨过的时间格数更新下一轮边界状态')
    active = mi('𝒜')
    va = at(sub(vector, active), n)
    sparse = equation(18, '活跃集整合', row(u_n, eq, v0, plus,
                      sup(mi('𝟏', True), mi('T', True)), va, plus,
                      sup(va, mi('T', True)), sub(mi('W'), row(active, active)), va),
                      '阶段一只保留零初始状态下整个窗口有输入的位点，得到同一驱动曲线')

    indices = table(['记号', '唯一含义', '与时间的关系'], [
        ['n，m', '离散时间格编号，0,…,T−1', '第 n 格的物理时间为 t<sub>n</sub>=nΔt；本轮 Δt=1 ms'],
        ['ℓ', 'reset 曲线更新轮次，ℓ=1,2,…', '每轮确认一次最早触发；一轮可以跨过多个时间格'],
        ['n<sub>ℓ</sub>，K', '第 ℓ 次触发的时间格；整个窗口确认的触发总数', 'n<sub>1</sub>&lt;…&lt;n<sub>K</sub>，0≤K≤T'],
        ['i，j，k', '输入位点编号', '沿用 PPT 的 a<sub>jk</sub>；与 reset 迭代轮次分开'],
        ['u[n]，v<sup>(ℓ)</sup>[n]，v[n]', 'reset 前驱动；第 ℓ 轮候选曲线；最终输出', '方括号内始终是时间格，上标括号内始终是更新轮次'],
    ])
    notation = table(['PPT / 本节符号', '含义', '代码 / 后文记号'], [
        ['x<sub>i</sub>[n]', '每个位点的带符号脉冲输入，兴奋性为正、抑制性为负', '<code>x</code>'],
        ['k<sub>i</sub>；ω<sub>i</sub>；τ<sub>r,i</sub>、τ<sub>d,i</sub>',
         '响应核；幅度；上升与衰减时间常数', '<code>kernels</code>；<code>omega</code>；<code>tau_rise</code>、<code>tau_decay</code>'],
        ['v<sub>i</sub>[n]，<b>v</b>[n]', '各位点的电压贡献及其向量，与最终胞体电压 v[n] 区分', '<code>y</code>，批量矩阵布局为 <code>Y</code>'],
        ['a<sub>i</sub>、c<sub>i</sub>；β<sub>i</sub>、γ<sub>i</sub>', '阶段一滤波状态及其逐格衰减系数',
         '<code>state_a</code>、<code>state_c</code>；<code>decay_a</code>、<code>decay_c</code>'],
        ['a<sub>jk</sub>，v<sub>0</sub>', '位点对耦合系数、基线电压；单下标 a<sub>i</sub> 是滤波状态',
         '<code>W[j,k]+W[k,j]</code>；<code>bilinear.bias</code>（后文 b）'],
        ['u[n]，v<sup>(ℓ−1)</sup>[n]，v[n]', '阶段一输出、当前轮候选后缀、最终胞体电压',
         '<code>v_drive</code>、<code>v_seg</code>（当前后缀）、<code>v</code>'],
        ['ℓ，n<sub>ℓ</sub>，s<sub>ℓ</sub>，ρ<sub>ℓ</sub>', '事件轮次、事件位置、待处理后缀起点、该处 reset 状态',
         '<code>while</code> 的轮次；原 Python 的 <code>end</code>、<code>t</code>、<code>r_t</code>'],
        ['r[n]，z[n]', '最终轨迹上的 reset 状态与触发指示；均以时间格为下标', '<code>r_t</code>、<code>s_t</code>（逐格版本）'],
        ['α，A<sub>r</sub>，τ<sub>reset</sub>，v<sub>th</sub>', '逐格衰减比例、抑制幅度、时间常数与触发阈值',
         '<code>alpha</code>、<code>abs(reset_amp)</code>、<code>max(abs(tau_reset), 10⁻⁴)</code>、<code>v_th</code>'],
    ])
    return f'''
<!-- DBNN_MODEL_START -->
<p>DBNN 的完整前向分成两个阶段。<strong>阶段一先计算整条驱动曲线 u[n]；阶段二以该曲线为输入，通过 reset 事件迭代得到最终电压 v[n]。</strong> DBNN_F 与 DBNN_R 是阶段一的两种等价计算路径，随后使用相同的 reset 规则。</p>
<div class="model-flow"><strong>阶段一：</strong>输入 x<sub>i</sub>[n] → 卷积 / 状态递推滤波 → 线性与双线性整合 → u[n]<br>
<strong>阶段二：</strong>u[n] → 寻找最早触发 → 修正其后整段曲线 → 继续寻找下一次触发 → v[n]</div>
<p class="model-source">符号参照 <code>DBNN-mouse V1 L5PC.pptx</code> 第 2 页，reset 依据冻结的 <code>DBNN.py::_apply_reset_closed_loop</code>。保留 k<sub>i</sub>、v<sub>i</sub>、a<sub>i</sub>、c<sub>i</sub>、β<sub>i</sub>、γ<sub>i</sub>、a<sub>jk</sub> 和 v<sub>0</sub>。N 为候选位点数，T 为时间格数，B 为独立样本数；下面省略 batch 下标。</p>
<div class="model-index-table">{indices}</div>
<p class="muted">上文讨论中的“第 k 轮曲线”在这里统一改记为第 ℓ 轮，以保留 PPT 中 k 的位点含义。连续时间用 t，离散时间用方括号 [n]，迭代轮次用上标 (ℓ)，公式按 (1.1)–(1.18) 连续编号。</p>

<h3 id="model-stage-one">1.1 阶段一：滤波与二次整合，输出 u[n]</h3>
<h4 id="model-convolution">路径 F：卷积形式（DBNN_F）</h4>
<p>x<sub>i</sub>[n] 为第 i 个位点在第 n 格的带符号脉冲输入。双指数核由幅度 ω<sub>i</sub> 和正时间常数 τ<sub>r,i</sub>、τ<sub>d,i</sub> 决定，得到位点贡献 v<sub>i</sub>[n]：</p>
{f_kernel}{f_filter}
<p class="muted">⊗ 表示离散因果卷积，核在 (n−m)Δt 取值。FFT 实现零填充至至少 2T−1，再截取前 T 格；k<sub>i</sub>(0)=0。</p>
<h4 id="model-recurrence">路径 R：状态递推形式（DBNN_R）</h4>
<p>双指数核可写为 k<sub>i</sub>(nΔt)=ω<sub>i</sub>(β<sub>i</sub><sup>n</sup>−γ<sub>i</sub><sup>n</sup>)，因此同一位点响应也可由两个状态计算：</p>
{r_coeff}{r_states}{r_voltage}
<p class="muted">这里的 n 始终沿物理时间格推进；并行 scan 加速的是这两个状态递推。相同输入、参数和零初始状态下，{eqref(2)} 与 {eqref(3)}–{eqref(5)} 代数等价。</p>
<h4>两条路径共用的整合与阶段输出</h4>
<p>将全部位点贡献线性求和，再加入位点对之间的双线性项和基线电压 v<sub>0</sub>：</p>
{integration}
<p><strong>阶段一到这里结束，得到完整的 u[0],…,u[T−1]。</strong> 在后续 reset 迭代中，u[n] 作为固定驱动使用。</p>
<p>记 <b>v</b>[n]=(v<sub>1</sub>[n],…,v<sub>N</sub>[n])<sup>T</sup>。PPT 每对位点只求和一次，代码使用零对角线的完整矩阵 W，二者对应为：</p>
{matrix}
<p class="muted">W 可以不对称，每对有效系数为 W<sub>jk</sub>+W<sub>kj</sub>。核内已经包含 ω<sub>i</sub>，线性项直接求和 v<sub>i</sub>；后文的 y / Y、b 分别对应这些位点贡献及 v<sub>0</sub>。</p>
<details id="model-ode"><summary>连续时间 ODE 与路径 R 的对应关系</summary>
<p>将输入写为 Dirac 脉冲列，δ 为 Dirac 分布，零初始状态下有：</p>
{ode}
<p>脉冲间状态指数衰减；脉冲到达时 a<sub>i</sub>、c<sub>i</sub> 同时增加 x<sub>i</sub>[n]。取到达后的状态即得到 {eqref(4)} 的精确离散更新。这组 ODE 完成阶段一的位点滤波，之后仍按 {eqref(6)} 得到 u[n]。</p></details>

<h3 id="model-stage-two">1.2 阶段二：按事件迭代修正曲线，输出 v[n]</h3>
<p>令 v<sup>(ℓ)</sup>[n] 表示<strong>完成 ℓ 次 reset 修正后的候选曲线</strong>。每轮在尚未确认的后缀中找出最早触发，只新增这一次触发的负向指数效应，然后重新检查修正后的后缀。α 是每个时间格的衰减比例，τ<sub>reset</sub> 才是时间常数：</p>
{parameters}
<h4 id="model-reset">初始化、寻找事件与更新曲线</h4>
{initialization}
<p class="muted">n<sub>0</sub>=−1 只是“尚无已确认事件”的起始标记。第一轮 ℓ=1，从 n=0 开始搜索。所有上标 (ℓ) 均为迭代编号。</p>
{selection}
<p>若 𝒮<sub>ℓ</sub> 为空，就结束迭代；否则确认 n<sub>ℓ</sub> 为本轮事件位置，并更新曲线：</p>
{curve_update}
<p><strong>触发格 n<sub>ℓ</sub> 的电压保留，本次抑制从 n<sub>ℓ</sub>+1 开始。</strong> α 的指数 n−n<sub>ℓ</sub>−1 表示抑制开始后经过的时间格数。若连续几格均达到阈值，也可在相邻时间格连续确认事件。</p>
<div class="model-algorithm" id="model-reset-algorithm"><h4>算法 1：reset 曲线迭代</h4>
<p><b>输入：</b>阶段一的 u[0],…,u[T−1]，以及 α、A<sub>r</sub>、v<sub>th</sub>。<b>输出：</b>最终电压 v[0],…,v[T−1] 与触发时间格 n<sub>1</sub>,…,n<sub>K</sub>。</p>
<ol class="model-step-list">
<li>按 {eqref(10)} 初始化 v<sup>(0)</sup> 与 n<sub>0</sub>，令 ℓ=1。</li>
<li>按 {eqref(11)} 检查当前候选曲线在 n&gt;n<sub>ℓ−1</sub> 的剩余后缀。</li>
<li>若没有达到阈值的点，令 K=ℓ−1，返回 v=v<sup>(K)</sup> 及已确认事件。</li>
<li>否则只确认最早的 n<sub>ℓ</sub>，按 {eqref(12)} 修正其后整段曲线。</li>
<li>令 ℓ←ℓ+1，返回步骤 2，依据修正后的曲线寻找下一次触发。</li></ol>
<p class="muted">事件位置严格递增，每个时间格最多确认一次，因此 K≤T。ℓ 统计曲线更新轮次，n 标记曲线横轴的位置。</p></div>
{final_output}
<p>例如 Δt=1 ms，第一轮 ℓ=1 在 n<sub>1</sub>=100 触发，则在 101 ms 减去 A<sub>r</sub>，102 ms 减去 A<sub>r</sub>α，103 ms 减去 A<sub>r</sub>α²。第二轮从修正后的 101 ms 起继续搜索。</p>
<p>迭代结束后，最终曲线也可以用累计 reset 状态表达：</p>
{history}
<p class="muted">n=0 时求和为空，r[0]=0；K=0 时最终 v[n]=u[n]、z[n]=0。z[n] 是最终事件序列在第 n 个时间格的指示值，其下标始终是时间格。</p>
<details id="model-reset-sequential"><summary>与逐格 reset 递推的等价关系</summary>
{sequential}
<p>每个时间格先输出 v[n]，再判断 z[n]，最后更新下一格的 r[n+1]。展开这个递推就得到 {eqref(14)}；按最早事件逐轮确认则得到算法 1。两种写法使用相同的闭环规则。</p></details>
<details id="model-reset-event"><summary>GPU 实现：用边界状态重建后缀，避免保存所有轮次的曲线</summary>
<p>第 ℓ 轮的待处理后缀从 s<sub>ℓ</sub> 开始，该处已累计的 reset 状态记为 ρ<sub>ℓ</sub>。初值 s<sub>1</sub>=0、ρ<sub>1</sub>=0。只考虑此前已确认的事件时，当前候选后缀可直接重建：</p>
{gpu_start}
<p>GPU 并行检查这些时间格，并归约得到最早达到阈值的 n<sub>ℓ</sub>。先写出 s<sub>ℓ</sub>≤n≤n<sub>ℓ</sub> 的最终输出，再更新下一轮边界：</p>
{gpu_next}
<p>若后缀没有触发，直接写出全部余段并结束。这样在数学上完成 {eqref(12)} 的整段修正，实际只维护驱动 u、输出 v 和当前边界状态。原始 PyTorch 的 B=1 推理已采用事件思路；优化核将搜索、写出和状态更新放在 GPU 内完成。这里的幂指数由跨过的时间格数决定。</p></details>
<p class="model-source">以上描述推理时的硬阈值前向。训练代码保持相同的前向触发规则，通过代理梯度反传。NEURON 基线由膜电位与离子通道动力学生成发放和恢复；DBNN 的 reset 状态用于近似发放后的影响。</p>

<h3 id="model-mapping">1.3 活跃位点压缩与实现对应</h3>
<p>整窗接收输入的位点集合记为 𝒜，A=|𝒜|。全窗无输入且初始滤波状态为零的位点始终满足 v<sub>i</sub>[n]=0。因此阶段一可只计算活跃位点，并提取 W<sub>𝒜𝒜</sub>：</p>
{sparse}
<p>递推滤波由 O(BTN) 缩为 O(BTA)，二次整合由 O(BTN²) 缩为 O(BTA²)。所得 u[n] 保留全部时间格，随后进入相同的阶段二。全量 W 的存储仍为 O(N²)；本轮保留完整指数尾部与活跃子矩阵。</p>
<p>每个 batch 当前共用同一组 active IDs。跨窗口推理需传递阶段一的 a<sub>i</sub>、c<sub>i</sub> 和阶段二的 reset 边界状态；活跃集合还需包含尚有残留滤波状态的位点。若全窗几乎所有位点都接受过输入，A 仍接近 N。</p>
<details id="model-notation"><summary>完整符号与代码对照表</summary><div class="model-reference-table">{notation}</div></details>
<!-- DBNN_MODEL_END -->
'''
