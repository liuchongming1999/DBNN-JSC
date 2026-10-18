"""Readable audit with immutable evidence; no manuscript mutation."""
from pathlib import Path
import json,html,datetime,shutil,hashlib,argparse

def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();S=a.source;O=a.output;O.mkdir(parents=True,exist_ok=True);(O/'evidence').mkdir(exist_ok=True)
 probes=read(S/'numerical_probes.json');cal=read(S/'data_and_calibration_audit.json');label=read(S/'sampling_and_label_audit.json');solver=read(S/'solver_equivalence_audit.json');whole=read(S/'tcn_whole_main379_complete.json') if (S/'tcn_whole_main379_complete.json').exists() else None
 sources=S/'sources';public={m:read(sources/'original'/m/'complete.json') for m in ['tcn','elm','reduce_legacy']};train={m:read(sources/'aligned'/f'{m}_1ms/complete.json') for m in ['tcn','elm','lstm']}
 for p in S.glob('*.json'):shutil.copy2(p,O/'evidence'/p.name)
 shutil.copy2(S/'Spieler_et_al_ICLR2024.pdf',O/'evidence/Spieler_et_al_ICLR2024.pdf')
 # Only required plain-text source copies; no redundant models/data.
 shutil.copytree(sources,O/'source_snapshot',dirs_exist_ok=True)
 now=datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
 wrap=lambda s:f'<p>{s}</p>'
 speed_html='<p>冻结TCN的全部379条整段前向核验尚在运行，完成前不使用三个样例的倍率替换论文结论。</p>'
 if whole:
  z=whole['summary'];pa=whole['parity'];wc=z['tcn_author_windows']['wall_ms'];wf=z['tcn_whole']['wall_ms'];db=z['dbnn_frozen']['wall_ms']
  speed_html=f'''<div class="callout"><b>全379条验证完成。</b>TCN整段推理与原切窗推理的尖峰完全一致：{pa['whole_events_equal_cases']}/379；最大电压差{pa['whole_voltage_max_mv']:.3g} mV。切窗总时间{wc:.2f} ms，整段{wf:.2f} ms，整段快{wc/wf:.2f}倍。同轮DBNN总时间{db:.2f} ms，因而本次DBNN相对整段TCN的时间比为{wf/db:.2f}倍。</div>
<table><thead><tr><th>固定模型、不同推理方式</th><th>总prepared wall time / ms</th><th>RMSE / mV</th><th>F1</th></tr></thead><tbody>'''+''.join(f'<tr><td>{n}</td><td>{v["wall_ms"]:.3f}</td><td>{v["rmse_mv"]:.6f}</td><td>{v["events"]["f1"]:.6f}</td></tr>' for n,v in z.items())+'''</tbody></table><p>这是本轮独立配对计时，不直接拼接成旧主图或用来比较不同日期硬件负载下的其他模型。原图、原始计时和checkpoint均保留。</p>'''
 tri='<table><thead><tr><th>方法</th><th>检查内容</th><th>核查结果</th></tr></thead><tbody>'
 rows=[('TCN','结构、损失、预算、初始化、优化器、BatchNorm','7层×128、物理历史152 ms（153个采样点）、400 ms训练窗、树突PCA辅助监督与720000次更新均有记录。NAdam数值核对和BatchNorm梯度核对通过。七层结构取自公开权重，不是把仓库当前三层示例误当作唯一默认。'),('Branch-ELM','模型公式、初始化、损失、预算、推理拼接','直接使用作者固定版本ELM类；损失与公开Python模块逐值相同；350000次更新。除已发现的输入映射外，窗口抽样还有两项明确差异，详见下文。'),('LSTM','门控公式、参数量、训练预算与初始化依据','50单元标准LSTM公式通过独立核对；500输入时110302个可训练参数，原1278输入时对应265902，与原表对上；342000次更新。但没有作者LSTM权重或原训练实现的逐项复现证据，初始化只能视为当前自定实现细节。'),('NEURON / CoreNEURON','相同输入、线程设置、实际引擎、步长、输出一致性','主线固定步长、secondorder=0；CoreContext在psolve期间确实启用CoreNEURON，当前对照明确为CPU1核。同一步长下379条的电压和尖峰一致性全部通过。'),('Neuron_Reduce','原作者回归、缩减设置、NEURON9兼容补丁','作者Hay回归在NEURON7.8.2通过；0 Hz缩减频率和自动分段与作者README示例一致。9.0.2适配仅过滤非数值API项并保留原细胞ena/ek；旧版本严格回归与当前细胞适配分开记录。')]
 tri+=''.join(f'<tr><td>{x}</td><td>{y}</td><td>{z}</td></tr>' for x,y,z in rows)+'</tbody></table>'
 defaults=[('更新预算','30 × 11400','35 × 10000','采用公开代码350000次'),('训练loss预热','前150 ms不计loss','全500 ms计loss；burn_in_time用于评估','与Python公开代码一致'),('记忆时间常数初始化','1–100 ms，等距','1–150 ms，对数间隔','与Python公开代码一致'),('记忆时间常数可训练性/范围','可学习；0–500 ms范围','固定；1–150 ms范围','与Python公开代码一致'),('记忆更新系数λ','10','5','与Python类默认一致')]
 deftable='<table><tr><th>设置</th><th>论文附录 / Table S1</th><th>公开Python代码</th><th>当前实验</th></tr>'+''.join(f'<tr><td>{x}</td><td>{y}</td><td>{z}</td><td>{v}</td></tr>' for x,y,z,v in defaults)+'</table>'
 traintable='<table><tr><th>模型</th><th>已完成更新数</th><th>可训练参数</th><th>验证选中轮次</th></tr>'+''.join(f'<tr><td>{m}</td><td>{d["updates"]:,}</td><td>{d["config"]["trainable_parameters"]:,}</td><td>{d["selected_epoch"]}</td></tr>' for m,d in train.items())+'</table>'
 caltable='<table><tr><th>模型</th><th>原始RMSE / mV</th><th>验证集校准后RMSE / mV</th></tr>'+''.join(f'<tr><td>{m}</td><td>{d["raw"]:.6f}</td><td>{d["calibrated"]:.6f}</td></tr>' for m,d in cal['voltage_calibration'].items())+'</table>'
 content=f'''<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DBNN 对比方法实现核查</title><style>body{{margin:0;background:#f2f5f7;color:#263f4f;font:16px/1.8 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1100px;margin:32px auto;background:white;padding:35px 44px;border-radius:12px}}h1{{font-size:29px}}h2{{font-size:23px;margin-top:38px;border-bottom:1px solid #d9e2e8}}h3{{font-size:18px;margin-top:25px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{padding:10px 12px;border-bottom:1px solid #d9e2e8;text-align:left;vertical-align:top}}th{{background:#edf3f7}}.callout{{background:#eef7f5;border-left:4px solid #008d7e;padding:15px 20px}}.warning{{background:#fff7eb;border-left:4px solid #c68b31;padding:15px 20px}}.muted{{font-size:14px;color:#738795}}code{{overflow-wrap:anywhere}}a{{color:#16759b}}@media(max-width:700px){{main{{padding:18px;margin:0}}table{{display:block;overflow:auto}}}}</style>
<main>
<h1>对比方法实现核查：TCN、Branch-ELM、LSTM与数值求解器</h1><p class="muted">更新于{now}。这是独立核查记录；未修改正文、主图、训练数据、DBNN或现有对照检查点。</p>

<div class="warning"><b>结论：核心实现已有多项直接验证，但“所有方法完全按原作者默认”仍不够准确。</b>需要优先处理Branch-ELM抽样器的实际差异、TCN可避免的推理切窗开销，以及LSTM尚缺作者结果训练复现这三点。输入/标签、校准和F1中也有合理适配，应明确写出来。</div>
<h2>1. 核查范围与证据层次</h2>
<p>逐项对照固定作者仓库版本、论文附录、真正写入checkpoint的配置、完整训练日志和实际评估代码。作者三个仓库的工作区均干净，没有悄悄改过vendor模型。已完成预算与公开权重复现是不同的证据：前者证明我们的任务没有提前停训；后者证明公开模型能够被正确执行；二者都不能单独证明换成小鼠数据后必然达到原文精度。</p>
{tri}
<h2>2. Branch-ELM：映射之外的两项实际抽样差异</h2>
<p>作者公开加载器先抽取8条不同轨迹，再从每条轨迹的500 ms之后抽取500 ms窗口。当前共用加载器先有放回抽取8条轨迹，再抽窗口终点：终点至少500 ms，因此窗口最早可以从0 ms开始。这是起点与终点概念混用造成的适配差异，不应被“窗口都是500 ms”掩盖。</p>

<table><tr><th>项目</th><th>作者公开ELM加载器</th><th>当前共用加载器</th></tr><tr><td>6秒轨迹的窗口起点</td><td>从500 ms开始，末端保留完整窗口</td><td>从0 ms开始，末端保留完整窗口</td></tr><tr><td>每个批次的8条轨迹</td><td>无放回选择，不重复</td><td>有放回选择，可能重复</td></tr><tr><td>与TCN作者加载器的关系</td><td>ELM单独采用该规则</td><td>更接近TCN原来的“抽窗口终点”规则</td></tr></table>
<p>在6秒轨迹上，当前ELM约9.1%的窗口会早于作者允许的起点。此项影响目前没有单独量化，不能预先说它一定使ELM更差或更好。建议最小修正是为ELM保留作者的独立抽样规则；其余loss、Adam、学习率、容量、种子和预算保持不变。当前形态映射重训只修正了映射，尚未包含此次发现的抽样修正，所以不能把它称为所有细节均校正后的最终对照。本轮检查没有中途改变该训练任务。</p>

<p class="muted">证据：<a href="source_snapshot/vendor/elmneuron/src/neuronio/neuronio_data_loader.py">作者generate_batch</a>与<a href="source_snapshot/panel.py">当前sample_batch</a>。作者按文件加载和TCN批内抽时刻不重复等细节也与当前统一数据池有细微区别；这些应列为数据适配，而不是隐藏成逐行相同的训练流水线。</p>

<h3>公开代码默认与论文设置确实不同</h3>
{deftable}<p>论文实验使用JAX/Equinox/Optax，而仓库提供的是最小PyTorch实现。我们直接使用该仓库的模型，公开20-memory检查点复现也通过，因此“按公开Python代码默认”有证据；“完全等同论文Table S1设置”没有。两种版本均有出处，不能把这些差异一概判成我们误写了模型。后续应先固定一种来源，明确版本；若需复现论文规格，则作为独立预设对照，不做测试集调参。</p>

<p>初始化、E/I正负输入、正突触权重约束、零初始状态、递归方程、输出缩放与500 ms/150 ms重叠拼接都已对照作者类/评估代码。公开loss模块与当前loss在相同输出上的差为0。此前发现的随机编号分支映射仍是单独的重要问题。</p>

<h2>3. TCN：训练主体通过；推理速度需要采用更合适的等价执行</h2>
<p>模型取自公开的七层NMDA TCN检查点结构，并非把仓库当前展示的三层例子冒充七层网络。卷积、ReLU后BatchNorm、头部初始化、正则项、三路监督及五阶段权重均有代码依据。NAdam与Keras 2.2.4原公式在600次受控更新中最大参数差为{probes['nadam_vs_keras224_numpy']['max_parameter_abs']:.3g}；BatchNorm前向、移动统计与有限差分梯度检查通过。这些是小规模实现检查，不是从头复现作者整次训练。</p>

<p>原评估用400 ms窗口、154 ms重叠，便于处理大批公开数据。对于本研究的单条轨迹，TCN在推理时没有递归隐状态，BatchNorm也使用固定统计，因此可一次完成整段因果卷积；每个输出仍只依赖前152 ms物理历史。这样既避免重叠计算，也让GPU有更多时间点并行，输出在浮点误差范围内应保持相同。三个预设时长样例先确认了这一点，随后才做全部379条冻结权重核查。</p>
{speed_html}
<p>本次配对采用同一张5090、FP32 IEEE、统一CUDA Graph、每输入预热3次和计时7次，方法顺序按固定随机种子打散。没有改变训练、权重、阈值或模型容量。若全379条输出一致性通过，建议后续用整段TCN作为更有说服力的速度对照，并重新核对“最快学习对照”的主要倍率；不能只因为切窗是原作者流程，就把明显可以避免的开销永久计入方法的速度劣势。</p>

<h2>4. LSTM：结构与预算有依据，完整原文训练复现仍缺失</h2>
<p>当前实现是50单元单层LSTM，门的sigmoid/tanh、顺序与显式递归公式相符；在原1278输入维数下，参数量对应论文表S10的265902。30轮×11400批次、Adam、余弦学习率、500 ms窗口和150 ms loss预热也有记录。</p>
<p>但当前采用了Glorot输入权重、正交递归权重和forget bias=1。这些不是从作者公开LSTM代码核验出来的。原文使用Equinox，<a href="https://github.com/patrick-kidger/equinox/blob/v0.8.0/equinox/nn/rnn.py">Equinox 0.8.0默认LSTM初始化</a>为均匀随机，和当前选择不同；作者是否覆盖默认初始化，现有材料无法确定。不能仅凭参数量对上，就宣称初始化、训练实现和原文完全一致。</p>
<p>最合适的表述是“按论文结构与训练方案实现的LSTM-50对照”，并列明初始化。若要把证据补到与TCN/ELM相同层次，应先在公开NeuronIO数据上复核LSTM训练结果。论文也报告了更大的LSTM；当前50单元是明确的容量选择，不能把它概括为整类LSTM能达到的最优精度。本轮没有新开LSTM调参或训练。</p>

<h2>5. 输入、尖峰标签、校准与指标：哪些属于合理适配</h2>
<p>TCN的输入通道非负，E/I身份由通道区分；ELM和LSTM使用E为正、I为负的编码。TCN和LSTM输入层对通道做可学习的稠密线性变换，不依赖位点相邻编号；固定符号可由输入权重吸收。ELM分支邻域则对编号有结构依赖，故其映射必须特别核验。主379条及训练数据的单bin输入最大均为1，没有发现学习模型输入多次事件而求解器只投递一次的计数不一致。</p>

<h3>尖峰时间标签不同，不能机械照抄减0.5 ms</h3>
<p>作者原数据的尖峰由AP波形局部峰值定义，加载时再用(t−0.5)转整数标签；我们以NEURON的−20 mV向上穿越时刻为事件，再映射到最近的1 ms网格。保存训练轨迹中，峰值相对穿越时刻的中位延后约0.3 ms；134个事件中，按两套定义重建标签有64个相差一个bin，70个相同。这是事件约定适配，不足以证明训练标签写错。应把统一事件定义讲清楚，不能直接在当前原生事件上照搬−0.5 ms造成额外平移。</p>

<h3>测试集矩匹配改成验证集锁定，是为了避免使用测试答案</h3>
<p>作者评估代码会用测试目标电压的均值和标准差调整预测；我们在验证集锁定一次仿射校准，主测试不再读取目标做校准。这是有意的评价协议改变。原379条同一0.0125 ms参考下，校准前后结果如下；三个对照都受益，不是校准把别人做差。</p>
{caltable}
<p>作者原文主要报告逐时间点ROC-AUC；我们的F1是独立事件、±5 ms逐一匹配，并用验证集选阈值。AUC不能直接当作F1。学习模型用局部概率峰且间隔至少5 ms，主379条的参考事件和DBNN预测事件都没有小于5 ms的间隔。本轮未据此改变阈值或尖峰规则。</p>

<h2>6. 数值求解器：未发现同输入下的核心配置错误</h2>
<p>NEURON multisplit与CoreNEURON在0.5和0.25 ms下，各有379/379条与同一步长单线程NEURON的尖峰完全一致；电压最大差不超过1.9×10⁻⁹ mV。固定步长与secondorder=0均明确记录。CoreNEURON是CPU1核版本，不能据此概括其GPU或大网络性能；DeepDendrite按照此前决定只放Related works，不纳入主表速度排名。</p>
<p>Neuron_Reduce的0 Hz频率与自动分段就是作者README示例设置，不应因回归测试使用38 Hz就认定0 Hz选错。原作者Hay测试在NEURON7.8.2的已保存轨迹最大复现差约1.34×10⁻⁷ mV。当前小鼠模型有NEURON9兼容性补丁，并显式恢复原本的ena=50、ek=−85；两类证据和适配记录应分开披露。</p>
<p>目前timing是预处理后完整计算：学习模型包含其输出计算和同步，求解器包含原生电压/事件记录；模型构建、输入整理、传输、warmup等成本单列。各方法的记录网格及引擎交接工作仍有差异，所以不能宣称每个内部操作相同。CPU单细胞、有限突触规模与预处理后边界均须明确，不外推到批量/网络场景。</p>

<h2>7. 训练预算与公开数据复现的边界</h2>
{traintable}<p>TCN与ELM的公开检查点均在12个测试文件、作者原始筛选和指标下通过复现：TCN RMSE={public['tcn']['metrics']['rmse_mv']:.6f} mV、AUC={public['tcn']['metrics']['spike_auc']:.6f}；ELM RMSE={public['elm']['metrics']['rmse_mv']:.6f} mV、AUC={public['elm']['metrics']['spike_auc']:.6f}。这是公开权重的推理/指标复现，不等于在公开训练集上从零复现了论文训练。本研究换成小鼠后，大家使用共同117条6秒拟合轨迹和32条验证轨迹；不能用换任务后的主测试值直接判定原方法是否被正确训练。</p>

<p>当前对照均是单个固定种子的已完成训练，原文部分表格报告三次训练的均值/标准差。按测试轨迹bootstrap不能替代不同训练种子的波动。现有预算证明没有少跑规定更新数，但不能据此保证超参数对新任务必然最优。遵照用户要求，本轮不额外搜索他人超参数。</p>

<h2>8. 建议后续的最小处理顺序</h2>
<ol><li>ELM：固定“公开Python代码默认”的来源，修正窗口起点与批内轨迹无放回抽样，与形态映射一起做清楚标记的对照，保留原配置和旧结果。</li><li>TCN：在全部379条等价性通过后采用整段前向计时，再核对主要速度倍率；不需要为这一项重新训练。</li><li>LSTM：明确为LSTM-50及自定初始化；若要声称严格复现，再补公开数据上的训练检验，不凭猜测更改成Equinox默认就宣称问题解决。</li><li>正文后续统一写明共同数据、事件定义、验证校准和固定容量。当前文章与主图均未修改。</li></ol>
<h2>证据与固定来源</h2>
<p><a href="evidence/Spieler_et_al_ICLR2024.pdf">Spieler et al. ICLR 2024原文</a>（附录A、B与表S10）；<a href="https://github.com/AaronSpieler/elmneuron/tree/52e68a6d39523ac6613a586699b116e8e606dda3">ELM固定代码版本</a>；<a href="https://github.com/SelfishGene/neuron_as_deep_net/tree/074c4666300a8ad246601dab179a97a6942f0f29">TCN固定代码版本</a>；<a href="https://github.com/orena1/neuron_reduce/tree/94b1850607f9c62a205ad8fb695f2fd91d84d87d">Neuron_Reduce固定代码版本</a>；<a href="https://github.com/keras-team/keras/blob/2.2.4/keras/optimizers.py">Keras NAdam原公式</a>。</p>
<p class="muted">JSON证据：<a href="evidence/numerical_probes.json">数值实现检查</a>、<a href="evidence/sampling_and_label_audit.json">抽样与标签</a>、<a href="evidence/data_and_calibration_audit.json">输入与电压校准</a>、<a href="evidence/solver_equivalence_audit.json">求解器等价性</a>。训练配置、作者源代码和公开检查点复现证明保存在source_snapshot目录；远程新增核查原始数据在 <code>/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/baseline_fidelity_audit_20261016</code>。</p>
</main></html>'''
 (O/'DBNN_对比方法实现核查_20261016.html').write_text(content)
 print(O/'DBNN_对比方法实现核查_20261016.html')
if __name__=='__main__':main()
