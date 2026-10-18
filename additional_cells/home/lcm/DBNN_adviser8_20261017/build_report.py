"""Build an independent report; never rewrite manuscript results or main figures."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import html
from zoneinfo import ZoneInfo
from common8 import read,write,sha


def val(x,digits=3):return '—' if x is None else f'{x:.{digits}f}'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args()
    work=a.work.resolve();protocol=read(work/'locked_protocol.json');jobs=protocol['jobs']
    rows=[];evidence=[]
    trained=sum((Path(j['base'])/'training/complete.json').exists() for j in jobs)
    references=sum(read(Path(j['base'])/'test/status.json')['completed'] if (Path(j['base'])/'test/status.json').exists() else 0 for j in jobs)
    for j in jobs:
        base=Path(j['base']);p=base/'evaluation/complete.json'
        if not p.exists():
            progress=base/'test/status.json'
            count=read(progress)['completed'] if progress.exists() else 0
            trained_label='训练已完成' if (base/'training/complete.json').exists() else '训练中'
            rows.append(f'<tr><td>{j["cell"]}</td><td>{j["profile"]}</td><td colspan="8">{trained_label}；参考解 {count}/182；完整评估待完成</td></tr>');continue
        d=read(p);train=read(base/'training/complete.json')
        assert d['all_inputs_retained'] and len(d['records'])==182
        assert d['checkpoint_sha256']==train['checkpoint_sha256']
        assert d['test_manifest_sha256']==j['test_manifest_sha256']
        assert sha(job_cp:=base/'training/best_checkpoint.pt')==d['checkpoint_sha256']
        manifest_rows={r['case_id']:r for r in read(j['test_manifest'])['cases']}
        assert {r['case_id'] for r in d['records']}==set(manifest_rows)
        assert all(r['input_sha256']==manifest_rows[r['case_id']]['input_sha256'] for r in d['records'])
        assert all(r['numerical']['events_equal'] and r['numerical']['max_abs_mv']<.001 for r in d['records'])
        e=d['events'];ci=d['bootstrap']['f1_ci95']
        counts=f'{e["tp"]}/{e["fp"]}/{e["fn"]}'
        fallbacks=sum(r['backend']=='prepared_fft_fp64_sequential_reset' for r in d['records'])
        rows.append(f'<tr><td>{j["cell"]}</td><td>{j["profile"]}</td><td>72 / 24 / 182</td>'
            f'<td>{j["validation_reference_events"]}</td><td>{e["tp"]+e["fn"]}</td>'
            f'<td>{val(d["voltage"]["pooled_rmse_mv"],4)}</td><td>{val(e["f1"])}</td>'
            f'<td>{counts}</td><td>{"—" if ci is None else val(ci[0])+"–"+val(ci[1])}</td>'
            f'<td>{val(d["prepared_wall_ms"],2)}<br><small>数值回退 {fallbacks}/182</small></td></tr>')
        evidence.append(dict(cell=j['cell'],profile=j['profile'],evaluation_sha256=sha(p),
            checkpoint_sha256=d['checkpoint_sha256'],test_manifest_sha256=j['test_manifest_sha256'],
            inputs=182,all_diagonal_trainable=train['diagonal_trainable'],
            diagonal_max_abs=train['diagonal_max_abs'],voltage=d['voltage'],events=e['f1']))
    frozen='冻结主线诊断尚未完成。'
    f=work/'frozen_primary/complete.json'
    if f.exists():
        d=read(f);b=d['bootstrap'];sections=[]
        for split,r in d['comparisons'].items():
            sections.append(f'<tr><td>{split}</td><td>{r["inputs"]}</td><td>{r["reference_spikes"]}</td>'
                f'<td>{val(r["reference_event_reset_voltage"]["pooled_rmse_mv"],4)}</td>'
                f'<td>{val(r["autonomous_reset_voltage"]["pooled_rmse_mv"],4)}</td>'
                f'<td>{val(r["events"]["f1"])}</td></tr>')
        frequencies=''.join(f'<tr><td>{r["v_th"]}</td><td>{r["reset_amp"]}</td><td>{r["tau_reset"]}</td><td>{val(100*r["fraction"],2)}%</td></tr>' for r in b['reset_frequencies'])
        frozen=f'''<p>主线全部权重及reset固定；验证集32条、23个参考尖峰。以完整轨迹为单位重采样{b['replicates']}次，
        仅反事实比较同一冻结电压模型上的reset候选，不将所得参数装回模型。原reset被选中的比例为{val(100*b['original_reset_fraction'],1)}%。</p>
        <table><tr><th>集合</th><th>输入数</th><th>参考尖峰</th><th>参考事件reset RMSE / mV</th><th>自主reset RMSE / mV</th><th>自主事件F1</th></tr>{''.join(sections)}</table>
        <p>参考事件reset使用真实事件时刻，是用于分解电压误差的诊断；它不构成可部署方法，也不将强制事件的一致性当作F1。两种电压误差与reset偏差之间有交叉项，RMSE差值不能直接解释为独立误差占比。</p><p>下表是原候选范围内的敏感性：被选中的幅度几乎均为4mV，阈值变化集中在原值附近；这不证明网格外的最优性，也没有据此更改主线参数。</p><table><tr><th>阈值/mV</th><th>reset幅度/mV</th><th>恢复时间/ms</th><th>重采样选中比例</th></tr>{frequencies}</table>'''
    snapshots=work/'frozen_primary/snapshot_stability.json'
    if snapshots.exists():
        c=read(snapshots)
        current_fraction=100*next(r['selection_fraction'] for r in c['candidates'] if r['name']=='refine_ema')
        cr=''.join(f'<tr><td>{html.escape(r["name"])}</td><td>{val(r["pooled_rmse_mv"],4)}</td><td>{val(r["events"]["f1"])}</td><td>{val(100*r["selection_fraction"],2)}%</td></tr>' for r in c['candidates'])
        frozen+=f'<h3>候选checkpoint的选择敏感性</h3><p>六个历史候选均保持各自权重及原reset不变，在相同32条验证轨迹上重采样10000次，仅比较排序。当前主线refine_ema的选中比例为{val(current_fraction,2)}%；部分joint和voltage候选是完全相同的物理参数，平分数时按原列表顺序计入首个，不能把重复候选的0%解读为性能较差。</p><table><tr><th>冻结候选</th><th>完整验证RMSE/mV</th><th>完整验证F1</th><th>重采样选中比例</th></tr>{cr}</table><p>这与上面的reset敏感性是两项不同诊断：前者固定同一最终权重比较reset，后者比较六个已有的权重/reset组合。没有用379条测试结果重新选择候选。</p>'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DBNN训练校准与跨细胞复核</title>
    <style>body{{font:16px/1.8 system-ui,sans-serif;color:#263c4b;max-width:1400px;margin:45px auto;padding:0 28px}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px 10px;border-bottom:1px solid #dce4e9;text-align:left}}th{{background:#edf4f7}}code{{overflow-wrap:anywhere}}.note{{background:#eff8f6;padding:16px 22px}}</style>
    <h1>DBNN训练校准与跨细胞复核</h1><p>生成于 {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}；训练已完成 {trained} / 12；新参考解 {references} / 2184；完整评估 {len(evidence)} / 12。</p>
    <p class="note">这是一份独立实验报告。主线小鼠L5PC checkpoint、阈值、训练、379条主统计及论文跨细胞表未因本实验被替换。新结果交由作者审阅后决定论文修改。</p>
    <h2>1．已经补清的主线训练流程</h2><p>初始reset固定拟合 → 验证集校准并选择第一阶段权重 → 按校准幅度和恢复时间重建参考事件reset及电压目标 → 继续400轮拟合 → 从六个验证候选选择并冻结模型。主线最终reset与第二阶段目标使用的reset一致。正文3.2和训练附录已据此补充说明。</p>
    <h2>2．冻结主线模型的诊断</h2>{frozen}
    <h2>3．跨细胞实验预先固定的协议</h2><p>六种细胞，各含AMPA及AMPA+NMDA，共12种条件；每种仍使用500个候选位点。每种模型均从随机初始化训练全矩阵DBNN，不从旧零对角checkpoint继续训练。每个条件保留原始72条训练和24条验证的完整6秒源轨迹，不按参考发放数筛选、不截取低发放前缀。预算固定为2000+400轮，500ms随机窗口，前150ms warm-up，仅验证集校准与选择。</p>
    <p>新测试每种182条：1/2/4/8/16s × 64/128/256/500个位点 × 10/25/50Hz × 平稳、兴奋性门控、速率切换三类输入，再加两个零输入对照；随机、分层、近端、远端位置按预先定义的循环规则均衡分配。AMPA和NMDA使用配对的新随机输入；种子与全部输入哈希在训练和参考解运行前锁定。所有发放数量及burst结果均保留。若±10ms电压mask覆盖整条轨迹，该条电压指标记为不可定义，事件仍计入；不把它当作零电压误差或删除该输入。参考解沿用每种细胞已固定的积分步长及物理工作点，DBNN采样1ms。</p>
    <p>电压RMSE排除参考尖峰±10ms；事件采用±5ms、边界容差10⁻⁶ms的一对一匹配；区间估计按完整输入轨迹进行10000次bootstrap。GPU时间为预处理后的完整推理，含同步，3次中位数；共享服务器上的测量不与旧31窗的总时间直接相除。参数始终保持FP32；快速实现需与FFT电压drive及顺序FP64 reset数值核验一致。罕见不一致的输入回退到FFT与FP64 GPU reset，保留原输入和模型，实际计时包含该完整回退实现。</p>
    <h2>4．完整新测试结果</h2><table><tr><th>细胞</th><th>突触</th><th>训练/验证/测试</th><th>验证尖峰</th><th>测试尖峰</th><th>RMSE/mV</th><th>F1</th><th>TP/FP/FN</th><th>F1 95%区间</th><th>182条GPU总时间/ms</th></tr>{''.join(rows)}</table>
    <p>新测试与旧31窗不同，训练源也不再按发放筛选，因此旧表与本表的差异不能单独归因于允许非零对角。各条件的完整电压、drive、事件、逐输入指标、分组和计时重复均保存。</p>
    <h2>5．可核验记录</h2><p>锁定协议 SHA-256：<code>{sha(work/'locked_protocol.json')}</code>；原始路径 <code>{html.escape(str(work))}</code>。</p></html>'''
    (work/'DBNN_训练校准与跨细胞更新_20261017.html').write_text(page)
    write(work/'verification.json',dict(complete=len(evidence)==12 and f.exists(),
        completed_conditions=len(evidence),locked_protocol_sha256=sha(work/'locked_protocol.json'),
        evidence=evidence,primary_and_manuscript_crosscell_results_unchanged=True))


if __name__=='__main__':main()
