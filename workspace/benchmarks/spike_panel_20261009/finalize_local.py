# -*- coding: utf-8 -*-
"""Final local integrity checks and synchronization of the manuscript outline."""
import hashlib
import json
from pathlib import Path
import re


def main():
    root=Path('/Users/lcmsjtu/Desktop/博士毕业/DBNN');data=root/'DBNN_spike_panel_data_20261009'
    record=json.loads((data/'completion_manifest.json').read_text())
    assert hashlib.sha256((data/'comparison.json').read_bytes()).hexdigest()==record['comparison_sha256']
    before=Path(__file__).resolve().parent/'results/main_report_before_independent_spike_update.html'
    original=before.read_text();updated=(root/'DBNN_L5PC_surrogate_report_20260928.html').read_text()
    images=lambda s:re.findall(r'<img\b[^>]*\bsrc="([^"]+)"',s)
    old_images,new_images=images(original),images(updated);assert new_images[:len(old_images)]==old_images
    ids=re.findall(r'\bid="([^"]+)"',updated);assert len(ids)==len(set(ids))
    print('local comparison hash verified; original figure sources preserved; section IDs unique')
    p=root/'JSC_投稿准备/DBNN_JSC_论文大纲_讨论版_20261009.html';s=p.read_text()
    start=s.index('<h3>5.3 Spike prediction and timing</h3>');end=s.index('<h3>5.4',start)
    replacement='''<h3>5.3 Spike prediction and timing</h3><p>事件统计以新生成的独立输入为主：637条输入共2087个参考尖峰；按照每窗0–3个尖峰、最小间隔50ms的范围，主统计使用379条输入、637个尖峰。原有31窗保留为覆盖性和计时面板，其10次尖峰计数包含共享前缀。</p><p>所有比较模型为1ms，参考NEURON为0.1ms。模型和事件阈值冻结，采用±5ms一对一匹配，并对完整独立轨迹进行5000次bootstrap。全矩阵DBNN的F1为0.530（95%CI 0.491–0.570）；Branch-ELM作者默认流程0.429、LSTM 0.271、TCN作者默认流程0.213。短时E/I比例变化输入的DBNN F1为0.879，连续Poisson为0.435；自然与结构化输入分别陈述。</p><p>正文报告precision／recall／F1、TP／FP／FN和时刻偏差。完整输入集、已有共同150ms训练缓冲对照及匹配容差敏感性放附录。<a href="../DBNN_spike_panel_data_20261009/DBNN_独立尖峰评估_1ms_20261009.html">独立尖峰评估的完整结果</a></p>
'''
    p.write_text(s[:start]+replacement+s[end:])
    print('outline event section synchronized')


if __name__=='__main__':main()
