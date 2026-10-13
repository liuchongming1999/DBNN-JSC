"""Verify paired inputs, numerical claims and preservation of the approved HTML."""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile
import numpy as np
from lxml import html
H=Path(__file__).resolve().parent;R=H/'results'
def load(p):return json.loads(p.read_text())
def values(name):return np.load(R/(name+'.npz'),allow_pickle=True)['voltage'][0]
g=load(H/'dbnn_gpu.json');a=load(H/'analysis.json')
assert g.get('finished') and len(g['cases'])==11
assert len(a['main'])==6
checks=[]
for T in [1001,6001]:
    for A in [32,128,500]:
        gg=next(r for r in g['cases'] if (r['A'],r['T'],r['B'],r['rate'])==(A,T,1,'sampled'))
        ref=values(f'main_A{A}_T{T}_neuron')
        for engine in ['neuron','core','multisplit_checked','cvode']:
            stem=f'main_A{A}_T{T}_{engine}';r=load(R/(stem+'.json'))
            assert r['finished'] and r['input_sha256']==gg['input_sha256']
            assert r['config']['dt']==.1
            v=values(stem);diff=float(abs(v-ref).max())
            if engine in ['core','multisplit_checked']:assert diff<1e-7,(stem,diff)
            checks.append(dict(case=stem,max_difference_mv=diff))
for row in g['cases']:
    for name,e in row['accuracy_vs_original'].items():
        if name.startswith('optimized') or name=='precision_strict':assert e['finite'] and e['max_abs_mv']<1e-3 and e['reset_bin_difference']==0,(row,name,e)
for n in [1,2,4,8]:
    c=load(H/f'dbnn_cpu{n}.json');assert c.get('finished')
    for row in c['cases']:
        e=row['accuracy_vs_original']['optimized_CPU'];assert e['finite'] and e['max_abs_mv']<1e-3 and e['reset_bin_difference']==0
old=values('original_mechanisms');fresh=values('pilot_neuron_n1');assert abs(old-fresh).max()<1e-9
for case in a['throughput']['cases']:assert case['voltage_max_diff_vs_serial']<1e-9
for e in a['main']:
    assert abs(e['speed_ratio_e2e']-e['neuron_best_ms']/e['gpu_e2e_ms'])<1e-9

before=(H/'report_before_neuron_focus.html').read_text();after=(H/'updated_report.html').read_text()
raw=lambda s:re.search(r'<script type="application/json" id="data">(.*?)</script>',s,re.S).group(1)
assert raw(before)==raw(after)
math=lambda s:re.findall(r'<math\b.*?</math>',s,re.S)
assert math(before)==math(after) and len(math(after))==18
images=lambda s:re.findall(r'src="(data:image/png;base64,[^"]+)"',s)
assert set(images(before)).issubset(set(images(after)))
dom=html.fromstring(after);ids=dom.xpath('//@id');assert len(ids)==len(set(ids))
missing=[h for h in dom.xpath('//a/@href') if h.startswith('#') and h[1:] not in set(ids)];assert not missing,missing
assert dom.xpath('//main/section/@id')[:7]==['conclusion','design','neuron-main','neuron-accuracy','neuron-throughput','neuron-protocol','next']
assert json.loads(dom.xpath('string(//script[@id="neuron-data"])'))['main']==a['main']
js=dom.xpath('//script[not(@type) or @type="text/javascript"]/text()')
node='/Users/lcmsjtu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
with tempfile.TemporaryDirectory() as td:
    for i,s in enumerate(js):
        p=Path(td)/f'script{i}.js';p.write_text(s);subprocess.run([node,'--check',str(p)],check=True,capture_output=True)
report=dict(verified=True,paired_cases=checks,old_raw_data_unchanged=True,mathml_preserved=18,previous_images_preserved=len(images(before)),
            total_embedded_images=len(images(after)),javascript_blocks_checked=len(js),ids=len(ids),broken_anchors=missing,
            output_sha256=hashlib.sha256(after.encode()).hexdigest(),bytes=len(after.encode()),
            browser_runtime='not performed; prior browser navigation approval rejection respected; structural checks and scientific PNG inspection used')
(H/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
