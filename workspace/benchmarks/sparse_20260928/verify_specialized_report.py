"""Structural and evidence checks for the updated self-contained report."""
import base64
import hashlib
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.anchors = []
        self.images = []
        self.layers = []
        self.math = 0
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'a' and a.get('href', '').startswith('#'):
            self.anchors.append(a['href'][1:])
        if tag == 'img':
            assert a.get('alt'), 'missing image alternative text'
            assert a['src'].startswith('data:image/png;base64,')
            image = base64.b64decode(a['src'].split(',', 1)[1], validate=True)
            assert image[:8] == b'\x89PNG\r\n\x1a\n'
            self.images.append(hashlib.sha256(image).hexdigest())
        if 'data-layer' in a:
            self.layers.append(a['data-layer'])
        if tag == 'math':
            self.math += 1


def main():
    old_path, new_path, result_path = map(Path, sys.argv[1:])
    old = old_path.read_text()
    new = new_path.read_text()
    before, after = Document(old), Document(new)
    assert not [k for k, n in Counter(after.ids).items() if n != 1]
    assert all(a in after.ids for a in after.anchors)
    assert len(after.layers) == len(set(after.layers)) == 5
    assert all('layer-detail-' + layer in after.ids for layer in after.layers)
    assert before.math == after.math == 18

    def formula(text):
        return text.split('<section id="design">', 1)[1].split('<div class="layer-guide"', 1)[0]

    assert formula(old) == formula(new), 'approved model formulas changed'

    def payload(text):
        raw = re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.S)
        return json.loads(raw.group(1))

    old_data, new_data = payload(old), payload(new)
    assert all(new_data[k] == v for k, v in old_data.items()), 'legacy evidence changed'
    assert re.findall(r'<script>(.*?)</script>', old, re.S) == re.findall(r'<script>(.*?)</script>', new, re.S)
    q = new_data['quadratic_specialized']
    f = q['confirmation']
    assert q['first']['finished'] and f['finished'] and q['source_replacement']['finished']
    assert len(q['first']['cases']) == len(f['cases']) == 9
    assert all(c['policy']['enabled'] for c in f['cases'])
    assert f['dispatch_checks_passed']
    for c in f['cases']:
        assert c['error_vs_original']['max_abs_mv'] < 1e-3
        assert c['error_vs_original']['spike_mismatch_bins'] == 0
        for scope in ['module_device', 'forward_device', 'forward_graph_wall']:
            assert all(v['n'] == 60 for v in c[scope].values())
        assert all(v['n'] == 30 for v in c['original_paired_wall'].values())
    assert all(c['samples'] == 10 and c['error']['spike_mismatch_bins'] == 0 for c in f['heldout_forward_checks'])
    for c in q['source_replacement']['cases']:
        assert all(v['n'] == 30 for v in c['wall'].values())
        assert all(v['spike_mismatch_bins'] == 0 for v in c['errors'].values())
    assert len(after.images) == len(before.images) + 2
    result = dict(
        passed=True, report=str(new_path), bytes=new_path.stat().st_size,
        sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
        embedded_pngs=len(after.images), mathml_formulas=after.math,
        approved_formulas_unchanged=True, legacy_json_unchanged=True,
        interaction_scripts_unchanged=True, anchors_valid=True,
        unique_ids=True, five_layer_controls=after.layers,
        new_shapes=9, second_pass_samples=60, original_control_samples=30,
        visual_check='The two exported new PNG plots were inspected with view_image.',
        browser_check='No browser navigation attempted; previous file navigation security rejection respected.',
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
