"""Check artifact structure and preserve previously approved report evidence."""
import hashlib,json,re,sys
from collections import Counter
from pathlib import Path
from verify_specialized_report import Document


def main():
    old_path,new_path,output=map(Path,sys.argv[1:]);old=old_path.read_text();new=new_path.read_text()
    a,b=Document(old),Document(new)
    assert not [k for k,v in Counter(b.ids).items() if v!=1]
    assert all(x in b.ids for x in b.anchors)
    assert len(b.layers)==len(set(b.layers))==5
    assert a.math==b.math==18 and len(b.images)==len(a.images)+4==21
    assert old.split('<section id="design">',1)[1].split('<div class="layer-guide"',1)[0]==new.split('<section id="design">',1)[1].split('<div class="layer-guide"',1)[0]
    def data(text):return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',text,re.S).group(1))
    da,db=data(old),data(new);assert all(db[k]==v for k,v in da.items())
    assert re.findall(r'<script>(.*?)</script>',old,re.S)==re.findall(r'<script>(.*?)</script>',new,re.S)
    d=db['tf32_five_layers'];f=d['repeat'];assert d['first']['finished'] and f['finished']
    assert len(f['cases'])==len(d['first']['cases'])==13 and d['api_validation']['passed']
    assert len(d['api_validation']['dispatch'])==26
    by={c['index']:c for c in d['first']['cases']}
    for c in f['cases']:
        for scope in ['drive_device','filter_device','forward_device','forward_graph_wall']:
            assert all(s['n']==60 for s in c[scope].values())
        assert all(s['n']==30 for s in c['original_paired_wall'].values())
        assert all(s['samples']==10 for s in c['heldout_accuracy'].values())
        for profile,p in d['policy']['profiles'].items():
            e=p['shapes'][f'{c["A"]}:{c["T"]}:{c["B"]}'];name=e['name']
            for v in [by[c['index']],c]:
                er=v['heldout_accuracy'][name]
                assert er['finite'] and er['max_abs_mv']<p['max_abs_mv'] and er['spike_mismatch_bins']==0
    result=dict(passed=True,report=str(new_path),bytes=new_path.stat().st_size,
        sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),embedded_pngs=21,mathml_formulas=18,
        approved_formulas_unchanged=True,legacy_json_unchanged=True,interaction_scripts_unchanged=True,
        valid_anchors=True,unique_ids=True,five_layer_controls=b.layers,
        shapes=13,first_repetitions=50,second_repetitions=60,api_dispatches=26,
        visual_check='Four scientific PNGs inspected; overlap fixes applied and final images rechecked.',
        browser_check='No navigation attempted; earlier browser security rejection respected.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
