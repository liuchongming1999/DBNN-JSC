"""Freeze measured precision profiles after two independent timing passes."""
import json
from pathlib import Path
from experiment import HERE,stamp


def main():
    a=json.loads((HERE/'tf32_evaluation.json').read_text())
    b=json.loads((HERE/'tf32_repeat.json').read_text())
    assert a.get('finished') and b.get('finished')
    aa={c['index']:c for c in a['cases']};bb={c['index']:c for c in b['cases']}
    assert len(aa)==len(bb)==13
    policy=dict(created=stamp(),hardware='RTX 5090 / SM120',exclusive_resources=False,
        mode='frozen-checkpoint inference, zero initial state',
        validation_scope='10 held-out sampled traces per shape in both passes; precision gates are empirical, not universal error bounds.',
        ieee_fallback=json.loads((HERE/'quad_backend_policy.json').read_text()),profiles={})
    for profile,limit in [('strict',1e-3),('approximate_0p01mv',1e-2)]:
        pp=dict(max_abs_mv=limit,max_reset_mismatch_bins=0,explicit_opt_in=profile!='strict',shapes={})
        for i,c in aa.items():
            r=bb[i];eligible=[]
            for name in c['specs']:
                assert c['specs'][name]==r['specs'][name]
                er1=c['heldout_accuracy'][name];er2=r['heldout_accuracy'][name]
                if not all(er['finite'] and er['max_abs_mv']<limit and er['spike_mismatch_bins']==0 for er in [er1,er2]):continue
                # The initial pass freezes the ranking; the second pass must
                # confirm a device benefit without a material wall regression.
                if name=='previous_ieee':eligible.append(name);continue
                gain=min(v['forward_device']['previous_ieee']['median_ms']/v['forward_device'][name]['median_ms'] for v in [c,r])
                wall=min(v['forward_graph_wall']['previous_ieee']['median_ms']/v['forward_graph_wall'][name]['median_ms'] for v in [c,r])
                if gain>1.05 and wall>.98:eligible.append(name)
            chosen=min(eligible,key=lambda name:c['forward_device'][name]['median_ms'])
            key=f'{c["A"]}:{c["T"]}:{c["B"]}'
            pp['shapes'][key]=dict(enabled=True,name=chosen,spec=c['specs'][chosen],
                uses_tensor_core=chosen not in ('previous_ieee','tuned_ieee') and chosen!='quad_cublas_tf32',
                cublas_tf32_allowed=chosen=='quad_cublas_tf32',
                max_observed_mv=max(v['heldout_accuracy'][chosen]['max_abs_mv'] for v in [c,r]),
                reset_mismatch_bins=0,
                forward_device_speed_ratio=r['forward_device']['previous_ieee']['median_ms']/r['forward_device'][chosen]['median_ms'],
                forward_wall_speed_ratio=r['forward_graph_wall']['previous_ieee']['median_ms']/r['forward_graph_wall'][chosen]['median_ms'])
        policy['profiles'][profile]=pp
    (HERE/'tf32_backend_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    for profile,p in policy['profiles'].items():
        print(profile,{k:v['name'] for k,v in p['shapes'].items()})


if __name__=='__main__':main()
