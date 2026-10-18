"""Check manifest isolation, paired inputs, causal reset, and trainable diagonals."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import torch
from common8 import read,write,sha,match_events,metric,mask,reset_function
from train_full_matrix import VariableDrive,random_state,fft_length


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args()
    work=a.work.resolve();protocol=read(work/'locked_protocol.json');pairs={};counts=[]
    for j in protocol['jobs']:
        assert sha(j['training_manifest'])==j['training_manifest_sha256']
        assert sha(j['test_manifest'])==j['test_manifest_sha256']
        train=read(j['training_manifest'])['cases'];test=read(j['test_manifest'])['cases']
        assert len(train)==96 and len(test)==182
        fit={r['seed'] for r in train if r['split']=='fit'}
        val={r['seed'] for r in train if r['split']=='validation'}
        assert len(fit)==72 and len(val)==24 and fit.isdisjoint(val)
        assert set(r['seed'] for r in test).isdisjoint(fit|val)
        fingerprints=[]
        for r in test:
            f=np.load(r['input_file'])
            assert hashlib.sha256(f['x'].tobytes()).hexdigest()==r['input_sha256']
            assert hashlib.sha256(f['ids'].tobytes()).hexdigest()==r['ids_sha256']
            assert np.all(f['ids'][:r['A']//2]<250) and np.all(f['ids'][r['A']//2:]>=250)
            assert f['x'].shape==(r['A'],r['duration_ms']+1)
            fingerprints.append((r['input_sha256'],r['ids_sha256'],r['seed']))
        if j['cell'] in pairs:assert pairs[j['cell']]==fingerprints
        else:pairs[j['cell']]=fingerprints
        counts.append(dict(cell=j['cell'],profile=j['profile'],fit=72,validation=24,test=182))
    reset=reset_function(work);rng=np.random.default_rng(3451)
    for n in [1,37,500,6001]:
        u=rng.normal(-60,8,n);v,e=reset(u,-58,4,40)
        r=0.;expected=[];vv=[]
        for i,value in enumerate(u):
            pred=value+r;vv.append(pred);fire=pred>=-58
            if fire:expected.append(float(i))
            r=np.exp(-1/40)*r-(4 if fire else 0)
        assert np.max(np.abs(v-np.array(vv)))<1e-10
        assert np.array_equal(e,expected)
    assert match_events([0],[5+5e-7])['tp']==1
    assert match_events([0],[5+2e-6])['tp']==0
    # A fully masked trace is retained, with an undefined voltage denominator.
    m=metric(np.zeros(11),np.zeros(11),[5])
    assert m['n']==0 and m['voltage_metric_defined'] is False
    torch.set_num_threads(1);state=random_state(291049)
    model=VariableDrive(state,torch.device('cpu'),1.,True)
    x=torch.zeros(1,500,64);x[0,0,1]=1;x[0,250,3]=-1
    y=model(torch.fft.rfft(x,n=fft_length(64)),64)
    y.square().mean().backward()
    assert model.weight.grad[0,0]!=0 and model.weight.grad[250,250]!=0
    assert np.array_equal(model.offdiag.numpy(),np.ones((500,500)))
    write(work/'contract_verification.json',dict(passed=True,conditions=counts,
        paired_ampa_nmda_inputs=True,source_seed_isolation=True,all_test_hashes_valid=True,
        reset_matches_independent_python=True,inclusive_event_boundary=True,
        fully_masked_inputs_not_removed=True,diagonal_gradients_nonzero=True,
        locked_protocol_sha256=sha(work/'locked_protocol.json')))
    print('CONTRACT_VERIFIED',len(counts),flush=True)


if __name__=='__main__':main()
