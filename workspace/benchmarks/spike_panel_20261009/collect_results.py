"""One shared 1-ms comparison, fixed inputs, masks, matches and case CIs."""
import argparse
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
from common_panel import read,write,sha
from score_records import grouped


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();manifest=read(work/'confirmatory/manifest.json');rows=manifest['cases']
    sys.path[:0]=[str(root/'benchmarks/accuracy_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    result=dict(created=datetime.now().isoformat(),input_cases=len(rows),reference_dt_ms=.1,all_comparison_models_dt_ms=1.,
                manifest_sha256=sha(work/'confirmatory/manifest.json'),reference_parity=read(work/'reference_parity.json'),
                primary_event_tolerance_ms=5,voltage_mask_ms=10,models={},statistical_scope='Fixed mouse V1 L5PC; variation over newly generated independent inputs',
                timing_scope='This campaign evaluates accuracy/events; solver runs are not a new publication timing benchmark')
    dbnn=read(work/'evaluation.json');result['models']['dbnn']=dbnn
    for method in ['tcn_common150','elm_common150','lstm_common150','tcn_author_default','elm_author_default']:
        d=read(work/'methods'/method/'complete.json');assert d['finished'];result['models'][method]=d
    for method in ['neuron1','neuron8','core1','reduce1']:
        records=[];validations=[]
        for row in rows:
            cid=row['case_id']
            if method=='neuron1':pjson=work/'methods/neuron1/raw'/f'neuron_dt1_{cid}.json';pnp=pjson.with_suffix('.npz')
            elif method in ['neuron8','core1']:
                matches=list((work/'methods'/method).glob(f'shard*/nt{8 if method=="neuron8" else 1}_dt1_{cid}.json'));assert len(matches)==1,(method,cid,len(matches))
                pjson=matches[0];pnp=pjson.with_suffix('.npz')
            else:
                matches=list((work/'methods/reduce1').glob(f'shard*/{cid}.json'));assert len(matches)==1,(method,cid,len(matches))
                pjson=matches[0];pnp=pjson.with_suffix('.npz')
            raw=read(pjson);assert raw.get('finished') and raw['input_sha256']==row['input_sha256'];assert raw['dt_ms']==1.
            if method in ['neuron8','core1']:
                assert raw['validation']['passed'];validations.append(raw['validation']['voltage_max_abs_mv'])
            pred=np.load(pnp);v=pred['voltage'];events=pred['spike_times_ms'] if 'spike_times_ms' in pred else pred['spike_times']
            ref=np.load(work/'confirmatory/references'/f'{cid}_reference.npz');rr=read(work/'confirmatory/references'/f'{cid}_reference.json')
            assert len(v)==len(ref['voltage'])==row['duration_ms']+1
            mm=metric(ref['voltage'],v,ref['spike_times'],1.);ee=match_events(ref['spike_times'],events)
            records.append(dict(**row,reference_spikes=rr['spike_count'],minimum_isi_ms=rr['minimum_isi_ms'],low_density=rr['low_density'],
                                voltage=mm,events=ee,reference_events_ms=ref['spike_times'].tolist(),predicted_events_ms=events.tolist(),prediction_file=str(pnp)))
        d=dict(finished=True,model=method,sample_interval_ms=1.,records=records,**grouped(records,aggregate,event_aggregate))
        if validations:d['single_thread_same_dt_parity']=dict(cases=len(validations),max_voltage_gap_mv=max(validations),all_event_times_match=True)
        write(work/'methods'/method/'scored.json',d);result['models'][method]=d
    hashes=[r['input_sha256'] for r in rows]
    for method,d in result['models'].items():
        assert len(d['records'])==len(rows)
        assert [r['input_sha256'] for r in d['records']]==hashes
        assert d['low_density']['reference_events']==dbnn['low_density']['reference_events']
        sensitivities={}
        low=[r for r in d['records'] if r['low_density']]
        for tol in [1,2,5,10,20]:
            scores=[match_events(np.array(r['reference_events_ms']),np.array(r['predicted_events_ms']),tol=tol) for r in low]
            sensitivities[str(tol)]=event_aggregate(scores)
        d['tolerance_sensitivity_ms']=sensitivities
        print(method,d['low_density']['pooled_rmse_mv'],d['low_density']['events']['f1'],d['low_density']['bootstrap_95ci']['f1'],flush=True)
    result['finished']=True;write(work/'comparison.json',result)


if __name__=='__main__':main()
