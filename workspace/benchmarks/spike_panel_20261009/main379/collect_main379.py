"""One auditable accuracy/timing payload for the new manuscript mainline."""
import argparse,json,sys
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();panel=a.work.resolve();work=panel/'main379'
    sys.path[:0]=[str(panel),str(root/'benchmarks/accuracy_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    from score_records import grouped
    old=read(panel/'comparison.json');rows=read(work/'manifest.json')['cases'];assert len(rows)==379
    low_ids={r['case_id'] for r in old['models']['dbnn']['records'] if r['low_density']};assert low_ids=={r['case_id'] for r in rows}
    result=dict(finished=True,cases=379,reference_events=637,total_duration_s=sum(r['duration_s'] for r in rows),
                reference_dt_ms=.1,output_interval_ms=1.,learned_model_interval_ms=1.,solver_dt_ms=.5,models={},
                protocol=read(work/'solver_launch.json'),examples=read(work/'selected_examples.json'))
    for key,source in [('gpu','dbnn'),('cpu1','dbnn'),('cpu8','dbnn'),('tcn','tcn_author_default'),('elm','elm_author_default'),('lstm','lstm_common150')]:
        timing=read(work/key/'complete.json');assert timing['finished'] and len(timing['records'])==379
        rr=[r for r in old['models'][source]['records'] if r['low_density']]
        assert [r['input_sha256'] for r in rr]==[r['input_sha256'] for r in timing['records']]
        assert all(r['validation']['passed'] for r in timing['records'])
        result['models'][key]=dict(accuracy=old['models'][source]['low_density'],groups=old['models'][source]['groups'],
                                  timing=timing,records=rr,interval_ms=1.)
    for key in ['neuron8','core1','reduce']:
        scored=[];timed=[]
        for row in rows:
            cid=row['case_id'];pattern=f'nt{8 if key=="neuron8" else 1}_dt0.5_{cid}.json' if key!='reduce' else f'{cid}.json'
            folder=work/key if key!='reduce' else work/'methods/reduce1'
            files=list(folder.glob(f'shard*/{pattern}'));assert len(files)==1,(key,cid,len(files))
            f=files[0];j=read(f);assert j['finished'] and j['dt_ms']==.5 and j['input_sha256']==row['input_sha256']
            if key!='reduce':assert j['validation']['passed'];t=j['timing']['solver_ms'];tt=dict(median_ms=t['median_ms'],runs_ms=t['runs_ms'])
            else:tt=j['wall']
            pred=np.load(f.with_suffix('.npz'));ref=np.load(panel/'confirmatory/references'/f'{cid}_reference.npz');assert np.isfinite(pred['voltage']).all()
            ev=pred['spike_times'] if 'spike_times' in pred else pred['spike_times_ms'];mm=metric(ref['voltage'],pred['voltage'],ref['spike_times'],1.);ee=match_events(ref['spike_times'],ev)
            rr=read(panel/'confirmatory/references'/f'{cid}_reference.json');assert rr['low_density']
            scored.append(dict(**row,reference_spikes=rr['spike_count'],minimum_isi_ms=rr['minimum_isi_ms'],low_density=True,voltage=mm,events=ee,
                               predicted_events_ms=ev.tolist(),reference_events_ms=ref['spike_times'].tolist()))
            timed.append(dict(case_id=cid,duration_s=row['duration_s'],A=row['A'],input_sha256=row['input_sha256'],wall=tt,
                              validation=j.get('validation'),raw_output=str(f)))
        groups=grouped(scored,aggregate,event_aggregate);assert groups['low_density']['reference_events']==637
        result['models'][key]=dict(accuracy=groups['low_density'],groups=groups['groups'],records=scored,interval_ms=.5,
                                  timing=dict(finished=True,records=timed,wall_ms=sum(r['wall']['median_ms'] for r in timed),
                                              runtime=dict(repeats=3,scope='Prepared psolve including native voltage and event recording; excludes initialization, event queue setup and host-grid interpolation. B=1, shared server, affinity per shard.')))
    durations={}
    for T in [1,2,4,8,16]:
        durations[str(T)]={}
        for key,d in result['models'].items():
            vals=[r['wall']['median_ms'] for r in d['timing']['records'] if r['duration_s']==T]
            durations[str(T)][key]=dict(cases=len(vals),median_ms=float(np.median(vals)),q1_ms=float(np.quantile(vals,.25)),q3_ms=float(np.quantile(vals,.75)),sum_ms=float(sum(vals)))
    result['durations']=durations
    (work/'main379_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,d in result['models'].items():print(k,d['accuracy']['pooled_rmse_mv'],d['accuracy']['events']['f1'],d['timing']['wall_ms'],flush=True)


if __name__=='__main__':main()
