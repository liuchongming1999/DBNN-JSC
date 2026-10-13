"""AMPA/NMDA comparison on matched inputs, after both locked 31-case evaluations."""
import json
from pathlib import Path
import numpy as np
from common import metric,aggregate

def paired_conditions(work,data,model):
    base=Path(work)/'results'/model
    summaries={};rows={};records={};conditions={}
    for profile in ['ampa','ampa_nmda']:
        gpu=base/profile/'gpu/complete.json'
        manifest=Path(data)/model/profile/'test/manifest.json'
        condition=manifest.parent/'reference_condition.json'
        if not gpu.exists() or not manifest.exists() or not condition.exists():return None
        conditions[profile]=json.loads(condition.read_text())
        summaries[profile]=json.loads(gpu.read_text())
        if not summaries[profile].get('finished') or '1.0' not in summaries[profile]['summary']:return None
        rows[profile]={z['case_id']:z for z in json.loads(manifest.read_text())['cases']}
        records[profile]={z['case_id']:z for z in summaries[profile]['records']}
    for key in ['holding_current_na','weight_us','inhibition_ratio','rest_mv','reference_dt_ms']:
        assert conditions['ampa']['operating_point'][key]==conditions['ampa_nmda']['operating_point'][key],('Different physical operating point',key)
    ids=sorted(set(rows['ampa'])&set(rows['ampa_nmda']))
    cases=[];metrics={p:[] for p in rows}
    for cid in ids:
        a,b=rows['ampa'][cid],rows['ampa_nmda'][cid]
        assert a['input_sha256']==b['input_sha256'] and a['ids_sha256']==b['ids_sha256'],('Unmatched input',cid)
        assert a['duration_ms']==b['duration_ms']
        refs={p:np.load(rows[p][cid]['reference_file']) for p in rows}
        # Both conditions use this same time mask; different reference spike
        # times must not give one condition a different scoring window.
        union=np.unique(np.r_[refs['ampa']['spike_times'],refs['ampa_nmda']['spike_times']])
        t=np.arange(round(a['duration_ms'])+1)
        result={'case_id':cid,'duration_ms':a['duration_ms'],'input_sha256':a['input_sha256'],'conditions':{}}
        for profile in rows:
            rv=np.interp(t,refs[profile]['time_ms'],refs[profile]['voltage'])
            pv=np.load(base/profile/'gpu/dt1.0ms'/(cid+'.npz'))['voltage']
            m=metric(rv,pv,union,1.)
            metrics[profile].append(m)
            result['conditions'][profile]={'metrics':m,'events':records[profile][cid]['variants']['1.0']['events']}
        cases.append(result)
    return {'cases':len(cases),'input_ids':ids,'dt_ms':1.,'summary':{p:aggregate(m) for p,m in metrics.items()} if cases else {},
            'records':cases,'scope':'Intersection of the independently reference-selected AMPA and AMPA+NMDA 31-window cohorts. Identical event raster, site IDs and duration. Both voltage errors exclude the union of the two reference spike +/-10ms windows. Each condition has its own independently validation-selected DBNN.'}
