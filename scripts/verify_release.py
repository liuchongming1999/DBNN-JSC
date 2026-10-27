"""Verify frozen files and recompute final paper metrics; no fitting or GPU needed."""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
from release_paths import REPO,resolve,read,digest,ORIGINAL
sys.path.insert(0,str(REPO/'paper/scripts'))
from event_metrics import match_events,aggregate_events

def close(a,b,label,tol=1e-10):
    if a is None or b is None:assert a is b,label
    else:assert np.isclose(a,b,rtol=tol,atol=tol),(label,a,b)

def metric(ref,pred,events):
    ref=np.asarray(ref,float);pred=np.asarray(pred,float)
    assert ref.shape==pred.shape and np.isfinite(ref).all() and np.isfinite(pred).all()
    keep=np.ones(len(ref),bool);t=np.arange(len(ref),dtype=float)
    for s in events:keep&=np.abs(t-float(s))>10.
    e=(pred-ref)[keep];r=ref[keep]
    return {'n':int(keep.sum()),'sse':float(e@e),'reference_ss':float(np.sum((r-r.mean())**2)) if len(r) else 0.,'error_ss':float(np.sum((e-e.mean())**2)) if len(e) else 0.}

def assert_metric(got,want,label):
    assert got['n']==want['n'],label
    for k in ['sse','reference_ss','error_ss']:
        if k in want:close(got[k],want[k],label+'/'+k,1e-9)

def input_check(row):
    a=np.load(resolve(row['input_file']))
    assert hashlib.sha256(a['x'].tobytes()).hexdigest()==row['input_sha256']
    if 'ids_sha256' in row:assert hashlib.sha256(a['ids'].tobytes()).hexdigest()==row['ids_sha256']

def primary():
    d=read(REPO/'paper/evidence/main379_results.json')
    assert d['reference_dt_ms']==.0125 and d['reference_events']==680 and len(d['models'])==13
    cp=resolve(ORIGINAL+'/benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt')
    assert digest(cp)==d['checkpoint_sha256']=='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
    train=read(resolve(ORIGINAL+'/benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json'))['cases']
    assert len(train)==149 and sum(r['split']=='fit' for r in train)==117
    train_hash={r['input_sha256'] for r in train}
    panel=d['models']['gpu']['records'];ids=[r['case_id'] for r in panel]
    assert len(ids)==len(set(ids))==379
    refs={};out={};raw_tcn_gap=0.
    for row in panel:
        input_check(row);assert row['input_sha256'] not in train_hash
        cid=row['case_id'];p=resolve(ORIGINAL+'/benchmarks/solver_main379_20261016/outputs/reference1/dt0.0125/'+cid+'.npz')
        a=np.load(p);assert np.array_equal(a['time_ms'],np.arange(len(a['voltage'])))
        assert a['spike_times_ms'].tolist()==row['reference_events_ms']
        refs[cid]=(a['voltage'].copy(),a['spike_times_ms'].copy())
    assert sum(len(x[1]) for x in refs.values())==680
    for key,m in d['models'].items():
        assert [r['case_id'] for r in m['records']]==ids
        sse=0.;n=0;evs=[]
        for row in m['records']:
            cid=row['case_id'];rv,re=refs[cid]
            if key in ['gpu','cpu1','cpu8']:
                p=resolve(ORIGINAL+'/benchmarks/random_init_main379_20261012/predictions/'+cid+'.npz');a=np.load(p);v=a['voltage'];events=a['predicted_events_ms']
            elif key=='elm':
                a=np.load(resolve(ORIGINAL+'/benchmarks/adviser6_elm_author_aligned_20261016/elm_evaluation/predictions/'+cid+'.npz'));v=a['voltage'];events=a['spike_times']
            elif key in ['tcn','lstm']:
                folder={'tcn':'tcn_author_default','lstm':'lstm_common150'}[key]
                a=np.load(resolve(ORIGINAL+'/benchmarks/spike_panel_20261009/methods/'+folder+'/predictions/'+cid+'.npz'));v=a['voltage']
                # The frozen event list is independently checked against the whole-forward export for TCN.
                events=np.asarray(row['predicted_events_ms'])
                if key=='tcn':
                    full=np.load(resolve(ORIGINAL+'/benchmarks/baseline_fidelity_audit_20261016/tcn_whole_main379/predictions/'+cid+'.npz'))
                    assert np.array_equal(full['spike_times_ms'],events)
                    gap=float(np.max(np.abs(full['voltage']-v)));raw_tcn_gap=max(raw_tcn_gap,gap);assert gap<.001
            elif key=='reduce':
                timed=next(t for t in m['timing']['records'] if t['case_id']==cid)
                a=np.load(resolve(str(Path(timed['raw_output']).with_suffix('.npz'))));v=a['voltage'];events=np.asarray(row['predicted_events_ms'])
            else:
                a=np.load(resolve(ORIGINAL+'/benchmarks/solver_main379_20261016/outputs/reference1/dt'+format(m['interval_ms'],'g')+'/'+cid+'.npz'));v=a['voltage'];events=a['spike_times_ms']
            assert np.array_equal(events,np.asarray(row['predicted_events_ms'])),(key,cid,'events')
            vm=metric(rv,v,re);assert_metric(vm,row['voltage'],key+'/'+cid)
            ev=match_events(re,events)
            for k in ['tp','fp','fn']:assert ev[k]==row['events'][k],(key,cid,k)
            sse+=vm['sse'];n+=vm['n'];evs.append(ev)
        rmse=float(np.sqrt(sse/n));event=aggregate_events(evs)
        close(rmse,m['accuracy']['pooled_rmse_mv'],key+' pooled',1e-10)
        for k in ['tp','fp','fn','f1']:close(event[k],m['accuracy']['events'][k],key+'/'+k)
        repetitions=7 if key in ['gpu','elm','tcn','lstm'] else 3
        for t in m['timing']['records']:
            assert len(t['wall']['runs_ms'])==repetitions
            close(float(np.median(t['wall']['runs_ms'])),t['wall']['median_ms'],key+' timing')
        close(sum(t['wall']['median_ms'] for t in m['timing']['records']),m['timing']['wall_ms'],key+' sum')
        out[key]={'inputs':379,'rmse_mv':rmse,'f1':event['f1'],'tp':event['tp'],'fp':event['fp'],'fn':event['fn'],'prepared_total_ms':m['timing']['wall_ms']}
        print('PRIMARY',key,json.dumps(out[key]),flush=True)
    budget=read(resolve(ORIGINAL+'/benchmarks/adviser6_elm_author_aligned_20261016/aligned/elm_1ms/complete.json'))
    assert budget['finished'] and budget['updates']==budget['total_updates']==350000
    assert digest(resolve(ORIGINAL+'/benchmarks/adviser6_elm_author_aligned_20261016/aligned/elm_1ms/best.pt'))==d['baseline_revision']['elm_checkpoint_sha256']
    assert d['baseline_revision']['tcn_whole_trajectory']
    return {'models':out,'reference_spikes':680,'reference_dt_ms':.0125,'train_test_disjoint':True,'tcn_whole_vs_frozen_max_gap_mv':raw_tcn_gap,'elm_updates':350000,'checkpoint_unchanged':True}

def crosscell():
    base=resolve('/home/lcm/DBNN_crosscell_final_selection_test_20261017');proto=read(base/'locked_protocol.json');locked=read(base/'models_locked.json')
    assert locked['all_models_locked'] and len(locked['models'])==12 and len(proto['jobs'])==12
    published=json.loads(gzip.decompress((REPO/'paper/evidence/crosscell_final_20261017/records.json.gz').read_bytes()))
    outputs=[]
    for i,job in enumerate(proto['jobs']):
        model=locked['models'][str(i)];cp=resolve(model['checkpoint']);assert digest(cp)==model['checkpoint_sha256']
        manifest=resolve(job['test_manifest']);assert digest(manifest)==job['test_manifest_sha256']
        rows=read(manifest)['cases'];assert len(rows)==182
        train=read(resolve(job['training_manifest']))['cases'];fit_hash={r['input_sha256'] for r in train};fit_seed={r['seed'] for r in train}
        ebase=resolve(job['base'])/'evaluation';record=read(ebase/'complete.json')
        assert record['finished'] and record['all_inputs_retained'] and not record['test_used_for_selection']
        scored={r['case_id']:r for r in record['records']};assert set(scored)=={r['case_id'] for r in rows}
        sse=0.;n=0;events=[]
        for row in rows:
            cid=row['case_id'];input_check(row);assert row['input_sha256'] not in fit_hash and row['seed'] not in fit_seed
            saved=np.load(ebase/'predictions'/(cid+'.npz'));refpath=resolve(row['reference_file']);ref=np.load(refpath);rec=scored[cid]
            assert digest(refpath)==rec['reference_sha256']
            target=np.interp(saved['time_ms'],ref['time_ms'],ref['voltage']);assert np.array_equal(target,saved['reference_voltage'])
            assert np.array_equal(ref['spike_times'],saved['reference_events_ms'])
            assert saved['predicted_events_ms'].tolist()==rec['predicted_events_ms']
            vm=metric(target,saved['voltage'],ref['spike_times']);assert_metric(vm,rec['voltage'],job['cell']+'/'+cid)
            ev=match_events(ref['spike_times'],saved['predicted_events_ms'])
            for k in ['tp','fp','fn']:assert ev[k]==rec['events'][k]
            close(float(np.median(rec['runs_ms'])),rec['prepared_wall_ms'],'cross timing')
            assert rec['numerical']['events_equal'] and rec['numerical']['captured_graph_events_equal']
            assert rec['numerical']['max_abs_mv']<.001 and rec['numerical']['captured_graph_max_abs_mv']<.001
            sse+=vm['sse'];n+=vm['n'];events.append(ev)
        rmse=float(np.sqrt(sse/n));event=aggregate_events(events);close(rmse,record['voltage']['pooled_rmse_mv'],'cross rmse')
        for k in ['tp','fp','fn','f1']:close(event[k],record['events'][k],'cross event')
        paper=next(x for x in published if x['cell']==job['cell'] and x['profile']==job['profile'])
        assert paper['model']['checkpoint_sha256']==digest(cp)
        close(rmse,paper['evaluation']['voltage']['pooled_rmse_mv'],'paper cross rmse')
        close(event['f1'],paper['evaluation']['events']['f1'],'paper cross f1')
        outputs.append({'cell':job['cell'],'profile':job['profile'],'inputs':182,'rmse_mv':rmse,'f1':event['f1'],'checkpoint_sha256':digest(cp),'diagonal_nonzero':model['diagonal_nonzero']})
        print('CROSSCELL',json.dumps(outputs[-1]),flush=True)
    return {'conditions':12,'inputs':2184,'panel_previously_evaluated':True,'output_filtering':False,'train_test_disjoint':True,'models':outputs}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-hashes',action='store_true');ap.add_argument('--primary-only',action='store_true');ap.add_argument('--output',type=Path,default=REPO/'build/release_verification.json');args=ap.parse_args()
    count=0
    if not args.skip_hashes:
        for row in read(REPO/'docs/release_manifest_20261027.json')['files']:
            p=REPO/row['path'];assert p.is_file(),row['path'];assert digest(p)==row['sha256'],row['path'];count+=1
        print('HASHES',count,flush=True)
    report={'passed':True,'files_verified':count,'primary':primary()}
    if not args.primary_only:
        report['crosscell']=crosscell()
        from verify_supplements import main as verify_supplements
        report['supplements']=verify_supplements()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'files_verified':count,'primary_models':13,'primary_inputs':379,'crosscell_inputs':report.get('crosscell',{}).get('inputs',0),'output':str(args.output)}),flush=True)
if __name__=='__main__':main()
