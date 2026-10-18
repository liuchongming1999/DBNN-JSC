"""Rebuild the final cross-cell tables from complete frozen input records.

No inference, model selection, reset calibration or outcome filtering occurs here.
The source panel has previously been evaluated; this is a fixed-panel reevaluation.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,tarfile
import numpy as np

R=Path(__file__).resolve().parents[1]
E=R/'evidence/crosscell_final_20261017'
ORDER=['granule','basket','golgi','hay_l5','human_l23','ca1']
NAMES=dict(granule='Granule',basket='Basket',golgi='Golgi',hay_l5='Rat L5b',human_l23='Human L2/3',ca1='CA1')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def aggregate(rows):
    n=sum(x['voltage']['n'] for x in rows);sse=sum(x['voltage']['sse'] for x in rows)
    reference_ss=sum(x['voltage']['reference_ss'] for x in rows)
    error_ss=sum(x['voltage']['error_ss'] for x in rows)
    tp,fp,fn=[sum(x['events'][key] for x in rows) for key in ['tp','fp','fn']]
    return dict(rmse=float(np.sqrt(sse/n)),ve=1-error_ss/reference_ss,
                f1=2*tp/(2*tp+fp+fn),tp=tp,fp=fp,fn=fn,n=n)

def bootstrap(rows):
    index=np.random.default_rng(20261017).integers(0,len(rows),(10000,len(rows)))
    sse=np.asarray([x['voltage']['sse'] for x in rows]);n=np.asarray([x['voltage']['n'] for x in rows])
    error=np.sqrt(sse[index].sum(1)/n[index].sum(1))
    tp,fp,fn=[np.asarray([x['events'][key] for x in rows])[index].sum(1) for key in ['tp','fp','fn']]
    denominator=2*tp+fp+fn
    f1=np.divide(2*tp,denominator,out=np.full(len(index),np.nan),where=denominator>0)
    return dict(rmse_ci95_mv=np.quantile(error,[.025,.975]).tolist(),f1_ci95=np.nanquantile(f1,[.025,.975]).tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--import-report',type=Path)
    args=parser.parse_args()
    if args.import_report:
        source=args.import_report;E.mkdir(exist_ok=True)
        verification=json.loads((source/'verification.json').read_text())
        assert verification['passed'] and verification['total_inputs']==2184
        locked=json.loads((source/'models_locked.json').read_text())
        protocol=json.loads((source/'locked_protocol.json').read_text())
        assert locked['all_models_locked'] and locked['before_first_test'] and len(locked['models'])==12
        records=[]
        archive=source/'evaluation_records_and_models.tar.gz'
        collector=json.loads((source/'collector_state.json').read_text())
        assert collector['status']=='complete' and sha(archive)==collector['archive_sha256']
        with tarfile.open(archive) as bundle:
            for i,job in enumerate(protocol['jobs']):
                prefix='cells/'+job['cell']+'/'+job['profile']
                data=json.load(bundle.extractfile(prefix+'/evaluation/complete.json'))
                checkpoint=bundle.extractfile(prefix+'/model.pt').read()
                model=locked['models'][str(i)]
                assert hashlib.sha256(checkpoint).hexdigest()==model['checkpoint_sha256']==data['checkpoint_sha256']
                records.append(dict(cell=job['cell'],profile=job['profile'],evaluation=data,model=model))
        (E/'records.json.gz').write_bytes(gzip.compress(json.dumps(records,allow_nan=False).encode(),mtime=0))
        for name in ['verification.json','models_locked.json','locked_protocol.json','summary.json']:
            (E/name).write_bytes((source/name).read_bytes())
        write(E/'source.json',dict(source_report=str(source),remote_work='/home/lcm/DBNN_crosscell_final_selection_test_20261017',
              evaluation_archive_sha256=sha(archive),selection_scope='validation only; final weights and reset frozen before this replay',
              panel_previously_evaluated=True,training_rerun=False,reset_recalibrated=False,output_filtering=False,
              records_sha256=sha(E/'records.json.gz')))
    source=json.loads((E/'source.json').read_text())
    assert sha(E/'records.json.gz')==source['records_sha256']
    all_rows=json.loads(gzip.decompress((E/'records.json.gz').read_bytes()))
    voltage_lines=[];event_lines=[];summary=[]
    for cell in ORDER:
        for profile in ['ampa','ampa_nmda']:
            item=next(x for x in all_rows if x['cell']==cell and x['profile']==profile)
            c=item['evaluation'];rows=c['records'];model=item['model']
            assert len(rows)==len({x['case_id'] for x in rows})==182 and c['all_inputs_retained']
            assert not c['test_used_for_selection']
            assert all(x['checkpoint_sha256']==model['checkpoint_sha256'] for x in rows)
            a=aggregate(rows);ci=bootstrap(rows)
            assert abs(a['rmse']-c['voltage']['pooled_rmse_mv'])<1e-12
            assert abs(a['ve']-c['voltage']['within_trace_weighted_ve'])<1e-12
            assert a['f1']==c['events']['f1']
            for key in ci:assert np.allclose(ci[key],c['bootstrap'][key],atol=1e-12,rtol=0)
            excitation='E' if profile=='ampa' else 'E+N'
            voltage_lines.append(f"{NAMES[cell]} & {excitation} & {a['rmse']:.3f} [{ci['rmse_ci95_mv'][0]:.3f}, {ci['rmse_ci95_mv'][1]:.3f}] & {a['f1']:.3f} [{ci['f1_ci95'][0]:.3f}, {ci['f1_ci95'][1]:.3f}] " + chr(92)*2)
            event_lines.append(f"{NAMES[cell]} & {excitation} & {a['tp']+a['fn']:,} & {a['tp']+a['fp']:,} & {a['f1']:.3f} & [{ci['f1_ci95'][0]:.3f}, {ci['f1_ci95'][1]:.3f}] \\\\")
            summary.append(dict(cell=cell,profile=profile,inputs=182,**a,**ci,
                checkpoint_sha256=model['checkpoint_sha256'],reset=model['reset'],
                diagonal_nonzero=model['diagonal_nonzero'],selected_candidate=model['selected_candidate']))
    (R/'sections/crosscell_table.tex').write_text('\\begin{tabular}{llrr}\n\\toprule\nCell & Exc. & RMSE (mV) [95\\% CI] & F1 [95\\% CI]'+chr(92)*2+'\n\\midrule\n'+'\n'.join(voltage_lines)+'\n\\bottomrule\n\\end{tabular}\n')
    (R/'sections/crosscell_event_table.tex').write_text('\\begin{tabular}{llrrrr}\n\\toprule\nCell & Excitation & Reference & Predicted & F1 & 95\\% CI\\\\\n\\midrule\n'+'\n'.join(event_lines)+'\n\\bottomrule\n\\end{tabular}\n')
    write(E/'paper_summary.json',dict(complete=True,conditions=12,total_inputs=2184,validation_selection_only=True,
         panel_previously_evaluated=True,rows=summary))
    result=dict(passed=True,conditions=12,total_inputs=2184,bootstrap_replicates=10000,
         voltage_event_and_ve_reaggregated=True,records_sha256=sha(E/'records.json.gz'),
         table_sha256={name:sha(R/'sections'/name) for name in ['crosscell_table.tex','crosscell_event_table.tex']})
    write(E/'table_verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
