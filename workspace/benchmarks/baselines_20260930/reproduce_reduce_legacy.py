"""Replay the saved NEURON-7.8 regression fixture in an isolated legacy copy."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();work=a.work.resolve()
    dest=work/'original/reduce_legacy';dest.mkdir(parents=True,exist_ok=True)
    src=work/'vendor/neuron_reduce';repo=dest/'vendor/neuron_reduce'
    if not repo.exists():
        shutil.copytree(src,repo,ignore=shutil.ignore_patterns('.git','x86_64','__pycache__'))
    helper=repo/'tests/test_script_helper.py';text=helper.read_text()
    old='''            h.nrn_load_dll("x86_64/.libs/libnrnmech.so.0")
            h.nrn_load_dll("x86_64/.libs/libnrnmech.0.so")
            h.nrn_load_dll("x86_64/.libs/libnrnmech.so")'''
    text=text.replace(old,'            import neuron\n            neuron.load_mechanisms(os.getcwd())')
    text=text.replace("['nrnivmodl', 'mod']","[os.path.join(sys.prefix, 'bin', 'nrnivmodl'), 'mod']")
    target=dest/'test_script_helper_paths.py';target.write_text(text)
    f=repo/'tests/TestsFiles/Test_5_Hay_2011'
    cmd=[sys.executable,str(target),str(f/'cell1.asc'),str(f/'L5PCtemplate.hoc'),'model.hoc','38','-1',str(f/'origRandomSynapses-10000'),str(f/'voltage_vectors_for_unit_test.txt'),'False','False','hay','37']
    env=os.environ.copy();env['PYTHONPATH']=str(repo);env['MPLBACKEND']='Agg';env.pop('LD_LIBRARY_PATH',None)
    started=time.time()
    with (dest/'run.log').open('w') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    txt=(dest/'run.log').read_text();passed=rc==0 and 'UNIT TEST PASSED' in txt
    import neuron,re
    metrics={}
    for label,key in [('current sim: rmsd between reduced vs complex model:','reduction_rmse_mv'),('unit test: rmsd between reduced vs complex model:','published_fixture_rmse_mv'),('rmsd between current sim reduced and unit test reduced :','fixture_replay_rmse_mv')]:
        m=re.search(re.escape(label)+r'\s*([0-9.eE+-]+)',txt)
        if m:metrics[key]=float(m[1])
    result=dict(finished=True,passed=passed,returncode=rc,neuron_version=neuron.__version__,elapsed_s=time.time()-started,metrics=metrics,scope='Official fully active Hay-2011 Test 5: 10000 synapses, dt 0.1 ms, 1500 ms. Unmodified model, ACG random inputs, reduction algorithm and assertion tolerances.',compatibility='Only compiler/shared-library paths in a separate NEURON-7.8.2 environment; vendor remains unchanged.')
    (dest/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
