"""Run the author's active Hay L5PC regression test, with library-path compatibility only."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
from prepare_compat import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();work=a.work.resolve()
    repo=work/'vendor/neuron_reduce';out=work/'original/reduce';out.mkdir(parents=True,exist_ok=True);compat=prepare(work)
    src=repo/'tests/test_script_helper.py';text=src.read_text()
    old='''            h.nrn_load_dll("x86_64/.libs/libnrnmech.so.0")
            h.nrn_load_dll("x86_64/.libs/libnrnmech.0.so")
            h.nrn_load_dll("x86_64/.libs/libnrnmech.so")'''
    assert old in text
    text=text.replace(old,'            import neuron\n            neuron.load_mechanisms(os.getcwd())')
    # The project venv was renamed; invoking its wrapper through Python avoids
    # the stale absolute shebang without changing the installed environment.
    assert "['nrnivmodl', 'mod']" in text
    text=text.replace("['nrnivmodl', 'mod']","[sys.executable, os.path.join(sys.prefix, 'bin', 'nrnivmodl'), 'mod']")
    helper=out/'test_script_helper_nrn9.py';helper.write_text(text)
    files=repo/'tests/TestsFiles/Test_5_Hay_2011'
    cmd=[sys.executable,str(helper),str(files/'cell1.asc'),str(files/'L5PCtemplate.hoc'),'model.hoc','38','-1',str(files/'origRandomSynapses-10000'),str(files/'voltage_vectors_for_unit_test.txt'),'False','False','hay','37']
    env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','');env['PYTHONPATH']=str(compat)+os.pathsep+str(repo)+os.pathsep+str(work/'deps')+os.pathsep+env.get('PYTHONPATH','');env['MPLBACKEND']='Agg'
    conda_lib=Path(sys.base_prefix)/'lib'
    env['LD_LIBRARY_PATH']=str(conda_lib)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    started=time.time()
    with (out/'run.log').open('w') as log:
        rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    log=(out/'run.log').read_text();passed=rc==0 and 'UNIT TEST PASSED' in log
    numbers={}
    for label,key in [('current sim: rmsd between reduced vs complex model:','reduction_rmse_mv'),('unit test: rmsd between reduced vs complex model:','published_fixture_rmse_mv'),('rmsd between current sim reduced and unit test reduced :','fixture_replay_rmse_mv')]:
        m=re.search(re.escape(label)+r'\s*([0-9.eE+-]+)',log)
        if m:numbers[key]=float(m[1])
    result=dict(finished=True,passed=passed,returncode=rc,elapsed_s=time.time()-started,scope='Author-provided fully active Hay L5PC + 10000 synapses, 1500 ms, dt 0.1 ms, frequency 38 Hz, exact saved control/reduced voltage vectors.',compatibility='Shared-library/compiler paths and NEURON 9 RANGE-variable introspection. Equations, model, input, solver and tolerances unchanged; compat/neuron9.patch records the API patch.',original_script_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),compatibility_script_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),metrics=numbers)
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
