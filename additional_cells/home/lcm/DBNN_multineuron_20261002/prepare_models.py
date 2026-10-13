"""Compile isolated density mechanisms for NEURON/CoreNEURON without edits to vendor."""
import argparse,json,re,shutil,subprocess,sys,hashlib,os
from pathlib import Path
from models import REGISTRY

NMDA = '''TITLE Controlled NMDA conductance for receptor sensitivity
NEURON { POINT_PROCESS DBNN_NMDA RANGE tau1, tau2, e, mg, g, i NONSPECIFIC_CURRENT i THREADSAFE }
UNITS { (nA)=(nanoamp) (mV)=(millivolt) (uS)=(microsiemens) }
PARAMETER { tau1=2 (ms) tau2=80 (ms) e=0 (mV) mg=1 }
ASSIGNED { v (mV) i (nA) g (uS) factor }
STATE { A (uS) B (uS) }
INITIAL { LOCAL tp A=0 B=0 tp=(tau1*tau2)/(tau2-tau1)*log(tau2/tau1) factor=1/(-exp(-tp/tau1)+exp(-tp/tau2)) }
BREAKPOINT { SOLVE state METHOD cnexp g=(B-A)/(1+mg*exp(-0.062*v)/3.57) i=g*(v-e) }
DERIVATIVE state { A'=-A/tau1 B'=-B/tau2 }
NET_RECEIVE(weight (uS)) { A=A+weight*factor B=B+weight*factor }
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',choices=REGISTRY,required=True);a=p.parse_args()
    work=a.work.resolve();s=REGISTRY[a.model];src=work/'vendor'/str(s['modeldb'])/s['variant'];out=work/'mechanisms'/a.model;mods=out/'mods';mods.mkdir(parents=True,exist_ok=True)
    entries=[];patches=[]
    for f in sorted((src/'mod_files').glob('*.mod')):
        txt=f.read_text()
        if re.search(r'\bSUFFIX\b',txt) and not re.search(r'\bPOINT_PROCESS\b',txt):
            original=txt
            # NMODL 9/CoreNEURON declaration compatibility only. Preserve
            # numerical constants, all rate expressions and channel densities.
            if 'FUNCTION r(' in txt and re.search(r'RANGE[^\n]*\br\b',txt):
                txt=re.sub(r'(RANGE[^\n]*)\br\s*,',r'\1',txt)
                patches.append(dict(file=f.name,change='Remove function r from RANGE (function is not a stored variable).'))
            if re.search(r'^CONSTANT\s*\{\s*cao\s*=\s*2',txt,re.M):
                txt=re.sub(r'^CONSTANT\s*\{\s*cao\s*=\s*2[^}]*\}', 'CONSTANT { cao_model = 2 (mM) }',txt,flags=re.M)
                # This read ion is otherwise unused; replacing declaration of
                # the unused constant leaves all kinetic expressions intact.
                txt=txt.replace('ASSIGNED {','ASSIGNED {\n cao (mM)',1)
                patches.append(dict(file=f.name,change='Rename unused constant cao to cao_model and declare read ion cao as ASSIGNED.'))
            if re.search(r'^LOCAL dsq, dsqvol',txt,re.M):
                txt=re.sub(r'^LOCAL dsq, dsqvol[^\n]*','',txt,flags=re.M)
                txt=txt.replace('ASSIGNED {','ASSIGNED {\n dsq\n dsqvol',1)
                patches.append(dict(file=f.name,change='Store geometry temporaries dsq/dsqvol per mechanism instance for KINETIC code generation.'))
            dest=mods/f.name;dest.write_text(txt);entries.append(dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),compiled_sha256=hashlib.sha256(txt.encode()).hexdigest(),original=str(f)))
    (mods/'DBNN_NMDA.mod').write_text(NMDA)
    (out/'source_audit.json').write_text(json.dumps(dict(model=a.model,spec=s,density_mechanisms=entries,compatibility_patches=patches,ionic_parameter_changes=[],synapse_note='Built-in ExpSyn/Exp2Syn; controlled Mg-blocked NMDA added only for explicit receptor sensitivity'),indent=2))
    # Invoke Python wrapper explicitly: its historical shebang precedes the
    # project rename. Wrapper supplies NMODL_PYLIB and embedded-Python paths.
    compiler=Path(sys.executable).with_name('nrnivmodl')
    cmd=[sys.executable,str(compiler),'-coreneuron','mods']
    env=os.environ.copy()
    with open(out/'compile.log','w') as f:
        result=subprocess.run(cmd,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT)
    (out/'build.json').write_text(json.dumps(dict(command=cmd,exit_code=result.returncode,neuron_library=(out/'x86_64/libnrnmech.so').exists(),core_library=(out/'x86_64/libcorenrnmech.so').exists()),indent=2))
    if result.returncode:print((out/'compile.log').read_text()[-6500:]);raise SystemExit(result.returncode)
    print('BUILT',a.model,flush=True)

if __name__=='__main__':main()
