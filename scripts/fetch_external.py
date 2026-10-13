"""Fetch pinned upstream comparison code; source licenses stay with upstream."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--models-only',action='store_true');a=p.parse_args();w=a.workspace.resolve()
items=[('https://github.com/ModelDBRepository/180373.git',None,w/'source/ShaiEtAl2015')]
if not a.models_only:
 for name,s in json.loads((w/'benchmarks/baselines_20260930/source_versions.json').read_text()).items():
  items.append((s['url'],s['commit'],w/'benchmarks/baselines_20260930/vendor'/name))
for url,commit,dest in items:
 if dest.exists():
  print('Already present:',dest);continue
 dest.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','clone',url,str(dest)],check=True)
 if commit:subprocess.run(['git','checkout',commit],cwd=dest,check=True)
print('Upstream source retrieval completed; see upstream documentation for mechanism compilation.')
