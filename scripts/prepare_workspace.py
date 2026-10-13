"""Materialize a relocatable working copy without editing frozen inputs/results."""
import argparse,json,shutil
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
OLD='/home/lcm/DBNN_L5PC_surrogate_20260928'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=REPO/'build/workspace');a=p.parse_args()
out=a.output.resolve()
if out.exists():raise SystemExit(f'Working directory already exists: {out}; use a fresh --output.')
shutil.copytree(REPO/'workspace',out)
changes=[]
for f in out.rglob('*'):
 if f.is_file() and f.suffix in {'.py','.json'}:
  text=f.read_text()
  if OLD in text:
   f.write_text(text.replace(OLD,str(out)));changes.append(str(f.relative_to(out)))
# Seq reset is compiled at each caller's expected location by build_native.py.
(REPO/'build').mkdir(exist_ok=True)
(REPO/'build/relocation.json').write_text(json.dumps({'workspace':str(out),'rewritten_paths':changes},indent=2)+'\n')
print(out)
