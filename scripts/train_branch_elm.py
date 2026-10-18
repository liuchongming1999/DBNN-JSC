"""Launch the archived fixed-default Branch-ELM recipe in a NEW output folder.

No budget, model, seed, optimizer, or threshold tuning options are introduced.
This is optional retraining, never used by release verification.
"""
import argparse,importlib.util,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args()
w=a.workspace.resolve();out=a.output.resolve()
if out.exists():raise SystemExit('Choose a new output directory; frozen models are never overwritten.')
source=w/'benchmarks/adviser6_elm_author_aligned_20261016';B=w/'benchmarks/baselines_20260930';out.mkdir(parents=True)
shutil.copy2(B/'source_versions.json',out/'source_versions.json')
gate=source/'original/elm/complete.json'
if not gate.exists():gate=B/'original/elm/complete.json'
if gate.exists():
 (out/'original/elm').mkdir(parents=True);shutil.copy2(gate,out/'original/elm/complete.json')
spec=importlib.util.spec_from_file_location('recorded_author_elm',source/'train_author_aligned_elm.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.argv=['train_author_aligned_elm.py','--root',str(w),'--work',str(out),'--data',str(w),'--method','elm','--dt','1','--device',a.device]+(['--smoke'] if a.smoke else [])
module.trainer.main()
