"""Regenerate figures/statistical tables in build/, keeping frozen files intact."""
from pathlib import Path
import argparse,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=R/'build/paper_reproduction');a=p.parse_args();dest=a.output.resolve()
if dest.exists():raise SystemExit('Use a fresh --output directory; no existing outputs are overwritten.')
dest.mkdir(parents=True);shutil.copytree(R/'paper',dest/'paper');(dest/'theory_data').symlink_to(R/'theory_data',target_is_directory=True)
for name in ['verify_refined_results.py','verify_completed_baselines.py','build_crosscell_final.py','build_figures.py','build_model_structure.py','build_event_diagnostics.py','build_frozen_training_diagnostics.py','build_workflow_cost_appendix.py','build_unfiltered_stress_appendix.py','build_engineering_ablation_appendix.py']:
 subprocess.run([sys.executable,str(dest/'paper/scripts'/name)],check=True,cwd=dest)
for name in ['unfiltered_stress_table.tex','engineering_ablation_table.tex']:
 assert (dest/'paper/sections'/name).read_bytes()==(R/'paper/sections'/name).read_bytes(),name
print(dest/'paper/figures')
