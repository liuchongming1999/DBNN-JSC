"""Create an audited NEURON 9 API compatibility copy; preserve original vendor code."""
import difflib,hashlib,json,shutil
from pathlib import Path

def prepare(work):
    src=Path(work)/'vendor/neuron_reduce/neuron_reduce';dest=Path(work)/'compat/neuron_reduce'
    dest.parent.mkdir(exist_ok=True);shutil.copytree(src,dest,dirs_exist_ok=True)
    target=dest/'subtree_reductor_func.py';before=target.read_text()
    old='                segment_to_mech_vals[seg][mech_name][n] = getattr(seg, n)'
    new='''                # NEURON 9 exposes NMODL procedures in dir(mech) as well.
                # Only scalar RANGE values existed in this list in the original API.
                if not hasattr(seg, n):
                    continue
                value = getattr(seg, n)
                if not isinstance(value, (int, float)):
                    continue
                segment_to_mech_vals[seg][mech_name][n] = value'''
    assert before.count(old)==1;after=before.replace(old,new);target.write_text(after)
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='vendor/subtree_reductor_func.py',tofile='compat/subtree_reductor_func.py'))
    (dest.parent/'neuron9.patch').write_text(diff)
    (dest.parent/'audit.json').write_text(json.dumps(dict(reason='Filter procedure names newly exposed by NEURON 9; copy every numeric scalar RANGE parameter exactly as before. Reduction equations and channel values unchanged.',original_sha256=hashlib.sha256(before.encode()).hexdigest(),compat_sha256=hashlib.sha256(after.encode()).hexdigest()),indent=2)+'\n')
    return dest.parent

if __name__=='__main__':prepare(Path(__file__).resolve().parent)
