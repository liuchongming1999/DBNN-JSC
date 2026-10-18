from pathlib import Path
import json,time,subprocess
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
start=time.monotonic()
while time.monotonic()-start<1200:
    j=json.loads((W/'status.json').read_text())
    if j['phase']=='failed':raise RuntimeError(j)
    if j.get('finished') and (W/'followup_complete.json').exists():break
    time.sleep(5)
else:raise TimeoutError('Timing campaign did not complete within the bounded wait')
for script in ['finalize_integrity.py','summarize.py']:
    subprocess.run([str(R/'.venv/bin/python'),str(W/script),str(W)] if script=='summarize.py' else [str(R/'.venv/bin/python'),str(W/script)],check=True)
