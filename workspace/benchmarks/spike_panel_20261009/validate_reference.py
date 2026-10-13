"""Replay one existing spiking raster to verify the new NEURON runner."""
import argparse
from pathlib import Path
import numpy as np
from common_panel import read,write
from run_references import initialize,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();out=work/'reference_parity';out.mkdir(exist_ok=True)
    rows=read(root/'benchmarks/dt_resolution_20260929/test/manifest.json')['cases']
    row=next(r for r in rows if r['case_id']=='case_0062');row={**row,'family':'reference_parity'}
    initialize(str(root),str(out));rec=run(row)
    old=np.load(root/'benchmarks/dt_resolution_20260929/test/references/case_0062_reference.npz')
    new=np.load(out/'case_0062_reference.npz')
    vv=np.interp(new['time_ms'],old['time_ms'],old['voltage'])
    voltage_gap=float(np.max(np.abs(vv-new['voltage'])))
    assert len(old['spike_times'])==len(new['spike_times'])
    event_gap=float(np.max(np.abs(old['spike_times']-new['spike_times'])))
    assert voltage_gap<1e-4,(voltage_gap,'voltage')
    assert event_gap<1e-6,(event_gap,'events')
    result=dict(passed=True,case_id=row['case_id'],reference_count=rec['spike_count'],voltage_max_gap_mv=voltage_gap,event_time_max_gap_ms=event_gap)
    write(work/'reference_parity.json',result);print(result,flush=True)


if __name__=='__main__':main()
