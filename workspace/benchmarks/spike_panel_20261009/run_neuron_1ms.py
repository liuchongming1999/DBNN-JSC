"""NEURON 1-ms single-thread outputs as an exact-dt solver validation base."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import sys
from datetime import datetime
from common_panel import read,write

S=None


def initialize(root,out):
    global S
    import run_references as ref
    ref.initialize(root,out);s=ref.STATE;h=s['h']
    sys.path.insert(0,str(Path(root)/'benchmarks/neuron_single_expanded_20260929'))
    from run_solver_grid import one_case
    from neuron import coreneuron
    nc=h.NetCon(s['network']['cell'].soma[0](.5)._ref_v,None,sec=s['network']['cell'].soma[0]);nc.threshold=-20;nc.delay=10
    spikes=h.Vector();nc.record(spikes)
    S=dict(**s,one_case=one_case,coreneuron=coreneuron,spike_nc=nc,spikes=spikes)


def run(row):
    s=S
    return s['one_case'](s['h'],s['coreneuron'],s['pc'],s['spike_nc'],s['spikes'],s['network'],s['active'],
                         row,Path('/'),s['out'],'neuron',1.,1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();out=work/'methods/neuron1/raw';out.mkdir(parents=True,exist_ok=True)
    rows=read(work/'confirmatory/manifest.json')['cases'];records=[]
    with ProcessPoolExecutor(16,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(root),str(out))) as pool:
        for future in as_completed([pool.submit(run,row) for row in rows]):
            r=future.result();assert r['finished'];records.append(dict(case_id=r['case_id'],spike_count=r['spike_count']))
            write(out.parent/'status.json',dict(updated=datetime.now().isoformat(),completed=len(records),total=len(rows)))
    write(out.parent/'raw_complete.json',dict(finished=True,dt_ms=1.,records=records))


if __name__=='__main__':main()
