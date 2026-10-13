"""Refresh project-level provenance after all results and the HTML exist."""
from pathlib import Path
import hashlib,json,datetime

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

C=read(HERE/'confirmation.json');N=read(HERE/'neuron_results.json')
V=read(HERE/'validation.json');F=read(HERE/'fidelity.json')
assert len(C['cases'])==len(N['cases'])==14 and 'finished' in C
assert sha(ROOT/'source/DBNN.py')=='34e5395760b423b9ce862cb0c2fb371a078e523bfde5f51bfc0696c2b216fe5d'
assert (ROOT/'DBNN_L5PC_surrogate_report_20260928.html').stat().st_size>100000
cs={r['index']:r for r in C['cases']};ns={r['index']:r for r in N['cases']}
status=read(ROOT/'status_20260928.json')
status['updated']=datetime.datetime.now().astimezone().isoformat()
status['fresh_checkpoints']=['results/train_cpu_bench/dbnn_best.pt','results/train_gpu5_b1_bench/dbnn_best.pt','results/train_gpu5_b1_lr2e4/dbnn_best.pt','results/dbnn_repro_finetune_5ep_lr2e4/dbnn_best.pt']
status['current_benchmark']='benchmarks/sparse_20260928/confirmation.json'
status['current_neuron_benchmark']='benchmarks/sparse_20260928/neuron_results.json'
status['sparse_benchmark']={'completed_cases':14,'gpu_repetitions':50,'neuron_repetitions':3,
 'neuron_case13_repetitions':5,'exclusive_resources':False,
 'full500_B1_speedup':cs[0]['speedup'],'sparse500_A32_B8_speedup':cs[5]['speedup'],
 'synthetic8000_A32_B1_speedup':cs[8]['speedup'],
 'sparse500_A32_cpu1_native_ms':ns[1]['cpu_one_thread']['packed_native_cpu_1t']['median_ms'],
 'sparse500_A32_neuron_cpu1_ms':ns[1]['core']['median_ms']}
status['checkpoint_evaluation']={k:{m:v[m] for m in ('mean_sample_rmse_mv','pooled_rmse_mv','max_abs_mv')} for k,v in V['checkpoints'].items()}
status['active_dbnn_processes']=[]
status['notes']=[n for n in status['notes'] if n!='No DBNN process is left running.']
note='The sparse benchmark finished all 14 scenarios. Timing remains shared-GPU, with pinned CPU affinity but no exclusive reservation. Complete converged training and physical dense-site fitting remain pending.'
if note not in status['notes']:status['notes'].append(note)
(ROOT/'status_20260928.json').write_text(json.dumps(status,indent=2))

p=ROOT/'README.md';text=p.read_text().replace('results/train_gpu5_bench/dbnn_best.pt','results/train_gpu5_b1_bench/dbnn_best.pt')
marker='## Sparse-site acceleration update (2026-09-28)'
if marker not in text:
    text+='\n'+marker+'\n\n'
    text+='The current measurements supersede the initial timing sections above. The project now includes CUDA/CUB, CuTe DSL, Triton parallel scans, CUTLASS SIMT and FastF32 TensorOp experiments, cached FFT, event-driven reset, CUDA Graph, and C++/OpenMP CPU inference.\n\n'
    text+='All 14 candidate/active-site, duration and batch scenarios completed. Main comparisons use 50 independent interleaved repetitions with the fastest untouched PyTorch F/R batch or per-trace path. A packed PyTorch control isolates the gain beyond exact support reduction. NEURON receives the same quantized events with inactive synapses removed; CPU comparisons use one thread. The GPU was shared.\n\n'
    text+='See `benchmarks/sparse_20260928/README.md` for the runnable protocol, `confirmation.json` for main GPU results, `neuron_results.json` for CPU/NEURON, `validation.json` and `fidelity.json` for accuracy. The root `DBNN_L5PC_surrogate_report_20260928.html` includes all plots, limitations and next steps. Exportable PNG/SVG figures are in `benchmarks/sparse_20260928/figures/`.\n\n'
    text+='N=2000/8000 measurements use synthetic parameter expansion, not retrained dense spatial models. Original training data were unavailable; full converged reproduction and high-fidelity spike/event prediction are still ongoing research tasks.\n'
p.write_text(text)

manifest=read(ROOT/'manifest_20260928.json')
paths={ROOT/r['path'] for r in manifest['files']}
for pattern in ['*.py','*.cpp','*.cu','*.json','*.so','*.md','*.npz']:
    paths.update(HERE.glob(pattern))
paths.update((HERE/'figures').glob('*.png'));paths.update((HERE/'figures').glob('*.svg'))
manifest['files']=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths) if p.is_file()]
manifest['updated']=status['updated'];manifest['cutlass_cpp']='3.9.2';manifest['cute_dsl']='4.2.0'
(ROOT/'manifest_20260928.json').write_text(json.dumps(manifest,indent=2))
print('Updated project status, README and hashes for',len(manifest['files']),'files')
