from pathlib import Path
def configurations(root,work):
    source=root/'benchmarks/fair_window_20261005/dbnn'
    return {name:(dt,source/f'dbnn_w500_dt{dt:g}ms_burn150/best_checkpoint.pt') for name,dt in [('B2',1.),('B6',.5),('B7',.25)]}
