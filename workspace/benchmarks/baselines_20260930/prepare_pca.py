"""Author-style 20-component whitened PCA, fit on training data only."""
import argparse,time
from pathlib import Path
import numpy as np
from panel import rows_for,save,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--aux',type=Path,required=True);p.add_argument('--dt',type=float,required=True);a=p.parse_args()
    out=a.aux/f'pca_{a.dt:g}ms';out.mkdir(exist_ok=True);rows,manifest=rows_for(a.root,a.dt)
    if (out/'complete.json').exists():return
    # One original training file contains 128 simulations; use at most 128 fit cases.
    fit=[r for r in rows if r['split']=='fit'][:128];stride=round(a.dt/.25);assert stride>=1
    started=time.time();n=0;s=np.zeros(639,np.float64);ss=np.zeros((639,639),np.float64)
    for i,r in enumerate(fit):
        v=np.load(a.aux/'raw'/f"{r['case_id']}.npy",mmap_mode='r')[::stride]
        # Original author clips raw dendrite voltages at +2mV before PCA.
        z=np.minimum(v,2).astype(np.float64);n+=len(z);s+=z.sum(0);ss+=z.T@z
        save(out/'status.json',dict(phase='fit',completed=i+1,total=len(fit)));print('PCA_FIT',i+1,len(fit),flush=True)
    mean=s/n;cov=(ss-n*np.outer(mean,mean))/(n-1);ev,vec=np.linalg.eigh(cov);idx=np.argsort(ev)[-20:][::-1]
    eigen=ev[idx];components=vec[:,idx];scale=np.sqrt(eigen);assert np.all(scale>0)
    np.savez(out/'pca.npz',mean=mean,components=components,scale=scale,eigenvalues=eigen,n_samples=n)
    for i,r in enumerate(rows):
        v=np.load(a.aux/'raw'/f"{r['case_id']}.npy",mmap_mode='r')[::stride]
        pred=((np.minimum(v,2).astype(np.float64)-mean)@components/scale).astype(np.float32)
        assert np.isfinite(pred).all();np.save(out/f"{r['case_id']}.npy",pred)
        save(out/'status.json',dict(phase='project',completed=i+1,total=len(rows)));print('PCA_PROJECT',i+1,len(rows),flush=True)
    save(out/'complete.json',dict(finished=True,dt_ms=a.dt,components=20,whiten=True,raw_clip_mv=2,projected_training_clip=3,fit_case_ids=[r['case_id'] for r in fit],fit_samples=n,explained_variance_ratio=float(eigen.sum()/ev.sum()),manifest_sha256=digest(manifest),elapsed_s=time.time()-started,scope='Author default first-training-file PCA size adapted to at most 128 fit cases; validation/main31 excluded from fitting.'))

if __name__=='__main__':main()
