"""Validation-only event thresholds spanning the model's actual score range."""
import numpy as np
from scipy.signal import find_peaks

VERSION='validation_peak_scores_v2'


def detected_events(prob,dt,threshold):
    peaks,_=find_peaks(prob,height=threshold,distance=max(1,round(5/dt)))
    return peaks*dt


def fit_threshold(pred,refs,dt,match_events,event_aggregate):
    scores=[]
    for p in pred:
        peaks,_=find_peaks(p,distance=max(1,round(5/dt)))
        scores.append(np.asarray(p)[peaks])
    scores=np.concatenate(scores) if scores else np.array([])
    quantiles=np.unique(np.r_[np.linspace(0,1,257),1-np.logspace(-7,0,129)])
    candidates=np.r_[0.,1.,.01,.025,.05,.075,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,
                     np.logspace(-9,-2,57)]
    if scores.size:candidates=np.r_[candidates,np.quantile(scores,quantiles)]
    grid=[]
    for threshold in np.unique(candidates):
        result=event_aggregate([match_events(r,detected_events(p,dt,float(threshold)))
                                for r,p in zip(refs,pred)])
        grid.append(dict(threshold=float(threshold),**result))
    best=max(grid,key=lambda r:(r['f1'],r['precision'] if r['precision'] is not None else -1,r['threshold']))
    return dict(threshold=best['threshold'],grid=grid,version=VERSION,
                validation_peak_score_range=[float(scores.min()),float(scores.max())] if scores.size else None,
                selection='Maximum validation micro-F1 on empirical peak-score quantiles including the high-score tail and logarithmic low-score thresholds; +/-5ms event matching, peak distance5ms. Ties: precision then higher threshold. No main31 data.')
