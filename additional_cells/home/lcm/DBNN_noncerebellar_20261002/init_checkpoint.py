"""Single-site physical filter initialization, same fixed DBNN architecture."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from models import build,run,REGISTRY
from screen import window,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',choices=REGISTRY,required=True);p.add_argument('--profile',default='ampa');a=p.parse_args()
    work=a.work.resolve();out=work/'results'/a.model/a.profile;out.mkdir(parents=True,exist_ok=True)
    if (out/'init_checkpoint.pt').exists():return
    env=build(work,a.model,a.profile);op=json.loads((work/'results'/a.model/'screen/operating_point.json').read_text());kw=dict(holding=op['holding_current_na'],weight=op['weight_us'],inhibition_ratio=op['inhibition_ratio'])
    base=window(run(env,300.,**kw),300);t=np.arange(1201)*.25-20.;records=[]
    tau_rise=np.full(500,2.,np.float32);tau_decay=np.full(500,20.,np.float32);omega=np.full(500,.01,np.float32)
    for sign,sites in [(1.,[5,50,100,150,200,245]),(-1.,[255,300,350,400,450,495])]:
        fit=[]
        for site in sites:
            x=np.zeros((1,301),np.uint8);x[0,20]=1;r=run(env,300.,[site],x,**kw);y=sign*(window(r,300)-base);tp=np.maximum(t,0)
            def residual(z):
                tr,td,om=np.exp(z);pred=om*(1-np.exp(-tp/tr))*np.exp(-tp/td);pred[t<0]=0;return pred-y
            result=least_squares(residual,np.log([2.,20.,max(float(np.max(y))*2,.0001)]),bounds=(np.log([.25,.5,1e-6]),np.log([50.,100.,100.])))
            pars=np.exp(result.x);fit.append(pars);records.append(dict(site=site,tau_rise_ms=float(pars[0]),tau_decay_ms=float(pars[1]),omega_mv=float(pars[2]),initial_single_site_rmse_mv=float(np.sqrt(np.mean(result.fun**2)))))
        span=slice(0,250) if sign>0 else slice(250,500);tr,td,om=np.median(fit,axis=0);tau_rise[span]=tr;tau_decay[span]=td;omega[span]=om
    import torch
    state=dict(tau_rise=torch.tensor(tau_rise),tau_decay=torch.tensor(tau_decay),omega=torch.tensor(omega),
               **{'bilinear.weight':torch.zeros((1,500,500)), 'bilinear.bias':torch.tensor([float(base[-1000:].mean())]),
                  'v_th':torch.tensor(op['rest_mv']+15.),'reset_amp':torch.tensor(8.),'tau_reset':torch.tensor(16.)})
    torch.save(dict(model_state=state,time_constant_units='ms',sample_interval_ms=1.,model=a.model,profile=a.profile,
        architecture='Original single double-exponential filter per site + zero-diagonal quadratic + linear sum + bias + event reset; no architecture additions',initialization='Independent fit of weak single-site responses for physical filter seed; quadratic weights zero; no L5PC parameters transferred.'),out/'init_checkpoint.pt')
    write(out/'initialization.json',dict(records=records,rest_mv=float(base[-1000:].mean()),initial_reset_threshold_mv=op['rest_mv']+15.,input_site_count=500))
    print('INITIALIZED',a.model,a.profile,flush=True)

if __name__=='__main__':main()
