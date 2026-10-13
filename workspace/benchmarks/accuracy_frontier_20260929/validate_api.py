"""Check physical-time adapter and graph input changes against FFT/CPU reset."""
import argparse,sys
from pathlib import Path
import numpy as np
import torch
from common import write
@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.root;work=a.work
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/channel_ablation_20260929')]
    from prepared_backend import prepare
    from backends import subset_state
    from train_channel_ablation import reset
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(841)
    state=torch.load(root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt',map_location='cuda',weights_only=False)['model_state']
    ids=torch.arange(64,device='cuda');st=subset_state(state,ids);st['aux_tau_rise']=st['tau_rise']*.5;st['aux_tau_decay']=st['tau_decay']*3.;st['aux_omega']=st['omega']*.1
    st['v_th']=torch.tensor(-71.,device='cuda');records=[]
    for input_dt,output_dt in [(1.,1.),(1.,.5),(.5,.5)]:
        for mode in ['compact','hier_store']:
            x=torch.zeros((1,64,2001),dtype=torch.int8,device='cuda');m,g=prepare(st,x,output_dt,input_dt,mode,512)
            for tag in ['random','zero','changed']:
                x.zero_() if tag=='zero' else x.copy_((torch.rand_like(x,dtype=torch.float32)<.02).to(torch.int8))
                v=g()[0];torch.cuda.synchronize();pred=v.cpu().numpy().copy();ratio=round(input_dt/output_dt);T=(x.shape[-1]-1)*ratio+1
                full=torch.zeros((1,64,T),device='cuda');full[...,::ratio]=x;t=torch.arange(T,device='cuda')*output_dt;nfft=1<<(2*T-2).bit_length()
                k=torch.zeros((64,T),device='cuda')
                for prefix in ['', 'aux_']:k+=st[prefix+'omega'][:,None]*(1-torch.exp(-t/st[prefix+'tau_rise'][:,None]))*torch.exp(-t/st[prefix+'tau_decay'][:,None])
                y=torch.fft.irfft(torch.fft.rfft(full,n=nfft)*torch.fft.rfft(k,n=nfft),n=nfft)[...,:T].transpose(1,2).contiguous()
                u=((y@st['bilinear.weight'][0])*y).sum(-1)+y.sum(-1)+st['bilinear.bias']
                oracle,ev=reset(u[0].cpu().numpy(),float(st['v_th']),float(st['reset_amp']),float(st['tau_reset'])/output_dt)
                rec=dict(input_interval_ms=input_dt,output_interval_ms=output_dt,mode=mode,input=tag,max_abs_mv=float(np.max(abs(pred-oracle))),events_equal=bool(np.array_equal(np.flatnonzero(pred>=float(st['v_th'])),ev)),reference_events=len(ev))
                rec['passed']=rec['max_abs_mv']<.001 and rec['events_equal'];assert rec['passed'],rec;records.append(rec)
    write(work/'api_validation.json',dict(finished=True,records=records,maximum_error_mv=max(r['max_abs_mv'] for r in records)));print('PASSED',len(records))
if __name__=='__main__':main()
