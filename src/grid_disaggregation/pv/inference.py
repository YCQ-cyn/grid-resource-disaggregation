"""Portable, causal MAIN-only runtime. No PV reference or future data accepted."""
from pathlib import Path
from collections import deque
import argparse,json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

class OnlinePV:
    def __init__(self,release=None,device='cpu'):
        parent=Path(__file__).resolve().parent
        self.release=Path(release) if release else parent if (parent/'manifest.json').exists() else parent/'release'
        self.norm=json.loads((self.release/'normalizer.json').read_text(encoding='utf-8'))
        self.manifest=json.loads((self.release/'manifest.json').read_text(encoding='utf-8'))
        self.fields=self.norm['fields'];self.device=device;self.method=self.manifest['method']
        self.history=deque(maxlen=241);self.last=None;self.last_sample=None;self.last_observed=None
        self.past=np.full(56,np.nan);self.ages=np.full(56,10000,dtype=int);self.members=[]
        if self.method.startswith('xgb'):
            import xgboost as xgb
            for seed in self.manifest['seeds']:
                reg=xgb.Booster();cls=xgb.Booster()
                reg.load_model(self.release/f'seed_{seed}_power.ubj');cls.load_model(self.release/f'seed_{seed}_activity.ubj')
                reg.set_param({'device':'cpu','nthread':4});cls.set_param({'device':'cpu','nthread':4});self.members.append((reg,cls))
        else:
            from .models import PVModelConfig,CausalTCNLSTMRegressor
            from .state_space import SmallCausalS4D
            for seed in self.manifest['seeds']:
                ck=torch.load(self.release/f'seed_{seed}_best_validation.pt',map_location=device,weights_only=True)
                cfg=PVModelConfig.from_dict(ck['model_config'])
                m=SmallCausalS4D(cfg) if self.method=='small_s4d' else CausalTCNLSTMRegressor(cfg)
                m.load_state_dict(ck['state_dict']);self.members.append(m.to(device).eval())

    def _step(self,at,record,observed):
        gap=at.hour==0 and at.minute<15
        raw=np.array([record.get(c,record.get('raw_main_'+c,np.nan)) for c in self.fields],float) if record is not None else np.full(56,np.nan)
        # Same deterministic quality rule as training; P=0 with normal voltage remains valid.
        voltage=raw[[self.fields.index('PhV_phs'+s) for s in 'ABC']]
        suspect=bool(np.all(np.isfinite(voltage)&(np.abs(voltage)<1)))
        valid=np.isfinite(raw)&bool(observed)&(not gap)&(not suspect)
        self.ages+=1;self.past[valid]=raw[valid];self.ages[valid]=0
        values=np.where(self.ages<=3,self.past,np.nan)
        if gap:values[:]=np.nan
        n=self.norm;z=(np.clip(values,n['lower'],n['upper'])-n['center'])/n['scale'];z=np.where(np.isfinite(z),z,0)
        if valid[0]:
            self.last_observed=at
            sample=record.get('sample_time_main',record.get('sample_time',at))
            if pd.notna(sample):self.last_sample=pd.Timestamp(sample)
        ma=(at-self.last_observed).total_seconds()/60 if self.last_observed is not None else 4
        sa=(at+pd.Timedelta('1min')-self.last_sample).total_seconds() if self.last_sample is not None else 300
        self.history.append(np.r_[z,valid,float(valid[0]),float(gap),min(4,max(0,ma))/4,min(300,max(0,sa))/300].astype(np.float32));self.last=at
        return bool(np.isfinite(values[0]) and not gap),suspect

    def _tree_vector(self):
        h=np.asarray(self.history,np.float32);parts=[h[-1,:114]]
        for c in ['TotW_MA','TotVar_MA']:
            s=h[:,self.fields.index(c)];parts.append(np.array([s[-k-1] if len(s)>k else 0. for k in [1,5,15,60]]))
            for width in [15,60,240]:
                p=s[max(0,len(s)-1-width):-1].astype(np.float64)
                parts.append(np.array([p.mean(),p.std(ddof=0),p.min(),p.max(),s[-1]-p.mean()]) if len(p) else np.zeros(5))
        parts.append(h[-1,114:]);return np.concatenate(parts).astype(np.float32)

    def update(self,minute_start,record=None,observed=True,predict=True):
        at=pd.Timestamp(minute_start)
        if at!=at.floor('min'):raise ValueError('Provide start of completed natural minute')
        if record is not None:
            for key in ['arrival_time_main','sample_time_main']:
                if pd.notna(record.get(key)) and pd.Timestamp(record[key])>=at+pd.Timedelta('1min'):
                    raise ValueError('Input not available by decision time')
        if self.last is not None:
            if at<=self.last:raise ValueError('Timestamps must increase')
            for t in pd.date_range(self.last+pd.Timedelta('1min'),at-pd.Timedelta('1min'),freq='min'):self._step(t,None,False)
        available,suspect=self._step(at,record,observed)
        if not predict:return None
        if self.method.startswith('xgb'):
            x=self._tree_vector()[None];powers=[];probs=[]
            for reg,cls in self.members:
                powers.append(max(float(reg.inplace_predict(x)[0]),0.));probs.append(float(cls.inplace_predict(x)[0]))
            p=float(np.mean(powers));q=float(np.mean(probs));mag=None
        else:
            h=np.asarray(self.history,np.float32)[-240:];x=np.concatenate([np.zeros((240-len(h),116),np.float32),h])
            powers=[];probs=[];mags=[]
            with torch.inference_mode():
                for m in self.members:
                    a,b=m(torch.tensor(x.T[None],device=self.device));q=float(a.sigmoid().item());mag=float((F.softplus(b)*self.norm['target_scale_kw']).item())
                    powers.append(q*mag);probs.append(q);mags.append(mag)
            p=float(np.mean(powers));q=float(np.mean(probs));mag=float(np.mean(mags))
        return dict(minute_start=str(at),available_after=str(at+pd.Timedelta('1min')),pv_power_kw=p,activity_probability=q,magnitude_kw=mag,input_available=available,suspect_input=suspect,gap_indicator=at.hour==0 and at.minute<15)

def canonical_csv(path):
    d=pd.read_csv(path,low_memory=False)
    if 'report_time' in d:
        for src,dst in [('report_time','arrival_time_main'),('data_time','sample_time_main')]:
            d[dst]=pd.to_datetime(d[src],utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        d=d.sort_values('arrival_time_main',kind='stable').drop_duplicates('sample_time_main',keep='first')
        d['minute_start']=d.arrival_time_main.dt.floor('min');d=d.groupby('minute_start').tail(1);d['observed']=True
    else:
        d['minute_start']=pd.to_datetime(d.minute_start);d['sample_time_main']=d.minute_start
        d['observed']=d.coverage_ratio.ge(.999) if 'coverage_ratio' in d else True
    return d.sort_values('minute_start').drop_duplicates('minute_start').set_index('minute_start')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--release');args=parser.parse_args()
    torch.set_num_threads(4);runner=OnlinePV(args.release);d=canonical_csv(args.input);rows=[]
    for t in pd.date_range(d.index.min(),d.index.max(),freq='min'):
        record=d.loc[t].to_dict() if t in d.index else None
        rows.append(runner.update(t,record,bool(record['observed']) if record else False))
    pd.DataFrame(rows).to_csv(args.output,index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
