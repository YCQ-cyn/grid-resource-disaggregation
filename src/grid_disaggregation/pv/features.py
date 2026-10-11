"""Deployable electrical-only transformations, no fitting or target access."""
import numpy as np
import pandas as pd

def encode(d,n):
    fields=n['fields'];x=d[['main_'+c for c in fields]].to_numpy(float)
    gap=d.gap_indicator.to_numpy(bool)
    observed=d[['source_valid_'+c for c in fields]].to_numpy(bool)&~gap[:,None]
    z=(np.clip(x,n['lower'],n['upper'])-n['center'])/n['scale']
    z=np.where(np.isfinite(z)&~gap[:,None],z,0)
    measurement=d.main_observed.to_numpy(bool)&~gap
    return np.column_stack([z,observed,measurement,gap]).astype(np.float32)

def tabular_encoded(current,n):
    features=[current];names=n['channel_order'].copy()
    for field in ['TotW_MA','TotVar_MA']:
        s=pd.Series(current[:,n['fields'].index(field)])
        for lag in [1,5,15,60]:
            features.append(s.shift(lag).fillna(0).to_numpy()[:,None]);names.append(f'{field}_lag{lag}')
        past=s.shift(1)
        for width in [15,60,240]:
            roll=past.rolling(width,min_periods=1)
            vals=np.column_stack([roll.mean(),roll.std(ddof=0),roll.min(),roll.max(),s-roll.mean()])
            features.append(np.nan_to_num(vals));names.extend(f'{field}_past{width}_{a}' for a in ['mean','std','min','max','delta'])
    return np.column_stack(features).astype(np.float32),names

def tree_online_vector(history,n):
    """Current vector + actual prior rows, no invented zero-padded rolling history."""
    h=np.asarray(history,np.float32);current=h[-1,:114];values=[current]
    for field in ['TotW_MA','TotVar_MA']:
        idx=n['fields'].index(field);s=h[:,idx]
        values.append(np.array([s[-lag-1] if len(s)>lag else 0. for lag in [1,5,15,60]]))
        for width in [15,60,240]:
            past=s[max(0,len(s)-1-width):-1].astype(np.float64)
            if len(past):values.append(np.array([past.mean(),past.std(ddof=0),past.min(),past.max(),float(s[-1])-past.mean()]))
            else:values.append(np.zeros(5))
    return np.concatenate(values).astype(np.float32)
