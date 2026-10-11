import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, f1_score
def score(table,pred,prob,n,threshold=.5,observed_only=True):
    ok=table.pv_observed.to_numpy(bool) if observed_only else np.ones(len(table),bool)
    y=table.pv_target_kw.to_numpy(float)[ok];p=pred[ok];q=prob[ok];sub=table.loc[ok]
    active=y>n['active_threshold_kw'];err=p-y
    energy=y.sum()/60;delta=err.sum()/60
    tmp=pd.DataFrame({'batch':sub.batch_id.to_numpy(),'date':sub.date.to_numpy(),'y':y/60,'p':p/60})
    daily=tmp.groupby(['batch','date'])[['y','p']].sum()
    return dict(n=len(y),mae=float(np.mean(np.abs(err))),generation_mae=float(np.mean(np.abs(err[active]))) if active.any() else None,
      rmse=float(np.sqrt(np.mean(err**2))),r2=float(r2_score(y,p)),activity_f1=float(f1_score(active,q>=threshold,zero_division=0)),
      energy_reference_kwh=float(energy),energy_predicted_kwh=float(p.sum()/60),energy_bias_kwh=float(delta),energy_bias_pct=float(delta/energy*100) if energy>0 else None,
      daily_absolute_energy_bias_kwh=float((daily.p-daily.y).abs().mean()),inactive_mae=float(np.abs(err[~active]).mean()) if (~active).any() else None,
      inactive_predicted_energy_kwh=float(p[~active].sum()/60),active_0_2_mae=float(np.abs(err[active&(y<=2)]).mean()) if (active&(y<=2)).any() else None)


def optimize_threshold(table,prob,n):
    observed=table.pv_observed.to_numpy(bool);a=table.pv_target_kw.to_numpy()[observed]>n['active_threshold_kw']
    grid=np.linspace(.05,.95,37)
    return float(grid[np.argmax([f1_score(a,prob[observed]>=t,zero_division=0) for t in grid])])
