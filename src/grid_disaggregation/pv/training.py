"""Domain-weighted dual-head training; no holdout-based model selection."""
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .models import PVModelConfig, CausalTCNLSTMRegressor
from .state_space import SmallCausalS4D
from .features import encode
from .metrics import score, optimize_threshold
from ..common import dump
SEEDS = [42, 2026, 3407]
WINDOW = 240
BATCH = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = None
def feature_matrix(g,n):
    return np.column_stack([encode(g,n),g[['measurement_age_scaled','sample_age_scaled']].to_numpy(np.float32)])


def model(method,config=None):
    cfg=PVModelConfig(input_channels=116) if config is None else PVModelConfig.from_dict(config)
    return SmallCausalS4D(cfg) if method=='small_s4d' else CausalTCNLSTMRegressor(cfg)


def pack(d,split,n,device=DEVICE):
    pieces=[];tables=[];ends=[];offset=0
    for _,g in d[d.split.eq(split)].groupby('segment_id',sort=True):
        g=g.sort_values('minute_start').reset_index(drop=True)
        x=np.concatenate([np.zeros((WINDOW,116),np.float32),feature_matrix(g,n)])
        ids=np.flatnonzero(g.eligible.to_numpy(bool));pieces.append(x);ends.extend(offset+WINDOW+ids)
        tables.append(g.iloc[ids].copy());offset+=len(x)
    table=pd.concat(tables,ignore_index=True)
    return dict(x=torch.tensor(np.concatenate(pieces),device=device),ends=torch.tensor(ends,device=device),table=table,
      y=torch.tensor(table.pv_target_kw.to_numpy(np.float32),device=device),weight=torch.tensor(np.where(table.pv_imputed,.25,1.).astype(np.float32),device=device))


def windows(p,ids):
    at=p['ends'][ids,None]+torch.arange(-WINDOW+1,1,device=p['x'].device)[None,:]
    return p['x'][at].transpose(1,2).contiguous()


def predict(m,p,n,batch=BATCH):
    m.eval();out=[]
    with torch.inference_mode():
        for j in range(0,len(p['ends']),batch):
            ids=torch.arange(j,min(j+batch,len(p['ends'])),device=p['x'].device)
            a,b=m(windows(p,ids));q=a.sigmoid();mag=F.softplus(b)*n['target_scale_kw']
            out.append(torch.stack([q*mag,q,mag],dim=1).cpu().numpy())
    return np.concatenate(out).T


def metrics(table,p,q,n,threshold=.5):
    r=score(table,p,q,n,threshold)
    y=table.pv_target_kw.to_numpy(float);ok=table.pv_observed.to_numpy(bool)
    at=pd.to_datetime(table.minute_start)
    adjacent=ok&np.r_[False,ok[:-1]]&(at.diff().dt.total_seconds().to_numpy()==60)
    dp=np.diff(p,prepend=p[0]);dy=np.diff(y,prepend=y[0])
    r['ramp_mae_kw_per_min']=float(np.abs(dp[adjacent]-dy[adjacent]).mean()) if adjacent.any() else None
    r['curve_correlation']=float(np.corrcoef(y[ok],p[ok])[0,1]) if np.std(p[ok])>1e-10 else None
    r['peak_error_kw']=float(p[ok].max()-y[ok].max())
    # Isolated excursion when both valid immediate neighbours return to similar power.
    isolated=ok[1:-1]&ok[:-2]&ok[2:]&(np.abs(p[2:]-p[:-2])<.2)&(p[1:-1]-np.maximum(p[:-2],p[2:])>.5)
    r['one_minute_spike_count']=int(isolated.sum())
    return r


def key(m):
    return (m['mae'],m['generation_mae'],m['rmse'],m['daily_absolute_energy_bias_kwh'])


def deep_fit(method,d,n):
    out=HERE/'training'/method;out.mkdir(parents=True,exist_ok=True)
    train=pack(d,'train',n);val=pack(d,'validation',n)
    new=torch.tensor(train['table'].batch_id.str.startswith('new').to_numpy(),device=DEVICE)
    inds_new=torch.where(new)[0];inds_old=torch.where(~new)[0]
    if not len(inds_new) or not len(inds_old):
        raise ValueError('Domain sampling requires new and historical TRAIN endpoints')
    on=train['y']>n['active_threshold_kw'];w=train['weight']
    # Classification balance reflects the same domain sampling distribution as minibatches.
    active_fraction=.75*float((w[new]*on[new]).sum()/w[new].sum())+.25*float((w[~new]*on[~new]).sum()/w[~new].sum())
    pos=torch.tensor((1-active_fraction)/active_fraction,device=DEVICE)
    for seed in SEEDS:
        completed=out/f'seed_{seed}_completed.json'
        if completed.exists():continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        m=model(method).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        best=None;stale=0;hist=[];started=time.perf_counter();ckpath=out/f'seed_{seed}_best_validation.pt'
        for epoch in range(1,51):
            tic=time.perf_counter();m.train();losses=[]
            # Stratified domain weighting, never use PV labels in input or future windows.
            sampled=torch.cat([inds_new[torch.randint(len(inds_new),(18000,),device=DEVICE)],inds_old[torch.randint(len(inds_old),(6000,),device=DEVICE)]])
            sampled=sampled[torch.randperm(len(sampled),device=DEVICE)]
            for ids in sampled.split(BATCH):
                a,b=m(windows(train,ids));yy=train['y'][ids]/n['target_scale_kw'];active=(train['y'][ids]>n['active_threshold_kw']).float();ww=w[ids]
                mag=F.softplus(b);p=a.sigmoid()*mag
                reg=F.smooth_l1_loss(p,yy,beta=.02,reduction='none')*(1+1.5*active)
                cls=F.binary_cross_entropy_with_logits(a,active,pos_weight=pos,reduction='none')
                amp=F.smooth_l1_loss(mag,yy,beta=.02,reduction='none')*active
                zero=F.smooth_l1_loss(p,torch.zeros_like(p),beta=.02,reduction='none')*(1-active)
                loss=((reg+.20*cls+.10*amp+.20*zero)*ww).sum()/ww.sum()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(loss.detach())
            pred,prob,mag=predict(m,val,n);met=metrics(val['table'],pred,prob,n);current=key(met)
            hist.append(dict(epoch=epoch,train_loss=float(torch.stack(losses).mean()),seconds=time.perf_counter()-tic,**met))
            if best is None or current<best:
                best=current;stale=0
                torch.save(dict(method=method,seed=seed,epoch=epoch,model_config=m.config.to_dict(),state_dict=m.state_dict(),validation=met),ckpath)
            else:stale+=1
            pd.DataFrame(hist).to_csv(out/f'seed_{seed}_epochs.csv',index=False)
            dump(HERE/'progress.json',dict(method=method,seed=seed,epoch=epoch,best_mae=best[0],best_generation_mae=best[1],stale=stale,epoch_seconds=hist[-1]['seconds']))
            if epoch==1 or epoch%10==0:print(f'{method} seed{seed} epoch{epoch} ValMAE{best[0]:.4f} GenMAE{best[1]:.4f}',flush=True)
            if stale>=8:break
        ck=torch.load(ckpath,map_location=DEVICE,weights_only=True);m.load_state_dict(ck['state_dict'])
        pred,prob,mag=predict(m,val,n);th=optimize_threshold(val['table'],prob,n)
        np.savez_compressed(out/f'seed_{seed}_validation.npz',power=pred,activity=prob,magnitude=mag)
        result=dict(method=method,seed=seed,best_epoch=ck['epoch'],threshold=th,parameters=sum(v.numel() for v in m.parameters()),checkpoint=str(ckpath),train_seconds=time.perf_counter()-started,**metrics(val['table'],pred,prob,n,th))
        dump(completed,result);print(f'COMPLETE {method} seed{seed} MAE{result["mae"]:.4f} GenMAE{result["generation_mae"]:.4f}',flush=True)
        del m;torch.cuda.empty_cache()
    del train,val

def run(table_path, normalizer_path, output, method='small_s4d'):
    global HERE
    import json
    import shutil
    HERE = Path(output)
    HERE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    d = pd.read_csv(table_path, parse_dates=['minute_start'], low_memory=False)
    n = json.loads(Path(normalizer_path).read_text(encoding='utf-8'))
    deep_fit(method, d, n)
    candidates = [json.loads((HERE/'training'/method/f'seed_{seed}_completed.json').read_text(encoding='utf-8')) for seed in SEEDS]
    selected = min(candidates, key=key)
    dump(HERE/'selection.json', {'method': method, 'seeds': [selected['seed']], 'threshold': selected['threshold'],
        'validation': selected, 'test_used_for_selection': False, 'rule': 'Validation MAE, generation MAE, RMSE, daily absolute energy bias'})
    release = HERE/'release'
    release.mkdir(exist_ok=True)
    shutil.copy2(normalizer_path, release/'normalizer.json')
    shutil.copy2(selected['checkpoint'], release/f"seed_{selected['seed']}_best_validation.pt")
    dump(release/'manifest.json', {'method': method, 'seeds': [selected['seed']], 'threshold': selected['threshold'], 'window': 240})
    m = model(method, torch.load(selected['checkpoint'], weights_only=True, map_location=DEVICE)['model_config']).to(DEVICE)
    m.load_state_dict(torch.load(selected['checkpoint'], weights_only=True, map_location=DEVICE)['state_dict'])
    rows = []
    for split in ['validation', 'test']:
        if not d.loc[d.split.eq(split), 'eligible'].any():
            continue
        packed = pack(d, split, n)
        power, activity, magnitude = predict(m, packed, n)
        rows.append({'split': split, **metrics(packed['table'], power, activity, n, selected['threshold'])})
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)
