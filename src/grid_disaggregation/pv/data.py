"""Source selection is configured explicitly; targets never enter model inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..common import dump, load_plan
def quality_age(g):
    measured=g.main_observed.to_numpy(bool)&~g.gap_indicator.to_numpy(bool)
    at=pd.to_datetime(g.minute_start);sample=pd.to_datetime(g.sample_time_main,format='mixed')
    ages=[];stale=[];last_sample=None;last_index=None
    for i,(t,s,v) in enumerate(zip(at,sample,measured)):
        if v:
            last_index=i
            if pd.notna(s):last_sample=s
        ages.append(min(4,i-last_index if last_index is not None else 4)/4)
        stale.append(min(300,max(0,(t+pd.Timedelta('1min')-last_sample).total_seconds()))/300 if last_sample is not None else 1.)
    return np.column_stack([ages,stale]).astype(np.float32)

def limited_linear(s,limit=3):
    missing=s.isna()
    lengths=missing.groupby((missing!=missing.shift()).cumsum()).transform('sum')
    candidate=s.interpolate(method='linear',limit_area='inside')
    return s.where(~missing|lengths.gt(limit),candidate)

def read_source(path, fields, asynchronous=False, role='main'):
    d = pd.read_csv(path, low_memory=False)
    d[fields] = d[fields].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if asynchronous:
        d['sample_time'] = pd.to_datetime(d.data_time, utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        d['arrival_time'] = pd.to_datetime(d.report_time, utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
        d = d.sort_values('arrival_time', kind='stable').drop_duplicates('sample_time', keep='first')
        clock = 'arrival_time' if role=='main' else 'sample_time'
        d['minute_start'] = d[clock].dt.floor('min')
        d = d.sort_values(clock, kind='stable').groupby('minute_start').tail(1)
    else:
        d['minute_start'] = pd.to_datetime(d.minute_start)
        if d.minute_start.duplicated().any() or not d.minute_start.eq(d.minute_start.dt.floor('min')).all():
            raise ValueError('Natural-minute source must have unique minute_start values')
        d['sample_time'] = d.minute_start
    return d.set_index('minute_start').sort_index()

def prepare(plan_path, fields_path, output):
    plan = load_plan(plan_path)
    fields = json.loads(Path(fields_path).read_text(encoding='utf-8'))['fields']
    blocks = []
    date_roles = {}
    for spec in plan['batches']:
        if not spec.get('pv'):
            continue
        roles = {d: role for role, dates in spec['dates'].items() for d in dates}
        for date, role in roles.items():
            if date in date_roles and date_roles[date]!=role:
                raise ValueError('A calendar date cannot span training and evaluation')
            date_roles[date] = role
        asynchronous = spec.get('async', False)
        main = read_source(spec['main'], fields, asynchronous, 'main')
        pv = read_source(spec['pv'], fields, asynchronous, 'pv')
        axis = pd.date_range(max(main.index.min(), pv.index.min()), min(main.index.max(), pv.index.max()), freq='min')
        main, pv = main.reindex(axis), pv.reindex(axis)
        d = pd.DataFrame({'minute_start': axis, 'batch_id': spec['id'], 'date': axis.strftime('%Y-%m-%d')})
        d['split'] = d.date.map(roles).fillna('excluded')
        d['main_observed'] = main.TotW_MA.notna().to_numpy() if asynchronous else main.coverage_ratio.ge(.999).to_numpy()
        d['pv_observed'] = pv.TotW_MA.notna().to_numpy() if asynchronous else (pv.coverage_ratio.ge(.999)&pv.TotW_MA.notna()).to_numpy()
        if asynchronous:
            for source, key in [(main, 'main_observed'), (pv, 'pv_observed')]:
                suspect = source[['PhV_phs'+s for s in 'ABC']].abs().lt(1).all(axis=1).to_numpy()
                d.loc[suspect, key] = False
        d['sample_time_main'] = main.sample_time.to_numpy()
        d['raw_pv_TotW_MA'] = pv.TotW_MA.to_numpy()
        d['gap_indicator'] = (axis.hour==0)&(axis.minute<15)
        columns = {}
        for field in fields:
            raw = main[field].to_numpy()
            columns['raw_main_'+field] = raw
            columns['source_valid_'+field] = d.main_observed.to_numpy(bool) & np.isfinite(raw)
            columns['main_'+field] = np.full(len(d), np.nan)
        columns.update(pv_target_kw=np.full(len(d), np.nan), measurement_age_scaled=np.full(len(d), np.nan),
                       sample_age_scaled=np.full(len(d), np.nan), segment_id=np.full(len(d), None, object))
        d = pd.concat([d, pd.DataFrame(columns, index=d.index)], axis=1)
        for number, (_, group) in enumerate(d.groupby((d.split!=d.split.shift()).cumsum())):
            ids = group.index
            role = group.split.iloc[0]
            if role=='excluded':
                continue
            if role!='train':
                d.loc[ids[:240], 'split'] = 'buffer'
                ids = ids[240:]
            if not len(ids):
                continue
            d.loc[ids, 'segment_id'] = spec['id']+'_'+str(number)
            gap = d.loc[ids, 'gap_indicator'].to_numpy(bool)
            valid = d.loc[ids, ['source_valid_'+f for f in fields]].to_numpy(bool) & ~gap[:, None]
            raw = d.loc[ids, ['raw_main_'+f for f in fields]].to_numpy(float)
            filled = pd.DataFrame(np.where(valid, raw, np.nan)).ffill(limit=3).to_numpy()
            filled[gap] = np.nan
            for k, field in enumerate(fields):
                d.loc[ids, 'main_'+field] = filled[:, k]
            target = pd.Series(np.where(d.loc[ids, 'pv_observed'], np.maximum(-d.loc[ids, 'raw_pv_TotW_MA'], 0), np.nan), index=ids)
            d.loc[ids, 'pv_target_kw'] = limited_linear(target, 3).to_numpy()
            d.loc[ids, ['measurement_age_scaled', 'sample_age_scaled']] = quality_age(d.loc[ids])
        d['pv_imputed'] = ~d.pv_observed & d.pv_target_kw.notna()
        d['eligible'] = d.pv_target_kw.notna() & d.main_TotW_MA.notna() & ~d.gap_indicator & d.segment_id.notna()
        blocks.append(d)
    data = pd.concat(blocks, ignore_index=True)
    train = data[data.split.eq('train')]
    new = train[train.batch_id.str.startswith('new')]
    if new.empty:
        raise ValueError('At least one TRAIN domain id must start with new for domain-weighted training')
    x = new[['main_'+f for f in fields]].to_numpy(float)
    center = np.nanmedian(x, axis=0)
    iqr = np.nanquantile(x, .75, axis=0)-np.nanquantile(x, .25, axis=0)
    scale = np.where(iqr>1e-8, iqr, np.nanstd(x, axis=0))
    scale = np.where(scale>1e-8, scale, 1.)
    lo, hi = np.nanquantile(x, [.001, .999], axis=0)
    xx = train[['main_'+f for f in fields]].to_numpy(float)
    ll, hh = np.nanquantile(xx, [.001, .999], axis=0)
    night = train[train.pv_observed & ((train.minute_start.dt.hour<5)|(train.minute_start.dt.hour>=22))]
    if night.empty:
        raise ValueError('Night-time TRAIN references are required')
    threshold = float(np.quantile(night.pv_target_kw, .995)+1e-6)
    active = new.loc[new.pv_observed & new.pv_target_kw.gt(threshold), 'pv_target_kw']
    target_scale = float(active.quantile(.95))
    if not np.isfinite(center).all() or not np.isfinite(target_scale) or target_scale<=0:
        raise ValueError('Insufficient finite TRAIN inputs or active PV targets')
    norm = dict(fields=fields, center=center.tolist(), scale=scale.tolist(), lower=np.minimum(lo,ll).tolist(), upper=np.maximum(hi,hh).tolist(),
        active_threshold_kw=threshold, target_scale_kw=target_scale, window=240, input_channels=116,
        channel_order=['main_'+f for f in fields]+['observed_'+f for f in fields]+['measurement_mask','gap_indicator','measurement_age_scaled','sample_age_scaled'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'normalizer.json', norm)
    data.to_csv(out/'training_table.csv', index=False)
    return data, norm
