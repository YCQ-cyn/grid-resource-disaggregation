"""Exact-minute supervised labels and masked causal resource windows."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from .models import TCNLSTMClassifier, ModelConfig
from ..common import dump, sha

LABELS = ['pv_active', 'energy_above_5kw', 'charger_above_1kw']
WINDOW = 120
def read_raw(path):
    df = pd.read_csv(path, parse_dates=['minute_start'])
    assert df.minute_start.notna().all() and not df.minute_start.duplicated().any(), path
    assert df.minute_start.eq(df.minute_start.dt.floor('min')).all(), path
    return df.set_index('minute_start').sort_index()


def encode(raw, complete, index, norm):
    gap = (index.hour == 0) & (index.minute < 15)
    valid = np.isfinite(raw) & complete[:, None] & ~gap[:, None]
    filled = np.where(valid, raw, np.array(norm['median']))
    scaled = (np.clip(filled, norm['lower'], norm['upper'])-norm['median'])/norm['scale']
    scaled[~valid] = 0
    measurement = valid.any(axis=1)
    return np.column_stack([scaled, valid.astype(float), measurement, gap]).astype('float32'), measurement


def pack(segments, split, device):
    xx, yy, mm, ends, meta = [], [], [], [], []
    cursor = 0
    for s in segments:
        if s['split'] != split:
            continue
        xx.append(np.pad(s['x'], ((WINDOW-1, 0), (0, 0))))
        yy.append(np.pad(s['y'], ((WINDOW-1, 0), (0, 0))))
        mm.append(np.pad(s['mask'], ((WINDOW-1, 0), (0, 0))))
        take = np.flatnonzero(s['mask'].any(axis=1))
        ends.extend((cursor+WINDOW-1+take).tolist())
        meta.extend([(s['batch'], str(s['time'][i]), float(s['p_main'][i])) for i in take])
        cursor += len(s['x'])+WINDOW-1
    return {'x': torch.tensor(np.concatenate(xx), device=device), 'y': torch.tensor(np.concatenate(yy), device=device),
            'mask': torch.tensor(np.concatenate(mm), device=device), 'ends': np.array(ends), 'meta': meta}


def get_batch(data, ends):
    e = torch.as_tensor(ends, device=data['x'].device)
    offsets = torch.arange(-WINDOW+1, 1, device=e.device)
    return data['x'][e[:, None]+offsets].transpose(1, 2), data['y'][e], data['mask'][e]


def predict(model, data, batch_size=256):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(data['ends']), batch_size):
            x, _, _ = get_batch(data, data['ends'][start:start+batch_size])
            result.append(model(x).sigmoid().cpu().numpy())
    return np.concatenate(result)


def metrics(y, mask, p, thresholds):
    rows = []
    for k, label in enumerate(LABELS):
        ok = mask[:, k]
        a, prob = y[ok, k], p[ok, k]
        if not len(a):
            continue
        pred = prob >= thresholds[k]
        both = len(np.unique(a)) == 2
        tn, fp, fn, tp = confusion_matrix(a, pred, labels=[0, 1]).ravel()
        rows.append({'label': label, 'n': len(a), 'positive': int(a.sum()),
                     'pr_auc_ap': float(average_precision_score(a, prob)) if both else None,
                     'roc_auc': float(roc_auc_score(a, prob)) if both else None,
                     'f1': float(f1_score(a, pred, zero_division=0)),
                     'precision': float(precision_score(a, pred, zero_division=0)),
                     'recall': float(recall_score(a, pred, zero_division=0)),
                     'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'threshold': float(thresholds[k])})
    return rows


def truth(data):
    return data['y'][data['ends']].cpu().numpy(), data['mask'][data['ends']].cpu().numpy()

def prepare(plan, fields, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    train_values = []
    night = []
    date_roles = {}
    for spec in plan['batches']:
        if spec.get('async'):
            continue  # Resource labels require exact natural-minute observations.
        roles = {date: role for role, dates in spec['dates'].items() for date in dates}
        for date, role in roles.items():
            if date in date_roles and date_roles[date] != role:
                raise ValueError('A physical calendar date cannot span training and evaluation')
            date_roles[date] = role
        main = read_raw(spec['main'])
        axis = pd.date_range(main.index.min().normalize(), main.index.max().normalize()+pd.Timedelta(hours=23, minutes=59), freq='min')
        source = main.reindex(axis)
        raw = source[fields].to_numpy(float)
        complete = source.coverage_ratio.ge(.999).to_numpy()
        powers = np.full((len(axis), 3), np.nan)
        known = np.zeros((len(axis), 3), bool)
        for k, role in enumerate(['pv', 'energy', 'charger']):
            if not spec.get(role):
                continue
            branch = read_raw(spec[role]).reindex(axis)
            powers[:, k] = branch.TotW_MA.to_numpy(float)
            known[:, k] = branch.coverage_ratio.ge(.999) & np.isfinite(powers[:, k])
        if not spec.get('energy_supervised', True):
            known[:, 1] = False
        train = np.isin(axis.strftime('%Y-%m-%d'), spec['dates'].get('train', []))
        train_values.append(np.where(complete[:, None] & train[:, None], raw, np.nan))
        take = train & known[:, 0] & ((axis.hour >= 20) | (axis.hour < 4))
        night.extend(np.maximum(-powers[take, 0], 0).tolist())
        frames.append((spec['id'], axis, raw, complete, powers, known, roles))
    if not train_values or not night:
        raise ValueError('TRAIN requires main observations and observed night-time PV references')
    x = np.concatenate(train_values)
    median = np.nanmedian(x, axis=0)
    q25, q75 = np.nanquantile(x, [.25, .75], axis=0)
    norm = dict(fields=fields, median=median.tolist(), scale=np.where(q75-q25 > 1e-6, q75-q25, 1).tolist(),
                lower=np.nanquantile(x, .001, axis=0).tolist(), upper=np.nanquantile(x, .999, axis=0).tolist(), window=120, input_channels=114)
    if not np.isfinite(median).all():
        raise ValueError('Every input field needs a finite TRAIN observation')
    threshold = float(np.quantile(night, .995)+.001)
    segments = []
    for batch, axis, raw, complete, powers, known, roles in frames:
        encoded, measured = encode(raw, complete, axis, norm)
        target = np.column_stack([np.maximum(-powers[:, 0], 0)>threshold, powers[:, 1]>=5, powers[:, 2]>=1]).astype(np.float32)
        mask = known & measured[:, None]
        for split in ['train', 'validation', 'internal_test', 'domain_test']:
            positions = np.flatnonzero(np.array([roles.get(d)==split for d in axis.strftime('%Y-%m-%d')]))
            for block in np.split(positions, np.flatnonzero(np.diff(positions)>1)+1):
                if len(block):
                    segments.append(dict(batch=batch, split=split, x=encoded[block], y=target[block], mask=mask[block],
                        time=axis[block], p_main=raw[block, fields.index('TotW_MA')], powers=powers[block]))
    dump(out/'normalizer.json', norm)
    dump(out/'target_definitions.json', {'pv_threshold_kw': threshold, 'energy_threshold_kw': 5, 'charger_threshold_kw': 1})
    return segments, norm

def prepare_auxiliary(plan, norm):
    train_dates = {d for b in plan['batches'] for d in b['dates'].get('train', [])}
    forbidden = {d for b in plan['batches'] for role, dates in b['dates'].items() if role!='train' for d in dates}
    forbidden.update(plan.get('excluded_shared_dates', []))
    if not train_dates:
        raise ValueError('No training calendar dates')
    groups = {}
    for source in plan.get('auxiliary_chargers', []):
        frame = read_raw(source['path'])
        dates = frame.index.strftime('%Y-%m-%d')
        allowed = (dates<=max(train_dates)) & ~np.isin(dates, list(forbidden))
        groups.setdefault(source['device'], []).append(frame.loc[allowed])
    segments = []
    for device, parts in groups.items():
        joined = pd.concat(parts).sort_index()
        if joined.empty:
            continue
        for at in joined.index[joined.index.duplicated()].unique():
            rows = joined.loc[[at], norm['fields']+['coverage_ratio']].to_numpy(float)
            if not np.allclose(rows, rows[0], equal_nan=True, atol=1e-8):
                raise ValueError('Conflicting duplicate charger observations')
        joined = joined[~joined.index.duplicated(keep='first')]
        allowed_dates = set(joined.index.strftime('%Y-%m-%d'))
        axis = pd.date_range(joined.index.min().normalize(), joined.index.max().normalize()+pd.Timedelta(hours=23, minutes=59), freq='min')
        frame = joined.reindex(axis)
        encoded, measured = encode(frame[norm['fields']].to_numpy(float), frame.coverage_ratio.ge(.999).to_numpy(), axis, norm)
        power = frame.TotW_MA.to_numpy(float)
        y = np.zeros((len(axis), 3), np.float32)
        y[:, 2] = power>=1
        mask = np.zeros_like(y, bool)
        mask[:, 2] = measured & np.isfinite(power)
        ids = np.flatnonzero(np.isin(axis.strftime('%Y-%m-%d'), list(allowed_dates)))
        for block in np.split(ids, np.flatnonzero(np.diff(ids)>1)+1):
            if len(block):
                segments.append(dict(batch='aux_'+device, split='train', x=encoded[block], y=y[block], mask=mask[block], time=axis[block],
                    p_main=power[block], powers=np.column_stack([np.full((len(block), 2), np.nan), power[block]])))
    return segments
