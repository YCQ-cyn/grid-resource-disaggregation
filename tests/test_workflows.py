import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from grid_disaggregation.pv.data import prepare
from grid_disaggregation.resource.data import prepare as prepare_resource

FIELDS = json.loads((Path(__file__).parents[1]/'config/fields.json').read_text())['fields']

def source(tmp_path, name, date, pv=False):
    at = pd.date_range(date, periods=720, freq='min')
    table = pd.DataFrame({field: np.ones(len(at)) for field in FIELDS})
    table['minute_start'] = at
    table['coverage_ratio'] = 1.
    power = np.maximum(np.sin((np.arange(len(at))-360)/240), 0)*3
    table['TotW_MA'] = -power if pv else 5+np.sin(np.arange(len(at))/10)
    table.loc[20, 'coverage_ratio'] = .5
    p = tmp_path/(name+'.csv')
    table.to_csv(p, index=False)
    return p.name

def test_pv_preparation_resets_splits_and_fits_train_only(tmp_path):
    batches = []
    for name, date, split in [('history', '2025-12-31', 'train'), ('new_train', '2026-01-01', 'train'),
                              ('new_validation', '2026-01-02', 'validation'), ('new_test', '2026-01-03', 'test')]:
        batches.append({'id': name, 'main': source(tmp_path, name+'_main', date),
            'pv': source(tmp_path, name+'_pv', date, True), 'async': False, 'dates': {split: [date]}})
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'batches': batches}))
    field_file = tmp_path/'fields.json'
    field_file.write_text(json.dumps({'fields': FIELDS}))
    data, norm = prepare(plan, field_file, tmp_path/'outputs')
    assert norm['input_channels']==116 and len(norm['fields'])==56
    assert norm['target_scale_kw']>0
    assert data.loc[data.split.eq('buffer'), 'eligible'].sum()==0
    assert data.loc[data.split.eq('validation'), 'minute_start'].min()==pd.Timestamp('2026-01-02 04:00')
    x = data.loc[data.batch_id.eq('new_train')&data.minute_start.eq('2026-01-01 00:20')].iloc[0]
    assert not x.main_observed and not x.source_valid_TotW_MA
    assert np.isfinite(x.main_TotW_MA)
    assert x.pv_imputed and not x.pv_observed
    # Evaluation values are modified after the first build; TRAIN fitting is unchanged.
    for batch in batches[-2:]:
        file = tmp_path/batch['main']
        table = pd.read_csv(file)
        table['TotW_MA'] *= 100
        table.to_csv(file, index=False)
    _, second = prepare(plan, field_file, tmp_path/'second_outputs')
    assert second==norm

def test_resource_preparation_masks_absent_branches(tmp_path):
    main = source(tmp_path, 'main', '2026-01-01')
    pv = source(tmp_path, 'pv', '2026-01-01', True)
    plan = {'batches': [{'id': 'historical', 'main': str(tmp_path/main), 'pv': str(tmp_path/pv),
        'energy': None, 'charger': None, 'dates': {'train': ['2026-01-01']}}]}
    segments, norm = prepare_resource(plan, FIELDS, tmp_path/'resource')
    assert norm['input_channels']==114
    assert segments and not any(s['mask'][:, 1:].any() for s in segments)
    s = segments[0]
    at = np.flatnonzero(s['time']==pd.Timestamp('2026-01-01 00:20'))[0]
    assert not s['mask'][at].any()
