import json
import numpy as np
import pandas as pd
import pytest
import torch

from grid_disaggregation.common import load_plan
from grid_disaggregation.resource.data import encode, metrics
from grid_disaggregation.resource.training import masked_bce
from grid_disaggregation.pv.data import limited_linear, quality_age
from grid_disaggregation.pv.metrics import score

def test_masked_unknown_labels_do_not_affect_loss():
    logits = torch.tensor([[.4, -.6, 1.]])
    mask = torch.tensor([[True, False, False]])
    a = masked_bce(logits, torch.zeros_like(logits), mask, torch.ones(3))
    b = masked_bce(logits, torch.tensor([[0., 1., 1.]]), mask, torch.ones(3))
    torch.testing.assert_close(a, b)

def test_gap_mask_and_observed_zero_are_distinct():
    norm = dict(median=[1., 1.], scale=[2., 2.], lower=[-5., -5.], upper=[5., 5.])
    at = pd.DatetimeIndex(['2026-01-01 00:14', '2026-01-01 00:15', '2026-01-01 00:16'])
    x, observed = encode(np.zeros((3, 2)), np.array([True, True, False]), at, norm)
    np.testing.assert_array_equal(observed, [False, True, False])
    np.testing.assert_array_equal(x[0, :2], [0, 0])
    np.testing.assert_array_equal(x[1, :2], [-.5, -.5])
    np.testing.assert_array_equal(x[:, -1], [1, 0, 0])

def test_short_offline_targets_only_fill_bounded_gaps():
    y = pd.Series([np.nan, 1., np.nan, np.nan, 4., np.nan, np.nan, np.nan, np.nan, 9., np.nan])
    filled = limited_linear(y, 3)
    np.testing.assert_allclose(filled.iloc[2:4], [2, 3])
    assert filled.iloc[[0, 5, 6, 7, 8, 10]].isna().all()

def test_freshness_uses_last_observation_only():
    d = pd.DataFrame({'minute_start': pd.date_range('2026-01-01 10:00', periods=3, freq='min'),
        'sample_time_main': ['2026-01-01 10:00:59', None, '2026-01-01 10:02:59'],
        'main_observed': [True, False, True], 'gap_indicator': [False]*3})
    ages = quality_age(d)
    np.testing.assert_allclose(ages[:, 0], [0, .25, 0])
    np.testing.assert_allclose(ages[:, 1], [1/300, 61/300, 1/300])

def test_unknown_resource_metrics_are_omitted():
    y = np.array([[0, 0, 0], [1, 0, 0]], float)
    mask = np.array([[1, 0, 0], [1, 0, 0]], bool)
    rows = metrics(y, mask, np.array([[.1, .9, .8], [.9, .1, .2]]), [.5]*3)
    assert len(rows)==1 and rows[0]['label']=='pv_active' and rows[0]['f1']==1

def test_energy_metrics_exclude_interpolated_targets():
    table = pd.DataFrame({'pv_observed': [True, True, False], 'pv_target_kw': [0., 2., 100.],
        'batch_id': ['one']*3, 'date': ['2026-01-01']*3})
    result = score(table, np.array([.1, 2., 0.]), np.array([.1, .9, .1]), {'active_threshold_kw': .02})
    assert result['n']==2
    assert result['mae']==pytest.approx(.05)
    assert result['energy_bias_kwh']==pytest.approx(.1/60)
    assert result['inactive_predicted_energy_kwh']==pytest.approx(.1/60)

@pytest.mark.parametrize('roles', [
    {'train': ['2026-01-01'], 'validation': ['2026-01-01']},
    {'train': ['2026-01-01', '2026-01-01']},
])
def test_duplicate_date_roles_rejected(tmp_path, roles):
    p = tmp_path/'plan.json'
    p.write_text(json.dumps({'batches': [{'dates': roles}]}))
    with pytest.raises(ValueError):
        load_plan(p)
