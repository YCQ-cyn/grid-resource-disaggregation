import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from grid_disaggregation.resource.models import ModelConfig, TCNLSTMClassifier
from grid_disaggregation.resource.inference import OnlineResource
from grid_disaggregation.pv.models import PVModelConfig
from grid_disaggregation.pv.state_space import SmallCausalS4D
from grid_disaggregation.pv.inference import OnlinePV

FIELDS = json.loads((Path(__file__).parents[1]/'config/fields.json').read_text())['fields']

def test_resource_replay_matches_encoded_window_and_rejects_future(tmp_path):
    torch.manual_seed(5)
    cfg = ModelConfig(input_channels=114)
    model = TCNLSTMClassifier(cfg).eval()
    checkpoint = tmp_path/'selected.pt'
    torch.save({'config': cfg.to_dict(), 'state_dict': model.state_dict()}, checkpoint)
    norm = {'fields': FIELDS, 'median': [0]*56, 'scale': [1]*56, 'lower': [-1000]*56, 'upper': [1000]*56}
    normalizer = tmp_path/'normalizer.json'
    normalizer.write_text(json.dumps(norm))
    runner = OnlineResource(checkpoint, normalizer)
    record = {field: 1. for field in FIELDS}
    for at in pd.date_range('2026-01-01 10:00', periods=125, freq='min'):
        result = runner.update(at, record)
    with torch.no_grad():
        expected = model(torch.tensor(np.asarray(runner.history).T[None])).sigmoid().numpy()[0]
    actual = [result[name+'_probability'] for name in ['pv_active','energy_above_5kw','charger_above_1kw']]
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    last = runner.last
    with pytest.raises(ValueError):
        runner.update(last+pd.Timedelta('1min'), {**record, 'sample_time': last+pd.Timedelta('2min')})
    assert runner.last==last

def test_pv_replay_matches_dual_head_and_rejects_duplicate(tmp_path):
    torch.manual_seed(7)
    cfg = PVModelConfig(input_channels=116)
    model = SmallCausalS4D(cfg).eval()
    torch.save({'model_config': cfg.to_dict(), 'state_dict': model.state_dict()}, tmp_path/'seed_7_best_validation.pt')
    norm = {'fields': FIELDS, 'center': [0]*56, 'scale': [1]*56, 'lower': [-1000]*56,
        'upper': [1000]*56, 'target_scale_kw': 4.}
    (tmp_path/'normalizer.json').write_text(json.dumps(norm))
    (tmp_path/'manifest.json').write_text(json.dumps({'method':'small_s4d', 'seeds':[7]}))
    runner = OnlinePV(tmp_path)
    record = {field: 1. for field in FIELDS}
    for at in pd.date_range('2026-01-01 10:00', periods=245, freq='min'):
        result = runner.update(at, record)
    h = np.asarray(runner.history)[-240:]
    with torch.no_grad():
        a, b = model(torch.tensor(h.T[None]))
        expected = float((a.sigmoid()*torch.nn.functional.softplus(b)*4).item())
    assert result['pv_power_kw']==pytest.approx(expected, abs=1e-7)
    size = len(runner.history)
    with pytest.raises(ValueError):
        runner.update(at, record)
    assert len(runner.history)==size and runner.last==at
