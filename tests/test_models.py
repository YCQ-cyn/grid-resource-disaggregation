import numpy as np
import torch

from grid_disaggregation.resource.models import ModelConfig, TCNLSTMClassifier
from grid_disaggregation.pv.models import PVModelConfig, CausalTCNLSTMRegressor
from grid_disaggregation.pv.state_space import SmallCausalS4D, S4DKernel, PlainS4DBlock

torch.set_num_threads(1)

def test_parameter_counts_and_independent_heads():
    resource = TCNLSTMClassifier(ModelConfig(input_channels=114)).eval()
    pv = SmallCausalS4D(PVModelConfig(input_channels=116)).eval()
    assert sum(p.numel() for p in resource.parameters()) == 38555
    assert sum(p.numel() for p in pv.parameters()) == 9314
    with torch.no_grad():
        logits = resource(torch.zeros(2, 114, 120))
        activity, magnitude = pv(torch.zeros(2, 116, 240))
    assert logits.shape == (2, 3)
    assert activity.shape == magnitude.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(activity).all() and torch.isfinite(magnitude).all()

def test_resource_prefix_is_causal():
    torch.manual_seed(7)
    m = TCNLSTMClassifier(ModelConfig(input_channels=114)).eval()
    x = torch.randn(1, 114, 120)
    with torch.no_grad():
        whole, _ = m.lstm(m.tcn(x).transpose(1, 2))
        prefix, _ = m.lstm(m.tcn(x[:, :, :70]).transpose(1, 2))
    torch.testing.assert_close(whole[:, :70], prefix, atol=1e-5, rtol=1e-4)

def test_s4d_fft_matches_real_causal_convolution():
    torch.manual_seed(11)
    block = PlainS4DBlock(width=4, state=8, dropout=0).eval()
    x = torch.randn(2, 4, 61)
    with torch.no_grad():
        z = block.norm(x.transpose(1, 2)).transpose(1, 2)
        kernel = block.kernel(61)
        real = torch.nn.functional.conv1d(torch.nn.functional.pad(z, (60, 0)), kernel.flip(-1)[:, None], groups=4)
        expected = x+block.mix(torch.nn.functional.gelu(real+z*block.skip[:, None]))
        actual = block(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

def test_s4d_prefix_is_causal():
    torch.manual_seed(13)
    m = SmallCausalS4D(PVModelConfig(input_channels=116)).eval()
    x = torch.randn(1, 116, 240)
    with torch.no_grad():
        whole = m.encoder(x)
        prefix = m.encoder(x[:, :, :140])
    torch.testing.assert_close(whole[:, :, :140], prefix, atol=2e-5, rtol=1e-4)

def test_tcn_lstm_power_is_nonnegative():
    m = CausalTCNLSTMRegressor(PVModelConfig(input_channels=116)).eval()
    with torch.no_grad():
        a, b = m(torch.randn(2, 116, 240))
        power = a.sigmoid()*torch.nn.functional.softplus(b)*10
    assert torch.isfinite(power).all() and (power>=0).all()
