"""Small causal S4D: same plain 32-wide, 24-state, two-block prior candidate.

Diagonal kernel follows https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
(Apache-2.0). No time pooling, bidirectional convolution, or temporal normalization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .models import PVOutputHeads


class S4DKernel(nn.Module):
    def __init__(self, width=32, state=24):
        super().__init__()
        self.log_dt = nn.Parameter(torch.rand(width)*(math.log(.1)-math.log(.001))+math.log(.001))
        self.C = nn.Parameter(torch.view_as_real(torch.randn(width,state//2,dtype=torch.cfloat)))
        self.log_A_real = nn.Parameter(torch.log(.5*torch.ones(width,state//2)))
        self.A_imag = nn.Parameter(math.pi*torch.arange(state//2).repeat(width,1))

    def forward(self, length):
        a = -self.log_A_real.exp()+1j*self.A_imag
        dt_a = a*self.log_dt.exp()[:,None]
        c = torch.view_as_complex(self.C.contiguous())*torch.expm1(dt_a)/a
        t = torch.arange(length,device=a.device)
        return 2*torch.einsum('hn,hnl->hl',c,torch.exp(dt_a[:,:,None]*t)).real


class PlainS4DBlock(nn.Module):
    def __init__(self, width=32, state=24, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.kernel = S4DKernel(width,state)
        self.skip = nn.Parameter(torch.randn(width))
        self.mix = nn.Conv1d(width,width,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x.transpose(1,2)).transpose(1,2)
        length = z.shape[-1]
        kernel = self.kernel(length)
        y = torch.fft.irfft(torch.fft.rfft(z,n=2*length)*torch.fft.rfft(kernel,n=2*length),n=2*length)[...,:length]
        return x+self.dropout(self.mix(F.gelu(y+z*self.skip[:,None])))


class SmallCausalS4D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(nn.Conv1d(config.input_channels,32,1),
            PlainS4DBlock(dropout=config.dropout),PlainS4DBlock(dropout=config.dropout))
        self.heads = PVOutputHeads(32,config.dropout)

    def forward(self, x):
        return self.heads(self.encoder(x)[:,:,-1])
