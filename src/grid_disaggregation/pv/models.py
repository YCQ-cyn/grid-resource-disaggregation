from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PVModelConfig:
    input_channels: int = 7
    channels: tuple[int, ...] = (16, 16, 24, 24, 32)
    kernel_size: int = 5
    dropout: float = 0.10
    lstm_hidden: int = 32
    lstm_layers: int = 1

    def to_dict(self) -> dict:
        output = asdict(self)
        output["channels"] = list(self.channels)
        return output

    @classmethod
    def from_dict(cls, value: dict) -> "PVModelConfig":
        normalized = dict(value)
        normalized["channels"] = tuple(normalized["channels"])
        return cls(**normalized)


class Chomp1d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return x
        return x[:, :, : -self.size].contiguous()


class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(x) + self.residual(x))


class CausalTCNEncoder(nn.Module):
    def __init__(self, config: PVModelConfig) -> None:
        super().__init__()
        blocks = []
        for level, output_channels in enumerate(config.channels):
            input_channels = (
                config.input_channels
                if level == 0
                else config.channels[level - 1]
            )
            blocks.append(
                CausalResidualBlock(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=config.kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PVOutputHeads(nn.Module):
    def __init__(self, input_features: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Dropout(dropout),
        )
        self.activity = nn.Linear(input_features, 1)
        self.magnitude = nn.Linear(input_features, 1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        return (
            self.activity(shared).squeeze(-1),
            self.magnitude(shared).squeeze(-1),
        )


class CausalTCNRegressor(nn.Module):
    def __init__(self, config: PVModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CausalTCNEncoder(config)
        self.heads = PVOutputHeads(config.channels[-1], config.dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.heads(encoded[:, :, -1])


class CausalTCNLSTMRegressor(nn.Module):
    def __init__(self, config: PVModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CausalTCNEncoder(config)
        self.lstm = nn.LSTM(
            input_size=config.channels[-1],
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=(
                config.dropout if config.lstm_layers > 1 else 0.0
            ),
        )
        self.heads = PVOutputHeads(config.lstm_hidden, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x).transpose(1, 2)
        sequence, _ = self.lstm(encoded)
        return self.heads(sequence[:, -1, :])


def build_pv_model(
    model_name: str,
    config: PVModelConfig,
) -> nn.Module:
    if model_name == "causal_tcn":
        return CausalTCNRegressor(config)
    if model_name == "causal_tcn_lstm":
        return CausalTCNLSTMRegressor(config)
    raise ValueError(f"未知光伏分离模型：{model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
