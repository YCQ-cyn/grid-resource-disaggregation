from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils import weight_norm


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 4
    output_labels: int = 3
    channels: tuple[int, ...] = (16, 16, 24, 32)
    kernel_size: int = 5
    dropout: float = 0.15
    lstm_hidden: int = 32
    lstm_layers: int = 1
    use_weight_norm: bool = False

    def to_dict(self) -> dict:
        output = asdict(self)
        output["channels"] = list(self.channels)
        return output

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        normalized = dict(value)
        normalized["channels"] = tuple(normalized["channels"])
        return cls(**normalized)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        conv1: nn.Module = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        conv2: nn.Module = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        if use_weight_norm:
            conv1 = weight_norm(conv1)
            conv2 = weight_norm(conv2)

        self.net = nn.Sequential(
            conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs
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
        return self.activation(self.net(x) + self.residual(x))


class TemporalConvNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        blocks = []
        for level, output_channels in enumerate(config.channels):
            input_channels = (
                config.input_channels
                if level == 0
                else config.channels[level - 1]
            )
            blocks.append(
                TemporalBlock(
                    n_inputs=input_channels,
                    n_outputs=output_channels,
                    kernel_size=config.kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                    use_weight_norm=config.use_weight_norm,
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNClassifier(nn.Module):
    """Causal TCN that classifies the state at the current endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tcn = TemporalConvNet(config)
        final_channels = config.channels[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(final_channels),
            nn.Dropout(config.dropout),
            nn.Linear(final_channels, config.output_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(x)
        return self.head(encoded[:, :, -1])


class TCNLSTMClassifier(nn.Module):
    """Causal TCN-LSTM that classifies the current endpoint state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tcn = TemporalConvNet(config)
        self.lstm = nn.LSTM(
            input_size=config.channels[-1],
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=(
                config.dropout if config.lstm_layers > 1 else 0.0
            ),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.lstm_hidden),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden, config.output_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(x).transpose(1, 2)
        sequence, _ = self.lstm(encoded)
        return self.head(sequence[:, -1, :])


def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    if model_name == "tcn":
        return TCNClassifier(config)
    if model_name == "tcn_lstm":
        return TCNLSTMClassifier(config)
    raise ValueError(f"未知模型：{model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
