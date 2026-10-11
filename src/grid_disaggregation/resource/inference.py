"""Past-only main-meter resource identification."""
from collections import deque
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .models import ModelConfig, TCNLSTMClassifier
from .data import encode, LABELS

class OnlineResource:
    def __init__(self, checkpoint, normalizer, thresholds=(.5, .5, .5), device='cpu'):
        self.norm = json.loads(Path(normalizer).read_text(encoding='utf-8'))
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model = TCNLSTMClassifier(ModelConfig.from_dict(ck['config'])).to(device).eval()
        self.model.load_state_dict(ck['state_dict'])
        self.thresholds = np.asarray(thresholds, float)
        self.device = device
        self.history = deque(maxlen=120)
        self.last = None

    def _append(self, at, record, complete):
        values = np.array([[record.get(f, np.nan) for f in self.norm['fields']]])
        x, measured = encode(values, np.array([complete], bool), pd.DatetimeIndex([at]), self.norm)
        self.history.append(x[0])
        self.last = at
        return bool(measured[0])

    def update(self, minute_start, record=None, complete=True):
        at = pd.Timestamp(minute_start)
        if at!=at.floor('min') or (self.last is not None and at<=self.last):
            raise ValueError('Supply strictly increasing completed natural minutes')
        if record is not None:
            for key in ['sample_time_main', 'sample_time']:
                if pd.notna(record.get(key)) and pd.Timestamp(record[key])>=at+pd.Timedelta('1min'):
                    raise ValueError('A source sample cannot be from the future')
        if self.last is not None:
            for missing in pd.date_range(self.last+pd.Timedelta('1min'), at-pd.Timedelta('1min'), freq='min'):
                self._append(missing, {}, False)
        observed = self._append(at, record or {}, complete and record is not None)
        history = np.asarray(self.history, np.float32)
        x = np.pad(history, ((120-len(history), 0), (0, 0)))
        with torch.inference_mode():
            p = self.model(torch.tensor(x.T[None], device=self.device)).sigmoid().cpu().numpy()[0]
        return {'minute_start': str(at), 'input_available': observed,
            **{name+'_probability': float(p[k]) for k, name in enumerate(LABELS)},
            **{name+'_state': int(p[k]>=self.thresholds[k]) for k, name in enumerate(LABELS)}}
