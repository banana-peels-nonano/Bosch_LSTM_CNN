import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Conv layers learn short local motion patterns (a stride, an arm swing)
    across all 15 channels; the LSTM then models how those patterns evolve
    over the 2 s window. Input: (B, T=100, C=15)."""

    def __init__(self, c_in=15, n_cls=12, conv=64, hidden=64, p=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(c_in, conv, 5, padding=2), nn.BatchNorm1d(conv), nn.ReLU(),
            nn.Conv1d(conv, conv, 5, padding=2), nn.BatchNorm1d(conv), nn.ReLU(),
            nn.MaxPool1d(2),               # 100 -> 50 time steps
            nn.Dropout(p),
        )
        self.lstm = nn.LSTM(conv, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, n_cls))

    def forward(self, x):
        z = self.cnn(x.transpose(1, 2))    # (B, conv, 50)
        out, (h, _) = self.lstm(z.transpose(1, 2))
        return self.head(h[-1])            # last hidden state -> logits


class CNNOnly(nn.Module):
    """Ablation: same conv stack, global average pooling instead of LSTM."""

    def __init__(self, c_in=15, n_cls=12, conv=64, p=0.5):
        super().__init__()
        self.cnn = CNNLSTM(c_in, n_cls, conv).cnn
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(conv, n_cls))

    def forward(self, x):
        z = self.cnn(x.transpose(1, 2)).mean(-1)
        return self.head(z)


class LSTMOnly(nn.Module):
    """Ablation: LSTM directly on raw (normalised) samples."""

    def __init__(self, c_in=15, n_cls=12, hidden=64, p=0.5):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, n_cls))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


MODELS = {"cnn_lstm": CNNLSTM, "cnn": CNNOnly, "lstm": LSTMOnly}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
