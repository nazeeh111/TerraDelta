"""Shared causal temporal baselines and an observation-conditioned transition model."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .features import FEATURE_NAMES, QUALITY_COLUMNS


class ObservationTransition(nn.Module):
    """Compact recurrent occupancy/transition emissions, with masked state updates.

    Missing evidence never updates the state. Elapsed time, previous seasonal evidence,
    valid coverage and registration enter the learned transition; a separate decoder
    enforces dwell and abstention. Neither this class nor the decoder predicts causes.
    """

    def __init__(self, no_quality: bool = False, spatial_residual: bool = True):
        super().__init__()
        self.no_quality = no_quality
        self.spatial_residual = spatial_residual
        self.cell = nn.GRUCell(len(FEATURE_NAMES), 32)
        self.head = nn.Linear(32, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
        if self.no_quality:
            x = x.clone()
            x[:, :, QUALITY_COLUMNS] = 0
        hidden = torch.zeros((len(x), 32), device=x.device, dtype=x.dtype)
        previous = torch.zeros((len(x), 2), device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            next_hidden = self.cell(x[:, t], hidden)
            observed = usable[:, t, None]
            hidden = torch.where(observed, next_hidden, hidden)
            emission = self.head(hidden)
            # Residual spatial log odds protect a meaningful simple baseline.
            spatial = x[:, t, 0].clamp(0.01, 0.99)
            if self.spatial_residual:
                emission = torch.stack(
                    (emission[:, 0] + torch.logit(spatial), emission[:, 1]), dim=1
                )
            previous = torch.where(observed, emission, previous)
            outputs.append(previous)
        return torch.stack(outputs, dim=1)


class WatchLSTM(nn.Module):
    """WATCH's unidirectional MLP128/LSTM128x2/time-head architecture.

    Task adaptation: shared candidate features and supervised building occupancy
    replace WATCH's archaeological-site features and weak event labels. No claim to
    reproduce WATCH's original dataset/results. Source revision and MIT attribution
    are recorded in docs/baselines.md and licenses/WATCH-MIT.txt.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        for layer in self.head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1e-3)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
        # Causal forward-fill, not future interpolation/full-sequence normalization.
        previous = torch.zeros_like(x[:, 0])
        fixed = []
        for t in range(x.shape[1]):
            previous = torch.where(usable[:, t, None], x[:, t], previous)
            fixed.append(previous)
        out, _ = self.lstm(self.encoder(torch.stack(fixed, dim=1)))
        occupancy = self.head(out)
        return torch.cat((occupancy, torch.zeros_like(occupancy)), dim=-1)


def make_temporal(kind: str, *, spatial_residual: bool = True) -> nn.Module:
    if kind == "observation":
        return ObservationTransition(spatial_residual=spatial_residual)
    if kind == "no_quality":
        return ObservationTransition(no_quality=True, spatial_residual=spatial_residual)
    if kind == "watch_lstm":
        return WatchLSTM()
    raise ValueError(f"Unknown temporal model {kind}")


@torch.inference_mode()
def predict_temporal(
    model: nn.Module, x: np.ndarray, usable: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    model.eval()
    results = []
    for start in range(0, len(x), batch_size):
        logits = model(
            torch.tensor(x[start : start + batch_size]),
            torch.tensor(usable[start : start + batch_size]),
        )
        results.append(logits[:, :, 0].sigmoid().numpy())
    return np.concatenate(results) if results else np.empty(usable.shape, dtype="float32")


def load_temporal(path: Path, kind: str = "observation") -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "terradelta-temporal-v1":
        raise ValueError("Unsupported temporal checkpoint format")
    model = make_temporal(
        kind, spatial_residual=state.get("model_config", {}).get("spatial_residual", True)
    )
    model.load_state_dict(state["models"][kind])
    return model.eval()


def hmm_filter(p: np.ndarray, usable: np.ndarray, transition: float = 0.015) -> np.ndarray:
    """Two-state causal HMM filtering; missing observations skip emission updates."""
    result = np.empty_like(p)
    posterior = np.full(len(p), 0.5)
    for t in range(p.shape[1]):
        prior = posterior * (1 - transition) + (1 - posterior) * transition
        emission = np.clip(p[:, t], 0.01, 0.99)
        next_p = (prior * emission) / (prior * emission + (1 - prior) * (1 - emission))
        posterior = np.where(usable[:, t], next_p, posterior)
        result[:, t] = posterior
    return result


def causal_smooth(p: np.ndarray, usable: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    result = np.empty_like(p)
    previous = np.full(len(p), 0.5)
    for t in range(p.shape[1]):
        previous = np.where(usable[:, t], alpha * p[:, t] + (1 - alpha) * previous, previous)
        result[:, t] = previous
    return result


def causal_ted(x: np.ndarray, usable: np.ndarray, lookback: int = 3) -> np.ndarray:
    """WATCH-inspired past-only median embedding distance on common RGB/spatial features.

    This baseline flags unsigned change; the spatial occupancy increase/decrease
    supplies direction. No future missing-month interpolation is performed.
    """
    result = np.zeros(usable.shape, dtype="float32")
    for i in range(len(x)):
        history = []
        for t in range(x.shape[1]):
            if not usable[i, t]:
                continue
            vector = x[i, t, [0, 1, 8, 9, 10]]
            if history:
                result[i, t] = np.sqrt(
                    np.mean((vector - np.median(history[-lookback:], axis=0)) ** 2)
                )
            history.append(vector)
    return result


def sn7_retrospective(p: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Controlled adaptation of SN7 fourth-place mean-refinement/persistence logic.

    Same local spatial backbone, not the original ten-model EfficientNet ensemble.
    Full-sequence mean and next-three support are RETROSPECTIVE by construction.
    End-of-sequence support follows upstream acceptance rather than inventing future
    evidence. Global candidate identity supplies overlap rejection, without copying
    future positions into a claimed online answer.
    """
    mean = (p * usable).sum(axis=1) / np.maximum(usable.sum(axis=1), 1)
    refined = 0.5 * p + 0.5 * mean[:, None]
    out = refined.copy()
    for t in range(p.shape[1] - 1):
        support = np.any((refined[:, t + 1 : t + 4] >= 0.5) & usable[:, t + 1 : t + 4], axis=1)
        out[:, t] = np.where(support, refined[:, t], np.minimum(refined[:, t], 0.49))
    return out
