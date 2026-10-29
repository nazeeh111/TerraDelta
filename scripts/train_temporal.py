"""Train only train regions, calibrate occupancy only development regions."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch.nn import functional as F

from terradelta.data import sha256
from terradelta.features import FEATURE_NAMES
from terradelta.temporal import make_temporal, predict_temporal


def load_split(root, regions):
    output = []
    for aoi in regions:
        f = np.load(root / aoi / "features.npz")
        y = np.load(root / aoi / "supervision.npz")
        output.append((aoi, f["x"], f["usable"], y["y"], y["supervised"]))
    return output


def calibrate(probability, truth, valid):
    scores = logit(np.clip(probability[valid], 0.001, 0.999))
    labels = truth[valid]

    def objective(params):
        z = np.exp(params[0]) * scores + params[1]
        return float(np.mean(np.logaddexp(0, z) - labels * z) + 1e-4 * np.dot(params, params))

    fit = minimize(objective, np.array([0.0, 0.0]), method="BFGS")
    a, b = float(np.exp(fit.x[0])), float(fit.x[1])
    calibrated = expit(a * scores + b)
    bins = []
    for lower in np.linspace(0, 0.9, 10):
        mask = (calibrated >= lower) & (calibrated < lower + 0.1 + 1e-8)
        if mask.any():
            bins.append(
                {
                    "lower": float(lower),
                    "n": int(mask.sum()),
                    "mean_score": float(calibrated[mask].mean()),
                    "observed_presence": float(labels[mask].mean()),
                }
            )
    return {
        "scale": a,
        "bias": b,
        "n": int(valid.sum()),
        "success": bool(fit.success),
        "brier": float(np.mean((calibrated - labels) ** 2)),
        "reliability_bins": bins,
        "meaning": "occupancy calibration on two development AOIs; not event correctness or arbitrary-shift guarantee",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    train = load_split(args.features, config["train"])
    dev = load_split(args.features, config["development"])
    # Equal per-region contribution; oversample real transition sequences without
    # changing development prevalence. No fabricated imagery or observations.
    models = {}
    traces = {}
    calibration = {}
    summary = {}
    for kind in ["observation", "no_quality", "watch_lstm"]:
        model = make_temporal(kind)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        best = float("inf")
        best_state = None
        trace = []
        started = time.perf_counter()
        for step in range(args.steps):
            _, x, usable, y, supervised = train[step % len(train)]
            changed = np.flatnonzero(np.any(np.diff(y, axis=1) != 0, axis=1))
            stable = np.flatnonzero(~np.any(np.diff(y, axis=1) != 0, axis=1))
            selected = np.concatenate(
                [
                    rng.choice(changed if len(changed) else np.arange(len(x)), 64),
                    rng.choice(stable if len(stable) else np.arange(len(x)), 64),
                ]
            )
            tx = torch.tensor(x[selected])
            mask = torch.tensor(usable[selected] & supervised[selected])
            ty = torch.tensor(y[selected])
            model.train()
            prediction = model(tx, torch.tensor(usable[selected]))
            loss = (
                F.binary_cross_entropy_with_logits(prediction[:, :, 0], ty, reduction="none") * mask
            ).sum() / mask.sum().clamp_min(1)
            if kind != "watch_lstm":
                # Auxiliary target is a change between successive USABLE annotations,
                # not a fabricated exact transition date inside a cloud gap.
                transitions = torch.zeros_like(ty)
                transition_mask = torch.zeros_like(mask)
                for i in range(len(ty)):
                    good = torch.nonzero(mask[i], as_tuple=True)[0]
                    if len(good) > 1:
                        transitions[i, good[1:]] = (ty[i, good[1:]] != ty[i, good[:-1]]).float()
                        transition_mask[i, good[1:]] = True
                aux = F.binary_cross_entropy_with_logits(
                    prediction[:, :, 1],
                    transitions,
                    reduction="none",
                    pos_weight=torch.tensor(10.0),
                )
                loss = loss + 0.2 * (aux * transition_mask).sum() / transition_mask.sum().clamp_min(
                    1
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if (step + 1) % 100 == 0 or step == 0:
                model.eval()
                dev_losses = []
                for _, dx, du, dy, ds in dev:
                    # Fixed deterministic whole sequences, capped sample count for runtime.
                    ix = np.arange(0, len(dx), max(1, len(dx) // 2000))[:2000]
                    dp = predict_temporal(model, dx[ix], du[ix])
                    dm = du[ix] & ds[ix]
                    pp = np.clip(dp[dm], 1e-6, 1 - 1e-6)
                    truth = dy[ix][dm]
                    dev_losses.append(
                        float(np.mean(-truth * np.log(pp) - (1 - truth) * np.log(1 - pp)))
                    )
                value = float(np.mean(dev_losses))
                trace.append(
                    {"step": step + 1, "train_loss": float(loss.detach()), "development_bce": value}
                )
                print(
                    kind, trace[-1], "seconds", round(time.perf_counter() - started, 1), flush=True
                )
                if value < best:
                    best = value
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        models[kind] = best_state
        traces[kind] = trace
        # Fit one transform globally over development AOIs; never per confirmation region.
        ps = []
        ys = []
        ms = []
        for _, dx, du, dy, ds in dev:
            ps.append(predict_temporal(model, dx, du).ravel())
            ys.append(dy.ravel())
            ms.append((du & ds).ravel())
        calibration[kind] = calibrate(np.concatenate(ps), np.concatenate(ys), np.concatenate(ms))
        summary[kind] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": time.perf_counter() - started,
            "selected_development_bce": best,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "terradelta-temporal-v1",
            "models": models,
            "feature_names": FEATURE_NAMES,
            "config_sha256": sha256(args.config),
            "train_regions": config["train"],
            "development_regions": config["development"],
            "seed": config["seed"],
        },
        args.output,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps_per_method": args.steps,
        "train_regions": config["train"],
        "development_regions": config["development"],
        "config_sha256": sha256(args.config),
        "weights_sha256": sha256(args.output),
        "feature_sha256": {
            aoi: sha256(args.features / aoi / "features.npz")
            for aoi in config["train"] + config["development"]
        },
        "models": summary,
        "calibration": calibration,
        "trace": traces,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
