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
from terradelta.features import FEATURE_NAMES, validate_feature_protocol
from terradelta.temporal import make_temporal, predict_temporal


def sequence_sampling_groups(y, mask):
    previous = np.zeros(len(y), dtype="float32")
    seen = np.zeros(len(y), bool)
    changed = np.zeros(len(y), bool)
    for t in range(y.shape[1]):
        changed |= mask[:, t] & seen & (y[:, t] != previous)
        previous = np.where(mask[:, t], y[:, t], previous)
        seen |= mask[:, t]
    eligible = mask.sum(axis=1) >= 2
    return np.flatnonzero(changed & eligible), np.flatnonzero(~changed & eligible)


def event_validation(model, dev, contexts):
    from terradelta.decoder import DecoderConfig, decode_events
    from terradelta.evaluation import evaluate_events

    raw = [predict_temporal(model, x, u) for _, x, u, _, _ in dev]
    fit = calibrate(
        np.concatenate([p.ravel() for p in raw]),
        np.concatenate([y.ravel() for _, _, _, y, _ in dev]),
        np.concatenate([(u & s).ravel() for _, _, u, _, s in dev]),
    )
    calibrated = [expit(fit["scale"] * logit(np.clip(p, 0.001, 0.999)) + fit["bias"]) for p in raw]
    candidates = []
    for threshold in [0.55, 0.65, 0.75]:
        reports = []
        for (_, _, usable, _, _), probability, (info, truth) in zip(dev, calibrated, contexts):
            events = decode_events(
                probability,
                usable,
                np.asarray(info["births"]),
                info,
                DecoderConfig(threshold=threshold),
                model_revision="development-checkpoint-selection",
            )
            result = evaluate_events(events, truth, info["metadata"])
            reports.append(
                {
                    "region_id": info["region_id"],
                    "event": result["event"],
                    "within_62_days": result["within_62_days"],
                }
            )
        timely = float(np.mean([r["within_62_days"]["f1"] or 0 for r in reports]))
        ordinary = float(np.mean([r["event"]["f1"] or 0 for r in reports]))
        candidates.append(((timely, ordinary), threshold, reports))
    key, threshold, reports = max(candidates, key=lambda item: (item[0], item[1]))
    return key, {
        "matched_62d_macro_f1": key[0],
        "ordinary_macro_f1": key[1],
        "selection_threshold": threshold,
        "region_reports": reports,
    }


def load_split(root, regions, *, physical_supervision: bool = False):
    output = []
    for aoi in regions:
        if physical_supervision:
            validate_feature_protocol(root / aoi, supervision=True)
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
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    repo = Path(__file__).resolve().parents[1]
    source_hashes = {
        name: sha256(repo / name)
        for name in [
            "scripts/train_temporal.py",
            "src/terradelta/temporal.py",
            "src/terradelta/features.py",
            "src/terradelta/decoder.py",
            "src/terradelta/evaluation.py",
        ]
    }
    if args.output.exists():
        raise ValueError("Trained artifact output must be new; preserve previous models")
    if args.steps < 1 or args.steps > config["resource_limits"].get(
        "temporal_training_steps", 1000
    ):
        raise ValueError("Temporal steps exceed the versioned resource limit")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    physical = config.get("truth_protocol") == "physical_coverage_v2"
    train = load_split(args.features, config["train"], physical_supervision=physical)
    dev = load_split(args.features, config["development"], physical_supervision=physical)
    event_selection = config.get("temporal_selection") == "event_62d"
    contexts = []
    if event_selection:
        if args.cache_root is None:
            raise ValueError("Event checkpoint selection requires development --cache-root")
        from terradelta.evaluation import ground_truth_events

        for aoi, _, _, _, _ in dev:
            info = json.loads((args.features / aoi / "sites.json").read_text())
            info["births"] = np.load(args.features / aoi / "features.npz")["births"].tolist()
            contexts.append((info, ground_truth_events(args.cache_root / aoi)))
    groups = [sequence_sampling_groups(y, u & s) for _, _, u, y, s in train]
    # Equal per-region contribution; oversample real transition sequences without
    # changing development prevalence. No fabricated imagery or observations.
    models = {}
    traces = {}
    calibration = {}
    summary = {}
    for kind in ["observation", "no_quality", "watch_lstm"]:
        # Paired GRU ablation gets the same initialization and real-example order.
        torch.manual_seed(config["seed"])
        rng = np.random.default_rng(config["seed"])
        model = make_temporal(kind, spatial_residual=config.get("temporal_spatial_residual", True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        best = float("inf")
        best_state = None
        best_key = None
        selected_step = None
        selected_event = None
        trace = []
        started = time.perf_counter()
        for step in range(args.steps):
            _, x, usable, y, supervised = train[step % len(train)]
            changed, stable = groups[step % len(train)]
            eligible = np.concatenate([changed, stable])
            if not len(eligible):
                raise ValueError("No supervised two-observation sequence in a training region")
            selected = np.concatenate(
                [
                    rng.choice(changed if len(changed) else eligible, 64),
                    rng.choice(stable if len(stable) else eligible, 64),
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
            if (step + 1) % (200 if event_selection else 100) == 0 or step == 0:
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
                selection_key = (-value,)
                event_metrics = None
                if event_selection:
                    selection_key, event_metrics = event_validation(model, dev, contexts)
                trace.append(
                    {
                        "step": step + 1,
                        "train_loss": float(loss.detach()),
                        "development_bce": value,
                        "event_selection": event_metrics,
                    }
                )
                print(
                    kind, trace[-1], "seconds", round(time.perf_counter() - started, 1), flush=True
                )
                eligible_checkpoint = step + 1 >= config.get("minimum_temporal_selection_step", 1)
                if eligible_checkpoint and (best_key is None or selection_key > best_key):
                    best = value
                    best_key = selection_key
                    selected_step = step + 1
                    selected_event = event_metrics
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
            "selected_step": selected_step,
            "selection_metric": "geographic_event_f1_at_62d"
            if event_selection
            else "occupancy_bce",
            "selected_development_event_metrics": selected_event,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "terradelta-temporal-v1",
            "models": models,
            "feature_names": FEATURE_NAMES,
            "model_config": {"spatial_residual": config.get("temporal_spatial_residual", True)},
            "config_sha256": sha256(args.config),
            "train_regions": config["train"],
            "development_regions": config["development"],
            "seed": config["seed"],
            "source_sha256": source_hashes,
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
        "supervision_sha256": {
            aoi: sha256(args.features / aoi / "supervision.npz")
            for aoi in config["train"] + config["development"]
        },
        "source_sha256": source_hashes,
        "models": summary,
        "calibration": calibration,
        "trace": traces,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
