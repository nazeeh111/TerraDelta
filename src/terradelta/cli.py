"""Typed command-line workflow for import, compute, query and independent exports."""

from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="Supported building-change histories from permitted RGBA satellite sequences.",
)


@app.command("fetch-sn7")
def fetch_sn7_command(
    output: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()],
    region: Annotated[list[str] | None, typer.Option()] = None,
    manifest: Annotated[Path | None, typer.Option()] = None,
):
    """Fetch bounded real sequences, with frozen checksums when --manifest is supplied."""
    from .acquire import fetch_sn7

    typer.echo(str(fetch_sn7(output, config, regions=region, expected_manifest=manifest)))


@app.command()
def ingest(region_dir: Path, output: Annotated[Path, typer.Option()], labels: bool = False):
    """Validate/import native SpaceNet RGBA imagery. --labels is offline research only."""
    from .data import ingest_region

    result = ingest_region(region_dir, output, labels=labels)
    typer.echo(
        json.dumps(
            {
                "region_id": result["region_id"],
                "observations": len(result["frames"]),
                "crs": result["crs"],
                "bbox": result["bbox"],
            },
            indent=2,
        )
    )


@app.command()
def run(
    cache_dir: Path,
    spatial_weights: Annotated[Path, typer.Option()],
    calibration: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    temporal_weights: Annotated[Path | None, typer.Option()] = None,
    event_gate_weights: Annotated[Path | None, typer.Option()] = None,
    method: Annotated[str | None, typer.Option(
        help="Event method; defaults to observation_gate with gate weights, otherwise observation."
    )] = None,
    device: str = "cpu",
    end: str | None = None,
):
    """Compute an image-only history into a NEW output directory. End restricts prefix."""
    from .pipeline import (
        GATE_METHODS, LEARNED_TEMPORAL_METHODS, METHODS,
        predict_sequence, reconstruct, prefix_frames,
    )
    from .features import prepare_features

    selected_method = method or ("observation_gate" if event_gate_weights else "observation")
    if selected_method not in METHODS:
        raise typer.BadParameter("method must be one of: " + ", ".join(METHODS))
    if selected_method in LEARNED_TEMPORAL_METHODS and temporal_weights is None:
        raise typer.BadParameter("This method requires --temporal-weights")
    if selected_method in GATE_METHODS[:2] and event_gate_weights is None:
        raise typer.BadParameter("This method requires --event-gate-weights")
    if output.exists() and any(output.iterdir()):
        raise typer.BadParameter("Output must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    predict_sequence(cache_dir, spatial_weights, output / "rasters", device=device, end=end)
    # Prefix features use a matching truncated cache, retaining native arrays by view.
    effective_cache = cache_dir
    if end:
        import numpy as np
        from .data import load_region

        region = load_region(cache_dir)
        metadata = region["metadata"]
        metadata["frames"] = prefix_frames(metadata["frames"], end)
        effective_cache = output / "prefix-cache"
        effective_cache.mkdir()
        for name in ["rgb", "valid"]:
            np.save(effective_cache / f"{name}.npy", region[name][: len(metadata["frames"])])
        (effective_cache / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    prepare_features(effective_cache, output / "rasters/probabilities.npy", output / "features")
    report = reconstruct(
        output / "features", temporal_weights, output / "history", calibration_path=calibration,
        method=selected_method, event_gate_weights=event_gate_weights,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command()
def query(
    store: Path,
    bbox: str | None = None,
    start: str | None = None,
    end: str | None = None,
    as_of: str | None = None,
    include_retracted: bool = False,
):
    """Query change-interval overlap at an immutable record-time snapshot."""
    from .store import EventStore

    ledger = EventStore(store)
    snapshot = ledger.snapshot_time(as_of)
    events = (
        ledger.query(
            bbox=[float(v) for v in bbox.split(",")] if bbox else None,
            start=start,
            end=end,
            as_of=snapshot,
            include_retracted=include_retracted,
        )
        if snapshot
        else []
    )
    typer.echo(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [e.as_feature() for e in events],
                "snapshot_time": snapshot.isoformat()
                if snapshot
                else datetime.min.replace(tzinfo=timezone.utc).isoformat(),
            }
        )
    )
    ledger.close()


@app.command("export")
def export_command(
    store: Path,
    output: Annotated[Path, typer.Option()],
    format: str = "geojson",
    bbox: str | None = None,
    start: str | None = None,
    end: str | None = None,
    as_of: str | None = None,
    include_retracted: bool = False,
):
    """Export the queried snapshot as RFC7946 GeoJSON or GeoParquet1.1."""
    from .store import EventStore

    ledger = EventStore(store)
    kwargs = dict(
        bbox=[float(v) for v in bbox.split(",")] if bbox else None,
        start=start,
        end=end,
        as_of=as_of,
        include_retracted=include_retracted,
    )
    if format == "geojson":
        path = ledger.export_geojson(output, **kwargs)
    elif format == "parquet":
        path = ledger.export_parquet(output, **kwargs)
    else:
        raise typer.BadParameter("format must be geojson or parquet")
    typer.echo(str(path))
    ledger.close()


@app.command()
def serve(
    cache_root: Annotated[Path, typer.Option()],
    run_root: Annotated[Path, typer.Option()],
    port: int = 8766,
):
    """Open the analyst workbench on loopback. Public compute exposure is excluded."""
    import uvicorn
    from .service import create_app

    uvicorn.run(create_app(cache_root, run_root), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
