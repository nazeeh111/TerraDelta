"""Loopback analyst API. It serves configured observations, never arbitrary local paths."""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
from io import BytesIO
import hashlib
import math
import json
from pathlib import Path
import tempfile
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, create_model

from .data import load_region
from .schema import EventRevision, utc_datetime
from .store import EventStore


EventState = Literal["proposed", "confirmed_by_model", "reviewed", "retracted"]


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=5, max_length=2000)


EventProperties = create_model(
    "EventProperties",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (field.annotation, deepcopy(field))
        for name, field in EventRevision.model_fields.items()
        if name != "geometry"
    },
    area_m2=(float, Field(ge=0)),
)


class EventFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    id: str
    geometry: dict[str, Any]
    properties: EventProperties


class EventCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[EventFeature]
    snapshot_time: datetime | None
    semantics: str
    coverage_note: str


class EventHistory(BaseModel):
    revisions: list[EventRevision]


def create_app(cache_root: Path, run_root: Path) -> FastAPI:
    cache_root, run_root = Path(cache_root).resolve(), Path(run_root).resolve()
    app = FastAPI(
        title="TerraDelta event-history API",
        version="0.1.0",
        description="Monthly observation support and append-only event revisions. Retrospective availability assumption; no exact construction dates.",
    )
    web = Path(__file__).parent / "web"
    app.mount("/web", StaticFiles(directory=web), name="web")
    run_root.mkdir(parents=True, exist_ok=True)
    store = EventStore(run_root / "events.duckdb")
    app.state.store = store
    regions = {}
    for metadata_file in sorted(cache_root.glob("*/metadata.json")):
        meta = json.loads(metadata_file.read_text())
        regions[meta["region_id"]] = (metadata_file.parent, meta)

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
                return JSONResponse(
                    {"detail": "Mutation requires the local application origin"}, status_code=403
                )
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def bad_input(_request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "regions": len(regions)}

    @app.get("/api/catalog")
    def catalog():
        public = []
        for _, meta in regions.values():
            clean = {k: v for k, v in meta.items() if k != "frames"}
            clean["temporal_precision"] = (
                "Mosaic filename month; individual source-scene acquisition timestamps are unavailable."
            )
            clean["frames"] = [
                {k: v for k, v in f.items() if k != "local_image"} for f in meta["frames"]
            ]
            public.append(clean)
        report_file = run_root / "catalog-info.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        return {
            "regions": public,
            "model_revision": report.get("model_revision", "No computed run loaded"),
            "limits": [
                "Monthly mosaics bracket observed support, not exact physical dates.",
                "Original publication times unknown; retrospective month-end availability assumed.",
                "No event is not proof of no physical change. Clouds and registration may exclude evidence.",
                "Building-resolution research model; no legal evidence or causal attribution.",
            ],
            **report,
        }

    def query_args(bbox, start, end, as_of):
        parsed = None
        if bbox:
            try:
                parsed = tuple(float(v) for v in bbox.split(","))
            except ValueError:
                raise ValueError("bbox must contain west,south,east,north numbers")
            if len(parsed) != 4:
                raise ValueError("bbox must contain four coordinates")
        if start and end and utc_datetime(start) > utc_datetime(end):
            raise ValueError("start must precede end")
        return {"bbox": parsed, "start": start, "end": end, "as_of": as_of}

    def selected(bbox, start, end, as_of, region_id, include_retracted, state=None):
        if region_id and region_id not in regions:
            raise HTTPException(404, "Unknown study area")
        kwargs = query_args(bbox, start, end, as_of)
        kwargs.update(region_id=region_id, states=[state] if state else None)
        include_retracted = include_retracted or state == "retracted"
        snapshot = store.snapshot_time(kwargs["as_of"])
        if snapshot is None:
            store.query(**kwargs, include_retracted=include_retracted)
            return [], datetime.min.replace(tzinfo=timezone.utc), kwargs
        kwargs["as_of"] = snapshot
        events = store.query(**kwargs, include_retracted=include_retracted)
        if region_id:
            events = [e for e in events if e.region_id == region_id]
        return events, snapshot, kwargs

    @app.get("/api/events", response_model=EventCollection)
    def events(
        bbox: str | None = None,
        start: str | None = None,
        end: str | None = None,
        as_of: str | None = None,
        region_id: str | None = None,
        include_retracted: bool = False,
        state: EventState | None = None,
    ):
        result, snapshot, _ = selected(bbox, start, end, as_of, region_id, include_retracted, state)
        return {
            "type": "FeatureCollection",
            "features": [e.as_feature() for e in result],
            "snapshot_time": snapshot.isoformat() if snapshot else None,
            "semantics": "interval overlap; as_of is record-time snapshot, not imagery time",
            "coverage_note": "See source validity/registration; an empty answer does not imply observed no change.",
        }

    @app.get("/api/events/{event_id}/history", response_model=EventHistory)
    def history(event_id: str):
        result = store.history(event_id)
        if not result:
            raise HTTPException(404, "Unknown event")
        return {"revisions": [e.model_dump(mode="json") for e in result]}

    @app.post("/api/events/{event_id}/review", response_model=EventRevision)
    def review(event_id: str, body: ReviewRequest):
        existing = store.history(event_id)
        if not existing:
            raise HTTPException(404, "Unknown event")
        old = existing[-1]
        if old.state == "retracted":
            raise ValueError("Retracted events require new evidence before review")
        payload = old.model_dump()
        payload.update(
            revision=old.revision + 1,
            state="reviewed",
            reason=body.reason,
            record_time=datetime.now(timezone.utc),
        )
        event = EventRevision(**payload)
        store.append(event)
        return event.model_dump(mode="json")

    @app.get("/api/regions/{region_id}/frames/{index}.png")
    def frame(
        region_id: str,
        index: int,
        detail_event: str | None = None,
        detail_revision: int | None = None,
    ):
        if region_id not in regions:
            raise HTTPException(404, "Unknown study area")
        path, meta = regions[region_id]
        if index < 0 or index >= len(meta["frames"]):
            raise HTTPException(404, "Unknown observation")
        if (detail_event is None) != (detail_revision is None):
            raise ValueError("Detail imagery requires both event identity and exact revision")
        left, top, width, height = 0, 0, int(meta["width"]), int(meta["height"])
        if detail_event is not None:
            if detail_revision < 1:
                raise ValueError("Detail revision must be positive")
            selected_event = next(
                (
                    event
                    for event in store.history(detail_event)
                    if event.revision == detail_revision
                ),
                None,
            )
            if selected_event is None or selected_event.region_id != region_id:
                raise HTTPException(404, "Unknown event revision in this area")
            source_frame = meta["frames"][index]
            if not any(
                source.source_id == source_frame["source_id"]
                and source.sha256 == source_frame["sha256"]
                for source in selected_event.sources
            ):
                raise HTTPException(409, "Exact cited image bytes are not loaded for this revision")
            from rasterio.features import bounds
            from rasterio.transform import Affine
            from rasterio.warp import transform_geom

            if "transform" not in meta:
                raise ValueError("Native transform is required for event detail")
            native = transform_geom("EPSG:4326", meta["crs"], selected_event.geometry)
            west, south, east, north = bounds(native)
            inverse = ~Affine(*meta["transform"])
            x0, y0 = inverse * (west, north)
            x1, y1 = inverse * (east, south)
            if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
                raise ValueError("Event geometry does not overlap the loaded observation")
            # Native pixels only: contextual crop, with no enhancement/resampling.
            crop_width = min(width, max(64, math.ceil(x1 - x0) + 32))
            crop_height = min(height, max(64, math.ceil(y1 - y0) + 32))
            left = max(0, min(width - crop_width, math.floor((x0 + x1 - crop_width) / 2)))
            top = max(0, min(height - crop_height, math.floor((y0 + y1 - crop_height) / 2)))
            width, height = crop_width, crop_height
        data = load_region(path)
        window = (slice(top, top + height), slice(left, left + width))
        rgba = np.concatenate(
            [
                np.asarray(data["rgb"][index, :, window[0], window[1]]),
                np.asarray(data["valid"][index, window[0], window[1]][None], dtype="uint8") * 255,
            ]
        ).transpose(1, 2, 0)
        stream = BytesIO()
        Image.fromarray(rgba).save(stream, format="PNG")
        return Response(
            stream.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "private,max-age=3600",
                "ETag": '"'
                + hashlib.sha256(
                    f"{meta['frames'][index]['sha256']}:{left},{top},{width},{height}".encode()
                ).hexdigest()
                + '"',
                "X-TerraDelta-Source-SHA256": meta["frames"][index]["sha256"],
                "X-TerraDelta-Pixel-Window": f"{left},{top},{width},{height}",
                "X-TerraDelta-View": "native-detail" if detail_event else "complete-native-area",
            },
        )

    def exported(kind, bbox, start, end, as_of, region_id, include_retracted, state=None):
        result, snapshot, query = selected(
            bbox, start, end, as_of, region_id, include_retracted, state
        )
        if kind == "geojson":
            payload = {
                "type": "FeatureCollection",
                "features": [e.as_feature() for e in result],
                "snapshot_time": snapshot.isoformat() if snapshot else None,
                "coordinate_reference_system": "OGC:CRS84",
            }
            names = {e.region_id for e in result} or ({region_id} if region_id else set())
            licenses = {
                name: regions[name][1].get("license") or "unspecified"
                for name in names
                if name in regions
            }
            payload["source_licenses"] = licenses
            terms = set(licenses.values())
            if len(terms) == 1 and "unspecified" not in terms:
                payload["license"] = next(iter(terms))
            return Response(
                json.dumps(payload),
                media_type="application/geo+json",
                headers={"Content-Disposition": 'attachment; filename="terradelta-events.geojson"'},
            )
        # Export the exact ledger revisions at the same immutable query snapshot.
        with tempfile.TemporaryDirectory(prefix="terradelta-export-") as temporary:
            output = Path(temporary) / "events.parquet"
            kwargs = {
                **query,
                "as_of": snapshot,
                "include_retracted": include_retracted or state == "retracted",
            }
            store.export_parquet(output, **kwargs)
            content = output.read_bytes()
        return Response(
            content,
            media_type="application/vnd.apache.parquet",
            headers={
                "Content-Disposition": 'attachment; filename="terradelta-events.parquet"',
                "X-TerraDelta-Snapshot": snapshot.isoformat() if snapshot else "",
            },
        )

    @app.get("/api/export.geojson")
    def export_geojson(
        bbox: str | None = None,
        start: str | None = None,
        end: str | None = None,
        as_of: str | None = None,
        region_id: str | None = None,
        include_retracted: bool = False,
        state: EventState | None = None,
    ):
        return exported("geojson", bbox, start, end, as_of, region_id, include_retracted, state)

    @app.get("/api/export.parquet")
    def export_parquet(
        bbox: str | None = None,
        start: str | None = None,
        end: str | None = None,
        as_of: str | None = None,
        region_id: str | None = None,
        include_retracted: bool = False,
        state: EventState | None = None,
    ):
        return exported("parquet", bbox, start, end, as_of, region_id, include_retracted, state)

    return app
