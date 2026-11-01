#!/usr/bin/env python3
"""Prepare immutable real-data receipts, then render a silent 28s data replay.

No browser recording, model execution, labels, generated imagery or enhancement.
Prepare uses GET-only loopback requests and read-only legacy DuckDB inspection.
Render needs only the resulting self-contained assets, Pillow, NumPy and FFmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

AOI = "L15-0457E-1135N_1831_3648_13"
W, H, FPS, DURATION = 1280, 720, 24, 28
BG, INK, MUTED = "#F0F1E9", "#172E35", "#51646A"
TEAL, AMBER, RED = "#12675F", "#F3B85F", "#AC3E2D"
MAP = (32, 100, 608, 676)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def media_binary(name):
    """Use the caller's PATH, with the established local install as a fallback."""
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / name
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(f"{name} is required; install it and add it to PATH")


def fonts_root(explicit=None):
    """Released bundles keep the licensed font beside the renderer in assets/."""
    root = explicit if explicit is not None else Path(__file__).resolve().parent / "assets"
    for name in ("inter.ttf", "inter-OFL.txt"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {name}; supply --fonts-root with Inter and its OFL")
    return root


def get_receipt(api, endpoint, params, output):
    url = api.rstrip("/") + endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read()
        status = response.status
    output.write_bytes(payload)
    value = json.loads(payload)
    return value, {"url": url, "http_method": "GET", "status": status,
                   "parameters": params, "response_file": output.name,
                   "response_sha256": hashlib.sha256(payload).hexdigest(),
                   "feature_count": len(value.get("features", [])),
                   "snapshot_time": value.get("snapshot_time")}


def prepare(args):
    import duckdb
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import shape

    root, task = args.output, args.task
    root.mkdir(parents=True, exist_ok=False)
    (root / "assets").mkdir()
    meta_path = task / "cache" / AOI / "metadata.json"
    meta = json.loads(meta_path.read_text())
    db_path = task / "products-v2" / AOI / "history" / "events.duckdb"
    with duckdb.connect(str(db_path), read_only=True) as db:
        event_id = db.execute("""
            SELECT event_id FROM event_revisions GROUP BY event_id
            HAVING min(CASE WHEN json_extract_string(payload,'$.state')='confirmed_by_model'
                            THEN revision END)
                 < max(CASE WHEN json_extract_string(payload,'$.state')='retracted'
                            THEN revision END)
            ORDER BY event_id LIMIT 1
        """).fetchone()[0]
        legacy = [json.loads(r[0]) for r in db.execute(
            "SELECT payload FROM event_revisions WHERE event_id=? ORDER BY revision", [event_id]
        ).fetchall()]
    history, history_receipt = get_receipt(
        args.api, f"/api/events/{event_id}/history", {}, root / "history.json"
    )
    if history["revisions"] != legacy:
        raise RuntimeError("API history does not equal selected frozen legacy history")
    confirmed = next(r for r in legacy if r["state"] == "confirmed_by_model")
    retracted = next(r for r in legacy if r["state"] == "retracted"
                     and r["revision"] > confirmed["revision"])
    run_path = task / "products-v2" / AOI / "history" / "run.json"
    run = json.loads(run_path.read_text())
    bbox = meta["bbox"]
    west, south, east, north = bbox
    smaller = [west, (south + north) / 2, (west + east) / 2, north]
    common = {"region_id": AOI, "state": "confirmed_by_model", "as_of": run["snapshot_time"]}
    queries = []
    for name, query_bbox, start, end in (
        ("query-full", bbox, "2018-01-01T00:00:00Z", "2019-12-31T23:59:59Z"),
        ("query-changed", smaller, "2019-01-01T00:00:00Z", "2019-12-31T23:59:59Z"),
    ):
        _, receipt = get_receipt(args.api, "/api/events", common | {
            "bbox": ",".join(str(v) for v in query_bbox), "start": start, "end": end,
        }, root / f"{name}.json")
        queries.append(receipt)
    _, old_receipt = get_receipt(args.api, "/api/events", {
        "region_id": AOI, "state": "confirmed_by_model", "as_of": confirmed["record_time"],
        "bbox": ",".join(str(v) for v in shape(confirmed["geometry"]).bounds),
    }, root / "query-old-answer.json")
    old = json.loads((root / "query-old-answer.json").read_text())
    if not any(f["id"] == event_id and f["properties"]["revision"] == confirmed["revision"]
               for f in old["features"]):
        raise RuntimeError("record-time query did not retain the old confirmed answer")

    rgb_path, valid_path = [task / "cache" / AOI / f"{n}.npy" for n in ("rgb", "valid")]
    rgb, valid = [np.load(p, mmap_mode="r") for p in (rgb_path, valid_path)]
    if len(meta["frames"]) != 24:
        raise ValueError("expected exact frozen Mexico24 monthly sequence")
    frame_sources = []
    for i, frame in enumerate(meta["frames"]):
        # Retain exact display RGB and real validity separately. No labels opened.
        Image.fromarray(np.moveaxis(rgb[i], 0, -1)).save(root / "assets" / f"rgb-{i:02d}.png")
        Image.fromarray(valid[i].astype(np.uint8) * 255).save(root / "assets" / f"valid-{i:02d}.png")
        frame_sources.append({k: frame[k] for k in (
            "source_id", "sha256", "uri", "acquisition_start", "acquisition_end",
            "available_at", "published_at", "valid_fraction", "registration")})
    transform = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    inverse = ~Affine(*meta["transform"])
    polygons = confirmed["geometry"]["coordinates"]
    if confirmed["geometry"]["type"] == "Polygon":
        polygons = [polygons]
    pixels = [[list(inverse * transform.transform(x, y)) for x, y in polygon[0]]
              for polygon in polygons]
    points = np.array([p for polygon in pixels for p in polygon])
    cx, cy = points.mean(axis=0)
    crop_size = 96
    left = min(max(0, int(cx - crop_size / 2)), meta["width"] - crop_size)
    top = min(max(0, int(cy - crop_size / 2)), meta["height"] - crop_size)
    comparison = args.repo / "artifacts/confirmation-v2/comparison.json"
    result = json.loads(comparison.read_text())["methods"]["observation_gate"]["event"]
    if (result["tp"], result["fp"], result["fn"]) != (27, 671, 1479):
        raise RuntimeError("unexpected frozen study result")
    fonts = fonts_root(args.fonts_root)
    for name in ("inter.ttf", "inter-OFL.txt"):
        shutil.copyfile(fonts / name, root / "assets" / name)
    manifest = {
        "format": "terradelta-data-replay-v2", "region_id": AOI,
        "kind": "edited real-data replay; not a browser recording or live detector",
        "video": {"width": W, "height": H, "fps": FPS, "seconds": DURATION,
                  "frames": FPS * DURATION, "audio": False},
        "selection": {"area": "Mexico preselected before outcomes",
                      "event": "lexicographically first ID confirmed then retracted; no accuracy ranking",
                      "event_id": event_id, "confirmed_revision": confirmed["revision"],
                      "retracted_revision": retracted["revision"]},
        "history_receipt": history_receipt, "query_receipts": queries,
        "retained_answer_receipt": old_receipt,
        "crs": meta["crs"], "affine": meta["transform"], "bbox": bbox,
        "native_shape": [meta["height"], meta["width"]],
        "detail_crop_native_pixels": [left, top, crop_size, crop_size],
        "event_polygon_native_pixels": pixels,
        "imagery": frame_sources,
        "display": "full-area imagery downsampled for fit; detail nearest-neighbor6x; no enhancement",
        "source_inputs_sha256": {"cache_rgb.npy": digest(rgb_path),
                                 "cache_valid.npy": digest(valid_path),
                                 "cache_metadata.json": digest(meta_path),
                                 "frozen_history_events.duckdb": digest(db_path),
                                 "run.json": digest(run_path),
                                 "confirmation-v2/comparison.json": digest(comparison)},
        "processing_revision": run["model_revision"], "study_event_metrics": result,
        "study_limit": "failed geographic transfer; not analyst-ready; new study is separate",
        "timing": "mosaic filename months; original publication unknown; retrospective month-end availability",
        "exclusions": "No target.npy, edge.npy, instances.npy or ground-truth geometries read",
        "credits": {"imagery": "SpaceNet Partners / Planet monthly mosaics",
                    "source_license": "CC-BY-SA-4.0",
                    "source_url": "https://spacenet.ai/sn7-challenge/",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "adaptation": "imagery resized, native detail cropped, event overlays and text added",
                    "video_license": "CC-BY-SA-4.0", "renderer_code_license": "MIT",
                    "font": "Inter; bundled SIL Open Font License"},
    }
    dump(root / "manifest.json", manifest)
    shutil.copyfile(__file__, root / "render_demo.py")
    print(json.dumps({"prepared": str(root), "event_id": event_id,
                      "query_counts": [q["feature_count"] for q in queries]}), flush=True)


class Replay:
    def __init__(self, root):
        self.root = root
        self.m = json.loads((root / "manifest.json").read_text())
        self.history = json.loads((root / "history.json").read_text())["revisions"]
        self.confirm = next(r for r in self.history if r["state"] == "confirmed_by_model")
        self.retract = next(r for r in self.history if r["state"] == "retracted"
                            and r["revision"] > self.confirm["revision"])
        self.frames = []
        for i in range(24):
            image = Image.open(root / "assets" / f"rgb-{i:02d}.png").convert("RGB")
            mask = Image.open(root / "assets" / f"valid-{i:02d}.png")
            image.paste(Image.new("RGB", image.size, "#758086"), mask=Image.eval(mask, lambda v: 255-v))
            self.frames.append(image)
        self.fonts = {size: ImageFont.truetype(str(root / "assets/inter.ttf"), size)
                      for size in (13, 15, 17, 19, 22, 24, 28, 32, 40, 46, 72)}
        self.full = json.loads((root / "query-full.json").read_text())
        self.changed = json.loads((root / "query-changed.json").read_text())

    def text(self, xy, text, size=24, fill=INK):
        self.d.text(xy, text, font=self.fonts[size], fill=fill, spacing=9)

    def card(self, box, fill="white"):
        self.d.rounded_rectangle(box, radius=14, fill=fill)

    def image(self, index, detail=False, event=False):
        image = self.frames[index]
        if detail:
            x, y, w, h = self.m["detail_crop_native_pixels"]
            image = image.crop((x, y, x+w, y+h)).resize((576, 576), Image.Resampling.NEAREST)
        else:
            image = image.resize((576, 576), Image.Resampling.LANCZOS)
        self.im.paste(image, MAP[:2])
        if event:
            for polygon in self.m["event_polygon_native_pixels"]:
                if detail:
                    coords = [(32 + (px-x)*6, 100 + (py-y)*6) for px, py in polygon]
                else:
                    height, width = self.m["native_shape"]
                    coords = [(32+px/width*576, 100+py/height*576) for px, py in polygon]
                self.d.line(coords, fill=AMBER, width=3, joint="curve")
        self.d.rectangle((32, 74, 608, 99), fill=INK)
        date = self.m["imagery"][index]["acquisition_start"][:7]
        label = f"{date}  |  " + ("NATIVE-PIXEL DETAIL · 6× display" if detail else "FULL STUDY AREA")
        self.text((44, 77), label, 17, "white")

    def frame_index(self, revision):
        source_id = revision["sources"][-1]["source_id"]
        return next(i for i, f in enumerate(self.m["imagery"]) if f["source_id"] == source_id)

    def render(self, t):
        self.im = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.im)
        self.text((32, 24), "TERRADELTA", 32)
        self.text((285, 35), "DATA REPLAY / FROZEN FIRST STUDY", 17, TEAL)
        self.text((1000, 35), "NOT LIVE · NO AUDIO", 15, MUTED)
        self.text((32, 695), "SpaceNet Partners / Planet · CC BY-SA 4.0 · Monthly mosaics · Unknown publication times", 13, MUTED)
        self.text((1075, 695), "Gray: invalid source", 13, MUTED)
        if t < 7:
            index = min(23, max(0, int((t - 1) / 6 * 24)))
            self.image(index)
            self.text((648, 112), "24 months.\nOne complete area.", 46)
            self.text((648, 260), "Mexico · January 2018–December 2019", 22, MUTED)
            self.card((648, 320, 1248, 492))
            self.text((672, 342), "Actual source mosaics", 28)
            self.text((672, 393), "1024 × 1023 native pixels\nEPSG:3857 · Source grid unchanged", 22, MUTED)
            fraction = self.m["imagery"][index]["valid_fraction"]
            self.text((672, 454), f"Source-valid pixels: {fraction:.1%}", 17, TEAL)
            self.text((648, 536), "Monthly support brackets change.\nIt cannot date construction exactly.", 24)
            for i in range(24):
                self.d.rectangle((650+i*24, 640, 668+i*24, 648), fill=TEAL if i<=index else "#CDD4CC")
        elif t < 12:
            self.image(23)
            changed = t >= 9.5
            self.text((648, 110), "Query real exported events.", 32)
            self.card((648, 173, 1248, 496))
            self.text((672, 193), "GET /api/events", 24, TEAL)
            self.text((672, 240), "2019 only · northwest quarter" if changed else "2018–2019 · full area", 28)
            count = len((self.changed if changed else self.full)["features"])
            self.text((672, 294), str(count), 72)
            self.text((672, 384), "latest confirmed-by-model events", 22)
            self.text((672, 430), "Exact date + bbox + record-time snapshot", 19, MUTED)
            if changed:
                self.d.rectangle((32, 100, 320, 388), outline=AMBER, width=4)
            self.text((648, 535), "Both responses saved with SHA256.\nNo event ≠ observed no change.", 24)
        elif t < 23:
            later = t >= 17
            revision = self.retract if later else self.confirm
            self.image(self.frame_index(revision), detail=True, event=True)
            self.text((648, 108), "A model answer can be corrected.", 32)
            self.text((648, 161), "Actual candidate geometry, not ground truth", 19, MUTED)
            self.card((648, 210, 1248, 351))
            self.text((672, 230), f"REVISION {self.confirm['revision']}  ·  JUNE 2018", 19, TEAL)
            self.text((672, 268), "Confirmed by model", 32)
            self.text((672, 312), "Retained by its original record-time query", 19, MUTED)
            if later:
                self.card((648, 374, 1248, 508))
                self.text((672, 394), f"REVISION {self.retract['revision']}  ·  AUGUST 2018", 19, RED)
                self.text((672, 432), "Retracted model support", 32, RED)
                self.text((672, 477), "Later observation reversed the raw support", 19, MUTED)
            else:
                self.text((648, 393), "Support interval: March–June 2018\nA monthly bracket, not an exact date.", 24)
            self.text((648, 553), "Old answer retained. New revision appended.\nNo claim about the physical cause.", 22)
            self.text((648, 633), self.m["selection"]["event_id"], 17, MUTED)
        else:
            self.image(23)
            self.text((648, 108), "Failed geographic transfer.", 40, RED)
            self.text((648, 166), "First study · three complete held-out regions", 22)
            self.card((648, 223, 1248, 433))
            self.text((672, 241), "3.9%", 72, RED)
            self.text((975, 241), "1.8%", 72, RED)
            self.text((674, 328), "PRECISION", 19, MUTED)
            self.text((979, 328), "RECALL", 19, MUTED)
            self.text((672, 382), "27 matched · 671 false alerts · 1,479 missed", 22)
            self.text((648, 481), "Not analyst-ready.", 40)
            self.text((648, 550), "Revision history works. Detection is inadequate.\nA separate new study is underway.", 22, MUTED)
        return self.im


def render(args):
    root = args.output
    output = root / "terradelta-data-replay-v2.mp4"
    if output.exists():
        raise FileExistsError("keep the last valid render; use a fresh output directory")
    if Path(__file__).resolve() != (root / "render_demo.py").resolve():
        shutil.copyfile(__file__, root / "render_demo.py")
    replay = Replay(root)
    ffmpeg = media_binary("ffmpeg")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-metadata", "title=TerraDelta: frozen real-data replay",
               "-metadata", "comment=Data replay, not browser capture. Imagery CC BY-SA4.0.",
               "-n", str(output)]
    with (root / "encode.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i in range(FPS * DURATION):
                process.stdin.write(replay.render(i/FPS).tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg failed; see encode.log")
    ffprobe = media_binary("ffprobe")
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)]))
    dump(root / "ffprobe.json", probe)
    for name, at in (("start", 0), ("query", 10.5), ("middle", 14), ("correction", 20), ("end", 27.5)):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(at),
                        "-i", str(output), "-frames:v", "1", "-n", str(root / f"encoded-{name}.png")],
                       check=True)
    shutil.copyfile(root / "encoded-middle.png", root / "poster.png")
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"], video["nb_frames"]) == (W, H, str(FPS*DURATION))
    assert abs(float(probe["format"]["duration"]) - DURATION) < .01
    assert not any(s["codec_type"] == "audio" for s in probe["streams"])
    manifest = replay.m
    manifest["artifacts_sha256"] = {p.name: digest(p) for p in root.glob("*") if p.is_file()
                                    and p.name != "manifest.json"}
    manifest["asset_sha256"] = {p.name: digest(p) for p in (root / "assets").glob("*")}
    manifest["verification"] = {"encoded_frames": video["nb_frames"],
                                "seconds": probe["format"]["duration"],
                                "codec": video["codec_name"], "pixel_format": video["pix_fmt"],
                                "audio": False, "encoded_stills": [0, 10.5, 14, 20, 27.5],
                                "limit": "sampled encoded stills and technical checks; no audible review needed"}
    dump(root / "manifest.json", manifest)
    print(json.dumps({"video": str(output), "sha256": digest(output),
                      "bytes": output.stat().st_size, "frames": video["nb_frames"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "render"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--api", default="http://127.0.0.1:54992")
    parser.add_argument("--fonts-root", type=Path,
                        help="Prepare: directory containing inter.ttf and inter-OFL.txt; defaults to bundled assets/")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.task is None:
            parser.error("prepare requires --task")
        prepare(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
