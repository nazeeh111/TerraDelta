"""Bounded, unsigned acquisition of permitted SpaceNet7 source objects."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://spacenet-dataset.s3.amazonaws.com/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def fetch_sn7(
    output: Path,
    config: Path,
    *,
    regions: list[str] | None = None,
    expected_manifest: Path | None = None,
    maximum_bytes: int = 3_000_000_000,
) -> Path:
    """Download whole AOI sequences, checking bytes/hashes. Never a global archive."""
    output = Path(output)
    settings = json.loads(Path(config).read_text())
    allowed = {aoi for split in ["train", "development", "confirmation"] for aoi in settings[split]}
    requested = regions or sorted(allowed)
    if not set(requested) <= allowed:
        raise ValueError("Regions must belong to the frozen config")
    expected = {}
    if expected_manifest:
        expected = {i["key"]: i for i in json.loads(Path(expected_manifest).read_text())}
    objects = []
    for aoi in requested:
        prefix = f"spacenet/SN7_buildings/train/{aoi}/"
        url = BASE + "?" + urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
        with urllib.request.urlopen(url, timeout=90) as response:
            listing = ET.fromstring(response.read())
        if listing.findtext("s:IsTruncated", namespaces=NS) != "false":
            raise ValueError("Unexpected oversized listing; no implicit unbounded pagination")
        for item in listing.findall("s:Contents", NS):
            key = item.findtext("s:Key", namespaces=NS)
            if not (
                "/images_masked/" in key
                or "/labels_match/" in key
                or ("/labels/" in key and key.endswith("_UDM.geojson"))
            ):
                continue
            objects.append(
                {
                    "key": key,
                    "bytes": int(item.findtext("s:Size", namespaces=NS)),
                    "s3_last_modified": item.findtext("s:LastModified", namespaces=NS),
                }
            )
    if sum(i["bytes"] for i in objects) > maximum_bytes:
        raise ValueError("Bounded download limit exceeded before any object downloads")

    def download(item):
        key = item["key"]
        relative = Path(*key.split("/")[3:])
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            content = destination.read_bytes()
        else:
            with urllib.request.urlopen(BASE + key, timeout=120) as response:
                content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != item["bytes"]:
            raise ValueError(f"Source length mismatch: {key}")
        if expected_manifest and (key not in expected or expected[key]["sha256"] != digest):
            raise ValueError(f"Source checksum differs from frozen manifest: {key}")
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(destination)
        return {**item, "sha256": digest, "local": str(relative)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, objects))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "acquisition-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
