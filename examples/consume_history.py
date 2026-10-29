"""Second client: consume exported GeoJSON without TerraDelta or a map library.

Usage: python examples/consume_history.py events.geojson
       python examples/consume_history.py http://127.0.0.1:8766/api/events
"""

import argparse
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description="Summarize a versioned TerraDelta event export")
parser.add_argument("source")
parser.add_argument(
    "--class", dest="change_class", choices=["building_addition", "building_removal"]
)
args = parser.parse_args()
if args.source.startswith(("https://", "http://")):
    with urllib.request.urlopen(args.source, timeout=30) as response:
        doc = json.load(response)
else:
    doc = json.loads(Path(args.source).read_text())
assert doc["type"] == "FeatureCollection"
summary = {}
for feature in doc["features"]:
    p = feature["properties"]
    if args.change_class and p["change_class"] != args.change_class:
        continue
    key = (p.get("region_id"), p["change_class"], p["state"])
    bucket = summary.setdefault(key, {"events": 0, "footprint_area_m2": 0.0, "sources": set()})
    bucket["events"] += 1
    bucket["footprint_area_m2"] += p["area_m2"]
    bucket["sources"].update(s["source_id"] for s in p["sources"])
print(
    json.dumps(
        {
            "snapshot_time": doc.get("snapshot_time"),
            "groups": [
                {
                    "region": k[0],
                    "change_class": k[1],
                    "state": k[2],
                    "events": v["events"],
                    "footprint_area_m2": round(v["footprint_area_m2"], 2),
                    "distinct_supporting_sources": len(v["sources"]),
                }
                for k, v in summary.items()
            ],
        },
        indent=2,
    )
)
