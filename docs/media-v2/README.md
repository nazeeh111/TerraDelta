# Frozen first-study data replay

This silent 28-second research demo uses actual frozen Mexico imagery, event
revisions and read-only API responses. It is an edited **data replay**, not a
browser recording, live warning service or successful detector demonstration.
No ground-truth geometries or target arrays were read to choose the example.

Mexico was preselected before outcomes. The correction example is the
lexicographically first frozen event ID with a confirmed-by-model revision
followed by a retraction: `td-008e8fad29b65e55d92f`. It was not ranked by accuracy
or visual attractiveness. Its June 2018 observation-supported revision 2 remains
available through its original record-time query. August 2018 observation support
led to revision 3 retracting model support. This makes no claim about physical
cause or the correctness of the original building classification.

| Video time | Actual material |
| --- | --- |
| 0–7 s | Complete 1024×1023-pixel AOI through all 24 monthly sources, Jan 2018–Dec 2019; invalid pixels gray |
| 7–12 s | Saved GET queries: full bbox / 2018–2019 returns 292 latest confirmed-by-model events; changed NW-quarter bbox / 2019 returns 16 at the same record-time snapshot |
| 12–17 s | June source pixels and actual revision 2; source-supported interval March–June |
| 17–23 s | Same native context from August and actual revision 3; prior answer retained |
| 23–28 s | Three-region frozen first-study results: 27 matched, 671 false alerts, 1,479 missed; precision 3.9%, recall 1.8%; not analyst-ready |

Detail is a 96×96 native-pixel crop enlarged 6× with nearest-neighbor display.
There is no sharpening, generated content, inferred missing observation or
physical-resolution improvement. WGS84 event geometry is transformed through
EPSG:3857 and the source's inverse affine grid. Monthly filenames and unknown
publication times require the retrospective month-end availability assumption.
The separate next study is not a result claimed by this clip.

The master, exact received API JSON, original display RGB/mask PNGs, encoded QC
stills, font/OFL, renderer snapshot and full manifest are retained in the task's
`demo-v2/render-01` bundle. `poster.png` and `manifest.json` beside this note are
review assets; publishing or distribution is a separate action. The manifest
contains the master hash, API URLs/parameters/counts/snapshot/digests, complete
source image URLs/SHA256 values, processing identity and technical receipt.

The offline source archive is `terradelta-research-v2-demo-source.zip`. Its
`bundle-manifest.json` lists every distributed member with SHA256 and binds the
original master, original manifest and original renderer identities. The original
master is unchanged; `editable/render_demo.py` is a separate portable copy.
Only font and FFmpeg discovery changed. The source archive includes all 24 display RGB/mask
pairs, three exact query responses, event history, licensed font, encoded QC stills,
poster and the original study manifest. It contains no ground-truth arrays,
model weights, backbone, private cache or database.

The editable source is [render_demo.py](../../scripts/render_demo.py). Preparation
uses only GET on the configured loopback API and DuckDB with `read_only=True`.
Rendering uses Pillow and FFmpeg with no network, paid service, audio or browser
automation. Extract the source archive into a new folder, then run from its
`terradelta-demo-v2` directory. The archive includes a ready `editable` directory
without an existing master:

```sh
python -m pip install -r requirements-render.txt
python editable/render_demo.py render --output editable
```

The render step is offline and reads only the bundled inputs. Installing the
free dependencies may require a network connection. Python 3.12, NumPy 2.2.6
and Pillow 12.0.0 are the recorded environment. Install FFmpeg with H.264 support
and ffprobe on `PATH`; `~/.local/bin` is a fallback. Preparation accepts
`--fonts-root PATH` containing `inter.ttf` and `inter-OFL.txt`; its default is
`assets` beside the renderer. Rendering always uses the output's bundled fonts.
The renderer refuses an existing master. Retain the selected master; use a fresh
copy of `editable` for another render. Different FFmpeg/library versions can
change encoded bytes even when frame pixels match. A fresh full portable render
matched the original master SHA256 and all five encoded QC PNGs locally. The
bundle includes Inter and its SIL Open Font License.
Technical checks verify H.264, 1280×720, 24 fps, 672 frames, exactly 28 seconds
and no audio. Encoded start/query/middle/correction/end frames are exported and
visually inspected. This does not substitute for a human's normal-speed review
of the final edit.

Imagery credit: **SpaceNet Partners / Planet monthly mosaics**, adapted by
resizing/cropping and adding model-vector/text overlays, under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
[SpaceNet 7 source](https://spacenet.ai/sn7-challenge/). Video imagery adaptation
is CC BY-SA 4.0; renderer source is MIT. Exact image URLs and original-source
hashes are in the manifest. No new commercial or data-rights claim is implied.
