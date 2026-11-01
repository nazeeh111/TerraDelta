# Research-release material: draft, not posted

**Publication status:** the [source](https://github.com/nazeeh111/TerraDelta) and
[research prerelease](https://github.com/nazeeh111/TerraDelta/releases/tag/v0.1.0-research)
are public. The release target passed
[Linux CI](https://github.com/nazeeh111/TerraDelta/actions/runs/41291280215), and all
17 downloaded assets match their complete-file SHA-256 digests and sizes. Every
ZIP passes its integrity check. Version-2 confirmation is complete and negative:
failed geographic transfer, not an analyst-ready detector. The
[full result](results-v2.md) and original artifacts accompany the release. This
announcement remains a draft: no outreach has been sent, and no adoption or
external endorsement is claimed.

## Short project introduction

TerraDelta is an experimental engine for a source-linked, revisable history of
building-footprint change hypotheses. Each event records geometry, an observation
interval, source identities, image-quality diagnostics and explicit revisions.
A cloud gap remains unknown; later evidence can confirm or retract a hypothesis
without rewriting an earlier query snapshot.

The software includes native-grid imagery ingestion, a locally trained compact
building model, an experimental event-compatibility gate, typed API/CLI queries,
GeoJSON/GeoParquet/raster exports and an analyst evidence interface. An independent
client can read the exported history without that interface.

The scientific result is a failure. On three complete held-out SpaceNet 7 regions,
the quality gate matched 27 alerts with 671 false alerts and 1,479 misses:
precision 0.03868, recall 0.01793 and F1 0.02450. At a 62-day detection horizon,
17 matches remained and F1 was 0.01543. The no-quality ablation had higher pooled
ordinary/timely F1, so no overall benefit from learned quality inputs was shown.
Only 391 of 1,506 truth events overlapped a proposed site at the stated geometry
threshold, exposing an upstream limitation as well.

The study retains all twelve methods, complete-region cases, exclusions,
calibration, timing diagnostics and regional uncertainty. Earlier negative
versions remain preserved. The implemented data/revision contract is not evidence
that the alerts are accurate enough for planning or environmental decisions.

## Executed evidence and remaining release gates

| Field | Current evidence or limit |
| --- | --- |
| Frozen held-out comparison | [Twelve methods, three complete regions](../artifacts/confirmation-v2/comparison.json); failed geographic transfer |
| Matched-delay comparison | Common 62-day horizon plus recorded delay; quality-gate F1 0.01543 |
| Quality-input ablation | No-quality gate ordinary/timely F1 0.02603 / 0.01673; no overall quality advantage |
| Timing support | Per-region raw/expanded label-interval containment alongside low recall; not exact-date coverage |
| False alerts and revisions | Quality gate 9.865 false alerts/source-valid km²-year and 131 confirmed IDs later retracted |
| Stress evidence | [Retained real-data perturbation controls](../artifacts/stress-v2.json), with fixed-candidate and one-frame radiometry limits |
| Online event-gate invariance | [All three gate/proposal methods pass on all three regions](../artifacts/verification-v2/online-gates-invariance.json); legacy GRU strict numerical confidence limitation remains separate |
| Independent execution | [Anonymous public clone and changed public cropped/prefix inputs](../artifacts/verification-v3/public-install-v1.json), [real installed API queries, GIS readback and standard-library consumer](../artifacts/verification-v3/public-api-v1.json) |
| Public repository | [nazeeh111/TerraDelta](https://github.com/nazeeh111/TerraDelta), public visibility verified |
| Public CI, clean clone and downloaded release | Release target `b2ed1be` passed Linux CI; all 17 downloaded files verified, with anonymous access to source/release pages and the asset manifest |
| Complete-area demonstration | [Genuine 28-second replay and editable sources](media-v2/README.md); complete 24-month Mexico area, supporting imagery and an actual correction |

## Compact research release-note draft

TerraDelta packages a bounded real-data experiment in observation-aware building
change histories, with native geospatial metadata, explicit timing uncertainty,
append-only corrections and independently readable exports. Its frozen
three-region evaluation failed geographic transfer. The quality gate retained
27 correct matches out of 698 confirmed alerts and missed 1,479 supported events;
its no-quality ablation had higher pooled F1. This is a reproducible research
artifact with known scientific failure, not an operational alerting product.

The release includes the full comparison and failure evidence rather than only
selected attractive image crops. Monthly mosaics and unavailable original
publication timestamps limit timing to observation-supported intervals under an
explicit retrospective month-end-availability assumption. They establish no
exact construction date, live-warning performance, legal fact or cause.

This version includes no global monitoring, land-cover adapter, SAR fusion,
individual inference or generated replacement observations. SpaceNet already
established building tracking and WATCH already studies temporal localization.
TerraDelta tests a specific observation-conditioned event method and revision
contract; it claims neither that persistence nor the project name is new.

## Future work and sharing

A new scientific revision declares nine training, four development and three
fresh confirmation regions because version-2 regions have now been examined.
Image production and source-bound temporal fitting are underway. A genuine frozen
neural embedding-distance comparison is specified before fitting and confirmation
scoring. No fresh confirmation result, delivery date or guarantee is promised.
Preserve the earlier negative versions.

Keep any eventual announcement relevant to the audience and link the actual
source, permitted artifacts, executed checks and failed comparison. Do not post
this draft or contact others without applicable user authorization. The current
repository/release links above have been verified; refresh their state before any
eventual announcement.
