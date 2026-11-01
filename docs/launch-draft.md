# Research-release material: draft, not posted

**Publication status:** prepared draft only. Public release, remote CI and
release-download verification remain pending. Version-2 confirmation is complete
and negative: failed geographic transfer, not an analyst-ready detector. The
[full result](results-v2.md) and original artifacts must accompany any eventual
research release. No announcement or outreach has been sent, and no adoption or
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
| Independent execution | [Fresh wheel and changed cropped/prefix inputs](../artifacts/verification-v2/independent-install.json), [actual API queries/readback](../artifacts/verification-v2/independent-api.json) and [second consumer](../artifacts/verification-v2/second-consumer.json) |
| Public repository | Intended namespace `nazeeh111/TerraDelta`; verify public state before sharing |
| Public CI, clean clone and downloaded release | Pending, separate from local wheel checks |
| Complete-area demonstration | Pending final current-source video and source/revision walkthrough |

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

A new scientific revision needs a declared method/protocol and fresh complete
geographic confirmation data because version-2 regions have now been examined.
Feasibility work is ongoing; no positive future score, delivery date or guarantee
is promised. Preserve both earlier negative versions.

Keep any eventual announcement relevant to the audience and link the actual
source, permitted artifacts, executed checks and failed comparison. Do not post
this draft or contact others without applicable user authorization. The current
repository/release state must be verified before filling public links.
