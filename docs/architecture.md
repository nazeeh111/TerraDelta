# Architecture

The editable source is [architecture.mmd](architecture.mmd). The same core exports
an event history for both the local analyst interface and an independent consumer.

```mermaid
flowchart LR
  A[Permitted RGBA imagery and UDM masks] --> B[Native grid, identities and anchored QA]
  B --> C[Trained compact U-Net]
  C --> D[Causal sites and spatial evidence]
  B --> E[Validity, elapsed time, seasonal reference]
  D --> F[Observation-conditioned temporal emissions]
  E --> F
  F --> G[Persistence, abstention and timing interval]
  G --> H[Append-only DuckDB revisions]
  H --> I[Typed area/date/snapshot API]
  I --> J[Analyst image evidence interface]
  I --> K[Independent export consumer]
  C --> L[Native source-linked GeoTIFFs]
```

The spatial model never sees confirmation labels. Image-only candidate geometry is
fixed at discovery. Missing or registration-rejected observations cannot update
the temporal state or become absence evidence. The decoder distinguishes inventory
present in the first usable image from a supported subsequent addition.

Acquisition time, assumed/known availability and ledger record time are distinct.
The current import workflow supports acquisition-ordered sequences with monotonic
availability. Out-of-order publication is rejected until availability-order replay
is implemented. Historical answers use immutable ledger snapshot tokens.

Retrospective SN7 mean-refinement uses future evidence and is named and evaluated
separately. Its detection availability includes the full sequence. Online methods
are tested under future extension and edits without rewriting earlier answers.
