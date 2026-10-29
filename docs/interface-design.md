# Analyst workspace

The embedded interface follows the analyst's evidence path: choose a complete imported area and an observation interval, query event history, inspect an event's timing support and its actual source images, then export a recorded answer. It makes no global coverage or causal attribution claim. Native monthly imagery occupies the center; a separate source-month selector does not change the event query dates.

The interface uses local system typography, warm neutral surfaces and restrained green accents. Proposed, reviewed and retracted states carry text labels as well as distinguishable outline styles and colors. Transparent source pixels expose a checkerboard exclusion pattern. Selected event outlines gain a white halo so their native geometry remains visible over varying imagery.

`index.html`, `style.css` and `app.js` are dependency-free package assets. The service should mount these files at `/web` and serve `index.html` at `/`. It must provide the agreed `/api/catalog`, `/api/events`, `/api/events/{id}/history`, `/api/regions/{id}/frames/{index}.png` and export endpoints. Canvas geometry projection is explicitly limited to EPSG:3857 source grids with WGS84 event geometries, matching the validated SpaceNet 7 adapter. Other source CRSs show imagery without silently drawing incorrectly positioned overlays.

The complete area remains visible, without initial event cropping. Two linked supporting source previews open their corresponding full-area observation; all cited original image links are retained. A model score is labeled as a model score, rather than a probability of legal or physical truth. The source panel exposes acquisition support, publication uncertainty, coverage, registration diagnostics and SHA-256 identity.

Query results preserve their returned snapshot token in export links. A snapshot may be copied and replayed. Newer history revisions are shown separately when an event is selected, so the UI does not silently replace a prior query's selected revision. Empty results explicitly differ from observed no change; unavailable/quality-rejected imagery receives its own status. Starting a new query clears the previous overlays and disables prior export links. Fetch failures show a persistent error rather than presenting stale results under new dates or displaying a fabricated empty success.

Accessibility includes native labeled form controls, keyboard-accessible result buttons, a skip link, visible focus rings, text alternatives for source previews, live query/error feedback and a native modal dialog. Event cards provide the keyboard equivalent of canvas selection. Layouts adapt from three desktop columns to stacked mobile content; transitions respect reduced-motion preferences.

## Verification status

Node 24 JavaScript syntax checking passed. Source review covered query races, stale exports/overlays, image-load races, safe source links, DOM text handling, projection/geometry handling and keyboard-equivalent event controls. A local static-layout browser attempt timed out in the CUA tool before returning a screenshot, so no rendered result is claimed. The parent implementation owner will verify the actual current data/API server once event products exist. No synthetic results are bundled as a product demonstration.
