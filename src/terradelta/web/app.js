"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const stateNames = { proposed: "Proposed", confirmed_by_model: "Model confirmed", reviewed: "Reviewed", retracted: "Retracted" };
  const stateColors = { proposed: "#b17c27", confirmed_by_model: "#75be80", reviewed: "#78b3db", retracted: "#d98069" };
  const state = { catalog: null, region: null, frameIndex: 0, features: [], selected: null, snapshot: null, paths: [], querySequence: 0, historySequence: 0, imageSequence: 0, imageLoaded: false };
  const element = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (key === "text") node.textContent = value;
      else if (key === "class") node.className = value;
      else node.setAttribute(key, value);
    });
    (Array.isArray(children) ? children : [children]).forEach((child) => {
      if (child != null) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return node;
  };
  const labelState = (value) => stateNames[value] || "Unknown state";
  const badge = (value) => element("span", { class: `state-tag ${Object.hasOwn(stateNames, value) ? value : ""}`, text: labelState(value) });
  const className = (value) => value === "building_addition" ? "Building addition" : value === "building_removal" ? "Building removal" : "Unsupported class";
  const month = (value) => {
    const date = new Date(value);
    return Number.isFinite(date.valueOf()) ? date.toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" }) : "Unknown month";
  };
  const timestamp = (value) => {
    const date = new Date(value);
    return Number.isFinite(date.valueOf()) ? date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" }) + " UTC" : "Unknown";
  };
  const interval = (value) => {
    if (!value) return "Timing support unavailable";
    return (value.lower ? month(value.lower) : "Before the first observation") + " – " + month(value.upper);
  };
  const area = (value) => Number.isFinite(Number(value)) ? `${Math.round(Number(value)).toLocaleString()} m²` : "Area unavailable";
  const frameUrl = (region, index) => `/api/regions/${encodeURIComponent(region.region_id)}/frames/${index}.png`;
  const feedback = (message, error = false) => {
    $("query-feedback").textContent = message;
    $("query-feedback").classList.toggle("error", error);
    $("query-feedback").setAttribute("role", error ? "alert" : "status");
  };
  const safeSourceLink = (uri) => {
    try {
      const url = new URL(uri);
      return ["https:", "http:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch { return null; }
  };
  async function getJSON(url) {
    const response = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
    if (!response.ok) {
      let detail = `Request failed (${response.status}).`;
      try {
        const body = await response.json();
        if (typeof body.detail === "string") detail += ` ${body.detail}`;
      } catch { /* Keep bounded status text if the server does not return JSON. */ }
      throw new Error(detail);
    }
    return response.json();
  }
  function definitionList(items) {
    return element("dl", {}, items.flatMap(([term, description]) => [element("dt", { text: term }), element("dd", {}, description)]));
  }
  function dateInput(value) { return String(value || "").slice(0, 10); }
  function regionCaption(region) {
    if (!region.bbox || region.bbox.length !== 4) return region.region_id;
    const [west, south, east, north] = region.bbox;
    const latitude = (south + north) / 2, longitude = (west + east) / 2;
    return `${Math.abs(latitude).toFixed(2)}° ${latitude < 0 ? "S" : "N"}, ${Math.abs(longitude).toFixed(2)}° ${longitude < 0 ? "W" : "E"}`;
  }
  function queryParameters() {
    const start = $("start").value, end = $("end").value;
    if (!state.region || !start || !end) throw new Error("Choose an area and both dates.");
    if (start > end) throw new Error("The starting date must be on or before the ending date.");
    const params = new URLSearchParams({ region_id: state.region.region_id, start: `${start}T00:00:00+00:00`, end: `${end}T23:59:59+00:00`, include_retracted: String($("include-retracted").checked) });
    const bboxText = $("bbox").value.trim();
    if (bboxText) {
      const box = bboxText.split(",").map((v) => Number(v.trim()));
      if (box.length !== 4 || box.some((v) => !Number.isFinite(v)) || box[0] >= box[2] || box[1] >= box[3] || box[0] < -180 || box[2] > 180 || box[1] < -90 || box[3] > 90) {
        throw new Error("Enter a valid box as west, south, east, north in longitude/latitude.");
      }
      params.set("bbox", box.join(","));
    } else if (state.region.bbox) params.set("bbox", state.region.bbox.join(","));
    const asOf = $("as-of").value.trim();
    if (asOf) {
      if (!/(Z|[+-]\d{2}:\d{2})$/i.test(asOf) || !Number.isFinite(new Date(asOf).valueOf())) throw new Error("The snapshot must be an ISO timestamp with a timezone, such as 2026-10-29T12:00:00+00:00.");
      params.set("as_of", asOf);
    }
    return params;
  }
  function enableExports(params, snapshot) {
    const exportParams = new URLSearchParams(params);
    if (snapshot) exportParams.set("as_of", snapshot);
    [ ["export-geojson", "/api/export.geojson"], ["export-parquet", "/api/export.parquet"] ].forEach(([id, url]) => {
      $(id).href = `${url}?${exportParams}`;
      $(id).classList.remove("disabled-link");
      $(id).removeAttribute("aria-disabled");
    });
  }
  async function queryHistory() {
    let params;
    try { params = queryParameters(); }
    catch (error) { feedback(error.message, true); return; }
    const sequence = ++state.querySequence;
    $("query-button").disabled = true;
    $("query-button").textContent = "Querying history…";
    $("event-list").setAttribute("aria-busy", "true");
    state.features = []; state.selected = null; state.snapshot = null;
    $("event-detail").hidden = true; $("clear-selection").hidden = true;
    $("event-count").textContent = "—";
    $("copy-snapshot").disabled = true;
    $("snapshot-status").textContent = "A new history query is pending. No new snapshot has been returned.";
    ["export-geojson", "export-parquet"].forEach((id) => { $(id).removeAttribute("href"); $(id).classList.add("disabled-link"); $(id).setAttribute("aria-disabled", "true"); });
    $("event-list").replaceChildren(emptyState("Reading recorded history…", "Waiting for the local event store."));
    drawOverlay();
    feedback("Reading the recorded event history…");
    try {
      const result = await getJSON(`/api/events?${params}`);
      if (sequence !== state.querySequence) return;
      state.features = Array.isArray(result.features) ? result.features : [];
      state.snapshot = result.snapshot_time || result.metadata?.snapshot_time || null;
      state.selected = null;
      $("event-detail").hidden = true;
      $("clear-selection").hidden = true;
      renderEvents();
      drawOverlay();
      enableExports(params, state.snapshot);
      const count = state.features.length;
      feedback(`${count.toLocaleString()} ${count === 1 ? "event" : "events"} returned for this area and date interval. Inspect coverage before interpreting an empty result.`);
      $("snapshot-status").textContent = state.snapshot ? `Recorded answer at ${state.snapshot}. This snapshot token replays the same historical revisions.` : "The ledger has no recorded revision timestamp. No immutable snapshot token is available.";
      $("copy-snapshot").disabled = !state.snapshot;
      $("snapshot-copy-feedback").textContent = "";
    } catch (error) {
      if (sequence === state.querySequence) {
        feedback(`${error.message} Check the dates, area and local API, then query again.`, true);
        $("event-list").replaceChildren(emptyState("History could not be loaded.", "No result is being presented for this query. Check the query error and try again."));
        $("event-count").textContent = "—";
        $("snapshot-status").textContent = "The query failed; no recorded answer or export snapshot was returned.";
      }
    } finally {
      if (sequence === state.querySequence) {
        $("query-button").disabled = false;
        $("query-button").replaceChildren("Query history ", element("span", { "aria-hidden": "true", text: "→" }));
        $("event-list").removeAttribute("aria-busy");
      }
    }
  }
  function emptyState(title, text) {
    return element("div", { class: "empty-state" }, [element("span", { "aria-hidden": "true", text: "↳" }), element("h3", { text: title }), element("p", { text })]);
  }
  function visibleFeatures() {
    const filter = $("state-filter").value;
    return state.features.filter((feature) => filter === "all" || feature.properties?.state === filter);
  }
  function renderEvents() {
    const features = visibleFeatures();
    $("event-count").textContent = features.length.toLocaleString();
    if (!features.length) {
      $("event-list").replaceChildren(emptyState("No supported events returned.", "This does not establish no physical change. Check usable observations, the query period and the model's limits."));
      return;
    }
    $("event-list").replaceChildren(...features.map((feature) => {
      const p = feature.properties || {};
      const id = p.event_id || feature.id;
      const button = element("button", { type: "button", class: `event-card${state.selected === id ? " selected" : ""}`, "aria-pressed": String(state.selected === id), "aria-label": `${className(p.change_class)}, ${labelState(p.state)}, ${interval(p.observation_interval)}` }, [
        element("span", { class: "event-card-top" }, [element("span", { class: "event-kind", text: className(p.change_class) }), badge(p.state)]),
        element("span", { class: "event-interval", text: interval(p.observation_interval) }),
        element("span", { class: "event-card-bottom" }, [element("span", { text: area(p.area_m2) }), element("span", { class: "event-id", text: `${String(id || "").slice(0, 12)} · r${p.revision || "?"}` })]),
      ]);
      button.addEventListener("click", () => selectEvent(feature));
      return button;
    }));
  }
  function supportingImages(sources) {
    const linked = sources.map((source) => ({ source, index: state.region.frames.findIndex((frame) => frame.source_id === source.source_id) })).filter((item) => item.index >= 0);
    const chosen = linked.length <= 2 ? linked : [linked[0], linked[linked.length - 1]];
    return element("div", { class: "support-grid" }, chosen.map(({ source, index }) => {
      const button = element("button", { class: "support-image", type: "button", "aria-label": `Inspect supporting observation ${month(source.acquisition_start)}` }, [element("img", { src: frameUrl(state.region, index), alt: `Complete area, supporting observation ${month(source.acquisition_start)}`, loading: "lazy" }), element("span", { text: month(source.acquisition_start) })]);
      button.addEventListener("click", () => { $("frame").value = String(index); showFrame(index); $("frame").focus(); });
      return button;
    }));
  }
  async function selectEvent(feature) {
    const p = feature.properties || {};
    const id = p.event_id || feature.id;
    state.selected = id;
    const sequence = ++state.historySequence;
    $("clear-selection").hidden = false;
    renderEvents(); drawOverlay();
    const sources = Array.isArray(p.sources) ? p.sources : [];
    const sourceItems = sources.map((source) => {
      const url = safeSourceLink(source.uri);
      return element("li", {}, url ? element("a", { href: url, target: "_blank", rel: "noopener", text: `${month(source.acquisition_start)} · original source ↗`, title: source.source_id }) : element("span", { text: `${month(source.acquisition_start)} · ${source.source_id}` }));
    });
    const timingCard = element("div", { class: "timing-card" }, [element("span", { class: "label", text: "SUPPORTED OBSERVATION INTERVAL" }), element("strong", { text: interval(p.observation_interval) }), element("p", { text: "This is timing support from observations, not a known construction or removal date." })]);
    const detail = $("event-detail");
    detail.replaceChildren(element("div", { class: "detail-title" }, [element("h3", { id: "event-detail-title", text: className(p.change_class) }), badge(p.state)]), timingCard,
      definitionList([["Event identity", element("span", { class: "event-id", text: String(id) })], ["Detected", timestamp(p.detected_at)], ["Recorded", timestamp(p.record_time)], ["Availability", sources.some((source) => source.availability_assumption === "retrospective_month_end") ? "Retrospective month-end assumption; publication unknown" : "Known publication timestamps"], ["Footprint area", area(p.area_m2)], ["Model score", Number.isFinite(Number(p.confidence)) && p.confidence != null ? Number(p.confidence).toFixed(3) : "Not supplied"], ["Model / revision", p.model_revision || "Not supplied"], ["Event revision", String(p.revision || "Unknown")]]),
      element("h4", { class: "detail-subheading", text: "Actual supporting observations" }), supportingImages(sources), element("p", { class: "source-instruction", text: "Select a source to inspect its complete area. All cited images remain available below." }), element("ul", { class: "source-list" }, sourceItems),
      element("details", {}, [element("summary", { class: "detail-subheading", text: "Quality and registration diagnostics" }), definitionList(Object.entries(p.quality || {}).map(([key, value]) => [key.replaceAll("_", " "), typeof value === "object" ? JSON.stringify(value) : String(value)]))]),
      element("h4", { class: "detail-subheading", text: "Revision history" }), element("p", { class: "source-instruction", text: "All recorded revisions, including revisions after the query snapshot. The selected revision above remains the returned answer." }), element("div", { id: "revision-content", role: "status", text: "Reading recorded revisions…" }));
    if (p.reason) detail.insertBefore(element("p", { class: "history-error", text: p.reason }), detail.children[2]);
    detail.hidden = false;
    try {
      const response = await getJSON(`/api/events/${encodeURIComponent(id)}/history`);
      if (sequence !== state.historySequence || state.selected !== id) return;
      const revisions = Array.isArray(response.revisions) ? response.revisions : [];
      const items = revisions.map((revision) => element("li", {}, [element("div", { class: "revision-top" }, [element("strong", { text: `r${revision.revision}` }), badge(revision.state)]), element("span", { text: timestamp(revision.record_time) }), element("p", { text: interval(revision.observation_interval) }), revision.reason ? element("p", { text: revision.reason }) : null]));
      $("revision-content").replaceChildren(revisions.length ? element("ol", { class: "revision-list" }, items) : element("p", { class: "field-help", text: "No revisions returned by the history endpoint." }));
      $("revision-content").removeAttribute("role");
    } catch (error) {
      if (sequence === state.historySequence && state.selected === id) $("revision-content").replaceChildren(element("p", { class: "history-error", text: `${error.message} The selected revision above remains visible.` }));
    }
  }
  function project(lon, lat) {
    const radius = 6378137;
    const [west, south, east, north] = state.region.bounds_native;
    const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat));
    const x = radius * lon * Math.PI / 180;
    const y = radius * Math.log(Math.tan(Math.PI / 4 + clamped * Math.PI / 360));
    return [(x - west) / (east - west) * state.region.width, (north - y) / (north - south) * state.region.height];
  }
  function geometryPath(geometry) {
    const path = new Path2D();
    const polygons = geometry?.type === "Polygon" ? [geometry.coordinates] : geometry?.type === "MultiPolygon" ? geometry.coordinates : [];
    for (const polygon of polygons) for (const ring of polygon) {
      ring.forEach(([lon, lat], index) => { const [x, y] = project(lon, lat); if (index === 0) path.moveTo(x, y); else path.lineTo(x, y); });
      path.closePath();
    }
    return path;
  }
  function drawOverlay() {
    const canvas = $("event-overlay"), context = canvas.getContext("2d");
    if (!state.region) return;
    canvas.width = state.region.width; canvas.height = state.region.height;
    context.clearRect(0, 0, canvas.width, canvas.height);
    state.paths = [];
    if (!$("show-overlay").checked || state.region.crs !== "EPSG:3857" || !state.region.bounds_native) return;
    const scale = state.region.width / Math.max(canvas.clientWidth, 1);
    const features = visibleFeatures();
    const selected = features.find((feature) => (feature.properties?.event_id || feature.id) === state.selected);
    const ordered = selected ? [...features.filter((feature) => feature !== selected), selected] : features;
    for (const feature of ordered) {
      const p = feature.properties || {}, id = p.event_id || feature.id;
      const path = geometryPath(feature.geometry), active = id === state.selected;
      context.setLineDash(["proposed", "retracted"].includes(p.state) ? [3 * scale, 2 * scale] : []);
      context.lineJoin = "round";
      if (active) { context.lineWidth = 4.5 * scale; context.strokeStyle = "#ffffff"; context.stroke(path); }
      context.lineWidth = (active ? 2 : 1.05) * scale;
      context.strokeStyle = active ? "#c6902c" : stateColors[p.state] || "#e4e6df";
      context.fillStyle = active ? "#e0b34d50" : (stateColors[p.state] || "#ffffff") + "25";
      context.fill(path, "evenodd"); context.stroke(path);
      state.paths.push({ path, feature });
    }
  }
  function renderSourceMetadata(frame) {
    const reg = frame.registration || {};
    const sourceItems = [["Source identity", frame.source_id || "Not supplied"], ["Acquisition support", `${timestamp(frame.acquisition_start)} to ${timestamp(frame.acquisition_end)}`], ["Publication", frame.published_at ? timestamp(frame.published_at) : "Unknown; original publication time is unavailable"], ["Availability", frame.availability_assumption === "known_publication" ? "Known publication" : "Assumed available at month end; retrospective use"], ["Usable pixels", Number.isFinite(frame.valid_fraction) ? `${(frame.valid_fraction * 100).toFixed(1)}% of full area` : "Not supplied"], ["Registration", reg.usable === true ? "Passed image-only diagnostic" : "Not usable or not established"], ["Estimated shift", Array.isArray(reg.shift_pixels) ? `${reg.shift_pixels.map((v) => Number(v).toFixed(2)).join(", ")} pixels (row, column)` : "Unavailable"], ["SHA-256", element("span", { class: "event-id", text: frame.sha256 || "Not supplied" })]];
    $("source-metadata").replaceChildren(...definitionList(sourceItems).children);
  }
  function showFrame(index) {
    if (!state.region || !state.region.frames[index]) return;
    const frame = state.region.frames[index], sequence = ++state.imageSequence;
    state.frameIndex = index; state.imageLoaded = false;
    $("image-loading").hidden = false;
    $("scene").hidden = true;
    $("scene-empty").hidden = true;
    $("observation-title").textContent = `Complete study area · ${month(frame.acquisition_start)}`;
    $("scene-image").alt = `Actual SpaceNet satellite monthly mosaic for ${state.region.region_id}, ${month(frame.acquisition_start)}. The complete study area is shown; transparent checkerboard marks unusable imagery.`;
    $("image-dimensions").textContent = `${state.region.width} × ${state.region.height} native pixels · full extent`;
    const valid = frame.valid_fraction;
    const usable = frame.registration?.usable === true && Number.isFinite(valid) && valid > 0;
    const status = usable ? `${(valid * 100).toFixed(1)}% usable area · image-only registration check passed` : "No usable observation established · inspect quality and exclusions";
    $("observation-status").replaceChildren(element("span", { class: `status-dot ${usable ? "" : "warning"}` }), element("span", { text: status }));
    const source = safeSourceLink(frame.uri);
    if (source) { $("source-link").href = source; $("source-link").classList.remove("disabled-link"); $("source-link").removeAttribute("aria-disabled"); }
    else { $("source-link").removeAttribute("href"); $("source-link").classList.add("disabled-link"); $("source-link").setAttribute("aria-disabled", "true"); }
    renderSourceMetadata(frame);
    const image = new Image();
    image.onload = () => {
      if (sequence !== state.imageSequence) return;
      $("scene-image").src = image.src; $("scene").hidden = false;
      $("image-loading").hidden = true; state.imageLoaded = true; drawOverlay();
    };
    image.onerror = () => {
      if (sequence !== state.imageSequence) return;
      $("image-loading").hidden = true; $("scene").hidden = true; $("scene-empty").hidden = false;
      $("scene-empty").replaceChildren(element("span", { "aria-hidden": "true", text: "▱" }), element("p", { text: "The source preview could not be loaded. Use the original GeoTIFF link or select another observation." }));
    };
    image.src = frameUrl(state.region, index);
  }
  function selectRegion() {
    state.querySequence++; state.historySequence++;
    state.region = state.catalog.regions.find((region) => region.region_id === $("region").value);
    state.features = []; state.selected = null; state.snapshot = null;
    $("event-detail").hidden = true; $("clear-selection").hidden = true;
    $("event-count").textContent = "—";
    $("query-button").disabled = false;
    ["export-geojson", "export-parquet"].forEach((id) => { $(id).removeAttribute("href"); $(id).classList.add("disabled-link"); $(id).setAttribute("aria-disabled", "true"); });
    if (!state.region) return;
    const frames = state.region.frames || [];
    $("region-location").textContent = `${regionCaption(state.region)} · ${frames.length} monthly observations`;
    $("extent-chip").textContent = `${state.region.crs || "CRS unknown"} · native grid`;
    $("frame").replaceChildren(...frames.map((frame, index) => element("option", { value: String(index), text: month(frame.acquisition_start) })));
    $("frame").disabled = !frames.length;
    $("bbox").value = ""; $("as-of").value = "";
    if (!frames.length) { feedback("This area has no observations available. Import or configure a complete sequence.", true); return; }
    $("start").value = dateInput(frames[0].acquisition_start);
    $("end").value = dateInput(frames[frames.length - 1].acquisition_end);
    $("frame").value = String(frames.length - 1);
    showFrame(frames.length - 1);
    queryHistory();
  }
  $("query-form").addEventListener("submit", (event) => { event.preventDefault(); queryHistory(); });
  $("region").addEventListener("change", selectRegion);
  $("frame").addEventListener("change", () => showFrame(Number($("frame").value)));
  $("show-overlay").addEventListener("change", drawOverlay);
  $("state-filter").addEventListener("change", () => { renderEvents(); drawOverlay(); });
  $("clear-selection").addEventListener("click", () => { state.selected = null; state.historySequence++; $("event-detail").hidden = true; $("clear-selection").hidden = true; renderEvents(); drawOverlay(); });
  $("event-overlay").addEventListener("click", (event) => {
    if (!state.imageLoaded) return;
    const canvas = $("event-overlay"), rect = canvas.getBoundingClientRect(), context = canvas.getContext("2d");
    const x = (event.clientX - rect.left) / rect.width * canvas.width, y = (event.clientY - rect.top) / rect.height * canvas.height;
    const match = [...state.paths].reverse().find((item) => context.isPointInPath(item.path, x, y, "evenodd"));
    if (match) selectEvent(match.feature);
  });
  $("copy-snapshot").addEventListener("click", async () => {
    if (!state.snapshot) return;
    try { await navigator.clipboard.writeText(state.snapshot); $("snapshot-copy-feedback").textContent = "Snapshot time copied."; }
    catch { $("snapshot-copy-feedback").textContent = `Clipboard unavailable. Copy this snapshot time: ${state.snapshot}`; }
  });
  $("import-open").addEventListener("click", () => $("import-dialog").showModal());
  $("import-close").addEventListener("click", () => $("import-dialog").close());
  $("import-dialog").addEventListener("click", (event) => { if (event.target === $("import-dialog")) { const rect = $("import-dialog").getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $("import-dialog").close(); } });
  const observer = new ResizeObserver(drawOverlay); observer.observe($("scene-shell"));
  async function initialize() {
    try {
      const catalog = await getJSON("/api/catalog");
      state.catalog = catalog;
      const regions = Array.isArray(catalog.regions) ? catalog.regions : [];
      state.catalog.regions = regions;
      $("model-revision").textContent = `Installed model: ${catalog.model_revision || "not supplied"}. Coverage is limited to imported, validated sequences.`;
      $("limits-list").replaceChildren(...(Array.isArray(catalog.limits) ? catalog.limits : []).map((limit) => element("li", { text: String(limit) })));
      if (!regions.length) {
        $("region").replaceChildren(element("option", { text: "No imported areas" }));
        feedback("No study areas are configured. Import permitted imagery with the CLI, compute a history, then restart the local workspace.", true);
        return;
      }
      $("region").replaceChildren(...regions.map((region) => element("option", { value: region.region_id, text: `${regionCaption(region)} · ${region.region_id.split("_")[0]}` })));
      $("query-fields").disabled = false;
      selectRegion();
    } catch (error) {
      feedback(`${error.message} The local dataset could not be loaded. Start the TerraDelta API with a configured dataset, then reload.`, true);
      $("region").replaceChildren(element("option", { text: "Local dataset unavailable" }));
      $("model-revision").textContent = "Model metadata is unavailable until the local API connects.";
    }
  }
  initialize();
})();
