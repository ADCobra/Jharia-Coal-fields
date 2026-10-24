"use strict";

const COLORS = { high: "#d23f1d", medium: "#e2b007", low: "#7d878d", excluded: "#3f6e8c", coal: "#1e2326" };
const DECISION = {
  close: { chip: "Road closed", order: 0 },
  escalate: { chip: "Asked you", order: 1 },
  log: { chip: "Watching", order: 2 },
  exclude: { chip: "Ignored", order: 3 },
};
const TOOL_WORDS = {
  fetch_history: "Checked earlier photos",
  refine_alignment: "Double-checked the photo alignment",
  check_industrial: "Checked whether it is a factory",
  fetch_highres: "Looked at a sharper Sentinel-2 photo",
  request_drone: "Asked for a drone survey",
};
const METHOD = { aliked_lightglue: "ALIKED + LightGlue", disk: "DISK", sift: "SIFT", orb: "ORB", reference: "reference photo", none: "none" };
const PLATFORM = { LANDSAT_8: "Landsat 8", LANDSAT_9: "Landsat 9" };
const SVGNS = "http://www.w3.org/2000/svg";
const THUMB = { w: 128, h: 96 };

const state = { places: [], alertsNewest: "", passClosed: new Set(), isLatest: true, summary: null, runs: [], runId: null, run: null, hotspots: [], hsGeo: null, cones: null, selected: null, layers: {}, map: null, view: "photo", roads: null, roadNames: {}, routes: null, approvals: [], play: null };

function $(id) { return document.getElementById(id); }
function asset(path) { return window.FW_ASSET ? window.FW_ASSET(path) : path; }

// Under a prefix (the read-only real-data view at /real), links to the site root and the API stay in that view.
if (window.FW_BASE) {
  document.querySelectorAll('a[href="/"]').forEach((a) => { a.href = `${window.FW_BASE}/`; });
  document.querySelectorAll('a[href^="/api/"]').forEach((a) => { a.href = window.FW_BASE + a.getAttribute("href"); });
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function showOffline(on, text) {
  const bar = $("offline");
  if (!bar) return;
  bar.hidden = !on;
  if (text) $("offline-text").textContent = text;
}

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(window.FW_API ? window.FW_API(path) : path, { headers: { "Content-Type": "application/json" }, ...options });
  } catch (err) {
    showOffline(true);
    throw new Error("the server can't be reached");
  }
  showOffline(false);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (err) { detail = res.statusText; }
    throw new Error(detail);
  }
  return res.json();
}

function fmtDate(iso, opts = { day: "numeric", month: "short", year: "numeric" }) {
  return iso ? new Date(iso).toLocaleDateString("en-GB", { ...opts, timeZone: "UTC" }) : "";
}
function fmtNum(v, digits = 0, unit = "") {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return `${Number(v).toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits })}${unit}`;
}
function plural(n, word, many = `${word}s`) { return `${n} ${n === 1 ? word : many}`; }
function joinWords(items) { return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`; }
function capital(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function fmtDist(m) {
  if (m === null || m === undefined) return "n/a";
  return m >= 1000 ? `${fmtNum(m / 1000, 1)} km` : `${fmtNum(Math.round(m / 10) * 10, 0)} m`;
}

function levelOf(h) { return h.decision === "exclude" ? "excluded" : (h.risk && h.risk.level) || "low"; }
function fireNumber(h) { return h.track_id ? parseInt(String(h.track_id).replace(/\D/g, ""), 10) : null; }
function fireName(h) {
  if (h.decision === "exclude") return "Factory heat";
  const n = fireNumber(h);
  return n ? `Fire ${n}` : "New hot spot";
}
function roadName(id) {
  if (!id) return "a road";
  const name = state.roadNames[id];
  return name && name !== id ? `${name} (${id})` : id;
}
function roadPlain(id) {
  const name = state.roadNames[id];
  return name || id || "a road";
}
function distPhrase(h) {
  const r = h.risk || {};
  if (!r.nearest_segment) return "No road nearby";
  if (r.distance_m !== null && r.distance_m < 15) return `On ${roadName(r.nearest_segment)}`;
  return `${fmtDist(r.distance_m)} from ${roadName(r.nearest_segment)}`;
}
function movingPhrase(h) {
  const r = h.risk || {};
  return r.approach_m_per_day > 0.2 ? `, moving toward it about ${fmtNum(r.approach_m_per_day, 1)} m a day` : "";
}
function fireLine(h) {
  if (h.decision === "exclude") return `${fmtNum(h.max_anomaly_k, 0)} \u00b0C hotter than its surroundings and shaped like a factory, so it is ignored.`;
  return `${distPhrase(h)}${movingPhrase(h)}.`;
}
function cloudWords(f) {
  const pct = Math.round((f || 0) * 100);
  return pct < 5 ? "clear sky" : `${pct}% cloud`;
}
function platformOf(run) { return PLATFORM[run && run.platform] || (run && run.platform ? String(run.platform).replace(/_/g, " ") : "Landsat"); }

const TRUCK_SVG = (loaded) => `<svg viewBox="0 0 30 14" width="30" height="14" aria-hidden="true"><rect x="1" y="2" width="19" height="10" rx="1.5" fill="#e2b007" stroke="#3a2f00" stroke-width="1"/><rect x="3.5" y="4" width="14" height="6" rx="1" fill="${loaded ? "#1b1d1f" : "#b98c00"}"/><rect x="21" y="2.5" width="7.5" height="9" rx="1.5" fill="#2b3134" stroke="#111" stroke-width="0.8"/><rect x="25.5" y="4" width="2" height="6" fill="#9fb4c0"/></svg>`;
const BARRIER_SVG = `<svg viewBox="0 0 26 12" width="26" height="12" aria-hidden="true"><rect x="1" y="1" width="24" height="7" rx="1" fill="#fff" stroke="#9c2a10" stroke-width="1"/><path d="M4 1l-3 4M10 1l-6 7M16 1l-6 7M22 1l-6 7M25 4l-4 4" stroke="#d23f1d" stroke-width="2.4"/><rect x="3" y="8" width="2" height="4" fill="#333"/><rect x="21" y="8" width="2" height="4" fill="#333"/></svg>`;
const SITE_SVG = {
  pit: `<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="#1e2326" stroke="#fff" stroke-width="1.5"/><path d="M5 13l5-7 5 7z" fill="none" stroke="#e2b007" stroke-width="1.8"/><path d="M7.5 13l2.5-3.5 2.5 3.5" fill="none" stroke="#e2b007" stroke-width="1.2"/></svg>`,
  washery: `<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="#3f6e8c" stroke="#fff" stroke-width="1.5"/><path d="M5 14V8l3 2V8l3 2V6h4v8z" fill="#fff"/></svg>`,
  plant: `<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true"><circle cx="10" cy="10" r="9" fill="#59636a" stroke="#fff" stroke-width="1.5"/><path d="M5 14V9h3v5M9 14V7h2v7M12 14V5h3v9" fill="#fff"/></svg>`,
};

function initMap() {
  const map = L.map("map", { zoomControl: true, zoomSnap: 0.25 });
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 17, attribution: "Imagery outside the coalfield &copy; Esri, Maxar, Earthstar Geographics" }).addTo(map);
  state.map = map;
  const b = state.summary && state.summary.bounds;
  if (b) map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [10, 10] });
  else map.setView([23.745, 86.3], 11);
  map.createPane("scenery").style.zIndex = 250;
  map.createPane("photo").style.zIndex = 260;
  map.createPane("heat").style.zIndex = 300;
  map.createPane("labels").style.zIndex = 620;
  map.createPane("trucks").style.zIndex = 640;
  const sb = state.summary && state.summary.scenery_bounds;
  if (sb) state.layers.scenery = L.imageOverlay(asset("/api/scenery"), [[sb[1], sb[0]], [sb[3], sb[2]]], { pane: "scenery", interactive: false }).addTo(map);
  state.layers.sites = L.layerGroup().addTo(map);
  state.layers.barriers = L.layerGroup().addTo(map);
  state.layers.glow = L.layerGroup().addTo(map);
  state.layers.labels = L.layerGroup().addTo(map);
  state.layers.trucks = L.layerGroup().addTo(map);
  const zoomClass = () => { $("map").classList.toggle("far", map.getZoom() < 12); $("map").classList.toggle("very-far", map.getZoom() < 11); };
  map.on("zoomend", zoomClass);
  zoomClass();
  state.layers.roads = L.geoJSON(null, { style: roadStyle, onEachFeature: (f, layer) => layer.bindTooltip(roadLabel(f.properties), { sticky: true }) }).addTo(map);
  state.layers.routes = L.geoJSON(null, { style: (f) => ({ color: "#ffd75a", weight: 2, opacity: f.properties.kind === "current" ? 0.5 : 0.0, dashArray: "2 7", lineCap: "round" }), interactive: false }).addTo(map);
  state.layers.cones = L.geoJSON(null, { style: coneStyle, interactive: false }).addTo(map);
  state.layers.hotspots = L.geoJSON(null, { style: hotspotStyle, pointToLayer: (f, ll) => L.circleMarker(ll, { radius: 6 }), onEachFeature: (f, layer) => {
    layer.on("click", () => { if (!ruler.active) selectHotspot(f.properties.hotspot_id, true); });
  } }).addTo(map);
  initRuler(map);
}

const RULER_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M3 16.5 16.5 3 21 7.5 7.5 21z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="m7 13 2 2M10 10l2 2M13 7l2 2M8.5 11.5l1 1M11.5 8.5l1 1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
const ruler = { active: false, points: [], casing: null, line: null, dots: null, label: null, button: null, hint: null };

function rulerDistance() {
  let d = 0;
  for (let i = 1; i < ruler.points.length; i++) d += state.map.distance(ruler.points[i - 1], ruler.points[i]);
  return d;
}

function rulerDraw() {
  ruler.line.setLatLngs(ruler.points);
  ruler.casing.setLatLngs(ruler.points);
  ruler.dots.clearLayers();
  ruler.points.forEach((p) => L.circleMarker(p, { radius: 4, color: "#1e2326", weight: 2, fillColor: "#fff", fillOpacity: 1, interactive: false }).addTo(ruler.dots));
  if (ruler.label) { ruler.label.remove(); ruler.label = null; }
  if (ruler.points.length > 1) {
    const last = ruler.points[ruler.points.length - 1];
    ruler.label = L.tooltip({ permanent: true, direction: "right", offset: [10, 0], className: "ruler-label" }).setLatLng(last).setContent(fmtDist(rulerDistance())).addTo(state.map);
  }
}

function rulerSet(on) {
  ruler.active = on;
  ruler.button.setAttribute("aria-pressed", String(on));
  ruler.button.classList.toggle("on", on);
  $("map").classList.toggle("measuring", on);
  ruler.hint.hidden = !on;
  if (on) {
    ruler.points = [];
    rulerDraw();
    state.map.doubleClickZoom.disable();
  } else {
    state.map.doubleClickZoom.enable();
  }
}

function rulerClear() {
  ruler.points = [];
  rulerDraw();
}

function initRuler(map) {
  ruler.casing = L.polyline([], { color: "#1e2326", weight: 6, opacity: 0.85, interactive: false }).addTo(map);
  ruler.line = L.polyline([], { color: "#ffd75a", weight: 3, dashArray: "10 6", interactive: false }).addTo(map);
  ruler.dots = L.layerGroup().addTo(map);
  const Control = L.Control.extend({
    options: { position: "topleft" },
    onAdd() {
      const box = L.DomUtil.create("div", "leaflet-bar ruler-ctl");
      const b = L.DomUtil.create("a", "", box);
      b.href = "#";
      b.setAttribute("role", "button");
      b.setAttribute("aria-label", "Measure a distance");
      b.setAttribute("aria-pressed", "false");
      b.title = "Measure a distance";
      b.innerHTML = RULER_SVG;
      L.DomEvent.on(b, "click", (e) => { L.DomEvent.stop(e); if (ruler.active) { rulerSet(false); rulerClear(); } else rulerSet(true); });
      L.DomEvent.disableClickPropagation(box);
      ruler.button = b;
      return box;
    },
  });
  new Control().addTo(map);
  ruler.hint = el("p", { class: "ruler-hint", hidden: true, text: "Click points on the map to measure. Double-click or press Enter to finish, Esc to clear." });
  $("map").parentElement.append(ruler.hint);
  map.on("click", (e) => {
    if (!ruler.active) return;
    ruler.points.push(e.latlng);
    rulerDraw();
  });
  map.on("dblclick", () => { if (ruler.active) { ruler.active = false; rulerSet(false); } });
}

function mapNote() {
  const note = $("map-note");
  const sc = state.summary && state.summary.scenery;
  const run = state.run;
  let text = "";
  if (state.view === "photo") {
    if (run && run.layers && run.layers.photo) text = `${platformOf(run)} photo of ${fmtDate(run.acquired)}`;
    else if (sc && sc.kind === "photo") text = `${PLATFORM[sc.platform] || "Landsat"} photo of ${fmtDate(sc.acquired)}`;
    else if (sc) text = "Simulated aerial view (demo data)";
  } else if (state.view === "anomaly") text = "Heat above what the ground should be, on this date";
  else if (state.view === "lst") text = "Ground temperature measured from space, on this date";
  note.hidden = !text;
  note.textContent = text;
}

async function loadSites() {
  const [sites, places] = await Promise.all([api("/api/sites").catch(() => ({ features: [] })), api("/api/places").catch(() => [])]);
  state.places = places;
  buildFindList();
  state.layers.sites.clearLayers();
  for (const f of sites.features) {
    const [lon, lat] = f.geometry.coordinates;
    const kind = f.properties.kind;
    if (kind === "town") {
      L.marker([lat, lon], { icon: L.divIcon({ className: "town-icon", html: `<span class="town">${escapeHtml(f.properties.name)}</span>`, iconSize: [0, 0] }), interactive: false, keyboard: false, pane: "labels" }).addTo(state.layers.sites);
      continue;
    }
    const icon = L.divIcon({ className: `site site-${kind}`, html: `${SITE_SVG[kind] || SITE_SVG.pit}<span class="site-label">${escapeHtml(f.properties.name)}</span>`, iconSize: [20, 20], iconAnchor: [10, 10] });
    L.marker([lat, lon], { icon, keyboard: false, interactive: false }).addTo(state.layers.sites);
  }
}

function drawBarriers(roads) {
  state.layers.barriers.clearLayers();
  const ends = {};
  for (const f of roads.features) {
    if (roadStatus(f.properties) !== "closed") continue;
    const coords = f.geometry.coordinates;
    for (const c of [coords[0], coords[coords.length - 1]]) {
      const key = `${c[0].toFixed(4)},${c[1].toFixed(4)}`;
      ends[key] = ends[key] ? { c, n: ends[key].n + 1 } : { c, n: 1 };
    }
  }
  for (const { c, n } of Object.values(ends)) {
    if (n > 1) continue;
    L.marker([c[1], c[0]], { icon: L.divIcon({ className: "barrier", html: BARRIER_SVG, iconSize: [26, 12], iconAnchor: [13, 6] }), interactive: false, keyboard: false }).addTo(state.layers.barriers);
  }
}

function drawFireMarks() {
  state.layers.glow.clearLayers();
  state.layers.labels.eachLayer((layer) => { if (layer.options.fire) state.layers.labels.removeLayer(layer); });
  for (const h of state.hotspots) {
    const level = levelOf(h);
    const ll = [h.centroid_lonlat[1], h.centroid_lonlat[0]];
    if (["high", "medium"].includes(level)) {
      const size = level === "high" ? 46 : 32;
      L.marker(ll, { icon: L.divIcon({ className: `glow glow-${level}`, html: "<span></span>", iconSize: [size, size], iconAnchor: [size / 2, size / 2] }), interactive: false, keyboard: false, pane: "shadowPane" }).addTo(state.layers.glow);
    }
    const label = L.marker(ll, { icon: L.divIcon({ className: `fire-label l-${level}`, html: `<span>${escapeHtml(fireName(h))}</span>`, iconSize: [0, 0] }), keyboard: false, pane: "labels", fire: true });
    label.on("click", () => selectHotspot(h.hotspot_id, true));
    label.addTo(state.layers.labels);
  }
}

const traffic = { trucks: [], raf: null, last: 0, still: window.matchMedia("(prefers-reduced-motion: reduce)").matches };

function pathOf(coords) {
  const pts = coords.map(([lon, lat]) => L.latLng(lat, lon));
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i - 1].distanceTo(pts[i]));
  return { pts, cum, length: cum[cum.length - 1] };
}

function pointAt(path, d) {
  const { pts, cum } = path;
  let i = 1;
  while (i < cum.length - 1 && cum[i] < d) i++;
  const seg = Math.max(1e-6, cum[i] - cum[i - 1]);
  const t = Math.min(1, Math.max(0, (d - cum[i - 1]) / seg));
  const a = pts[i - 1], b = pts[i];
  return { ll: L.latLng(a.lat + (b.lat - a.lat) * t, a.lng + (b.lng - a.lng) * t), a, b };
}

function truckIcon(loaded, waiting) {
  return L.divIcon({ className: `truck${waiting ? " waiting" : ""}`, html: `<span class="truck-body">${TRUCK_SVG(loaded)}</span>${waiting ? '<span class="truck-alert">!</span>' : ""}`, iconSize: [30, 14], iconAnchor: [15, 7] });
}

function closedVertexKeys() {
  const keys = new Set();
  for (const f of (state.roads && state.roads.features) || []) {
    if (roadStatus(f.properties) !== "closed") continue;
    for (const ring of rings(f.geometry)) for (const c of ring) keys.add(`${c[0].toFixed(5)},${c[1].toFixed(5)}`);
  }
  return keys;
}

function buildTraffic(routes) {
  state.layers.trucks.clearLayers();
  traffic.trucks = [];
  const byRoute = {};
  for (const f of routes.geojson.features) (byRoute[f.properties.route] = byRoute[f.properties.route] || {})[f.properties.kind] = f;
  const blocked = new Set(routes.report.filter((r) => r.blocked).map((r) => r.route));
  if (!state.isLatest) {
    blocked.clear();
    const keys = closedVertexKeys();
    for (const [name, kinds] of Object.entries(byRoute)) {
      const normal = kinds.normal;
      if (!normal) continue;
      const hits = normal.geometry.coordinates.filter((c) => keys.has(`${c[0].toFixed(5)},${c[1].toFixed(5)}`)).length;
      if (hits >= 2) blocked.add(name);
      else kinds.current = normal;
    }
  }
  for (const [name, kinds] of Object.entries(byRoute)) {
    const current = kinds.current;
    if (blocked.has(name) || !current) {
      const normal = kinds.normal;
      if (!normal) continue;
      const [lon, lat] = normal.geometry.coordinates[0];
      for (let k = 0; k < 2; k++) {
        const m = L.marker([lat + k * 0.0009, lon + k * 0.0013], { icon: truckIcon(true, true), pane: "trucks", keyboard: false });
        m.bindTooltip(`${escapeHtml(name)}: waiting, the road is closed`, { direction: "top" });
        m.addTo(state.layers.trucks);
      }
      continue;
    }
    const path = pathOf(current.geometry.coordinates);
    if (path.length < 50) continue;
    const detour = state.isLatest && routes.report.find((r) => r.route === name && r.extra_m);
    const count = path.length > 5000 ? 2 : 1;
    for (let k = 0; k < count; k++) {
      const m = L.marker(path.pts[0], { icon: truckIcon(true, false), pane: "trucks", keyboard: false });
      m.bindTooltip(`${escapeHtml(name)}${detour ? `: on a ${fmtNum(detour.extra_m / 1000, 1)} km detour` : ""}`, { direction: "top" });
      m.addTo(state.layers.trucks);
      const seed = [...name].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) % 997, 7) / 997;
      traffic.trucks.push({ marker: m, path, phase: (seed + k / count) % 1, loaded: null, period: Math.max(24, path.length / 160) });
    }
  }
  if (!traffic.raf) traffic.raf = requestAnimationFrame(tick);
}

function tick(now) {
  traffic.raf = requestAnimationFrame(tick);
  if (now - traffic.last < 40 || !$("toggle-traffic").checked) return;
  traffic.last = now;
  const seconds = traffic.still ? 0 : now / 1000;
  for (const t of traffic.trucks) {
    const u = (seconds / t.period + t.phase) % 1;
    const outbound = u < 0.5;
    const d = (outbound ? u * 2 : (1 - u) * 2) * t.path.length;
    const { ll, a, b } = pointAt(t.path, d);
    t.marker.setLatLng(ll);
    if (t.loaded !== outbound) {
      t.loaded = outbound;
      t.marker.setIcon(truckIcon(outbound, false));
    }
    const pa = state.map.latLngToLayerPoint(a), pb = state.map.latLngToLayerPoint(b);
    let angle = Math.atan2(pb.y - pa.y, pb.x - pa.x) * 180 / Math.PI;
    if (!outbound) angle += 180;
    const body = t.marker.getElement() && t.marker.getElement().querySelector(".truck-body");
    if (body) body.style.transform = `rotate(${angle.toFixed(1)}deg)`;
  }
}

function roadStatus(p) {
  if (state.isLatest) return p.status;
  return state.passClosed.has(p.id) ? "closed" : "open";
}
function roadStyle(f) {
  const p = { ...f.properties, status: roadStatus(f.properties) };
  if (p.status === "closed") return { color: COLORS.high, weight: 6, opacity: 0.95 };
  if (p.status === "review") return { color: COLORS.medium, weight: 5, opacity: 0.95 };
  return { color: p.haul ? "#e8dcc0" : "#f4f1ea", weight: p.haul ? 1.6 : 2.6, opacity: p.haul ? 0.55 : 0.75 };
}
function roadLabel(p) {
  const st = roadStatus(p);
  const status = st === "closed" ? "closed" : st === "review" ? "waiting for a decision" : "open";
  return `${escapeHtml(p.name)} (${escapeHtml(p.id)}): ${status}`;
}
function coneStyle(f) {
  const core = f.properties.kind === "core";
  return { color: COLORS.high, weight: core ? 1 : 1.5, dashArray: core ? null : "6 5", fillColor: COLORS.high, fillOpacity: core ? 0.08 : 0.02 };
}
function hotspotStyle(f) {
  const p = f.properties;
  const level = p.decision === "exclude" ? "excluded" : p.level || "low";
  const selected = state.selected && state.selected === p.hotspot_id;
  return { color: selected ? "#ffffff" : COLORS[level], weight: selected ? 4 : 2, fillColor: COLORS[level], fillOpacity: level === "low" ? 0.25 : 0.5 };
}

function showOverview() {
  $("overview").hidden = false;
  $("detail").hidden = true;
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  const s = state.summary;
  const badge = $("data-badge");
  const real = s.source === "real" || s.source === "landsat";
  badge.textContent = real ? "Real Landsat data" : "Demo data (simulated)";
  badge.className = `data-badge${real ? " real" : ""}`;
  badge.setAttribute("aria-label", `${badge.textContent}. About the data`);
  if (!real) $("about-lead").textContent = "This copy runs on simulated demo data: a made-up coalfield with fires placed by a simulator, so every part of the system can be shown. The real setup uses the sources below.";
  $("help-data").textContent = real
    ? "This dashboard shows real Landsat 8 and 9 photos of Jharia from Microsoft Planetary Computer, with roads, mines and plants from OpenStreetMap."
    : "This dashboard is running on simulated demo data: a computer-made coalfield with known fires, used to test the system.";
  if (s.read_only) badge.textContent = "Real Landsat data · read-only";
  const link = $("view-link");
  if (s.other_view) {
    link.href = `${s.other_view.href}dashboard`;
    link.textContent = `Switch to ${s.other_view.label.toLowerCase()} →`;
    link.hidden = false;
  }
  document.body.classList.toggle("read-only", Boolean(s.read_only));
  $("ro-note").hidden = !s.read_only;
  if (s.read_only) $("decisions-block").querySelector("h2").firstChild.textContent = "Decisions raised for a person ";
  const canRun = s.can_run && (s.source === "synthetic" || (s.source === "real" && s.unprocessed > 0));
  $("run-next").hidden = !canRun;
  $("run-next").disabled = Boolean(s.running);
  $("run-status").textContent = s.running ? `Processing ${s.running.label}...` : s.last_error || "";
}

async function loadRoads() {
  const [roads, routes] = await Promise.all([api("/api/roads"), api("/api/routes")]);
  state.roads = roads;
  state.routes = routes;
  state.roadNames = Object.fromEntries(roads.features.map((f) => [f.properties.id, f.properties.name]));
  state.layers.roads.clearLayers().addData(roads);
  state.layers.routes.clearLayers().addData(routes.geojson);
  refreshRoadsForPass();
  const closed = roads.features.filter((f) => f.properties.status === "closed");
  const byName = {};
  for (const f of closed) (byName[f.properties.name] = byName[f.properties.name] || []).push(f.properties.id);
  const list = $("closed-roads");
  list.replaceChildren();
  for (const [name, ids] of Object.entries(byName)) list.append(el("li", {}, el("b", { text: name }), el("span", { class: "note", text: `${plural(ids.length, "section")}: ${ids.join(", ")}` })));
  if (!closed.length) list.append(el("li", { class: "empty", text: "Every road is open." }));
  const routeList = $("routes");
  routeList.replaceChildren();
  let affected = 0;
  for (const r of routes.report) {
    let note;
    if (r.blocked) { note = el("span", { class: "bad", text: "Blocked, trucks waiting" }); affected++; }
    else if (r.extra_m) { note = el("span", { class: "note", text: `${fmtNum(r.extra_m / 1000, 1)} km detour` }); affected++; }
    else note = el("span", { class: "note", text: r.before_m === null ? "No road link" : "Normal route" });
    routeList.append(el("li", {}, el("span", { text: r.route }), note));
  }
  if (!routes.report.length) routeList.append(el("li", { class: "empty", text: "No truck routes set up." }));
  $("roads-sub").textContent = closed.length ? `${plural(closed.length, "section")} closed${affected ? `, ${plural(affected, "truck route")} affected` : ""}` : "all open";
}

async function loadRuns() {
  state.runs = (await api("/api/runs?limit=500")).filter((r) => r.status === "done");
  const dots = $("dots");
  dots.replaceChildren();
  let lastYear = null;
  for (const r of state.runs) {
    const levels = r.by_level || {};
    const worst = levels.high ? "high" : levels.medium ? "medium" : "low";
    const year = new Date(r.acquired).getUTCFullYear();
    const label = year !== lastYear ? fmtDate(r.acquired, { month: "short", year: "2-digit" }) : fmtDate(r.acquired, { day: "numeric", month: "short" });
    lastYear = year;
    const title = `${fmtDate(r.acquired)}: ${plural(r.hotspots || 0, "hot spot")}${r.closed ? `, ${plural(r.closed, "road section")} closed` : ""}`;
    const button = el("button", { class: `dot-btn${r.agent ? "" : " history"}`, title, "aria-label": title, "aria-current": "false", onclick: () => { stopPlay(); selectRun(r.run_id); } },
      el("i", { class: `l-${worst}${r.closed ? " closed" : ""}` }),
      el("span", { text: label }));
    dots.append(el("li", {}, button));
  }
  if (!state.runs.length) dots.append(el("li", { class: "empty", text: "No photos processed yet." }));
}

function refreshRoadsForPass() {
  if (!state.roads || !state.routes) return;
  state.layers.roads.setStyle(roadStyle);
  drawBarriers(state.roads);
  buildTraffic(state.routes);
  state.layers.routes.setStyle((f) => ({ color: "#ffd75a", weight: 2, opacity: state.isLatest && f.properties.kind === "current" ? 0.5 : 0.0, dashArray: "2 7", lineCap: "round" }));
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".map-mode button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.view === view ? "true" : "false"));
  for (const name of ["lst", "anomaly"]) {
    const layer = state.layers[name];
    if (!layer) continue;
    if (name === view) layer.addTo(state.map); else state.map.removeLayer(layer);
  }
  if (state.layers.photo) {
    if (view === "photo") state.layers.photo.addTo(state.map); else state.map.removeLayer(state.layers.photo);
  }
  const scale = $("heat-scale");
  scale.hidden = view === "photo";
  scale.classList.toggle("lst", view === "lst");
  scale.querySelector("span:last-child").textContent = view === "lst" ? "hotter ground" : "hotter than normal";
  scale.querySelector("span:first-child").textContent = view === "lst" ? "cooler ground" : "normal";
  document.querySelectorAll(".step-card").forEach((b) => b.setAttribute("aria-pressed", b.dataset.view === view ? "true" : "false"));
  mapNote();
}

function setOverlay(name, url, bounds, pane, opacity) {
  if (state.layers[name]) { state.map.removeLayer(state.layers[name]); state.layers[name] = null; }
  if (!url || !bounds) return;
  state.layers[name] = L.imageOverlay(url, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], { opacity, interactive: false, pane });
}

async function selectRun(runId, hotspotId = null) {
  state.runId = runId;
  const [run, hotspots, hsGeo, cones] = await Promise.all([
    api(`/api/runs/${runId}`),
    api(`/api/runs/${runId}/hotspots`),
    api(`/api/runs/${runId}/geojson/hotspots`),
    api(`/api/runs/${runId}/geojson/cones`).catch(() => ({ type: "FeatureCollection", features: [] })),
  ]);
  state.run = run;
  state.hotspots = hotspots;
  state.hsGeo = hsGeo;
  state.cones = cones;
  state.passClosed = new Set(run.closed_segments || []);
  state.isLatest = state.runs.length > 0 && state.runs[state.runs.length - 1].run_id === runId;
  refreshRoadsForPass();
  const bounds = run.layers && run.layers.bounds;
  setOverlay("lst", asset(`/api/runs/${runId}/layers/lst`), bounds, "heat", 0.75);
  setOverlay("anomaly", asset(`/api/runs/${runId}/layers/anomaly`), bounds, "heat", 0.9);
  setOverlay("photo", run.layers && run.layers.photo ? asset(`/api/runs/${runId}/layers/photo`) : null, bounds, "photo", 1);
  setView(state.view);
  const fireIds = new Set(hotspots.filter((h) => h.decision !== "exclude" && ((h.risk || {}).level !== "low" || ((h.projection || {}).reach_probability || 0) > 0)).map((h) => h.hotspot_id));
  state.layers.cones.clearLayers().addData({ type: "FeatureCollection", features: cones.features.filter((f) => fireIds.has(f.properties.hotspot_id)) });
  state.layers.hotspots.clearLayers().addData(hsGeo);
  state.layers.cones.bringToFront();
  state.layers.hotspots.bringToFront();
  drawFireMarks();
  renderDateBar();
  renderSteps();
  renderFires();
  renderPassFacts();
  buildFindList();
  document.querySelectorAll(".dot-btn").forEach((b, i) => b.setAttribute("aria-current", state.runs[i] && state.runs[i].run_id === runId ? "true" : "false"));
  const current = document.querySelector('.dot-btn[aria-current="true"]');
  if (current) current.scrollIntoView({ block: "nearest", inline: "center" });
  if (hotspotId && hotspots.some((h) => h.hotspot_id === hotspotId)) await selectHotspot(hotspotId, false);
  else closeDetail();
  writeHash();
}

function closedShown() {
  if (!state.isLatest || !state.roads) return state.run ? state.run.closed_segments || [] : [];
  return state.roads.features.filter((f) => f.properties.status === "closed").map((f) => f.properties.id);
}

function storyFor(run, hotspots) {
  if (!run.agent) return `Fire Watch was still learning what normal looks like on this date, so it only recorded the ${plural(hotspots.length, "hot spot")} it saw.`;
  const count = (k) => hotspots.filter((h) => h.decision === k).length;
  const newly = (run.actions || []).filter((a) => a.kind === "close_road" && a.executed).flatMap((a) => a.target || []);
  const closed = closedShown();
  const sentences = [];
  if (newly.length) {
    sentences.push(`Fire Watch closed ${plural(newly.length, "section")} of ${roadsShort(newly)} because a fire got too close.`);
  } else if (closed.length) {
    sentences.push(`${plural(closed.length, "section")} of ${roadsShort(closed)} ${closed.length === 1 ? "stays" : "stay"} closed.`);
  }
  const parts = [];
  if (count("escalate")) parts.push(`${plural(count("escalate"), "fire")} ${count("escalate") === 1 ? "needs" : "need"} your decision`);
  if (count("log")) parts.push(`${count("log")} ${count("log") === 1 ? "is" : "are"} being watched`);
  if (count("exclude")) parts.push(`${plural(count("exclude"), "factory", "factories")} ignored`);
  if (parts.length) sentences.push(`${capital(joinWords(parts))}.`);
  if (!sentences.length) return "No fires found on this date.";
  return sentences.join(" ");
}

function renderDateBar() {
  const run = state.run;
  const idx = state.runs.findIndex((r) => r.run_id === state.runId);
  $("pass-date").textContent = fmtDate(run.acquired);
  $("pass-sub").textContent = `${platformOf(run)}, photo ${idx + 1} of ${state.runs.length}`;
  $("prev-pass").disabled = idx <= 0;
  $("next-pass").disabled = idx < 0 || idx >= state.runs.length - 1;
  $("pass-story").textContent = storyFor(run, state.hotspots);
}

function topFire() {
  const fires = state.hotspots.filter((h) => h.decision !== "exclude");
  return fires.sort((a, b) => ((b.risk && b.risk.score) || 0) - ((a.risk && a.risk.score) || 0))[0] || state.hotspots[0] || null;
}

function cropAround(lon, lat, spanLon = 0.05) {
  const spanLat = spanLon * Math.cos((lat * Math.PI) / 180) * (THUMB.h / THUMB.w);
  return [lon - spanLon / 2, lat - spanLat / 2, lon + spanLon / 2, lat + spanLat / 2];
}
function centerOf(coords) {
  const n = coords.length || 1;
  return [coords.reduce((s, p) => s + p[0], 0) / n, coords.reduce((s, p) => s + p[1], 0) / n];
}
function cropBg(url, b, c) {
  if (!url || !b) return null;
  const cw = c[2] - c[0], ch = c[3] - c[1];
  const sx = (b[2] - b[0]) / cw, sy = (b[3] - b[1]) / ch;
  const px = ((b[0] - c[0]) / cw) / (1 - sx), py = ((c[3] - b[3]) / ch) / (1 - sy);
  return `url("${url}") ${px * 100}% ${py * 100}% / ${sx * 100}% ${sy * 100}% no-repeat`;
}
function rings(g) {
  if (!g) return [];
  if (g.type === "LineString") return [g.coordinates];
  if (g.type === "MultiLineString" || g.type === "Polygon") return g.coordinates;
  if (g.type === "MultiPolygon") return g.coordinates.flat();
  return [];
}
function overlay(c, shapes) {
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${THUMB.w} ${THUMB.h}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");
  const cw = c[2] - c[0], ch = c[3] - c[1];
  const pt = ([lon, lat]) => `${(((lon - c[0]) / cw) * THUMB.w).toFixed(1)},${(((c[3] - lat) / ch) * THUMB.h).toFixed(1)}`;
  for (const s of shapes) {
    for (const ring of rings(s.geometry)) {
      const path = document.createElementNS(SVGNS, "path");
      path.setAttribute("d", `M${ring.map(pt).join("L")}${s.closed ? "Z" : ""}`);
      path.setAttribute("vector-effect", "non-scaling-stroke");
      for (const [k, v] of Object.entries(s.style)) path.setAttribute(k, v);
      svg.append(path);
    }
  }
  return svg;
}
function pic(layers, shapes = null, crop = null, extra = "") {
  const box = el("span", { class: `step-pic ${extra}`, "aria-hidden": "true" });
  const img = el("span", { class: "img" });
  img.style.background = layers.filter(Boolean).join(", ") || "#6d7568";
  box.append(img);
  if (shapes && crop) box.append(overlay(crop, shapes));
  return box;
}

function renderSteps() {
  const run = state.run;
  const lb = run.layers && run.layers.bounds;
  const sb = state.summary && state.summary.scenery_bounds;
  const scenery = (c) => (sb ? cropBg(asset("/api/scenery"), sb, c) : null);
  const photo = (c) => (run.layers && run.layers.photo ? cropBg(asset(`/api/runs/${run.run_id}/layers/photo`), lb, c) : scenery(c));
  const anomaly = (c) => cropBg(asset(`/api/runs/${run.run_id}/layers/anomaly`), lb, c);
  const fire = topFire();
  const mid = lb ? [(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2] : [86.3, 23.745];
  const focus = cropAround(...(fire ? fire.centroid_lonlat : mid));
  const wide = cropAround(...mid, (lb ? lb[2] - lb[0] : 0.4) * 0.9);
  const fires = state.hotspots.filter((h) => h.decision !== "exclude");
  const factories = state.hotspots.length - fires.length;
  const threatening = fires.filter((h) => {
    const r = h.risk || {};
    return (r.projected_distance_m !== null && r.projected_distance_m !== undefined && r.projected_distance_m < 250) || (r.distance_m !== null && r.distance_m !== undefined && r.distance_m < 250) || ((h.projection || {}).reach_probability || 0) >= 0.2;
  });
  const closed = closedShown();
  const pending = state.approvals.length;
  const roadFeatures = (state.roads && state.roads.features) || [];
  const closedLines = roadFeatures.filter((f) => closed.includes(f.properties.id));
  const roadCrop = closedLines.length ? cropAround(...centerOf(closedLines.flatMap((f) => rings(f.geometry).flat())), 0.07) : focus;
  const hotShapes = [
    ...((state.cones && state.cones.features) || []).filter((f) => f.properties.kind !== "core").map((f) => ({ geometry: f.geometry, closed: true, style: { fill: "none", stroke: COLORS.high, "stroke-width": 1.2, "stroke-dasharray": "4 3" } })),
    ...((state.hsGeo && state.hsGeo.features) || []).map((f) => {
      const level = f.properties.decision === "exclude" ? "excluded" : f.properties.level || "low";
      return { geometry: f.geometry, closed: true, style: { fill: COLORS[level], "fill-opacity": 0.6, stroke: COLORS[level], "stroke-width": 1.5 } };
    }),
  ];
  const roadShapes = roadFeatures.flatMap((f) => {
    const shut = closed.includes(f.properties.id);
    if (shut) return [{ geometry: f.geometry, style: { fill: "none", stroke: COLORS.high, "stroke-width": 4, "stroke-linecap": "round" } }];
    return [{ geometry: f.geometry, style: { fill: "none", stroke: f.properties.haul ? "#f0e4c4" : "#ffffff", "stroke-width": f.properties.haul ? 1 : 1.8, "stroke-opacity": 0.85 } }];
  });
  const steps = [
    { n: 1, title: "Satellite photo", result: [el("b", { text: platformOf(run) }), `, ${cloudWords(run.cloud_fraction)}`], art: pic([photo(wide)]), view: "photo", go: () => setView("photo") },
    { n: 2, title: "Hot ground found", result: fires.length ? [el("b", { text: plural(fires.length, "hot spot") }), factories ? `, ${plural(factories, "factory", "factories")} ignored` : ""] : ["Nothing unusual"], art: pic([anomaly(focus), scenery(focus)]), view: "anomaly", go: () => setView("anomaly") },
    { n: 3, title: "Spread predicted", result: threatening.length ? [el("b", { text: `${threatening.length} could reach a road` }), " in 48 days"] : ["No fire close to a road"], art: pic([scenery(focus)], hotShapes, focus), go: () => { if (fire) selectHotspot(fire.hotspot_id, true); } },
    { n: 4, title: "Roads protected", result: !run.agent ? ["Agent not running yet on this date"] : [el("b", { text: closed.length ? `${plural(closed.length, "section")} closed` : "All roads open" }), pending ? `, ${plural(pending, "decision")} for you` : ""], art: pic([scenery(roadCrop)], roadShapes, roadCrop), alert: pending > 0 && run.agent, go: () => { closeDetail(); if (state.approvals.length) $("decisions-block").scrollIntoView({ behavior: "smooth", block: "start" }); else { $("roads-block").open = true; $("roads-block").scrollIntoView({ behavior: "smooth" }); } } },
  ];
  $("steps").replaceChildren(...steps.map((s) => el("li", {},
    el("button", { class: `step-card${s.alert ? " alert" : ""}`, "data-view": s.view || "", "aria-pressed": s.view && s.view === state.view ? "true" : "false", onclick: s.go },
      s.art,
      el("span", {}, el("span", { class: "step-num", text: `Step ${s.n}` }), el("span", { class: "step-title", text: s.title }), el("span", { class: "step-result" }, ...s.result))))));
}

function renderFires() {
  const list = $("fires");
  list.replaceChildren();
  const items = [...state.hotspots].sort((a, b) => {
    const da = (DECISION[a.decision] || DECISION.log).order, db = (DECISION[b.decision] || DECISION.log).order;
    return da - db || ((b.risk || {}).score || 0) - ((a.risk || {}).score || 0);
  });
  for (const h of items) {
    const level = levelOf(h);
    const d = DECISION[h.decision] || DECISION.log;
    list.append(el("li", {}, el("button", { class: "fire", onclick: () => selectHotspot(h.hotspot_id, true) },
      el("span", { class: `mark l-${level}`, "aria-hidden": "true" }),
      el("span", {},
        el("span", { class: "fire-top" }, el("span", { class: "fire-name", text: fireName(h) }), el("span", { class: `chip chip-${h.decision || "log"}`, text: d.chip })),
        el("span", { class: "fire-line", text: fireLine(h) })))));
  }
  if (!items.length) list.append(el("li", { class: "empty", text: "No fires found on this date." }));
}

function renderPassFacts() {
  const run = state.run;
  const al = run.alignment || {};
  const facts = [
    ["Cloud over the coalfield", fmtNum((run.cloud_fraction || 0) * 100, 0, "%")],
    ["Lined up with earlier photos using", METHOD[al.method] || al.method || "n/a"],
    ["Alignment error left", al.residual_px !== null && al.residual_px !== undefined ? fmtNum(al.residual_px, 2, " px") : "n/a"],
    ["Second, independent check", al.check_method ? (al.agreement_px !== null && al.agreement_px !== undefined ? `${METHOD[al.check_method] || al.check_method} agrees within ${fmtNum(al.agreement_px, 2, " px")}` : `${METHOD[al.check_method] || al.check_method} could not align`) : "not run"],
    ["Processing time", fmtNum(run.seconds, 1, " s")],
    ["Scene", run.scene_id],
  ];
  $("pass-facts").replaceChildren(...facts.map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: String(v) }))));
}

function closeDetail() {
  state.selected = null;
  showOverview();
  if (state.layers.hotspots) state.layers.hotspots.setStyle(hotspotStyle);
  writeHash();
}

function firePopup(h) {
  const d = DECISION[h.decision] || DECISION.log;
  const ha = h.area_m2 / 10000;
  const box = el("div", { class: "fire-pop" },
    el("div", { class: "fire-pop-top" }, el("b", { text: fireName(h) }), el("span", { class: `chip chip-${h.decision || "log"}`, text: d.chip })),
    el("p", { text: fireLine(h) }),
    el("p", { class: "fire-pop-nums", text: `+${fmtNum(h.max_anomaly_k, 0)} °C, ${fmtNum(ha, ha < 10 ? 1 : 0)} ha` }));
  return box;
}

async function selectHotspot(hotspotId, pan) {
  const h = state.hotspots.find((x) => x.hotspot_id === hotspotId);
  if (!h) return;
  state.selected = hotspotId;
  state.layers.hotspots.setStyle(hotspotStyle);
  if (pan) {
    const ll = [h.centroid_lonlat[1], h.centroid_lonlat[0]];
    state.map.panTo(ll);
    L.popup({ className: "fire-popup", offset: [0, -6], autoPan: false, maxWidth: 260 }).setLatLng(ll).setContent(firePopup(h)).openOn(state.map);
  }
  const [track, trace] = await Promise.all([
    h.track_id ? api(`/api/tracks/${h.track_id}`).catch(() => null) : Promise.resolve(null),
    api(`/api/runs/${state.runId}/trace?hotspot=${encodeURIComponent(hotspotId)}`).catch(() => []),
  ]);
  renderDetail(h, track, trace);
  $("overview").hidden = true;
  $("detail").hidden = false;
  document.querySelector(".panel").scrollTop = 0;
  writeHash();
}

function summaryFor(h) {
  const r = h.risk || {};
  if (h.decision === "exclude") return "This heat looks like a factory or coke ovens: it is very hot, has a neat rectangular shape and sits on a mapped industrial site. Fire Watch ignores it so it doesn't raise false alarms.";
  const where = `${distPhrase(h)}${movingPhrase(h)}.`;
  if (h.decision === "close") {
    const segs = ((state.run.actions || []).find((a) => a.kind === "close_road" && a.track_id === h.track_id) || {}).target || state.run.closed_segments || [];
    return `${where} That is close enough to be dangerous, so Fire Watch closed ${segs.length ? `${plural(segs.length, "section")} of ${roadsShort(segs)}` : "the road"}. It will suggest reopening once the fire stays quiet for two photos in a row.`;
  }
  if (h.decision === "escalate") {
    const why = r.projected_distance_m !== null && r.projected_distance_m !== undefined && r.distance_m > 250 && r.projected_distance_m < 250
      ? "The fire isn't at the road yet, but it could reach it within 48 days."
      : "Fire Watch isn't sure enough to act on its own.";
    return `${where} ${why} It has asked you to decide.`;
  }
  return `${where} It isn't a danger to a road right now, so Fire Watch keeps an eye on it in every new photo.`;
}

function bigFact(label, value, small) {
  return el("div", {}, el("dt", { text: label }), el("dd", {}, value, small ? el("small", { text: small }) : null));
}

function renderDetail(h, track, trace) {
  const r = h.risk || {};
  const p = h.projection || {};
  const d = DECISION[h.decision] || DECISION.log;
  $("detail-title").textContent = fireName(h);
  $("detail-chip").textContent = d.chip;
  $("detail-chip").className = `chip chip-${h.decision || "log"}`;
  $("detail-summary").textContent = summaryFor(h);
  const ha = h.area_m2 / 10000;
  const pitches = Math.max(1, Math.round(h.area_m2 / 7140));
  const size = bigFact("Size", `${fmtNum(ha, ha < 10 ? 1 : 0)} ha`, `about ${plural(pitches, "football pitch", "football pitches")}`);
  const heat = bigFact("Heat", `+${fmtNum(h.max_anomaly_k, 0)} \u00b0C`, "hotter than around it");
  const facts = h.decision === "exclude"
    ? [heat, size, bigFact("On a mapped plant", fmtNum((h.industrial_overlap || 0) * 100, 0, "%"), "of its area")]
    : [bigFact("Nearest road", r.nearest_segment ? (r.distance_m < 15 ? "On it" : fmtDist(r.distance_m)) : "None near", r.nearest_segment ? roadPlain(r.nearest_segment) : ""), size, heat];
  $("detail-facts").replaceChildren(...facts);
  const analogues = (h.evidence || {}).analogues;
  const more = [
    ["Risk score", `${fmtNum(r.score, 2)} (${r.level})`],
    ["Confidence", fmtNum(r.confidence, 2)],
    ["Chance of reaching a road in 48 days", p.reach_probability !== undefined && p.reach_probability !== null ? fmtNum(p.reach_probability * 100, 0, "%") : "n/a"],
    ["Predicted spread's distance to a road", fmtDist(r.projected_distance_m)],
    ["Average heat above surroundings", fmtNum(h.mean_anomaly_k, 1, " K")],
    ["Peak heat above surroundings", fmtNum(h.max_anomaly_k, 1, " K")],
    ["Similar past fires", analogues && analogues.n ? `${analogues.n}; ${analogues.reached_road_share === null || analogues.reached_road_share === undefined ? "none could reach a road" : `${fmtNum(analogues.reached_road_share * 100, 0, "%")} reached a road`}` : "none yet"],
    ["Agent's reasoning", capital(h.decision_reason || "none recorded")],
    ["Location", `${fmtNum(h.centroid_lonlat[1], 4)} N, ${fmtNum(h.centroid_lonlat[0], 4)} E`],
    ["Track", h.track_id || "new"],
  ];
  if (r.confidence_notes && r.confidence_notes.length) more.push(["What lowered confidence", r.confidence_notes.join("; ")]);
  $("detail-more").replaceChildren(...more.map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: String(v) }))));
  renderSpark(track);
  renderFilm(h, track);
  renderTrail(trace.filter((e) => e.hotspot_id === h.hotspot_id));
}

function renderFilm(h, track) {
  const box = $("detail-film");
  const strip = $("film-strip");
  strip.replaceChildren();
  const bounds = state.run && state.run.layers && state.run.layers.bounds;
  const byScene = new Map(state.runs.map((r) => [r.scene_id, r]));
  const pts = track && track.points ? track.points.filter((p) => byScene.has(p.scene_id)) : [];
  if (!bounds || pts.length < 2) { box.hidden = true; return; }
  box.hidden = false;
  const crop = cropAround(h.centroid_lonlat[0], h.centroid_lonlat[1], 0.03);
  for (const p of pts) {
    const run = byScene.get(p.scene_id);
    const here = run.run_id === state.runId;
    const label = `${fmtDate(run.acquired, { day: "numeric", month: "short" })}, ${fmtNum(p.area_m2 / 10000, 1)} ha`;
    const art = pic([cropBg(asset(`/api/runs/${run.run_id}/layers/anomaly`), bounds, crop), cropBg(asset(`/api/runs/${run.run_id}/layers/photo`), bounds, crop)], null, null, "film-pic");
    strip.append(el("li", {}, el("button", { class: `film-frame${here ? " on" : ""}`, "aria-current": here ? "true" : "false", "aria-label": `${fireName(h)} on the ${fmtDate(run.acquired)} photo, ${fmtNum(p.area_m2 / 10000, 1)} hectares`, onclick: () => { stopPlay(); selectRun(run.run_id, p.hotspot_id); } },
      art, el("span", { text: label }))));
  }
  const on = strip.querySelector(".film-frame.on");
  if (on) on.scrollIntoView({ block: "nearest", inline: "center" });
}

function buildFindList() {
  const list = $("find-list");
  if (!list) return;
  list.replaceChildren();
  const seen = new Set();
  const add = (value, label) => {
    const key = value.toLowerCase();
    if (!value || seen.has(key)) return;
    seen.add(key);
    list.append(el("option", { value, label }));
  };
  for (const h of state.hotspots || []) add(fireName(h), "Fire on this date");
  for (const p of state.places || []) add(p.name, { town: "Town", mine: "Mine", road: "Road" }[p.kind] || "Place");
}

let findMarker = null;
function flyToPlace(lat, lon, name, zoom = 14) {
  state.map.flyTo([lat, lon], zoom, { duration: 0.9 });
  if (findMarker) findMarker.remove();
  findMarker = L.marker([lat, lon], { icon: L.divIcon({ className: "find-pin", html: `<span></span><b>${escapeHtml(name)}</b>`, iconSize: [0, 0] }), keyboard: false, interactive: false, pane: "labels" }).addTo(state.map);
}

function findPlace(q) {
  const t = q.trim().toLowerCase();
  if (!t) return false;
  const fire = (state.hotspots || []).find((h) => fireName(h).toLowerCase() === t);
  if (fire) { selectHotspot(fire.hotspot_id, true); state.map.setZoom(Math.max(state.map.getZoom(), 14)); return true; }
  if (state.roads && (state.places || []).some((p) => p.kind === "road" && p.name.toLowerCase() === t)) {
    const parts = state.roads.features.filter((f) => (f.properties.name || "").toLowerCase() === t);
    if (parts.length) {
      const b = L.geoJSON({ type: "FeatureCollection", features: parts }).getBounds();
      state.map.flyToBounds(b, { padding: [40, 40], maxZoom: 15, duration: 0.9 });
      const c = b.getCenter();
      if (findMarker) findMarker.remove();
      findMarker = L.marker(c, { icon: L.divIcon({ className: "find-pin", html: `<span></span><b>${escapeHtml(parts[0].properties.name)}</b>`, iconSize: [0, 0] }), keyboard: false, interactive: false, pane: "labels" }).addTo(state.map);
      return true;
    }
  }
  const all = state.places || [];
  const p = all.find((x) => x.name.toLowerCase() === t) || all.find((x) => x.name.toLowerCase().startsWith(t)) || all.find((x) => x.name.toLowerCase().includes(t));
  if (p) { flyToPlace(p.lat, p.lon, p.name, p.kind === "town" ? 14 : 15); return true; }
  return false;
}

function setupFind() {
  const form = $("find");
  const input = $("find-input");
  const fail = (msg) => { input.setCustomValidity(msg); input.reportValidity(); setTimeout(() => input.setCustomValidity(""), 2500); };
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!findPlace(input.value)) fail(`Nothing called “${input.value.trim()}” on the map. Try a town, mine, road or a fire such as Fire 3.`);
  });
  input.addEventListener("change", () => { if ([...$("find-list").options].some((o) => o.value.toLowerCase() === input.value.trim().toLowerCase())) findPlace(input.value); });
}

const THEME_KEY = "fw-theme";
function setupTheme() {
  const button = $("theme");
  const apply = (theme) => {
    document.documentElement.dataset.theme = theme;
    button.setAttribute("aria-pressed", String(theme === "dark"));
    button.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
  };
  apply(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  button.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    apply(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (err) { return; }
  });
}

const BELL_KEY = "fw-alerts-seen";
function alertTitle(n) {
  const d = n.data || {};
  const fire = d.track ? `Fire ${parseInt(String(d.track).replace(/\D/g, ""), 10)}` : "A fire";
  const subj = n.subject || "";
  if (/closure/i.test(subj)) return `${fire}: road closed`;
  if (/reopen/i.test(subj)) return `${fire}: reopening suggested`;
  if (/review|decision|escalat|approval/i.test(subj)) return `${fire}: needs your decision`;
  if (/drone/i.test(subj)) return `${fire}: drone survey requested`;
  return subj.split(":")[0] || "Alert";
}
function alertBody(n) {
  const first = String(n.message || "").split("\n")[1] || "";
  return first.replace(/\b([LR]\d+-\d+)(, [LR]\d+-\d+){3,}/g, (m) => `${m.split(", ").length} road sections`).slice(0, 160);
}
async function loadAlerts() {
  const items = await api("/api/notifications?limit=30").catch(() => null);
  const list = $("bell-list");
  list.replaceChildren();
  if (!items) { list.append(el("li", { class: "empty", text: "Alerts can't be loaded right now." })); return; }
  let seen = "";
  try { seen = localStorage.getItem(BELL_KEY) || ""; } catch (err) { seen = ""; }
  const fresh = items.filter((n) => (n.ts || "") > seen).length;
  const count = $("bell-count");
  count.hidden = !fresh;
  count.textContent = fresh > 9 ? "9+" : String(fresh);
  $("bell").setAttribute("aria-label", fresh ? `Alerts, ${fresh} new` : "Alerts");
  $("bell-sub").textContent = items.length ? `Latest ${items.length}` : "";
  for (const n of items) {
    const d = n.data || {};
    const run = state.runs.find((r) => r.scene_id === d.pass);
    const when = d.date ? fmtDate(d.date) : fmtDate(n.ts);
    const item = el("button", { class: `bell-item sev-${n.severity || "info"}${(n.ts || "") > seen ? " new" : ""}`, onclick: () => { closeBell(); if (run) selectRun(run.run_id, d.hotspot); else if (d.lat) flyToPlace(d.lat, d.lon, alertTitle(n)); } },
      el("b", { text: alertTitle(n) }), el("span", { text: alertBody(n) }), el("small", { text: `${when}${run ? "" : ", photo not loaded"}` }));
    list.append(el("li", {}, item));
  }
  if (!items.length) list.append(el("li", { class: "empty", text: "No alerts yet. Road closures and decisions for you appear here." }));
  state.alertsNewest = items.length ? items[0].ts : seen;
}
function closeBell() {
  $("bell-panel").hidden = true;
  $("bell").setAttribute("aria-expanded", "false");
}
function setupBell() {
  const button = $("bell");
  const panel = $("bell-panel");
  button.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = panel.hidden;
    panel.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (open) {
      try { if (state.alertsNewest) localStorage.setItem(BELL_KEY, state.alertsNewest); } catch (err) { return; }
      $("bell-count").hidden = true;
      $("bell").setAttribute("aria-label", "Alerts");
    }
  });
  document.addEventListener("click", (e) => { if (!panel.hidden && !e.target.closest(".bell-wrap")) closeBell(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) { closeBell(); button.focus(); } });
}

const cmp = { maps: {}, layers: {}, view: "photo", syncing: false, ready: false };
function compareLayers(side, run, hsGeo) {
  const map = cmp.maps[side];
  for (const l of cmp.layers[side] || []) map.removeLayer(l);
  const bounds = run.layers && run.layers.bounds;
  const box = bounds ? [[bounds[1], bounds[0]], [bounds[3], bounds[2]]] : null;
  const out = [];
  if (box) {
    const sb = state.summary && state.summary.scenery_bounds;
    if (cmp.view === "photo") {
      if (run.layers.photo) out.push(L.imageOverlay(asset(`/api/runs/${run.run_id}/layers/photo`), box, { interactive: false }));
      else if (sb) out.push(L.imageOverlay(asset("/api/scenery"), [[sb[1], sb[0]], [sb[3], sb[2]]], { interactive: false }));
    } else {
      if (sb) out.push(L.imageOverlay(asset("/api/scenery"), [[sb[1], sb[0]], [sb[3], sb[2]]], { interactive: false, opacity: 0.55 }));
      out.push(L.imageOverlay(asset(`/api/runs/${run.run_id}/layers/anomaly`), box, { interactive: false, opacity: 0.9 }));
    }
  }
  out.push(L.geoJSON(hsGeo, { style: hotspotStyle, interactive: false }));
  for (const l of out) l.addTo(map);
  cmp.layers[side] = out;
}
async function compareLoad(side) {
  const runId = $(`compare-${side}`).value;
  const [run, hotspots, hsGeo] = await Promise.all([api(`/api/runs/${runId}`), api(`/api/runs/${runId}/hotspots`), api(`/api/runs/${runId}/geojson/hotspots`)]);
  cmp[side] = { run, hotspots, hsGeo };
  compareLayers(side, run, hsGeo);
  const fires = hotspots.filter((h) => h.decision !== "exclude");
  const area = fires.reduce((a, h) => a + h.area_m2, 0) / 1e6;
  const closed = (run.closed_segments || []).length;
  $(`compare-${side}-sum`).textContent = `${plural(fires.length, "hot spot")}, ${fmtNum(area, 2)} km² hot ground, ${closed ? `${plural(closed, "section")} closed` : "no closures"}`;
  compareChange();
}
function compareChange() {
  if (!cmp.a || !cmp.b) return;
  const ids = (x) => new Set(x.hotspots.filter((h) => h.decision !== "exclude" && h.track_id).map((h) => h.track_id));
  const a = ids(cmp.a), b = ids(cmp.b);
  const both = [...b].filter((t) => a.has(t)).length;
  const gone = [...a].filter((t) => !b.has(t)).length;
  const fresh = [...b].filter((t) => !a.has(t)).length;
  const days = Math.round((new Date(cmp.b.run.acquired) - new Date(cmp.a.run.acquired)) / 86400000);
  const span = days === 0 ? "the same day" : `${Math.abs(days)} days ${days > 0 ? "apart" : "apart, in reverse order"}`;
  $("compare-change").textContent = `${capital(span)}. ${plural(both, "fire")} on both dates, ${fresh} new on the later one, ${gone} no longer seen.`;
}
function setupCompare() {
  const dlg = $("compare");
  const open = $("compare-open");
  open.disabled = state.runs.length < 2;
  open.addEventListener("click", async () => {
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", "");
    if (!cmp.ready) {
      cmp.ready = true;
      for (const side of ["a", "b"]) {
        const sel = $(`compare-${side}`);
        sel.replaceChildren(...state.runs.map((r) => el("option", { value: r.run_id, text: `${fmtDate(r.acquired)} (${platformOf(r)})` })));
        const map = L.map(`compare-map-${side}`, { zoomControl: side === "a", attributionControl: false, zoomSnap: 0.25 });
        L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 17 }).addTo(map);
        cmp.maps[side] = map;
        sel.addEventListener("change", () => compareLoad(side));
      }
      const sync = (from, to) => from.on("move", () => {
        if (cmp.syncing) return;
        cmp.syncing = true;
        to.setView(from.getCenter(), from.getZoom(), { animate: false });
        cmp.syncing = false;
      });
      sync(cmp.maps.a, cmp.maps.b);
      sync(cmp.maps.b, cmp.maps.a);
      document.querySelectorAll("[data-cview]").forEach((b) => b.addEventListener("click", () => {
        cmp.view = b.dataset.cview;
        document.querySelectorAll("[data-cview]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        for (const side of ["a", "b"]) if (cmp[side]) compareLayers(side, cmp[side].run, cmp[side].hsGeo);
      }));
    }
    let later = Math.max(0, state.runs.findIndex((r) => r.run_id === state.runId));
    let earlier = Math.max(0, later - 6);
    if (earlier === later) later = Math.min(state.runs.length - 1, earlier + 6);
    $("compare-a").value = state.runs[earlier].run_id;
    $("compare-b").value = state.runs[later].run_id;
    requestAnimationFrame(() => {
      for (const side of ["a", "b"]) cmp.maps[side].invalidateSize();
      cmp.maps.a.setView(state.map.getCenter(), state.map.getZoom(), { animate: false });
    });
    await Promise.all([compareLoad("a"), compareLoad("b")]);
  });
  $("compare-close").addEventListener("click", () => dlg.close());
}

function setupAbout() {
  const dlg = $("about-data");
  $("data-badge").addEventListener("click", () => { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", ""); });
  const s = state.summary || {};
  if ((s.source === "real" || s.source === "landsat") && state.runs.length) {
    $("about-photos").textContent = `${plural(state.runs.length, "Landsat 8 and 9 scene")} from ${fmtDate(state.runs[0].acquired)} to ${fmtDate(state.runs[state.runs.length - 1].acquired)}: Collection 2 Level-2 (USGS), read from Microsoft Planetary Computer. Very cloudy scenes are skipped.`;
  }
  if (state.roads) $("about-osm").textContent = `OpenStreetMap: ${fmtNum(state.roads.features.length)} road sections of about 400 m each, plus mines, plants and towns. The agent closes mine haul roads by itself; public roads always go to a person.`;
}

function renderSpark(track) {
  const fig = $("detail-spark");
  fig.replaceChildren();
  const cutoff = state.run ? new Date(state.run.acquired).getTime() : Infinity;
  const pts = track && track.points ? track.points.filter((p) => new Date(p.acquired).getTime() <= cutoff + 1000) : [];
  if (pts.length < 2) {
    fig.append(el("figcaption", { text: "First time this spot has been seen. Its history builds up with every new photo." }));
    return;
  }
  const W = 360, H = 76, pad = 8;
  const t0 = new Date(pts[0].acquired).getTime(), t1 = new Date(pts[pts.length - 1].acquired).getTime();
  const x = (p) => pad + ((new Date(p.acquired).getTime() - t0) / Math.max(1, t1 - t0)) * (W - 2 * pad);
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Size of this fire over time");
  const vals = pts.map((p) => p.area_m2);
  const hi = Math.max(...vals);
  const y = (v) => H - pad - (v / Math.max(1e-9, hi)) * (H - 2 * pad);
  const area = document.createElementNS(SVGNS, "path");
  area.setAttribute("d", `M${x(pts[0]).toFixed(1)},${H - pad} ${pts.map((p) => `L${x(p).toFixed(1)},${y(p.area_m2).toFixed(1)}`).join(" ")} L${x(pts[pts.length - 1]).toFixed(1)},${H - pad}Z`);
  area.setAttribute("fill", "rgba(210,63,29,0.12)");
  svg.append(area);
  const line = document.createElementNS(SVGNS, "path");
  line.setAttribute("d", pts.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(p.area_m2).toFixed(1)}`).join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", COLORS.high);
  line.setAttribute("stroke-width", "2");
  svg.append(line);
  const last = pts[pts.length - 1];
  const dot = document.createElementNS(SVGNS, "circle");
  dot.setAttribute("cx", x(last).toFixed(1));
  dot.setAttribute("cy", y(last.area_m2).toFixed(1));
  dot.setAttribute("r", "3.5");
  dot.setAttribute("fill", COLORS.high);
  svg.append(dot);
  fig.append(svg);
  const first = pts[0];
  const change = last.area_m2 - first.area_m2;
  const trend = Math.abs(change) < 0.1 * first.area_m2 ? "about the same size" : change > 0 ? "growing" : "shrinking";
  fig.append(el("figcaption", { text: `Size over ${plural(pts.length, "photo")} since ${fmtDate(first.acquired)}: ${trend} (${fmtNum(first.area_m2 / 10000, 1)} to ${fmtNum(last.area_m2 / 10000, 1)} ha).` }));
}

function toolResult(tool, r) {
  if (tool === "fetch_highres") {
    if (r.checked === false) return r.reason ? capital(r.reason) : "No clear photo available";
    return r.confirmed ? "It shows hot pixels, so the fire is confirmed" : "No hot pixels, so not confirmed";
  }
  if (tool === "fetch_history") {
    if (!r.observable) return "No clear earlier photos of this spot";
    return `It was hot here in ${r.present || 0} of ${r.observable} earlier clear photos`;
  }
  if (tool === "refine_alignment") {
    if (r.ok) return "A third method agrees the photo is lined up correctly";
    return "A third method disagrees, so the measurement is less certain";
  }
  if (tool === "check_industrial") return r.verdict === "industrial" ? "Yes, it looks like a factory" : r.verdict === "fire" ? "No, it looks like a coal fire" : "Unclear";
  if (tool === "request_drone") return r.requested ? "Survey requested" : "A survey was already requested";
  return "";
}

function stepFor(e) {
  if (e.node === "observe") return { kind: "look", what: "Looked at the fire", why: distPhrase({ risk: e.observation || {} }) };
  if (e.node === "tool") return { kind: "tool", what: TOOL_WORDS[e.tool] || e.tool, why: toolResult(e.tool, e.result || {}) };
  if (e.node === "decide") {
    const words = { close: "Decided to close the road", escalate: "Decided to ask a person", log: "Decided to keep watching", exclude: "Decided it is factory heat" };
    return { kind: "decide", what: words[e.decision] || `Decided: ${e.decision}`, why: "" };
  }
  if (e.node === "act") {
    const acts = ((e.result || {}).actions || []).map((a) => {
      const t = a.target && a.target.length ? a.target.join(", ") : "";
      const words = { close_road: `Closed ${t}`, keep_closed: `Kept ${t} closed`, escalate: "Sent it to you for a decision", notify: "Sent an alert", exclude: "Marked it as factory heat", log: "Logged it", propose_reopen: `Suggested reopening ${t}`, request_drone: "Asked for a drone survey", blocked: `Wanted to close ${t}, but the safety check said no` };
      return `${words[a.kind] || a.kind.replace(/_/g, " ")}${a.executed === false && a.kind !== "blocked" ? " (stopped by the safety check)" : ""}`;
    });
    return acts.length ? { kind: "act", what: acts.join(". "), why: "" } : null;
  }
  if (e.node === "skip") return { kind: "plan", what: "Kept watching, no action needed", why: "" };
  return null;
}

function renderTrail(events) {
  const log = $("detail-log");
  log.replaceChildren();
  for (const e of events) {
    const s = stepFor(e);
    if (!s) continue;
    log.append(el("li", { class: `t-${s.kind}` }, el("span", { class: "what", text: s.what }), s.why ? el("span", { class: "why", text: s.why }) : null));
  }
  if (!log.children.length) log.append(el("li", { class: "empty", text: "No steps recorded." }));
}

async function loadApprovals() {
  state.approvals = await api("/api/approvals?status=pending");
  const box = $("decisions");
  box.replaceChildren(...state.approvals.map(decisionCard));
  $("decisions-block").hidden = !state.approvals.length;
  $("decision-count").textContent = state.approvals.length;
}

const GENERIC_ROADS = new Set(["Local road", "Service road", "Mine track", "road"]);
function roadsShort(segs) {
  const names = [...new Set(segs.map(roadPlain))];
  const named = names.filter((n) => !GENERIC_ROADS.has(n));
  const lanes = names.length > named.length;
  const shown = named.slice(0, 3);
  const extra = named.length - shown.length;
  const parts = [...shown];
  if (extra > 0) parts.push(`${plural(extra, "other road")}`);
  if (lanes) parts.push(named.length ? "nearby lanes" : "local lanes");
  return parts.length ? joinWords(parts) : "the road";
}
function segsShort(segs) {
  return segs.length <= 5 ? segs.join(", ") : `${segs.slice(0, 4).join(", ")} and ${segs.length - 4} more`;
}

function decisionCard(a) {
  const reopen = a.kind === "reopen_road";
  const segs = a.segments || [];
  const where = roadsShort(segs);
  const fireNo = a.track_id ? parseInt(String(a.track_id).replace(/\D/g, ""), 10) : null;
  const fire = fireNo ? `Fire ${fireNo}` : "A fire";
  let title, body, yes, no;
  if (reopen) {
    title = `Reopen ${where}?`;
    body = `${fire} has stayed quiet for two photos in a row. Fire Watch suggests reopening ${plural(segs.length, "section")} (${segsShort(segs)}).`;
    yes = "Yes, reopen"; no = "Keep closed";
  } else if (a.recommended === "close") {
    title = `Close ${where}?`;
    const pub = segs.some((sid) => { const f = state.roads && state.roads.features.find((x) => x.properties.id === sid); return f && !f.properties.haul; });
    body = `${fire} could put ${plural(segs.length, "section")} (${segsShort(segs)}) at risk. ${pub && state.summary && state.summary.auto_close === "haul" ? "Public roads are never closed without a person." : "Fire Watch isn't sure enough to close it alone."}`;
    yes = "Yes, close it"; no = "Not now";
  } else {
    title = `Check ${fire.toLowerCase()}?`;
    body = "Fire Watch can't tell how serious this is from space and suggests someone takes a look.";
    yes = "Mark as checked"; no = "Dismiss";
  }
  const status = el("p", { class: "meta", role: "status" });
  const note = el("textarea", { "aria-label": "Note (optional)", placeholder: "Note (optional)", hidden: true });
  const act = async (approve, button) => {
    button.disabled = true;
    try {
      await api(`/api/approvals/${a.approval_id}`, { method: "POST", body: JSON.stringify({ approve, note: note.value }) });
      status.textContent = "Saved.";
      await Promise.all([loadApprovals(), loadRoads(), loadSummary()]);
      if (state.run) { renderSteps(); renderDateBar(); }
    } catch (err) {
      status.textContent = `Could not save: ${err.message}`;
      button.disabled = false;
    }
  };
  const when = a.pass_date ? `Raised after the ${fmtDate(a.pass_date)} photo` : `Raised ${fmtDate(a.created)}`;
  const show = a.track_id ? el("button", { class: "link", text: `Show ${fire.toLowerCase()} on the map`, onclick: () => {
    const h = state.hotspots.find((x) => x.track_id === a.track_id);
    if (h) selectHotspot(h.hotspot_id, true);
  } }) : null;
  return el("article", { class: `decision${reopen ? " reopen" : ""}` },
    el("h3", { text: title }),
    el("p", { text: body }),
    el("p", { class: "meta" }, when, a.passes > 1 ? `, seen in ${a.passes} photos` : ""),
    note,
    el("div", { class: "actions" },
      el("button", { class: "btn btn-go", onclick: (ev) => act(true, ev.currentTarget), text: yes }),
      el("button", { class: "btn", onclick: (ev) => act(false, ev.currentTarget), text: no }),
      show,
      el("button", { class: "link", text: "Add a note", onclick: (ev) => { note.hidden = false; note.focus(); ev.currentTarget.remove(); } })),
    status);
}

async function runNext() {
  const button = $("run-next");
  button.disabled = true;
  try {
    await api("/api/runs/next", { method: "POST" });
    $("run-status").textContent = "Processing the next photo...";
    const before = state.runs.length;
    for (let i = 0; i < 150; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      await loadSummary();
      if (!state.summary.running) break;
    }
    await loadRuns();
    await Promise.all([loadRoads(), loadApprovals()]);
    if (state.runs.length > before) await selectRun(state.runs[state.runs.length - 1].run_id);
  } catch (err) {
    $("run-status").textContent = `Could not start: ${err.message}`;
  } finally {
    await loadSummary();
  }
}

function readAt() {
  const m = /(?:^|[#&])at=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/.exec(location.hash);
  return m ? { lat: parseFloat(m[1]), lon: parseFloat(m[2]) } : null;
}

function writeHash() {
  const parts = [`run=${encodeURIComponent(state.runId || "")}`];
  if (state.selected) parts.push(`hotspot=${encodeURIComponent(state.selected)}`);
  try { history.replaceState(null, "", `#${parts.join("&")}`); } catch (err) { return; }
}
function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  return { run: params.get("run"), hotspot: params.get("hotspot") };
}

function step(delta) {
  const idx = state.runs.findIndex((r) => r.run_id === state.runId);
  const next = state.runs[idx + delta];
  if (next) return selectRun(next.run_id, null);
  return null;
}

function stopPlay() {
  if (state.play) clearTimeout(state.play);
  state.play = null;
  $("play").setAttribute("aria-pressed", "false");
  $("play").innerHTML = "&#9654; Play";
}

function togglePlay() {
  if (state.play) { stopPlay(); return; }
  $("play").setAttribute("aria-pressed", "true");
  $("play").innerHTML = "&#10074;&#10074; Pause";
  let idx = state.runs.findIndex((r) => r.run_id === state.runId);
  if (idx >= state.runs.length - 1) idx = -1;
  const advance = async () => {
    idx += 1;
    if (idx >= state.runs.length) { stopPlay(); return; }
    await selectRun(state.runs[idx].run_id);
    if (state.play !== null) state.play = setTimeout(advance, 1500);
  };
  state.play = setTimeout(advance, 0);
}

async function main() {
  await loadSummary();
  initMap();
  document.querySelectorAll(".map-mode button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("prev-pass").addEventListener("click", () => { stopPlay(); step(-1); });
  $("next-pass").addEventListener("click", () => { stopPlay(); step(1); });
  $("run-next").addEventListener("click", runNext);
  $("detail-back").addEventListener("click", closeDetail);
  $("play").addEventListener("click", togglePlay);
  $("open-help").addEventListener("click", () => { const d = $("help"); if (d.showModal) d.showModal(); else d.setAttribute("open", ""); });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, dialog")) return;
    if (e.key === "ArrowLeft") { stopPlay(); step(-1); }
    if (e.key === "ArrowRight") { stopPlay(); step(1); }
    if (e.key === "Escape" && state.selected) closeDetail();
  });
  setupTheme();
  setupFind();
  setupBell();
  $("offline-retry").addEventListener("click", () => location.reload());
  document.addEventListener("keydown", (e) => {
    if (!ruler.active && !ruler.points.length) return;
    if (e.key === "Escape") { rulerSet(false); rulerClear(); }
    if (e.key === "Enter" && ruler.active && !e.target.closest("input, textarea")) { rulerSet(false); }
  });
  $("toggle-traffic").addEventListener("change", (e) => { if (e.target.checked) state.layers.trucks.addTo(state.map); else state.map.removeLayer(state.layers.trucks); });
  $("key-truck").innerHTML = TRUCK_SVG(true);
  $("key-barrier").innerHTML = BARRIER_SVG;
  const at = readAt();
  await loadRoads();
  await Promise.all([loadRuns(), loadApprovals(), loadSites()]);
  setupAbout();
  setupCompare();
  const wanted = readHash();
  const known = state.runs.find((r) => r.run_id === wanted.run);
  const latest = state.runs.length ? state.runs[state.runs.length - 1].run_id : null;
  if (known) await selectRun(known.run_id, wanted.hotspot);
  else if (latest) await selectRun(latest);
  else {
    $("pass-date").textContent = "No photos yet";
    $("pass-story").textContent = state.summary.source === "real" ? "Run the download and processing steps to see real photos here." : "Run the demo to see photos here.";
    renderFires();
  }
  buildFindList();
  if (at) {
    const near = (state.places || []).map((p) => ({ p, d: state.map.distance([p.lat, p.lon], [at.lat, at.lon]) })).sort((x, y) => x.d - y.d)[0];
    flyToPlace(at.lat, at.lon, near && near.d < 300 ? near.p.name : "Your place", 14);
  }
  loadAlerts();
}

main().catch((err) => {
  $("run-status").textContent = `Could not load: ${err.message}`;
  $("pass-date").textContent = "Not loaded";
  $("pass-story").textContent = "The dashboard couldn't load its data. Check that the Fire Watch server is running, then try again.";
  $("fires").replaceChildren(el("li", { class: "empty", text: "Nothing to show until the data loads." }));
  showOffline(true, `Couldn't load the dashboard: ${err.message}.`);
  console.error(err);
});
