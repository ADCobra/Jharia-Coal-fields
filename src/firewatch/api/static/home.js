"use strict";

const PLATFORMS = { LANDSAT_8: "Landsat 8", LANDSAT_9: "Landsat 9" };

function byId(id) { return document.getElementById(id); }
const API = (path) => (window.FW_API ? window.FW_API(path) : path);
const DASH = window.FW_DASH || "/dashboard";
// Under a prefix (the read-only real-data view at /real), links to the dashboard and the site root stay in that view.
if (window.FW_BASE) {
  document.querySelectorAll('a[href="/dashboard"]').forEach((a) => { a.href = DASH; });
  document.querySelectorAll('a[href="/"]:not([data-view])').forEach((a) => { a.href = `${window.FW_BASE}/`; });
}

function renderViewSwitch(status) {
  const inReal = Boolean(window.FW_BASE);
  if (!status.other_view && !inReal) return;
  const box = byId("view-switch");
  box.querySelector('[data-view="sim"]').classList.toggle("on", !inReal);
  box.querySelector('[data-view="real"]').classList.toggle("on", inReal);
  box.querySelectorAll("a").forEach((a) => a.setAttribute("aria-current", a.classList.contains("on") ? "page" : "false"));
  box.hidden = false;
  const hero = byId("hero-real");
  if (hero) hero.hidden = inReal;
}
function fmtDate(iso) { return iso ? new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }) : ""; }
function li(cls, ...parts) {
  const item = document.createElement("li");
  if (cls) item.className = cls;
  item.append(...parts);
  return item;
}
function node(tag, text) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  return n;
}

function renderRoads(status, filter = "") {
  const list = byId("road-list");
  list.replaceChildren();
  const q = filter.trim().toLowerCase();
  const roads = status.closed_roads.filter((r) => !q || r.name.toLowerCase().includes(q) || r.sections.some((s) => s.toLowerCase().includes(q)));
  for (const r of roads) {
    const name = node("b", r.name);
    const sub = node("small", `${r.sections.length} ${r.sections.length === 1 ? "section" : "sections"}: ${r.sections.join(", ")}`);
    const text = node("span");
    text.append(name, sub);
    list.append(li("", node("i"), text, node("span", "Closed")));
    list.lastChild.lastChild.className = "state";
  }
  if (!status.closed_roads.length) {
    const text = node("span");
    text.append(node("b", "Every road is open"), node("small", "No fire is close enough to a road to close it."));
    list.append(li("ok", node("i"), text, node("span", "Open")));
    list.lastChild.lastChild.className = "state";
  } else if (!roads.length) {
    const text = node("span");
    text.append(node("b", `“${filter.trim()}” is not closed`), node("small", "It is not on the list of closed roads."));
    list.append(li("ok", node("i"), text, node("span", "Open")));
    list.lastChild.lastChild.className = "state";
  }
}

const KIND = { town: "Town", mine: "Mine", road: "Road" };
const LEVEL_WORD = { high: "High risk", medium: "Medium risk", low: "Low risk", excluded: "Factory heat, ignored" };
const ACTION = { close: "Fire Watch closed the road next to it", escalate: "Fire Watch asked a person to look at it", log: "Fire Watch keeps watching it", exclude: "Recognised as factory heat and ignored" };

function haversine(a, b) {
  const R = 6371000, rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad, dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
function direction(from, to) {
  const y = to.lat - from.lat, x = (to.lon - from.lon) * Math.cos(from.lat * Math.PI / 180);
  const deg = (Math.atan2(x, y) * 180 / Math.PI + 360) % 360;
  return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][Math.round(deg / 45) % 8];
}
function km(m) { return m < 950 ? `${Math.round(m / 10) * 10} m` : `${(m / 1000).toFixed(m < 9500 ? 1 : 0)} km`; }

function setupCompare(status) {
  const frame = byId("compare-frame");
  const cap = byId("cmp-cap");
  const layers = status.layers;
  const bounds = layers && layers.bounds;
  if (!layers || !bounds) {
    frame.classList.remove("loading");
    byId("cmp-empty").hidden = false;
    byId("cmp-range").hidden = true;
    cap.textContent = "";
    return null;
  }
  const photo = byId("cmp-photo"), lst = byId("cmp-lst");
  let pending = 2;
  const done = () => { pending -= 1; if (pending <= 0) frame.classList.remove("loading"); };
  for (const img of [photo, lst]) { img.addEventListener("load", done, { once: true }); img.addEventListener("error", done, { once: true }); }
  photo.src = API(layers.photo ? "/api/public/layers/photo" : "/api/public/layers/scenery");
  lst.src = API("/api/public/layers/lst");
  photo.alt = `Ordinary satellite photo of the Jharia coalfield, ${fmtDate(status.latest_photo)}`;
  lst.alt = "Heat image of the same area. Hotter ground is brighter orange and yellow.";
  const range = byId("cmp-range");
  const setPos = (v) => frame.style.setProperty("--pos", `${v}%`);
  range.addEventListener("input", () => setPos(range.value));
  setPos(range.value);
  const toPct = (lon, lat) => [((lon - bounds[0]) / (bounds[2] - bounds[0])) * 100, ((bounds[3] - lat) / (bounds[3] - bounds[1])) * 100];
  const dots = byId("cmp-dots");
  const fires = status.fire_list || [];
  for (const f of fires) {
    const [x, y] = toPct(f.lon, f.lat);
    if (x < 0 || x > 100 || y < 0 || y > 100) continue;
    const d = node("span");
    d.className = `cmp-dot ${f.level}`;
    d.style.left = `${x}%`;
    d.style.top = `${y}%`;
    dots.append(d);
  }
  const counts = { high: 0, medium: 0, low: 0 };
  for (const f of fires) if (counts[f.level] !== undefined) counts[f.level] += 1;
  cap.replaceChildren();
  const when = node("span", `${PLATFORMS[status.platform] || "Landsat"}, ${fmtDate(status.latest_photo)}.`);
  const scale = node("span");
  scale.append("Heat: cooler", node("i"), "hotter");
  scale.children[0].className = "cmp-scale";
  cap.append(when, scale);
  for (const lvl of ["high", "medium", "low"]) {
    if (!counts[lvl]) continue;
    const k = node("span");
    k.className = "cmp-key";
    const i = node("i");
    i.className = lvl;
    k.append(i, `${counts[lvl]} ${lvl} risk`);
    cap.append(k);
  }
  return {
    pin(place) {
      dots.querySelectorAll(".cmp-pin").forEach((p) => p.remove());
      if (!place) return;
      const [x, y] = toPct(place.lon, place.lat);
      if (x < 0 || x > 100 || y < 0 || y > 100) return;
      const p = node("span");
      p.className = "cmp-pin";
      p.style.left = `${x}%`;
      p.style.top = `${y}%`;
      p.append(node("span", place.name));
      dots.append(p);
    },
  };
}

function setupPlaces(status, compare) {
  const places = status.places || [];
  const input = byId("place-input");
  const list = byId("place-list");
  const out = byId("place-result");
  const unique = new Map();
  for (const p of places) if (p.name && !unique.has(p.name.toLowerCase())) unique.set(p.name.toLowerCase(), p);
  for (const p of unique.values()) {
    const o = document.createElement("option");
    o.value = p.name;
    o.label = KIND[p.kind] || "";
    list.append(o);
  }
  const fires = (status.fire_list || []).filter((f) => f.level !== "excluded");
  const closed = new Set(status.closed_roads.map((r) => r.name.toLowerCase()));

  function find(q) {
    const t = q.trim().toLowerCase();
    if (!t) return null;
    if (unique.has(t)) return unique.get(t);
    const all = [...unique.values()];
    return all.find((p) => p.name.toLowerCase().startsWith(t)) || all.find((p) => p.name.toLowerCase().includes(t)) || null;
  }

  function show(q) {
    out.replaceChildren();
    const place = find(q);
    if (!place) {
      const c = node("div");
      c.className = "pr-card miss";
      c.append(node("div", `No place called “${q.trim()}” on the map`));
      c.firstChild.className = "pr-name";
      c.append(node("p", "Try a town such as Jharia or Katras, a mine, or a road name from the list."));
      out.append(c);
      if (compare) compare.pin(null);
      return;
    }
    const ranked = fires.map((f) => ({ f, d: haversine(place, f) })).sort((a, b) => a.d - b.d);
    const near = ranked[0];
    const c = node("div");
    const nearLevel = near && near.d < 3000 ? near.f.level : "";
    c.className = `pr-card ${nearLevel}`;
    const name = node("div", place.name);
    name.className = "pr-name";
    const kind = node("div", KIND[place.kind] || "Place");
    kind.className = "pr-kind";
    c.append(name, kind);
    if (!near) {
      c.append(node("p", "No fires are being followed on the latest photo."));
    } else {
      const within = ranked.filter((r) => r.d <= 2000).length;
      const p1 = node("p");
      const b = node("b", near.f.name);
      p1.append(b, ` is the nearest fire, ${km(near.d)} ${direction(place, near.f)}. ${LEVEL_WORD[near.f.level] || ""}. ${ACTION[near.f.decision] || ""}.`);
      c.append(p1);
      c.append(node("p", within ? `${within} ${within === 1 ? "fire is" : "fires are"} within 2 km of here.` : "No fire is within 2 km of here."));
    }
    const roadsHere = new Map();
    const generic = new Set(["local road", "service road", "mine track", "road"]);
    for (const r of ranked.filter((r) => r.d <= 2000)) if (r.f.road && !generic.has(r.f.road.toLowerCase())) roadsHere.set(r.f.road, closed.has(r.f.road.toLowerCase()));
    if (place.kind === "road") roadsHere.set(place.name, closed.has(place.name.toLowerCase()));
    if (roadsHere.size) {
      const ul = node("ul");
      ul.className = "pr-roads";
      for (const [road, shut] of [...roadsHere.entries()].slice(0, 4)) {
        const li = node("li", `${road}: ${shut ? "has closed sections" : "open"}`);
        if (shut) li.className = "shut";
        ul.append(li);
      }
      c.append(ul);
    }
    const a = node("a", "Open this place on the map");
    a.className = "btn";
    a.href = `${DASH}#at=${place.lat.toFixed(5)},${place.lon.toFixed(5)}`;
    c.append(a);
    if (!window.FW_DASH) {
      const small = node("p", "The map is in the dashboard, which needs a sign-in.");
      small.className = "pr-kind";
      c.append(small);
    }
    out.append(c);
    if (compare) compare.pin(place);
  }

  byId("place-form").addEventListener("submit", (e) => { e.preventDefault(); show(input.value); });
  input.addEventListener("change", () => { if (unique.has(input.value.trim().toLowerCase())) show(input.value); });
  const tryRow = byId("place-try");
  const picks = ["Jharia", "Katras", "Sindri", "Bhowra"].map((n) => unique.get(n.toLowerCase())).filter(Boolean);
  if (picks.length) {
    tryRow.append("Try:");
    for (const p of picks) {
      const b = node("button", p.name);
      b.type = "button";
      b.addEventListener("click", () => { input.value = p.name; show(p.name); });
      tryRow.append(b);
    }
  }
  if (!places.length) {
    input.disabled = true;
    out.append(node("p", "Place names load with the map data. None are available yet."));
  }
}

function setupArea(status) {
  if (typeof status.hot_area_km2 !== "number" || !status.latest_photo) return;
  const v = status.hot_area_km2;
  byId("area-ours").hidden = false;
  byId("area-ours-yr").textContent = new Date(status.latest_photo).getUTCFullYear();
  byId("area-ours-bar").style.width = `${Math.min(100, (v / 8.9) * 100)}%`;
  byId("area-ours-val").textContent = `${v.toFixed(2)} km²`;
  const note = node("p", `The striped bar is the hot ground Fire Watch found on the ${fmtDate(status.latest_photo)} photo. It is measured differently from NRSC's maps, so treat it as a rough check, not the same number.`);
  note.className = "area-ours-note";
  byId("area-bars").after(note);
}

function setupSink() {
  const svg = byId("sink-svg");
  if (!svg) return;
  const buttons = [...document.querySelectorAll("#sink-steps button")];
  const g = window.gsap;
  const flat = { road: "M0 104 L160 104 C 240 104 270 104 320 104 C 370 104 400 104 480 104 L640 104", rock: "M0 120 L160 120 C 240 120 270 120 320 120 C 370 120 400 120 480 120 L640 120 L640 340 L0 340 Z", soil: "M0 104 L160 104 C 240 104 270 104 320 104 C 370 104 400 104 480 104 L640 104 L640 122 L480 122 C 400 122 370 122 320 122 C 270 122 240 122 160 122 L0 122 Z", void: "M310 236 C 315 236 325 236 330 236 L330 270 L310 270 Z" };
  const sunk = { road: "M0 104 L160 104 C 240 104 262 134 320 136 C 378 134 400 104 480 104 L640 104", rock: "M0 120 L160 120 C 240 120 262 150 320 152 C 378 150 400 120 480 120 L640 120 L640 340 L0 340 Z", soil: "M0 104 L160 104 C 240 104 262 134 320 136 C 378 134 400 104 480 104 L640 104 L640 122 L480 122 C 400 122 378 150 320 152 C 262 150 240 122 160 122 L0 122 Z", void: "M250 254 C 280 250 360 250 390 254 L390 270 L250 270 Z" };
  const wide = "M250 236 C 280 236 360 236 390 236 L390 270 L250 270 Z";
  const states = [
    { void: flat.void, glow: 1, heat: 0, cracks: 0, sat: 0, ground: flat, barrier: 0, truck: 520 },
    { void: wide, glow: 0.85, heat: 1, cracks: 1, sat: 1, ground: flat, barrier: 0, truck: 520 },
    { void: sunk.void, glow: 0.5, heat: 0.6, cracks: 1, sat: 0, ground: sunk, barrier: 1, truck: 120 },
  ];
  const q = (id) => svg.querySelector(id);
  let at = -1;
  function go(i, instant) {
    if (i === at) return;
    at = i;
    const s = states[i];
    buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
    const d = instant || !g || window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 1.2;
    const set = (sel, vars) => (g ? g.to(q(sel), { duration: d, ease: "power2.inOut", ...vars }) : null);
    if (!g) {
      q("#sk-void").setAttribute("d", s.void);
      for (const [id, key] of [["#sk-road", "road"], ["#sk-lines", "road"], ["#sk-rockfill", "rock"], ["#sk-soil", "soil"]]) q(id).setAttribute("d", s.ground[key]);
      q("#sk-burnglow").setAttribute("opacity", s.glow);
      q("#sk-heat").setAttribute("opacity", s.heat);
      q("#sk-cracks").setAttribute("opacity", s.cracks);
      q("#sk-sat").setAttribute("opacity", s.sat);
      q("#sk-barrier").setAttribute("opacity", s.barrier);
      q("#sk-truck").setAttribute("transform", `translate(${s.truck} 90)`);
      return;
    }
    set("#sk-void", { attr: { d: s.void } });
    set("#sk-road", { attr: { d: s.ground.road } });
    set("#sk-lines", { attr: { d: s.ground.road } });
    set("#sk-rockfill", { attr: { d: s.ground.rock } });
    set("#sk-soil", { attr: { d: s.ground.soil } });
    set("#sk-burnglow", { opacity: s.glow });
    set("#sk-heat", { opacity: s.heat });
    set("#sk-cracks", { opacity: s.cracks });
    set("#sk-sat", { opacity: s.sat });
    set("#sk-barrier", { opacity: s.barrier });
    set("#sk-truck", { x: s.truck, y: 90, duration: d ? 1.8 : 0 });
  }
  if (g) g.set(q("#sk-truck"), { x: 70, y: 90 });
  buttons.forEach((b, i) => b.addEventListener("click", () => { stop(); go(i); }));
  go(0, true);
  let timer = null;
  const stop = () => { if (timer) clearInterval(timer); timer = null; };
  if (g && !window.matchMedia("(prefers-reduced-motion: reduce)").matches && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (!entries[0].isIntersecting || timer !== null || at !== 0) return;
      io.disconnect();
      g.to(q("#sk-truck"), { x: 520, duration: 2.5, ease: "none" });
      let i = 0;
      timer = setInterval(() => { i += 1; if (i > 2) { stop(); return; } go(i); }, 3200);
    }, { threshold: 0.6 });
    io.observe(svg);
  }
}

function setupAbout(status) {
  const dlg = byId("about-data");
  if (!dlg) return;
  if (!dlg.dataset.bound) {
    dlg.dataset.bound = "1";
    const open = () => { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", ""); };
    byId("live-badge").addEventListener("click", open);
    document.querySelectorAll("[data-about]").forEach((b) => b.addEventListener("click", open));
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }
  if (!status) return;
  const real = status.source === "real" || status.source === "landsat";
  if (!real) {
    byId("about-lead").textContent = "This copy runs on demo data: a made-up coalfield with fires placed by a simulator, so every part of the system can be shown. The real setup uses the sources below.";
  } else if (status.photos) {
    byId("about-photos").textContent = `${status.photos} Landsat 8 and 9 Collection 2 Level-2 scenes (USGS) from ${fmtDate(status.first_photo)} to ${fmtDate(status.latest_photo)}, read from Microsoft Planetary Computer. Very cloudy scenes are skipped. Each pass brings a surface temperature image and an ordinary colour photo.`;
  }
}

function setLoading(on) {
  byId("live-off").hidden = true;
  document.querySelector(".live").classList.remove("off");
  if (on) {
    byId("live-badge").textContent = "Loading";
    byId("live-badge").className = "live-badge";
    byId("live-date").innerHTML = '<span class="skel" style="width:70%"></span>';
    byId("live-sub").innerHTML = '<span class="skel" style="width:90%"></span>';
    for (const id of ["n-fires", "n-closed", "n-asks"]) byId(id).innerHTML = '<span class="skel num"></span>';
  }
}

function offline() {
  byId("live-badge").textContent = "Offline";
  byId("live-badge").className = "live-badge off";
  byId("live-date").textContent = "Status unavailable";
  byId("live-sub").textContent = "";
  for (const id of ["n-fires", "n-closed", "n-asks"]) byId(id).textContent = "–";
  byId("live-off").hidden = false;
  document.querySelector(".live").classList.add("off");
  byId("roads-count").textContent = "Road status unavailable";
  byId("roads-when").textContent = "";
  const list = byId("road-list");
  list.replaceChildren();
  const text = node("span");
  text.append(node("b", "Can't load road status"), node("small", "The Fire Watch server didn't answer. Try again in a minute."));
  list.append(li("", node("i"), text, node("span", "")));
  list.firstChild.className = "off";
  byId("compare-frame").classList.remove("loading");
  byId("cmp-empty").textContent = "The heat image can't load while Fire Watch is offline.";
  byId("cmp-empty").hidden = false;
  byId("cmp-cap").textContent = "";
}

let started = false;
async function main() {
  setLoading(true);
  let status;
  try {
    const res = await fetch(API("/api/public/status"));
    if (!res.ok) throw new Error(res.statusText);
    status = await res.json();
  } catch (err) {
    offline();
    if (!started) { started = true; setupAbout(null); setupSink(); }
    return;
  }
  window.FW_STATUS = status;
  window.dispatchEvent(new CustomEvent("fw-status", { detail: status }));
  setupAbout(status);
  renderViewSwitch(status);
  if (!started) { started = true; setupSink(); }
  setLoading(false);
  byId("cmp-empty").hidden = true;
  const real = status.source === "real" || status.source === "landsat";
  byId("live-badge").textContent = real ? "Real Landsat data" : "Demo data";
  byId("live-badge").className = `live-badge${real ? " real" : ""}`;
  byId("live-badge").setAttribute("aria-label", `${real ? "Real Landsat data" : "Demo data"}. About the data`);
  if (status.latest_photo) {
    byId("live-date").textContent = fmtDate(status.latest_photo);
    const cloud = Math.round((status.cloud_fraction || 0) * 100);
    const sky = cloud < 5 ? "clear sky" : `${cloud}% cloud`;
    byId("live-sub").textContent = `${PLATFORMS[status.platform] || "Landsat"}, ${sky}. ${status.photos} photos since ${fmtDate(status.first_photo)}.`;
  } else {
    byId("live-date").textContent = "No photos yet";
    byId("live-sub").textContent = "Processing hasn't started.";
  }
  const sections = status.closed_roads.reduce((n, r) => n + r.sections.length, 0);
  byId("n-fires").textContent = status.fires;
  byId("n-closed").textContent = sections;
  byId("n-asks").textContent = status.pending_decisions;
  byId("roads-count").textContent = status.closed_roads.length ? `${status.closed_roads.length} ${status.closed_roads.length === 1 ? "road has" : "roads have"} closed sections` : "No road closures";
  byId("roads-when").textContent = status.latest_photo ? `As of the ${fmtDate(status.latest_photo)} photo` : "";
  renderRoads(status);
  byId("road-search").addEventListener("input", (e) => renderRoads(status, e.target.value));
  const compare = setupCompare(status);
  setupPlaces(status, compare);
  setupArea(status);
}

byId("live-retry").addEventListener("click", main);
main();
