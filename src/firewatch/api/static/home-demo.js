"use strict";

(function () {
  const scene = document.getElementById("scene");
  if (!scene || !window.gsap) return;
  const q = (sel) => scene.querySelector(sel);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const CHAPTERS = [
    { id: "pass", at: 0, title: "Satellite pass", text: "Every 8 days, a Landsat satellite photographs the Jharia coalfield, including a heat image." },
    { id: "heat", at: 7, title: "Heat image", text: "Sun-warmed coal and dumps are filtered out. What stays is ground hotter than it should be." },
    { id: "find", at: 13, title: "Fires found", text: "OpenCV outlines each hot spot. The coke ovens are recognised as factory heat and ignored." },
    { id: "spread", at: 20, title: "Spread predicted", text: "Fire 6 is followed photo to photo, and Fire Watch predicts where it could reach in 48 days." },
    { id: "close", at: 28.5, title: "Road closed", text: "It is 218 m from the haul road and the agent is confident, so it closes the road. Trucks take a detour." },
    { id: "reopen", at: 38, title: "A person reopens", text: "Weeks later the fire has gone quiet. Fire Watch asks a person, and only a person can reopen the road." },
  ];
  const END = 50;

  const roadMain = q("#main-road");
  const roadHaul = q("#haul-road");
  const lenMain = roadMain.getTotalLength();
  const lenHaul = roadHaul.getTotalLength();

  function placeOn(path, length, el, p, flip) {
    const d = Math.max(0, Math.min(1, p)) * length;
    const a = path.getPointAtLength(d);
    const b = path.getPointAtLength(Math.min(length, d + 1));
    let angle = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;
    if (flip) angle += 180;
    el.setAttribute("transform", `translate(${a.x.toFixed(1)} ${a.y.toFixed(1)}) rotate(${angle.toFixed(1)})`);
  }

  const truckH = q("#truck-h");
  const truckHBody = q("#truck-h-body");
  const haul = { p: 0, back: false };
  function drawHaul() {
    const d = Math.max(0, Math.min(1, haul.p)) * lenHaul;
    const a = roadHaul.getPointAtLength(d);
    const b = roadHaul.getPointAtLength(Math.min(lenHaul, d + 1));
    let angle = (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI;
    if (haul.back) angle += 180;
    truckH.setAttribute("transform", `translate(${a.x.toFixed(1)} ${a.y.toFixed(1)})`);
    truckHBody.setAttribute("transform", `rotate(${angle.toFixed(1)})`);
  }

  const drawable = (el) => {
    const len = el.getTotalLength ? el.getTotalLength() : 400;
    gsap.set(el, { strokeDasharray: len, strokeDashoffset: len });
    return len;
  };

  const ringFire = q("#ring-fire circle");
  const ringOven = q("#ring-oven rect");
  const closed = q("#haul-closed");
  const detour = q("#detour");
  drawable(ringFire);
  drawable(ringOven);
  drawable(closed);
  gsap.set(detour, { opacity: 0 });
  gsap.set(["#tag-fire", "#tag-oven", "#measure", "#card", "#cursor", "#click-ring", "#endcard", "#truck-h-alert"].map(q), { opacity: 0 });
  gsap.set(q("#cone"), { opacity: 0, scale: 0.2, svgOrigin: "660 350" });
  gsap.set(scene.querySelectorAll("#barriers > g"), { opacity: 0, scale: 0, transformOrigin: "50% 50%" });
  gsap.set(q("#sat"), { x: -100, y: 60 });
  gsap.set(q("#beam"), { x: -700, opacity: 0 });
  gsap.set(q("#card"), { x: 60 });
  gsap.set(q("#cursor"), { x: 1180, y: 460 });
  drawHaul();

  const tl = gsap.timeline({ paused: true, defaults: { ease: "power2.inOut" }, onUpdate: sync, onComplete: () => setPlaying(false) });

  tl.to("#sat", { x: 1400, duration: 6.5, ease: "none" }, 0)
    .to("#beam", { opacity: 1, duration: 0.6 }, 0.3)
    .to("#beam", { x: 800, duration: 6.5, ease: "none" }, 0)
    .to("#beam", { opacity: 0, duration: 0.6 }, 6)
    .to("#scan-rect", { attr: { width: 1280 }, duration: 6, ease: "none" }, 0.3)
    .to(haul, { p: 1, duration: 9, ease: "sine.inOut", onUpdate: drawHaul }, 0.5)
    .set(haul, { back: true, onComplete: drawHaul }, 9.6)
    .to("#warm-spots", { opacity: 0, duration: 2.5 }, 8.5)
    .fromTo("#heat-fire", { scale: 0.9, svgOrigin: "660 350" }, { immediateRender: false, scale: 1.15, duration: 1.2, yoyo: true, repeat: 3, ease: "sine.inOut" }, 8)
    .to(ringFire, { strokeDashoffset: 0, duration: 1.4 }, 13.4)
    .to("#tag-fire", { opacity: 1, y: -6, duration: 0.6, ease: "back.out(2)" }, 14.6)
    .to(ringOven, { strokeDashoffset: 0, duration: 1.4 }, 15.6)
    .to("#tag-oven", { opacity: 1, y: -6, duration: 0.6, ease: "back.out(2)" }, 16.8)
    .to("#heat-oven", { opacity: 0.35, duration: 1.2 }, 17.6)
    .to(haul, { p: 0, duration: 8, ease: "sine.inOut", onUpdate: drawHaul }, 10)
    .to("#thermal", { opacity: 0.45, duration: 1.2 }, 20.2)
    .to("#cone", { opacity: 1, scale: 1, duration: 2.6, ease: "power3.out" }, 20.6)
    .to("#measure", { opacity: 1, duration: 0.8 }, 24.2)
    .set(haul, { back: false, onComplete: drawHaul }, 26.5)
    .to(haul, { p: 0.12, duration: 3, ease: "power1.out", onUpdate: drawHaul }, 26.6)
    .to(closed, { strokeDashoffset: 0, duration: 1.4, ease: "power1.inOut" }, 29)
    .to(scene.querySelectorAll("#barriers > g"), { opacity: 1, scale: 1, duration: 0.6, ease: "back.out(3)", stagger: 0.25 }, 30.2)
    .to("#truck-h-alert", { opacity: 1, duration: 0.3 }, 31)
    .fromTo("#truck-h-alert", { scale: 0.6, transformOrigin: "50% 50%" }, { immediateRender: false, scale: 1.15, duration: 0.5, yoyo: true, repeat: 3, ease: "sine.inOut" }, 31)
    .to(detour, { opacity: 1, duration: 1 }, 32.4)
    .fromTo(detour, { strokeDashoffset: 120 }, { immediateRender: false, strokeDashoffset: 0, duration: 4, ease: "none" }, 32.4)
    .to("#measure", { opacity: 0, duration: 0.6 }, 37.6)
    .to(["#heat-fire", "#ring-fire", "#tag-fire"], { opacity: 0.25, duration: 2 }, 38.2)
    .to("#cone", { opacity: 0, duration: 1.5 }, 38.6)
    .to("#card", { opacity: 1, x: 0, duration: 0.9, ease: "power3.out" }, 40)
    .to("#cursor", { opacity: 1, duration: 0.3 }, 41)
    .to("#cursor", { x: 972, y: 262, duration: 1.6, ease: "power2.inOut" }, 41.1)
    .fromTo("#click-ring", { opacity: 1, scale: 0.4, svgOrigin: "978 268" }, { immediateRender: false, opacity: 0, scale: 2.4, duration: 0.6, ease: "power1.out" }, 42.8)
    .to("#btn-yes rect", { fill: "#4f7343", duration: 0.3 }, 42.8)
    .to("#card", { opacity: 0, x: 40, duration: 0.6 }, 43.8)
    .to("#cursor", { opacity: 0, duration: 0.4 }, 43.8)
    .to(closed, { strokeDashoffset: closed.getTotalLength(), duration: 1 }, 44.2)
    .to(scene.querySelectorAll("#barriers > g"), { opacity: 0, scale: 0, duration: 0.4, stagger: 0.15 }, 44.2)
    .to("#truck-h-alert", { opacity: 0, duration: 0.3 }, 44.6)
    .to(detour, { opacity: 0, duration: 0.8 }, 44.6)
    .to(haul, { p: 1, duration: 4, ease: "sine.inOut", onUpdate: drawHaul }, 44.8)
    .to("#endcard", { opacity: 1, duration: 1 }, 48.6)
    .set({}, {}, END);

  const ambient = gsap.timeline({ repeat: -1, paused: true });
  const mainA = { p: 0 };
  const mainB = { p: 1 };
  const truckA = q("#truck-a");
  const truckB = q("#truck-b");
  const drawA = () => placeOn(roadMain, lenMain, truckA, mainA.p, false);
  const drawB = () => placeOn(roadMain, lenMain, truckB, mainB.p, true);
  ambient.fromTo(mainA, { p: -0.05 }, { p: 1.05, duration: 16, ease: "none", onUpdate: drawA }, 0)
    .fromTo(mainB, { p: 1.05 }, { p: -0.05, duration: 16, ease: "none", onUpdate: drawB }, 4);
  ambient.progress(0.2);
  drawA();
  drawB();

  const caption = document.getElementById("cap-text");
  const step = document.getElementById("cap-step");
  const hud = q("#hud-date");
  const fill = document.getElementById("scrub-fill");
  const scrub = document.getElementById("scrub");
  const time = document.getElementById("time");
  const toggle = document.getElementById("toggle");
  const bigPlay = document.getElementById("big-play");
  const endActions = document.getElementById("end-actions");
  const chapterList = document.getElementById("chapters");
  const marks = document.getElementById("scrub-marks");
  let current = -1;

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  CHAPTERS.forEach((c, i) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    const label = document.createElement("b");
    label.textContent = `${i + 1}  ${fmt(c.at)}`;
    b.append(label, c.title);
    b.addEventListener("click", () => { tl.seek(c.at); play(); });
    li.append(b);
    chapterList.append(li);
    if (i) {
      const m = document.createElement("span");
      m.style.left = `${(c.at / END) * 100}%`;
      marks.append(m);
    }
  });
  const chapterButtons = [...chapterList.querySelectorAll("button")];

  function sync() {
    const t = tl.time();
    let idx = 0;
    CHAPTERS.forEach((c, i) => { if (t >= c.at) idx = i; });
    if (idx !== current) {
      current = idx;
      caption.textContent = CHAPTERS[idx].text;
      step.textContent = `${idx + 1} / ${CHAPTERS.length}`;
      chapterButtons.forEach((b, i) => b.classList.toggle("on", i === idx));
      hud.textContent = idx >= 5 ? "Landsat 8 · 9 Feb 2026" : "Landsat 9 · 23 Dec 2025";
    }
    const pct = (t / END) * 100;
    fill.style.width = `${pct}%`;
    scrub.setAttribute("aria-valuenow", Math.round(pct));
    time.textContent = `${fmt(t)} / ${fmt(END)}`;
    endActions.hidden = t < END - 1.2;
  }

  function setPlaying(on) {
    toggle.classList.toggle("playing", on);
    toggle.setAttribute("aria-label", on ? "Pause" : "Play");
    bigPlay.hidden = on || tl.time() >= END - 0.01;
    if (on) ambient.play(); else ambient.pause();
  }
  function play() {
    if (tl.time() >= END - 0.01) tl.seek(0);
    tl.play();
    setPlaying(true);
  }
  function pause() {
    tl.pause();
    setPlaying(false);
  }

  toggle.addEventListener("click", () => (tl.isActive() ? pause() : play()));
  bigPlay.addEventListener("click", play);
  document.getElementById("replay").addEventListener("click", () => { tl.seek(0); play(); });

  function seekFromEvent(e) {
    const r = scrub.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    tl.seek(x * END);
  }
  let dragging = false;
  scrub.addEventListener("pointerdown", (e) => { dragging = true; scrub.setPointerCapture(e.pointerId); seekFromEvent(e); });
  scrub.addEventListener("pointermove", (e) => { if (dragging) seekFromEvent(e); });
  scrub.addEventListener("pointerup", () => { dragging = false; });
  scrub.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { tl.seek(Math.min(END, tl.time() + 5)); e.preventDefault(); }
    if (e.key === "ArrowLeft") { tl.seek(Math.max(0, tl.time() - 5)); e.preventDefault(); }
    if (e.key === " " || e.key === "Enter") { tl.isActive() ? pause() : play(); e.preventDefault(); }
  });

  const watch = document.getElementById("watch-demo");
  if (watch) watch.addEventListener("click", () => setTimeout(play, 500));

  sync();
  setPlaying(false);

  const saveBtn = document.getElementById("save-video");
  const saveStatus = document.getElementById("save-status");
  const canRecord = saveBtn && !window.FW_NO_DOWNLOAD && window.MediaRecorder && HTMLCanvasElement.prototype.captureStream;
  const MIME = canRecord ? ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"].find((m) => MediaRecorder.isTypeSupported(m)) : null;
  if (canRecord && MIME) {
    saveBtn.hidden = false;
    const SCENE_CSS = [".sc-label{font:600 17px Barlow,sans-serif;fill:#f3f1e8;paint-order:stroke;stroke:rgba(20,20,20,.55);stroke-width:4px}", ".sc-road{font-size:15px;fill:#fff7dd}", ".tag text{font:600 16px Barlow,sans-serif;fill:#fff}", ".alert-text{font:700 14px Barlow,sans-serif;fill:#fff}", ".hud-text{font:600 15px Barlow,sans-serif;fill:#fff}", ".card-title{font:600 24px 'Barlow Condensed',sans-serif;fill:#1e2326}", ".card-text{font:400 17px Barlow,sans-serif;fill:#5a636a}", ".btn-text{font:600 16px Barlow,sans-serif;fill:#fff}", ".btn-text.dark{fill:#1e2326}", ".end-title{font:700 72px 'Barlow Condensed',sans-serif;fill:#fff}", ".end-sub{font:400 24px Barlow,sans-serif;fill:#cfd5d3}"].join("");
    const FONTS = [["Barlow", 400, "barlow-latin-400-normal"], ["Barlow", 600, "barlow-latin-600-normal"], ["Barlow Condensed", 600, "barlow-condensed-latin-600-normal"], ["Barlow Condensed", 700, "barlow-condensed-latin-700-normal"]];
    let fontCss = null;
    async function loadFonts() {
      if (fontCss !== null) return fontCss;
      const base = window.FW_FONT_BASE || "/static/vendor/fonts/";
      const parts = await Promise.all(FONTS.map(async ([family, weight, file]) => {
        try {
          const blob = await (await fetch(`${base}${file}.woff2`)).blob();
          const url = await new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(blob); });
          return `@font-face{font-family:'${family}';font-weight:${weight};src:url(${url}) format('woff2')}`;
        } catch (e) { return ""; }
      }));
      fontCss = parts.join("");
      return fontCss;
    }
    function wrapText(ctx, text, maxWidth) {
      const words = text.split(" ");
      const lines = [];
      let line = "";
      for (const w of words) {
        const next = line ? `${line} ${w}` : w;
        if (ctx.measureText(next).width > maxWidth && line) { lines.push(line); line = w; } else line = next;
      }
      if (line) lines.push(line);
      return lines;
    }
    function vint(bytes, at) {
      const first = bytes[at];
      let len = 1;
      while (len <= 8 && !(first & (0x80 >> (len - 1)))) len += 1;
      let value = first & (0xff >> len);
      for (let i = 1; i < len; i++) value = value * 256 + bytes[at + i];
      return { len, value };
    }
    function elementId(bytes, at) {
      const first = bytes[at];
      let len = 1;
      while (len <= 4 && !(first & (0x80 >> (len - 1)))) len += 1;
      let id = 0;
      for (let i = 0; i < len; i++) id = id * 256 + bytes[at + i];
      return { len, id };
    }
    async function withDuration(blob, ms) {
      try {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        let at = 0;
        const head = elementId(bytes, at);
        if (head.id !== 0x1a45dfa3) return blob;
        const headSize = vint(bytes, at + head.len);
        at += head.len + headSize.len + headSize.value;
        const seg = elementId(bytes, at);
        if (seg.id !== 0x18538067) return blob;
        at += seg.len + vint(bytes, at + seg.len).len;
        for (let guard = 0; guard < 16 && at < bytes.length; guard++) {
          const el = elementId(bytes, at);
          const size = vint(bytes, at + el.len);
          const body = at + el.len + size.len;
          if (el.id === 0x1549a966) {
            const content = bytes.subarray(body, body + size.value);
            const dur = new Uint8Array(11);
            dur.set([0x44, 0x89, 0x88]);
            new DataView(dur.buffer).setFloat64(3, ms);
            const total = content.length + dur.length;
            const sizeBytes = new Uint8Array(8);
            sizeBytes[0] = 0x01;
            let rest = total;
            for (let i = 7; i >= 1; i--) { sizeBytes[i] = rest & 0xff; rest = Math.floor(rest / 256); }
            return new Blob([bytes.subarray(0, at), bytes.subarray(at, at + el.len), sizeBytes, content, dur, bytes.subarray(body + size.value)], { type: blob.type });
          }
          at = body + size.value;
        }
      } catch (e) {
        return blob;
      }
      return blob;
    }
    let recording = null;
    saveBtn.addEventListener("click", async () => {
      if (recording) { recording.cancel = true; return; }
      pause();
      const resume = tl.time();
      const css = `${await loadFonts()}${SCENE_CSS}`;
      const canvas = document.createElement("canvas");
      canvas.width = 1280;
      canvas.height = 720;
      const ctx = canvas.getContext("2d");
      const stream = canvas.captureStream(30);
      const chunks = [];
      const rec = new MediaRecorder(stream, { mimeType: MIME, videoBitsPerSecond: 6e6 });
      rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      recording = { cancel: false };
      saveBtn.classList.add("rec");
      saveStatus.textContent = "Recording the demo as a video. This takes about a minute.";
      const img = new Image();
      const serializer = new XMLSerializer();
      async function draw(t) {
        tl.seek(t, false);
        ambient.time(3.2 + t, false);
        sync();
        const clone = scene.cloneNode(true);
        clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        clone.setAttribute("width", "1280");
        clone.setAttribute("height", "720");
        const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
        style.textContent = css;
        clone.insertBefore(style, clone.firstChild);
        const url = URL.createObjectURL(new Blob([serializer.serializeToString(clone)], { type: "image/svg+xml" }));
        img.src = url;
        try { await img.decode(); } catch (e) { URL.revokeObjectURL(url); return; }
        ctx.drawImage(img, 0, 0, 1280, 720);
        URL.revokeObjectURL(url);
        const grad = ctx.createLinearGradient(0, 560, 0, 720);
        grad.addColorStop(0, "rgba(14,16,18,0)");
        grad.addColorStop(1, "rgba(14,16,18,0.9)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 560, 1280, 160);
        ctx.font = "500 30px Barlow, sans-serif";
        const lines = wrapText(ctx, caption.textContent, 1040);
        ctx.fillStyle = "#e2b007";
        const pill = step.textContent;
        ctx.font = "700 20px 'Barlow Condensed', sans-serif";
        const pw = ctx.measureText(pill).width + 22;
        const top = 720 - 34 - lines.length * 38;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(40, top - 4, pw, 30, 14); else ctx.rect(40, top - 4, pw, 30);
        ctx.fill();
        ctx.fillStyle = "#1e2326";
        ctx.fillText(pill, 51, top + 18);
        ctx.font = "500 30px Barlow, sans-serif";
        ctx.fillStyle = "#ffffff";
        lines.forEach((l, i) => ctx.fillText(l, 60 + pw, top + 20 + i * 38));
      }
      await document.fonts.ready;
      await draw(0);
      rec.start(1000);
      const t0 = performance.now();
      let lastPct = -1;
      while (!recording.cancel) {
        const t = (performance.now() - t0) / 1000;
        if (t > END + 0.5) break;
        await draw(Math.min(END, t));
        const pct = Math.floor((t / END) * 100);
        if (pct !== lastPct) { lastPct = pct; saveBtn.textContent = `Recording ${Math.min(100, pct)}%, click to stop`; }
        await new Promise((ok) => requestAnimationFrame(ok));
      }
      const cancelled = recording.cancel;
      const recordedMs = performance.now() - t0;
      await new Promise((ok) => { rec.onstop = ok; rec.stop(); });
      stream.getTracks().forEach((tr) => tr.stop());
      recording = null;
      saveBtn.classList.remove("rec");
      saveBtn.textContent = "Save as video";
      tl.seek(resume);
      sync();
      if (cancelled) { saveStatus.textContent = "Recording stopped. Nothing was saved."; return; }
      const type = MIME.split(";")[0];
      const raw = new Blob(chunks, { type });
      const blob = type === "video/webm" ? await withDuration(raw, recordedMs) : raw;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `jharia-fire-watch-demo.${type === "video/mp4" ? "mp4" : "webm"}`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      saveStatus.textContent = "Video saved to your downloads.";
    });
  }

  if (!reduced && "IntersectionObserver" in window) {
    let started = false;
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && !started) { started = true; play(); }
        else if (!e.isIntersecting && tl.isActive()) pause();
      }
    }, { threshold: 0.55 });
    io.observe(document.getElementById("stage"));
  }
})();
