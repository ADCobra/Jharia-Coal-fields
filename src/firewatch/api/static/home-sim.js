"use strict";

(function () {
  const canvas = document.getElementById("sim-canvas");
  if (!canvas || !window.THREE) return;
  const T = window.THREE;
  const stage = document.getElementById("sim-stage");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const tween = (target, vars) => (window.gsap ? window.gsap.to(target, { duration: 0.6, ease: "power2.out", ...vars }) : Object.assign(target, vars));

  const RULES = { weights: { heat: 0.35, spread: 0.3, near: 0.35 }, d0: 400, high: 0.62, medium: 0.4, closeWithin: 250, autoConfidence: 0.7, cloudEdge: 15 };
  const M_PER_UNIT = 1000 / 12;
  const ROAD_Z = 5;
  const SPAN = 3.4;
  const EDGE = 34;

  const inputs = { heat: 14, dist: 450, speed: 2, cloud: 0, seen: true, factory: false };

  function assess(v) {
    const peak = v.heat;
    const mean = 0.55 * peak;
    const clamp = (x) => Math.max(0, Math.min(1, x));
    const aHat = clamp((0.5 * mean) / 12 + (0.5 * peak) / 24);
    const sHat = clamp(v.speed / 3);
    const reach = v.speed * 48;
    const projected = Math.max(0, v.dist - reach - 60);
    const effective = Math.min(v.dist, projected);
    const pHat = Math.exp(-effective / RULES.d0);
    const w = RULES.weights;
    const score = w.heat * aHat + w.spread * sHat + w.near * pHat;
    const level = score >= RULES.high ? "high" : score >= RULES.medium ? "medium" : "low";
    let conf = 1;
    const notes = [];
    if (!v.seen) { conf -= 0.15; notes.push("first sighting"); }
    const cloudEdge = v.cloud >= RULES.cloudEdge;
    if (cloudEdge) { conf -= 0.35; notes.push("partly under cloud"); }
    if (v.cloud > 50) conf -= 0.1;
    if (v.factory) conf -= 0.3;
    if (v.seen) conf += 0.15;
    conf = clamp(conf);
    const near = v.dist <= RULES.closeWithin || projected <= RULES.closeWithin;
    let decision, why;
    if (v.cloud >= 90) {
      decision = "hidden"; why = "Cloud covers the fire completely, so the satellite can't see it. Fire Watch waits for the next clear photo.";
    } else if (v.factory) {
      decision = "exclude"; why = "Very hot, a neat rectangle and on a mapped plant: this is coke ovens, not a coal fire. Ignored, so no false alarm.";
    } else if (cloudEdge && (near || level !== "low")) {
      decision = "ask"; why = "Part of the fire is under cloud, so its real size is unknown. Fire Watch asks a person instead of guessing.";
    } else if (level === "high") {
      if (conf < RULES.autoConfidence) { decision = "ask"; why = `High risk, but it is only ${Math.round(conf * 100)}% sure. Below 70%, a person decides.`; }
      else if (!near) { decision = "ask"; why = `High risk, but no road is within ${RULES.closeWithin} m of the fire or its spread. A person takes a look.`; }
      else if (v.dist > RULES.closeWithin) { decision = "ask"; why = `The fire is ${v.dist} m away. Only its 48-day spread reaches the road, and the next photo comes first, so it asks a person.`; }
      else { decision = "close"; why = `High risk, ${v.dist} m from the road and ${Math.round(conf * 100)}% sure. The agent closes the road by itself.`; }
    } else if (level === "medium") {
      decision = "ask"; why = "Medium risk: worth a person's look, not an automatic closure.";
    } else if (conf < 0.5 && near) {
      decision = "ask"; why = "Low risk, but the measurement is doubtful and a road is close. A person checks.";
    } else {
      decision = "watch"; why = "Not a danger to a road right now. Fire Watch checks it again on every new photo.";
    }
    return { score, level, conf, decision, why, reach, projected, notes };
  }

  // Fixed seed so the trees, rocks and people land in the same place on every visit.
  let seed = 20260;
  const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  const pick = (list) => list[Math.floor(rand() * list.length)];
  const smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

  const renderer = new T.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  const scene = new T.Scene();
  const HAZE = new T.Color(0xcbbda2);
  scene.background = HAZE;
  scene.fog = new T.Fog(HAZE, 34, 125);
  const camera = new T.PerspectiveCamera(45, 1, 0.1, 1200);
  const target = new T.Vector3(0, 0, 2.6);
  const lookAt = new T.Vector3();
  const view = { yaw: -0.35, pitch: 0.42, radius: 24 };

  scene.add(new T.HemisphereLight(0xc4d6ea, 0x6a5a40, 0.72));
  const sun = new T.DirectionalLight(0xfff1d8, 1.0);
  sun.position.set(-26, 24, -8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -28, right: 28, top: 24, bottom: -24, near: 1, far: 90 });
  sun.shadow.bias = -0.0006;
  scene.add(sun);

  // Tileable value noise, for the textures and the outer hills.
  const perm = new Uint8Array(512), lattice = new Float32Array(256);
  for (let i = 0; i < 256; i++) { perm[i] = i; lattice[i] = rand(); }
  for (let i = 255; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  for (let i = 0; i < 256; i++) perm[256 + i] = perm[i];
  function noise(x, y, period = 256) {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const h = (i, j) => lattice[perm[(perm[((i % period) + period) % period & 255] + (((j % period) + period) % period)) & 255]];
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  }
  function fbm(x, y, octaves = 5, period = 256) {
    let sum = 0, amp = 0.5, norm = 0, f = 1;
    for (let o = 0; o < octaves; o++) { sum += amp * noise(x * f, y * f, period * f); norm += amp; amp *= 0.5; f *= 2; }
    return sum / norm;
  }

  const W = 64, D = 46;
  const TW = 170, TD = 140, TSX = 300, TSZ = 250;
  const pit = { x: -9, z: -5, r: 5.2 };
  const dump = { x: 10, z: -7 };
  const village = { x: 14.5, z: 11.3 };
  const pitDist = (x, z) => Math.hypot(x - pit.x, (z - pit.z) * 1.25);
  const dumpDist = (x, z) => Math.hypot(x - dump.x, (z - dump.z) * 1.4);
  const grassAmt = (x, z) => 0.5 + 0.5 * Math.sin(x * 0.13 - z * 0.19 + 2) * Math.sin(x * 0.31 + z * 0.05 + 0.5);
  function height(x, z) {
    let h = 0.25 * Math.sin(x * 0.35) * Math.cos(z * 0.3) + 0.15 * Math.sin(x * 0.9 + z * 0.7) + 0.08 * Math.cos(x * 1.7 - z * 1.3);
    h += 0.05 * Math.sin(x * 2.3 + z * 1.1) * Math.sin(z * 2.9 - x * 0.7) + 0.025 * Math.sin(x * 5.1 - z * 4.3);
    // Rolling hills beyond the coalfield, which the haze fades into the horizon.
    const rim = smooth(24, 62, Math.hypot(x * 0.8, z));
    if (rim > 0) h += rim * (1.6 + 3.2 * fbm(x * 0.035 + 40, z * 0.035 + 40, 4));
    const rp = pitDist(x, z);
    if (rp < pit.r) h -= Math.ceil((pit.r - rp) / 1.1) * 0.55;
    else if (rp < pit.r + 1.2) h += 0.25 * (1 - (rp - pit.r) / 1.2);
    const rd = dumpDist(x, z);
    if (rd < 4) h += 1.6 * (1 - rd / 4) ** 1.4;
    const dz = Math.abs(z - ROAD_Z);
    if (dz < 1.6) h *= dz / 1.6;
    return h;
  }

  function canvasTexture(w, h, draw) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new T.CanvasTexture(c);
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return t;
  }
  function pixelTexture(size, fn) {
    return canvasTexture(size, size, (g) => {
      const img = g.createImageData(size, size);
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const [r, gg, b, a = 255] = fn(x, y);
        const i = (y * size + x) * 4;
        img.data[i] = r; img.data[i + 1] = gg; img.data[i + 2] = b; img.data[i + 3] = a;
      }
      g.putImageData(img, 0, 0);
    });
  }

  // Ground detail: soil grain at several scales, pebbles and small cracks, near neutral so the
  // vertex colours carry the hue. The same texture drives the bump map.
  const groundTex = pixelTexture(512, (x, y) => {
    const n = fbm(x / 64, y / 64, 5, 8);
    const grain = noise(x / 2.2, y / 2.2, 233);
    const v = 150 + 85 * n + 30 * (grain - 0.5);
    return [v + 6, v + 2, v - 6];
  });
  (() => {
    const g = groundTex.image.getContext("2d");
    for (let i = 0; i < 900; i++) {
      const light = rand() < 0.5;
      g.fillStyle = light ? `rgba(235,225,205,${0.25 + rand() * 0.35})` : `rgba(60,50,40,${0.2 + rand() * 0.3})`;
      g.beginPath(); g.ellipse(rand() * 512, rand() * 512, 0.8 + rand() * 2.2, 0.6 + rand() * 1.6, rand() * 3, 0, 7); g.fill();
    }
    g.strokeStyle = "rgba(70,58,45,0.35)"; g.lineWidth = 1;
    for (let k = 0; k < 18; k++) {
      let x = rand() * 512, y = rand() * 512, a = rand() * 6.3;
      g.beginPath(); g.moveTo(x, y);
      for (let s2 = 0; s2 < 10; s2++) { a += (rand() - 0.5) * 1.2; x += Math.cos(a) * 5; y += Math.sin(a) * 5; g.lineTo(x, y); }
      g.stroke();
    }
    groundTex.needsUpdate = true;
  })();
  groundTex.wrapS = groundTex.wrapT = T.RepeatWrapping;
  groundTex.repeat.set(TW / 4.5, TD / 4.5);

  const geo = new T.PlaneGeometry(TW, TD, TSX, TSZ);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colors = [];
  const cEarth = new T.Color(0x8f7b5a), cDry = new T.Color(0xa69168), cGreen = new T.Color(0x5f6e38), cLaterite = new T.Color(0x9b6244);
  const cDust = new T.Color(0x4a4540), cCoal = new T.Color(0x26282a), cDump = new T.Color(0x857f75), cFar = new T.Color(0x7a7a4e);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    pos.setY(i, height(x, z));
    const c = cEarth.clone().lerp(cDry, fbm(x * 0.08 + 7, z * 0.08 + 3, 4));
    c.lerp(cGreen, smooth(0.55, 0.85, grassAmt(x, z)) * 0.8);
    c.lerp(cLaterite, smooth(0.62, 0.8, fbm(x * 0.11 + 19, z * 0.11 + 5, 3)) * 0.7);
    c.lerp(cFar, smooth(28, 60, Math.hypot(x * 0.8, z)) * 0.55);
    const dz = Math.abs(z - ROAD_Z);
    if (dz < 2.8) c.lerp(cDust, (1 - dz / 2.8) * 0.5);
    const rp = pitDist(x, z);
    if (rp < pit.r) c.lerp(cCoal, 0.72 + 0.2 * (Math.ceil((pit.r - rp) / 1.1) % 2));
    else if (rp < pit.r + 1.4) c.lerp(cCoal, 0.35);
    if (dumpDist(x, z) < 4) c.lerp(cDump, 0.8);
    const shade = 0.9 + 0.2 * (fbm(x * 0.6, z * 0.6, 3) - 0.5);
    colors.push(c.r * shade, c.g * shade, c.b * shade);
  }
  geo.setAttribute("color", new T.Float32BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  const groundColors = Float32Array.from(colors);
  const ground = new T.Mesh(geo, new T.MeshPhongMaterial({ vertexColors: true, map: groundTex, bumpMap: groundTex, bumpScale: 0.05, shininess: 3, specular: 0x0d0d0d }));
  ground.receiveShadow = true;
  scene.add(ground);

  // Ground scorched around the fire, written into the vertex colours whenever the fire moves or changes.
  const cScorched = new T.Color(0x1c1714);
  let scorchKey = "";
  function scorchGround(fx, fz, radius, strength) {
    const key = `${fx.toFixed(2)},${fz.toFixed(2)},${radius.toFixed(2)},${strength.toFixed(2)}`;
    if (key === scorchKey) return;
    scorchKey = key;
    const col = geo.attributes.color;
    const reach = radius * 1.5;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i);
      const d = Math.hypot(x - fx, z - fz);
      let k = 0;
      if (d < reach) k = strength * (1 - smooth(radius * 0.35, reach, d)) * (0.75 + 0.25 * noise(x * 3, z * 3));
      const r = groundColors[i * 3], g2 = groundColors[i * 3 + 1], b = groundColors[i * 3 + 2];
      col.setXYZ(i, r + (cScorched.r - r) * k, g2 + (cScorched.g - g2) * k, b + (cScorched.b - b) * k);
    }
    col.needsUpdate = true;
  }

  // Water collected at the bottom of the open-cast pit.
  const water = new T.Mesh(new T.CircleGeometry(1, 48), new T.MeshPhongMaterial({ color: 0x2f4a4a, specular: 0x8fa3a3, shininess: 70, transparent: true, opacity: 0.88 }));
  water.rotation.x = -Math.PI / 2;
  water.scale.set(2.6, 2.6 / 1.25, 1);
  water.position.set(pit.x, height(pit.x, pit.z) + 0.32, pit.z);
  scene.add(water);

  // Sky: a gradient dome from a dusty horizon haze to blue, with a glow around the sun.
  const sky = new T.Mesh(new T.SphereGeometry(800, 32, 16), new T.ShaderMaterial({
    side: T.BackSide, depthWrite: false, fog: false,
    uniforms: { zenith: { value: new T.Color(0x4f80bb) }, mid: { value: new T.Color(0x9dbad6) }, horizon: { value: HAZE }, sunDir: { value: sun.position.clone().normalize() } },
    vertexShader: "varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: [
      "uniform vec3 zenith; uniform vec3 mid; uniform vec3 horizon; uniform vec3 sunDir; varying vec3 vDir;",
      "void main() {",
      "  float h = vDir.y;",
      "  vec3 col = mix(horizon, mid, smoothstep(0.0, 0.1, h));",
      "  col = mix(col, zenith, smoothstep(0.08, 0.55, h));",
      "  col = mix(col, horizon * 0.92, smoothstep(0.0, -0.08, h));",
      "  float s = max(dot(vDir, normalize(sunDir)), 0.0);",
      "  col += vec3(1.0, 0.86, 0.62) * (pow(s, 900.0) * 2.0 + pow(s, 24.0) * 0.18 + pow(s, 4.0) * 0.06);",
      "  gl_FragColor = vec4(col, 1.0);",
      "}",
    ].join("\n"),
  }));
  scene.add(sky);

  // A layer of fair-weather cumulus high above, drifting slowly.
  const skyCloudTex = pixelTexture(512, (x, y) => {
    const n = fbm(x / 128, y / 128, 6, 4);
    const a = smooth(0.52, 0.78, n);
    const shade = 225 + 30 * smooth(0.6, 0.9, n);
    return [shade, shade, Math.min(255, shade + 6), Math.round(a * 235)];
  });
  skyCloudTex.wrapS = skyCloudTex.wrapT = T.RepeatWrapping;
  skyCloudTex.repeat.set(3, 3);
  const skyClouds = new T.Mesh(new T.PlaneGeometry(1600, 1600), new T.MeshBasicMaterial({ map: skyCloudTex, transparent: true, depthWrite: false, fog: false, opacity: 0.9 }));
  skyClouds.rotation.x = Math.PI / 2;
  skyClouds.position.y = 120;
  scene.add(skyClouds);

  // Worn asphalt: aggregate grain, darker tyre tracks, cracks, patches, coal dust at the edges and faded markings.
  const RL = TW;
  const roadTex = canvasTexture(1024, 160, (g, w, h) => {
    const img = g.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const n = fbm(x / 48, y / 48, 4, 64);
      const grain = rand();
      const across = y / h;
      const track = Math.exp(-(((across - 0.3) / 0.07) ** 2)) + Math.exp(-(((across - 0.7) / 0.07) ** 2));
      let v = 74 + 26 * n + 22 * (grain - 0.5) - 12 * track;
      const edge = Math.min(across, 1 - across);
      const dust = smooth(0.09, 0.0, edge);
      const i = (y * w + x) * 4;
      img.data[i] = v + dust * 38; img.data[i + 1] = v - 1 + dust * 30; img.data[i + 2] = v - 4 + dust * 18; img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    g.fillStyle = "rgba(40,38,36,0.55)";
    for (let k = 0; k < 4; k++) g.fillRect(rand() * w, 20 + rand() * (h - 60), 60 + rand() * 120, 18 + rand() * 30);
    g.strokeStyle = "rgba(25,24,22,0.7)"; g.lineWidth = 1.4;
    for (let k = 0; k < 14; k++) {
      let x = rand() * w, y = rand() * h, a = rand() * 6.3;
      g.beginPath(); g.moveTo(x, y);
      for (let s2 = 0; s2 < 12; s2++) { a += (rand() - 0.5) * 1.1; x += Math.cos(a) * 7; y += Math.sin(a) * 7; g.lineTo(x, y); }
      g.stroke();
    }
    g.fillStyle = "rgba(225,218,196,0.5)";
    for (let x = 0; x < w; x += 3) { if (rand() < 0.85) { g.fillRect(x, 11, 3, 4); g.fillRect(x, h - 15, 3, 4); } }
    g.fillStyle = "rgba(236,214,120,0.75)";
    for (let x = 0; x < 380; x += 2) if (rand() < 0.9) g.fillRect(x, h / 2 - 3, 2, 6);
  });
  roadTex.wrapS = T.RepeatWrapping;
  roadTex.repeat.set(RL / 8, 1);
  const road = new T.Mesh(new T.BoxGeometry(RL, 0.1, 1.3), new T.MeshPhongMaterial({ map: roadTex, bumpMap: roadTex, bumpScale: 0.025, shininess: 6, specular: 0x1a1a1a }));
  road.position.set(0, 0.05, ROAD_Z);
  road.receiveShadow = true;
  scene.add(road);
  const shoulderTex = groundTex.clone();
  shoulderTex.needsUpdate = true;
  shoulderTex.repeat.set(RL / 3, 0.25);
  for (const side of [-1, 1]) {
    const shoulder = new T.Mesh(new T.BoxGeometry(RL, 0.06, 0.42), new T.MeshPhongMaterial({ color: 0x7c6d58, map: shoulderTex, bumpMap: shoulderTex, bumpScale: 0.04, shininess: 2 }));
    shoulder.position.set(0, 0.03, ROAD_Z + side * 0.86);
    shoulder.receiveShadow = true;
    scene.add(shoulder);
  }
  const span = new T.Mesh(new T.BoxGeometry(SPAN * 2, 0.04, 1.34), new T.MeshBasicMaterial({ color: 0xd23f1d, transparent: true, opacity: 0 }));
  span.position.set(0, 0.13, ROAD_Z);
  scene.add(span);

  function shadowAll(obj) { obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } }); return obj; }

  const stripesTex = canvasTexture(128, 32, (g) => {
    g.fillStyle = "#ffffff"; g.fillRect(0, 0, 128, 32);
    g.fillStyle = "#d23f1d";
    for (let i = -32; i < 160; i += 32) { g.beginPath(); g.moveTo(i, 32); g.lineTo(i + 16, 32); g.lineTo(i + 32, 0); g.lineTo(i + 16, 0); g.fill(); }
  });
  const barrierMat = new T.MeshLambertMaterial({ map: stripesTex });
  const legMat = new T.MeshLambertMaterial({ color: 0x333333 });
  const barriers = [-SPAN, SPAN].map((x) => {
    const g = new T.Group();
    const board = new T.Mesh(new T.BoxGeometry(0.12, 0.35, 1.5), barrierMat);
    board.position.y = 0.55;
    g.add(board);
    for (const dz of [-0.6, 0.6]) {
      const leg = new T.Mesh(new T.BoxGeometry(0.08, 0.6, 0.08), legMat);
      leg.position.set(0, 0.3, dz);
      g.add(leg);
    }
    g.position.set(x, 0.1, ROAD_Z);
    g.scale.set(1, 0.001, 1);
    g.visible = false;
    scene.add(shadowAll(g));
    return g;
  });

  // Textures for fire, smoke and glowing cracks.
  function glowTexture(inner, mid) {
    return canvasTexture(128, 128, (g) => {
      const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      grad.addColorStop(0, inner);
      grad.addColorStop(0.35, mid);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 128, 128);
    });
  }
  const fireTex = glowTexture("rgba(255,244,180,1)", "rgba(255,120,40,0.85)");
  const ovenTex = glowTexture("rgba(255,255,255,1)", "rgba(255,220,90,0.8)");
  const flameTex = glowTexture("rgba(255,250,215,1)", "rgba(255,140,40,0.9)");
  const sootTex = glowTexture("rgba(52,48,44,0.95)", "rgba(40,37,34,0.55)");
  const cracksTex = canvasTexture(256, 256, (g) => {
    g.lineCap = "round"; g.lineJoin = "round";
    g.shadowColor = "rgba(255,80,10,1)"; g.shadowBlur = 10;
    for (let k = 0; k < 16; k++) {
      let x = 128 + (rand() - 0.5) * 40, y = 128 + (rand() - 0.5) * 40, a = rand() * Math.PI * 2;
      g.strokeStyle = `rgba(255,${120 + Math.floor(rand() * 90)},40,0.95)`;
      g.lineWidth = 1.5 + rand() * 2.5;
      g.beginPath(); g.moveTo(x, y);
      const len = 40 + rand() * 80;
      for (let s = 0; s < len; s += 7) { a += (rand() - 0.5) * 0.9; x += Math.cos(a) * 7; y += Math.sin(a) * 7; g.lineTo(x, y); }
      g.stroke();
    }
    g.globalCompositeOperation = "destination-in";
    const fade = g.createRadialGradient(128, 128, 20, 128, 128, 128);
    fade.addColorStop(0, "rgba(0,0,0,1)"); fade.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = fade; g.fillRect(0, 0, 256, 256);
  });

  // A flame is a handful of additive sprites that rise and fade on a loop.
  function makeFlames(n, size, spread) {
    const g = new T.Group();
    for (let i = 0; i < n; i++) {
      const s = new T.Sprite(new T.SpriteMaterial({ map: flameTex, transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending }));
      s.userData = { t: i / n, dx: (rand() - 0.5) * spread, dz: (rand() - 0.5) * spread, speed: 1.3 + rand() * 0.8 };
      g.add(s);
    }
    g.userData.size = size;
    g.visible = false;
    return g;
  }
  function animateFlames(g, dt, level) {
    g.visible = level > 0.01;
    if (!g.visible) return;
    const size = g.userData.size;
    for (const s of g.children) {
      const u = s.userData;
      u.t = (u.t + dt * u.speed) % 1;
      s.position.set(u.dx * (1 - u.t * 0.6), u.t * 0.9 * size, u.dz * (1 - u.t * 0.6));
      const k = size * Math.min(1.2, level) * (0.45 + 0.6 * Math.sin(u.t * Math.PI));
      s.scale.set(k * 0.75, k * 1.15, 1);
      s.material.opacity = (1 - u.t) * Math.min(1, level);
      s.material.color.setRGB(1, 0.85 - u.t * 0.5, 0.55 - u.t * 0.5);
    }
  }

  // The fire itself: glow, cracks, vent flames, light, spread cone.
  const glowMat = new T.MeshBasicMaterial({ map: fireTex, transparent: true, depthWrite: false, depthTest: false, blending: T.AdditiveBlending });
  const glow = new T.Mesh(new T.PlaneGeometry(1, 1), glowMat);
  glow.rotation.x = -Math.PI / 2;
  glow.renderOrder = 3;
  scene.add(glow);
  const cracks = new T.Mesh(new T.PlaneGeometry(1, 1), new T.MeshBasicMaterial({ map: cracksTex, transparent: true, depthWrite: false, depthTest: false, blending: T.AdditiveBlending }));
  cracks.rotation.x = -Math.PI / 2;
  cracks.renderOrder = 4;
  scene.add(cracks);
  const ventFlames = makeFlames(9, 1.25, 1.4);
  scene.add(ventFlames);
  const fireLight = new T.PointLight(0xff7a2a, 2, 9, 2);
  scene.add(fireLight);

  const coneMat = new T.MeshBasicMaterial({ color: 0xd23f1d, transparent: true, opacity: 0.2, depthWrite: false, depthTest: false });
  const cone = new T.Mesh(new T.CircleGeometry(1, 64), coneMat);
  cone.rotation.x = -Math.PI / 2;
  scene.add(cone);
  const ringPts = [];
  for (let i = 0; i <= 64; i++) { const a = (i / 64) * Math.PI * 2; ringPts.push(new T.Vector3(Math.cos(a), 0, Math.sin(a))); }
  cone.renderOrder = 2;
  const coneEdge = new T.Line(new T.BufferGeometry().setFromPoints(ringPts), new T.LineDashedMaterial({ color: 0xff8a65, dashSize: 0.08, gapSize: 0.06, scale: 1, depthTest: false, transparent: true }));
  coneEdge.computeLineDistances();
  coneEdge.renderOrder = 2;
  scene.add(coneEdge);

  const distLine = new T.Line(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3()]), new T.LineDashedMaterial({ color: 0xffffff, dashSize: 0.25, gapSize: 0.18 }));
  scene.add(distLine);

  // Fire breaking through the road: glowing cracks and flames across the threatened section.
  const roadCracks = new T.Mesh(new T.PlaneGeometry(SPAN * 2.2, 1.6), new T.MeshBasicMaterial({ map: cracksTex, transparent: true, opacity: 0, depthWrite: false, blending: T.AdditiveBlending }));
  roadCracks.rotation.x = -Math.PI / 2;
  roadCracks.position.set(0, 0.115, ROAD_Z);
  roadCracks.renderOrder = 4;
  scene.add(roadCracks);
  const roadVents = [-2.6, -1.3, -0.2, 0.9, 2.2].map((x) => {
    const f = makeFlames(6, 0.9, 0.5);
    f.position.set(x, 0.1, ROAD_Z + (rand() - 0.5) * 0.7);
    f.userData.phase = rand() * 6;
    scene.add(f);
    return f;
  });

  const ovens = new T.Group();
  const ovenBody = new T.Mesh(new T.BoxGeometry(3.4, 1.0, 1.2), new T.MeshLambertMaterial({ color: 0x5a5d61 }));
  ovenBody.position.y = 0.5;
  ovens.add(ovenBody);
  for (let i = 0; i < 8; i++) {
    const slot = new T.Mesh(new T.BoxGeometry(0.16, 1.02, 1.22), new T.MeshLambertMaterial({ color: 0x3e4144 }));
    slot.position.set(-1.45 + i * 0.41, 0.5, 0);
    ovens.add(slot);
  }
  const stack = new T.Mesh(new T.CylinderGeometry(0.16, 0.2, 2.4, 12), new T.MeshLambertMaterial({ color: 0x3a3c3e }));
  stack.position.set(1.4, 1.2, -0.4);
  ovens.add(stack);
  ovens.visible = false;
  scene.add(shadowAll(ovens));

  const smoke = [];
  const smokeTex = glowTexture("rgba(210,210,210,0.9)", "rgba(190,190,190,0.45)");
  for (let i = 0; i < 14; i++) {
    const m = new T.Sprite(new T.SpriteMaterial({ map: smokeTex, color: 0xc7c4bf, transparent: true, opacity: 0.35, depthWrite: false }));
    m.userData = { t: i / 14, dx: (Math.random() - 0.5) * 0.8, dz: (Math.random() - 0.5) * 0.8 };
    scene.add(m);
    smoke.push(m);
  }

  // Cloud over the fire (the cloud slider): soft puffs built from sprites.
  const puffTex = canvasTexture(256, 256, (g) => {
    for (let k = 0; k < 16; k++) {
      const x = 60 + rand() * 136, y = 80 + rand() * 96, r = 34 + rand() * 46;
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, "rgba(255,255,255,0.55)");
      grad.addColorStop(0.6, "rgba(245,246,248,0.25)");
      grad.addColorStop(1, "rgba(240,242,246,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 256, 256);
    }
  });
  const clouds = [];
  const cloudMat = new T.SpriteMaterial({ map: puffTex, transparent: true, opacity: 0, depthWrite: false });
  for (let k = 0; k < 6; k++) {
    const g = new T.Group();
    for (let j = 0; j < 7; j++) {
      const sp = new T.Sprite(cloudMat);
      const sc = 2.2 + rand() * 1.4;
      sp.scale.set(sc * 1.3, sc, 1);
      sp.position.set(j * 0.7 - 2.1, rand() * 0.5, (rand() - 0.5) * 1.4);
      g.add(sp);
    }
    g.userData = { ox: (k % 3) * 3.4 - 3.4, oz: Math.floor(k / 3) * 3 - 1.5, phase: k };
    scene.add(g);
    clouds.push(g);
  }

  // Scenery: rocks, grass, coal heaps and a small village.
  const blocked = (x, z, pad) => Math.abs(z - ROAD_Z) < 1.5 + pad || pitDist(x, z) < pit.r + 1.4 + pad || dumpDist(x, z) < 4.2 + pad || Math.hypot(x - village.x, z - village.z) < 3 + pad;
  const rockGeo = new T.DodecahedronGeometry(1, 0);
  const rockMats = [0x6f6a62, 0x5b564f, 0x7c7266].map((c) => new T.MeshPhongMaterial({ color: c, map: groundTex, bumpMap: groundTex, bumpScale: 0.08, flatShading: true, shininess: 4 }));
  for (let i = 0; i < 60; i++) {
    const x = (rand() - 0.5) * (W - 4), z = (rand() - 0.5) * (D - 4);
    if (blocked(x, z, 0)) continue;
    const r = new T.Mesh(rockGeo, pick(rockMats));
    const s = 0.06 + rand() * 0.2;
    r.scale.set(s * (1 + rand()), s * 0.7, s * (1 + rand() * 0.5));
    r.rotation.set(rand() * 3, rand() * 3, rand() * 3);
    r.position.set(x, height(x, z) + s * 0.2, z);
    scene.add(shadowAll(r));
  }
  // Grass: clumps of curved blades, green where it is wet and straw-coloured where it is dry. Clumps near the fire char.
  const bladeTex = canvasTexture(128, 128, (g) => {
    for (let k = 0; k < 48; k++) {
      const x0 = 10 + rand() * 108, hgt = 48 + rand() * 76, lean = (rand() - 0.5) * 44, w0 = 1.8 + rand() * 2.6;
      const grad = g.createLinearGradient(0, 128, 0, 128 - hgt);
      grad.addColorStop(0, "#56603a");
      grad.addColorStop(1, rand() < 0.4 ? "#e4dcb4" : "#f2f6dc");
      g.fillStyle = grad;
      g.beginPath();
      g.moveTo(x0 - w0, 128);
      g.quadraticCurveTo(x0 + lean * 0.3, 128 - hgt * 0.6, x0 + lean, 128 - hgt);
      g.quadraticCurveTo(x0 + lean * 0.3 + w0 * 0.3, 128 - hgt * 0.55, x0 + w0, 128);
      g.fill();
    }
  });
  function grassGeometry() {
    const positions = [], normals = [], uvs = [], index = [];
    // Each blade card is drawn twice with opposite winding instead of double-sided, so both sides keep an
    // upward normal and are lit like the ground around them.
    [0, Math.PI / 3, (2 * Math.PI) / 3].forEach((a) => {
      const c = Math.cos(a) * 0.22, sn = Math.sin(a) * 0.22;
      for (const flip of [false, true]) {
        const b = positions.length / 3;
        positions.push(-c, 0, -sn, c, 0, sn, c, 0.3, sn, -c, 0.3, -sn);
        normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0);
        uvs.push(0, 0, 1, 0, 1, 1, 0, 1);
        if (flip) index.push(b, b + 2, b + 1, b, b + 3, b + 2);
        else index.push(b, b + 1, b + 2, b, b + 2, b + 3);
      }
    });
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(positions, 3));
    g.setAttribute("normal", new T.Float32BufferAttribute(normals, 3));
    g.setAttribute("uv", new T.Float32BufferAttribute(uvs, 2));
    g.setIndex(index);
    return g;
  }
  const GRASS = 9000;
  const grass = new T.InstancedMesh(grassGeometry(), new T.MeshLambertMaterial({ map: bladeTex, alphaTest: 0.5 }), GRASS);
  grass.receiveShadow = true;
  const grassX = new Float32Array(GRASS), grassZ = new Float32Array(GRASS), grassBase = new Float32Array(GRASS * 3);
  const m4 = new T.Matrix4(), q = new T.Quaternion(), e = new T.Euler(), sv = new T.Vector3(), pv = new T.Vector3();
  const cWet = new T.Color(0x8a9a56), cStraw = new T.Color(0xb5a46e), tint = new T.Color();
  let nGrass = 0;
  for (let i = 0; i < 60000 && nGrass < GRASS; i++) {
    const x = (rand() - 0.5) * 90, z = (rand() - 0.5) * 70;
    const wet = smooth(0.45, 0.8, grassAmt(x, z));
    if (rand() > 0.18 + 0.82 * wet) continue;
    if (blocked(x, z, -0.6) || Math.abs(z - ROAD_Z) < 1.2 || pitDist(x, z) < pit.r + 0.4) continue;
    const sc = 0.65 + rand() * 0.9;
    q.setFromEuler(e.set((rand() - 0.5) * 0.25, rand() * 3.2, (rand() - 0.5) * 0.25));
    m4.compose(pv.set(x, height(x, z) - 0.01, z), q, sv.set(sc, sc * (0.8 + rand() * 0.5), sc));
    grass.setMatrixAt(nGrass, m4);
    tint.copy(cStraw).lerp(cWet, wet * (0.6 + 0.4 * rand())).multiplyScalar(0.85 + rand() * 0.3);
    grass.setColorAt(nGrass, tint);
    grassX[nGrass] = x; grassZ[nGrass] = z;
    grassBase[nGrass * 3] = tint.r; grassBase[nGrass * 3 + 1] = tint.g; grassBase[nGrass * 3 + 2] = tint.b;
    nGrass++;
  }
  grass.count = nGrass;
  scene.add(grass);
  let grassTimer = 0;
  const cAsh = new T.Color(0x1e1a16);
  function charGrass(dt, active, radius) {
    grassTimer -= dt;
    if (grassTimer > 0) return;
    grassTimer = 0.25;
    let changed = false;
    for (let i = 0; i < nGrass; i++) {
      const d = Math.hypot(grassX[i] - fire.x, grassZ[i] - fire.z);
      const k = active ? 1 - smooth(radius * 0.6, radius * 1.25, d) : 0;
      if (k <= 0 && !grass.userData.dirty) continue;
      tint.setRGB(grassBase[i * 3], grassBase[i * 3 + 1], grassBase[i * 3 + 2]).lerp(cAsh, k * 0.9);
      grass.setColorAt(i, tint);
      changed = true;
    }
    grass.userData.dirty = active;
    if (changed) grass.instanceColor.needsUpdate = true;
  }
  for (const [x, z, r] of [[-3.2, -6.2, 0.7], [-2.6, -4.6, 0.5], [-3.8, -2.9, 0.6], [6.2, -4.5, 0.55]]) {
    const heap = new T.Mesh(new T.ConeGeometry(r, r * 0.8, 9), new T.MeshLambertMaterial({ color: 0x1e1e1f, flatShading: true }));
    heap.position.set(x, height(x, z) + r * 0.38, z);
    scene.add(shadowAll(heap));
  }
  for (const [x, z, roof] of [[13, 11, 0x6b5636], [14.7, 10.1, 0x8d8f91], [15.9, 11.9, 0x6b5636], [12.5, 12.7, 0x7a6040], [16.9, 10.3, 0x8d8f91], [14.3, 13.1, 0x6b5636]]) {
    const hut = new T.Group();
    const walls = new T.Mesh(new T.BoxGeometry(0.8, 0.45, 0.65), new T.MeshPhongMaterial({ color: 0xc89c70, map: groundTex, bumpMap: groundTex, bumpScale: 0.04, shininess: 2 }));
    walls.position.y = 0.22;
    const top = new T.Mesh(new T.ConeGeometry(0.66, 0.4, 4), new T.MeshLambertMaterial({ color: roof, flatShading: true }));
    top.position.y = 0.64; top.rotation.y = Math.PI / 4; top.scale.set(1.15, 1, 0.95);
    const door = new T.Mesh(new T.BoxGeometry(0.02, 0.24, 0.16), new T.MeshLambertMaterial({ color: 0x3b2a1c }));
    door.position.set(0.41, 0.12, 0);
    hut.add(walls, top, door);
    hut.position.set(x, height(x, z), z);
    hut.rotation.y = rand() * 0.6 - 0.3 + (x > 14.5 ? Math.PI : 0);
    scene.add(shadowAll(hut));
  }

  // Trees. Lumpy crowns covered in a leaf texture with gaps, so light and shadow fall through them, on barked
  // trunks: broad-leaved sal and mahua, tall eucalyptus, and low bushes. They catch fire inside the burning area
  // and grow back slowly once it moves away.
  const leafTex = canvasTexture(256, 256, (g) => {
    for (let k = 0; k < 1500; k++) {
      const l = 4 + rand() * 7;
      g.fillStyle = `hsl(${80 + rand() * 32},${28 + rand() * 30}%,${30 + rand() * 38}%)`;
      g.beginPath(); g.ellipse(rand() * 256, rand() * 256, l, l * (0.35 + rand() * 0.2), rand() * Math.PI, 0, 7); g.fill();
    }
  });
  leafTex.wrapS = leafTex.wrapT = T.RepeatWrapping;
  leafTex.repeat.set(2, 2);
  const leafDepth = new T.MeshDepthMaterial({ depthPacking: T.RGBADepthPacking, map: leafTex, alphaTest: 0.5 });
  const barkTex = canvasTexture(64, 128, (g) => {
    g.fillStyle = "#8a7a68"; g.fillRect(0, 0, 64, 128);
    for (let k = 0; k < 260; k++) {
      const v = rand() < 0.5 ? 50 + rand() * 40 : 150 + rand() * 60;
      g.fillStyle = `rgba(${v},${v * 0.9},${v * 0.78},0.55)`;
      g.fillRect(rand() * 64, rand() * 128, 1 + rand() * 2, 6 + rand() * 30);
    }
  });
  barkTex.wrapS = barkTex.wrapT = T.RepeatWrapping;
  function crownGeometry(offset) {
    const g = new T.IcosahedronGeometry(0.5, 2);
    const p = g.attributes.position, v = new T.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i).normalize();
      const k = 0.74 + 0.5 * fbm(v.x * 1.7 + offset, v.y * 1.7 + v.z * 1.2 + offset * 2, 3);
      p.setXYZ(i, v.x * 0.5 * k, v.y * 0.5 * k, v.z * 0.5 * k);
    }
    g.computeVertexNormals();
    return g;
  }
  const crownGeos = [0, 11, 23, 37].map(crownGeometry);
  const trunkGeo = new T.CylinderGeometry(0.035, 0.07, 1, 8);
  trunkGeo.translate(0, 0.5, 0);
  const branchGeo = new T.CylinderGeometry(0.012, 0.03, 0.55, 5);
  branchGeo.translate(0, 0.27, 0);
  const leafTints = [0x7f985c, 0x8a9e62, 0x738c52, 0x95a46a, 0x7b9156, 0x8f9a5c];
  const KINDS = {
    broad: { trunk: 0.8, bark: 0x9a8a78, crowns: [[0, 1.0, 0, 1.05], [0.3, 1.12, 0.14, 0.78], [-0.28, 1.1, -0.12, 0.74], [0.04, 1.38, -0.05, 0.7], [-0.1, 0.92, 0.3, 0.6]], squash: 0.82, flames: 1.0 },
    tall: { trunk: 1.55, bark: 0xd6cbb8, crowns: [[0, 1.55, 0, 0.62], [0.08, 1.9, 0.04, 0.52], [-0.07, 1.3, -0.05, 0.55], [0.02, 2.15, 0, 0.38]], squash: 1.35, flames: 1.6 },
    bush: { trunk: 0, bark: 0x6a5a48, crowns: [[0, 0.24, 0, 0.62], [0.28, 0.2, 0.1, 0.46], [-0.22, 0.22, -0.12, 0.5]], squash: 0.75, flames: 0.3 },
  };
  const cScorch = new T.Color(0x8a4b1c), cChar = new T.Color(0x17130f);
  const trees = [];
  function addTree(x, z, s, kindName = "broad") {
    if (blocked(x, z, -0.1) && Math.abs(z - ROAD_Z) < 1.5) return false;
    if (trees.some((t) => Math.hypot(t.x - x, t.z - z) < 0.9)) return false;
    const kind = KINDS[kindName];
    const g = new T.Group();
    const trunkMat = new T.MeshLambertMaterial({ color: kind.bark, map: barkTex });
    const leafMat = new T.MeshLambertMaterial({ color: pick(leafTints), map: leafTex, alphaTest: 0.5, side: T.DoubleSide });
    const coreMat = new T.MeshLambertMaterial({ color: leafMat.color.clone().multiplyScalar(0.42) });
    const crowns = [];
    kind.crowns.forEach(([dx, dy, dz, r], k) => {
      const geo = crownGeos[(k + Math.floor(rand() * 4)) % 4];
      const m = new T.Mesh(geo, leafMat);
      m.customDepthMaterial = leafDepth;
      m.position.set(dx, dy, dz);
      m.rotation.y = rand() * 6.3;
      m.userData.size = new T.Vector3(r, r * kind.squash * (0.9 + rand() * 0.2), r);
      m.scale.copy(m.userData.size);
      g.add(m);
      crowns.push(m);
      const core = new T.Mesh(geo, coreMat);
      core.position.copy(m.position);
      core.userData.size = m.userData.size.clone().multiplyScalar(0.72);
      core.scale.copy(core.userData.size);
      g.add(core);
      crowns.push(core);
    });
    if (kind.trunk) {
      const trunk = new T.Mesh(trunkGeo, trunkMat);
      trunk.scale.set(1, kind.trunk, 1);
      g.add(trunk);
      for (let b = 0; b < 2; b++) {
        const branch = new T.Mesh(branchGeo, trunkMat);
        branch.position.y = kind.trunk * (0.6 + 0.15 * b);
        branch.rotation.set(0, rand() * 6.3, 0.6 + rand() * 0.4);
        g.add(branch);
      }
    }
    const flames = makeFlames(5, 1.1, 0.6);
    flames.position.y = kind.flames;
    g.add(flames);
    g.position.set(x, height(x, z), z);
    g.scale.setScalar(s);
    g.rotation.y = rand() * Math.PI * 2;
    scene.add(shadowAll(g));
    trees.push({ g, x, z, trunkMat, leafMat, coreMat, crowns, base: leafMat.color.clone(), bark: new T.Color(kind.bark), burn: 0, flames });
    return true;
  }
  const kindAt = () => { const r = rand(); return r < 0.62 ? "broad" : r < 0.82 ? "tall" : "bush"; };
  for (let i = 0, n = 0; i < 500 && n < 46; i++) {
    const x = (rand() - 0.5) * 16, z = -10 + rand() * 12.5;
    if (!blocked(x, z, 0) && addTree(x, z, 0.6 + rand() * 0.35, kindAt())) n++;
  }
  for (const x of [-5.4, -2.9, -0.4, 1.9, 4.6, 6.8]) {
    addTree(x + (rand() - 0.5) * 0.6, ROAD_Z - 2.0 - rand() * 0.4, 0.6 + rand() * 0.25, kindAt());
    addTree(x + 1 + (rand() - 0.5) * 0.6, ROAD_Z + 2.0 + rand() * 0.4, 0.6 + rand() * 0.25, kindAt());
  }
  for (let i = 0, n = 0; i < 500 && n < 40; i++) {
    const x = (rand() - 0.5) * (W - 6), z = (rand() - 0.5) * (D - 6);
    if (!blocked(x, z, 0) && addTree(x, z, 0.55 + rand() * 0.45, kindAt())) n++;
  }

  // A belt of trees beyond the coalfield that fades into the haze: instanced, and out of reach of the fire.
  const FAR_TREES = 420;
  const farLeafMat = new T.MeshLambertMaterial({ map: leafTex, alphaTest: 0.5, side: T.DoubleSide });
  const farCrowns = new T.InstancedMesh(crownGeos[1], farLeafMat, FAR_TREES);
  const farTrunks = new T.InstancedMesh(trunkGeo, new T.MeshLambertMaterial({ color: 0x7a6a58, map: barkTex }), FAR_TREES);
  let nFar = 0;
  for (let i = 0; i < 8000 && nFar < FAR_TREES; i++) {
    const x = (rand() - 0.5) * 150, z = (rand() - 0.5) * 120;
    const ring = Math.hypot(x * 0.8, z);
    if (ring < 27 || Math.abs(z - ROAD_Z) < 2.5 || rand() > 0.25 + 0.75 * fbm(x * 0.05 + 3, z * 0.05 + 9, 3)) continue;
    const sc = 0.9 + rand() * 0.9, y = height(x, z);
    q.setFromEuler(e.set(0, rand() * 6.3, 0));
    m4.compose(pv.set(x, y + 1.0 * sc, z), q, sv.set(sc * 1.15, sc * 0.95, sc * 1.15));
    farCrowns.setMatrixAt(nFar, m4);
    tint.set(pick(leafTints)).multiplyScalar(0.75 + rand() * 0.3);
    farCrowns.setColorAt(nFar, tint);
    m4.compose(pv.set(x, y, z), q, sv.set(sc, sc * 0.75, sc));
    farTrunks.setMatrixAt(nFar, m4);
    nFar++;
  }
  farCrowns.count = farTrunks.count = nFar;
  scene.add(farCrowns, farTrunks);

  // Roadside place-name board, in the green and white of Indian highway signs.
  const SIGN = { x: -6.4, z: ROAD_Z - 2.4 };
  const signTex = canvasTexture(1024, 440, (g, w, h) => {
    g.fillStyle = "#0f6b3a"; g.fillRect(0, 0, w, h);
    g.strokeStyle = "#ffffff"; g.lineWidth = 14;
    g.beginPath(); g.roundRect ? g.roundRect(22, 22, w - 44, h - 44, 26) : g.rect(22, 22, w - 44, h - 44); g.stroke();
    g.fillStyle = "#ffffff"; g.textAlign = "center"; g.textBaseline = "middle";
    g.font = "700 128px 'Kohinoor Devanagari', 'Noto Sans Devanagari', 'Mangal', sans-serif";
    g.fillText("झरिया", w / 2, 132);
    g.font = "800 132px 'Helvetica Neue', Arial, sans-serif";
    g.fillText("JHARIA", w / 2, 272);
    g.font = "600 46px 'Helvetica Neue', Arial, sans-serif";
    g.fillText("COALFIELD  ·  DHANBAD, JHARKHAND", w / 2, 370);
  });
  const signFace = new T.MeshLambertMaterial({ map: signTex, emissive: 0xffffff, emissiveMap: signTex, emissiveIntensity: 0.35 });
  const signEdge = new T.MeshLambertMaterial({ color: 0x0b4f2b });
  const sign = new T.Group();
  const board = new T.Mesh(new T.BoxGeometry(3.2, 1.375, 0.08), [signEdge, signEdge, signEdge, signEdge, signFace, signFace]);
  board.position.y = 1.75;
  sign.add(board);
  const postMat = new T.MeshLambertMaterial({ color: 0x9aa0a4 });
  for (const dx of [-1.15, 1.15]) {
    const post = new T.Mesh(new T.CylinderGeometry(0.06, 0.07, 2.45, 10), postMat);
    post.position.set(dx, 1.2, -0.06);
    sign.add(post);
    const foot = new T.Mesh(new T.BoxGeometry(0.28, 0.12, 0.28), new T.MeshLambertMaterial({ color: 0xc9c3b6 }));
    foot.position.set(dx, 0.04, -0.06);
    sign.add(foot);
  }
  sign.position.set(SIGN.x, height(SIGN.x, SIGN.z), SIGN.z);
  sign.rotation.y = -0.2;
  scene.add(shadowAll(sign));
  for (let i = trees.length - 1; i >= 0; i--) {
    if (Math.abs(trees[i].x - SIGN.x) < 2.4 && Math.abs(trees[i].z - SIGN.z) < 1.6) { scene.remove(trees[i].g); trees.splice(i, 1); }
  }

  // People and vehicles.
  const cBurnt = new T.Color(0x14110e);
  function tracked(list) {
    return (c, extra) => {
      const m = new T.MeshLambertMaterial({ color: c, transparent: true, ...extra });
      m.userData.base = new T.Color(c);
      list.push(m);
      return m;
    };
  }
  function person(load) {
    const g = new T.Group(), mats = [], mat = tracked(mats);
    const shirt = mat(pick([0xc0392b, 0xd35400, 0x8e44ad, 0x2e86c1, 0xe5b100, 0x16a085, 0xd6336c, 0xe8e2d4]));
    const skin = mat(pick([0x6b4a35, 0x7a5236, 0x5a3d2b, 0x8a5c3c]));
    const legs = mat(rand() < 0.5 ? shirt.userData.base.getHex() : 0x3b3a36);
    const limb = (m, w, h) => { const geo = new T.BoxGeometry(w, h, w); geo.translate(0, -h / 2, 0); return new T.Mesh(geo, m); };
    const hips = [-0.045, 0.045].map((dz) => { const p = new T.Group(); p.position.set(0, 0.24, dz); p.add(limb(legs, 0.065, 0.24)); g.add(p); return p; });
    const torso = new T.Mesh(new T.BoxGeometry(0.11, 0.2, 0.15), shirt);
    torso.position.y = 0.34;
    const head = new T.Mesh(new T.SphereGeometry(0.058, 10, 8), skin);
    head.position.y = 0.5;
    g.add(torso, head);
    const arms = [-0.1, 0.1].map((dz) => { const p = new T.Group(); p.position.set(0, 0.43, dz); p.add(limb(skin, 0.045, 0.19)); g.add(p); return p; });
    if (load) {
      const basket = new T.Mesh(new T.CylinderGeometry(0.1, 0.065, 0.07, 10), mat(0x8a6a3e));
      basket.position.y = 0.58;
      const coal = new T.Mesh(new T.SphereGeometry(0.085, 8, 5, 0, Math.PI * 2, 0, Math.PI / 2), mat(0x1a1a1a));
      coal.position.y = 0.61;
      g.add(basket, coal);
    }
    g.scale.setScalar(1.25);
    return { g: shadowAll(g), mats, hips, arms };
  }
  function vehicle(kind, color) {
    const g = new T.Group(), mats = [], mat = tracked(mats);
    const dark = mat(0x1d2124);
    const wheel = new T.CylinderGeometry(0.075, 0.075, 0.06, 12);
    wheel.rotateX(Math.PI / 2);
    if (kind === "truck") {
      const body = new T.Mesh(new T.BoxGeometry(0.9, 0.38, 0.5), mat(color));
      body.position.set(-0.15, 0.32, 0);
      const load = new T.Mesh(new T.BoxGeometry(0.75, 0.12, 0.4), mat(0x1b1d1f));
      load.position.set(-0.15, 0.55, 0);
      const cab = new T.Mesh(new T.BoxGeometry(0.34, 0.42, 0.48), mat(0x2b3134));
      cab.position.set(0.47, 0.33, 0);
      g.add(body, load, cab);
      for (const [x, z] of [[-0.45, 0.24], [-0.45, -0.24], [0.1, 0.24], [0.1, -0.24], [0.5, 0.24], [0.5, -0.24]]) { const w = new T.Mesh(wheel, dark); w.position.set(x, 0.08, z); g.add(w); }
    } else {
      const body = new T.Mesh(new T.BoxGeometry(0.64, 0.16, 0.32), mat(color));
      body.position.y = 0.17;
      const cabin = new T.Mesh(new T.BoxGeometry(0.34, 0.14, 0.28), mat(0x2a3440));
      cabin.position.set(-0.04, 0.32, 0);
      const roof = new T.Mesh(new T.BoxGeometry(0.3, 0.025, 0.285), mat(color));
      roof.position.set(-0.04, 0.4, 0);
      g.add(body, cabin, roof);
      for (const [x, z] of [[-0.2, 0.16], [-0.2, -0.16], [0.2, 0.16], [0.2, -0.16]]) { const w = new T.Mesh(wheel, dark); w.position.set(x, 0.08, z); g.add(w); }
    }
    return { g: shadowAll(g), mats };
  }

  const movers = [];
  function addMover(kind, x, dir, lane, speed, opts = {}) {
    const body = kind === "walker" ? person(opts.load) : vehicle(kind, opts.color);
    const flames = makeFlames(kind === "walker" ? 6 : 7, kind === "walker" ? 0.95 : 1.1, kind === "walker" ? 0.2 : 0.6);
    scene.add(body.g, flames);
    const m = { kind, x, dir, lane, speed, gap: kind === "walker" ? 0.35 : kind === "truck" ? 1.5 : 1.1, state: "ok", timer: 0, fall: 0, phase: rand() * 6, flames, flameLevel: 0, ...body };
    m.g.rotation.y = dir > 0 ? 0 : Math.PI;
    movers.push(m);
    return m;
  }
  addMover("truck", -14, 1, ROAD_Z + 0.3, 2.2, { color: 0xe2b007 });
  addMover("car", -9.5, 1, ROAD_Z + 0.3, 2.2, { color: 0xb8c4cc });
  addMover("truck", 4, 1, ROAD_Z + 0.3, 2.2, { color: 0xe2b007 });
  addMover("car", 10, 1, ROAD_Z + 0.3, 2.2, { color: 0x2e6fb0 });
  addMover("truck", 12, -1, ROAD_Z - 0.3, 2.0, { color: 0xd9a400 });
  addMover("car", 7, -1, ROAD_Z - 0.3, 2.0, { color: 0xc23b2a });
  addMover("truck", -6, -1, ROAD_Z - 0.3, 2.0, { color: 0xd9a400 });
  addMover("car", -16, -1, ROAD_Z - 0.3, 2.0, { color: 0xf1f1ee });
  for (const [x, load] of [[-17, true], [-11, false], [-6.5, true], [-4.6, false], [-2.2, true], [8, false], [14, true]]) addMover("walker", x, 1, ROAD_Z + 0.66, 0.42 + rand() * 0.1, { load });
  for (const [x, load] of [[16, false], [9.5, true], [6, false], [4.2, true], [1.5, false], [-7, true], [-13.5, false]]) addMover("walker", x, -1, ROAD_Z - 0.66, 0.4 + rand() * 0.1, { load });

  // Villagers and coal carriers walking paths away from the road.
  const strollers = [
    { path: [[13.4, 10.6], [12.4, 8.6], [10.8, 7.0]], load: false },
    { path: [[15.6, 11.4], [16.6, 9.0], [16.2, 7.1]], load: false },
    { path: [[14.4, 12.2], [11.6, 13.4], [8.4, 12.6]], load: true },
    { path: [[-4.8, -9.4], [0.5, -11.6], [6.8, -9.6]], load: true },
    { path: [[-5.2, -8.6], [0.2, -10.8], [7.2, -8.8]], load: true },
    { path: [[6.8, -9.6], [0.5, -11.6], [-4.8, -9.4]], load: false },
  ].map((s) => {
    const p = person(s.load);
    scene.add(p.g);
    const segs = [];
    let total = 0;
    for (let i = 1; i < s.path.length; i++) { const l = Math.hypot(s.path[i][0] - s.path[i - 1][0], s.path[i][1] - s.path[i - 1][1]); segs.push(l); total += l; }
    return { ...p, path: s.path, segs, total, d: rand() * total, dir: 1, speed: 0.32 + rand() * 0.12, phase: rand() * 6 };
  });
  function walkPose(p, phase, amount) {
    const s = Math.sin(phase) * amount;
    p.hips[0].rotation.z = s; p.hips[1].rotation.z = -s;
    p.arms[0].rotation.z = -s * 0.9; p.arms[1].rotation.z = s * 0.9;
  }

  // Explosions: flash, fireball, flying debris and soot.
  const blasts = [];
  let shake = 0;
  // Lights stay in the scene: adding or removing one would recompile every material mid-explosion.
  const blastLights = [0, 1].map(() => { const l = new T.PointLight(0xffa040, 0, 11, 2); scene.add(l); return l; });
  let nextLight = 0;
  const debrisGeo = new T.BoxGeometry(0.08, 0.05, 0.1);
  function blast(x, y, z) {
    const g = new T.Group();
    g.position.set(x, y, z);
    const sprite = (map, blending) => new T.Sprite(new T.SpriteMaterial({ map, transparent: true, depthWrite: false, blending }));
    const fireballs = [], debris = [], soot = [];
    for (let i = 0; i < 18; i++) {
      const s = sprite(flameTex, T.AdditiveBlending);
      const a = rand() * Math.PI * 2, up = 0.4 + rand() * 0.9, v = 1.6 + rand() * 2.6;
      fireballs.push({ s, v: new T.Vector3(Math.cos(a) * v * (1 - up * 0.5), up * v, Math.sin(a) * v * (1 - up * 0.5)) });
      g.add(s);
    }
    for (let i = 0; i < 12; i++) {
      const d = new T.Mesh(debrisGeo, new T.MeshLambertMaterial({ color: i % 3 ? 0x2a2a2a : 0x7a5a20 }));
      const a = rand() * Math.PI * 2;
      debris.push({ d, v: new T.Vector3(Math.cos(a) * (1 + rand() * 2.5), 3 + rand() * 3.5, Math.sin(a) * (1 + rand() * 2.5)), spin: new T.Vector3(rand() * 9, rand() * 9, rand() * 9), rest: false });
      g.add(d);
    }
    for (let i = 0; i < 10; i++) {
      const s = sprite(sootTex, T.NormalBlending);
      s.userData = { dx: (rand() - 0.5) * 0.9, dz: (rand() - 0.5) * 0.9, delay: rand() * 0.4 };
      soot.push(s);
      g.add(s);
    }
    const flash = sprite(fireTex, T.AdditiveBlending);
    g.add(flash);
    const light = blastLights[nextLight++ % blastLights.length];
    light.position.set(x, y + 0.6, z);
    scene.add(g);
    blasts.push({ g, fireballs, debris, soot, flash, light, t: 0 });
    shake = 0.35;
  }
  function animateBlasts(dt) {
    for (let i = blasts.length - 1; i >= 0; i--) {
      const b = blasts[i];
      b.t += dt;
      const t = b.t;
      b.flash.scale.setScalar(1 + t * 9);
      b.flash.material.opacity = Math.max(0, 1 - t / 0.35);
      b.light.intensity = 5 * Math.max(0, 1 - t / 0.7);
      for (const f of b.fireballs) {
        f.s.position.addScaledVector(f.v, dt);
        f.v.multiplyScalar(Math.max(0, 1 - dt * 3));
        f.v.y += dt * 0.8;
        f.s.scale.setScalar(0.5 + t * 1.6);
        f.s.material.opacity = Math.max(0, 1 - t / 1.1);
        f.s.material.color.setRGB(1, Math.max(0.25, 0.95 - t), Math.max(0.1, 0.7 - t * 1.2));
      }
      for (const d of b.debris) {
        if (d.rest) continue;
        d.v.y -= 9.8 * dt;
        d.d.position.addScaledVector(d.v, dt);
        d.d.rotation.x += d.spin.x * dt; d.d.rotation.y += d.spin.y * dt; d.d.rotation.z += d.spin.z * dt;
        const floor = height(b.g.position.x + d.d.position.x, b.g.position.z + d.d.position.z) - b.g.position.y + 0.03;
        if (d.d.position.y < floor) { d.d.position.y = floor; d.rest = true; }
      }
      for (const s of b.soot) {
        const k = Math.max(0, t - s.userData.delay);
        s.position.set(s.userData.dx * (1 + k), 0.5 + k * 1.1, s.userData.dz * (1 + k));
        s.scale.setScalar(0.6 + k * 1.4);
        s.material.opacity = Math.min(0.75, k * 2) * Math.max(0, 1 - k / 3.2);
      }
      if (t > 4.5) {
        scene.remove(b.g);
        b.g.traverse((o) => { if (o.material) o.material.dispose(); });
        blasts.splice(i, 1);
      }
    }
  }

  const fire = { x: 0, z: 0, heat: 14, reach: 0, closed: 0, review: 0, cloud: 0, factory: 0, hazard: 0 };
  const fireZ = (dist) => ROAD_Z - 0.9 - dist / M_PER_UNIT;
  const fireRadius = () => 0.9 + fire.heat * 0.1;

  const el = (id) => document.getElementById(id);
  const out = { heat: el("o-heat"), dist: el("o-dist"), speed: el("o-speed"), cloud: el("o-cloud") };
  const sliders = { heat: el("s-heat"), dist: el("s-dist"), speed: el("s-speed"), cloud: el("s-cloud") };
  const toggles = { seen: el("t-seen"), factory: el("t-factory") };
  const banner = el("sim-banner");
  const alertBox = el("sim-alert");
  const icons = { close: "✕", ask: "?", watch: "●", exclude: "⊘", hidden: "☁" };
  const titles = { close: "Road closed", ask: "Asks a person", watch: "Keeps watching", exclude: "Ignored: factory heat", hidden: "Can't see it" };
  let lastDecision = null;
  // Barriers go up the moment the road is closed. Whoever is already between them is trapped on the burning section.
  const danger = { on: false, barriers: false, people: 0, vehicles: 0 };

  // Sound, synthesised with Web Audio. Off until the viewer turns it on.
  const sound = (() => {
    let ctx = null, master = null, noiseBuf = null, fireBed = null, windBed = null, on = false;
    function loop(filterType, freq, q, gain) {
      const src = ctx.createBufferSource();
      src.buffer = noiseBuf; src.loop = true;
      const f = ctx.createBiquadFilter();
      f.type = filterType; f.frequency.value = freq; f.Q.value = q;
      const g = ctx.createGain();
      g.gain.value = gain;
      src.connect(f).connect(g).connect(master);
      src.start();
      return g;
    }
    function init() {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      ctx = new AC();
      master = ctx.createGain();
      master.gain.value = 0.9;
      master.connect(ctx.destination);
      noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
      const d = noiseBuf.getChannelData(0);
      let b = 0;
      for (let i = 0; i < d.length; i++) { const w = Math.random() * 2 - 1; b = (b + 0.02 * w) / 1.02; d[i] = i % 2 ? w : b * 3.5; }
      windBed = loop("lowpass", 380, 0.6, 0.06);
      fireBed = loop("bandpass", 900, 0.6, 0);
      return true;
    }
    function burst(type, f0, f1, dur, peak, delay = 0) {
      const t0 = ctx.currentTime + delay;
      const src = ctx.createBufferSource();
      src.buffer = noiseBuf;
      const f = ctx.createBiquadFilter();
      f.type = type;
      f.frequency.setValueAtTime(f0, t0);
      f.frequency.exponentialRampToValueAtTime(f1, t0 + dur);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(peak, t0 + Math.min(0.03, dur / 4));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      src.connect(f).connect(g).connect(master);
      src.start(t0, Math.random() * 1.5);
      src.stop(t0 + dur + 0.05);
    }
    function tone(type, f0, f1, dur, peak, delay = 0) {
      const t0 = ctx.currentTime + delay;
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.setValueAtTime(f0, t0);
      o.frequency.exponentialRampToValueAtTime(f1, t0 + dur);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(peak, t0 + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      o.connect(g).connect(master);
      o.start(t0);
      o.stop(t0 + dur + 0.05);
    }
    const live = () => on && ctx;
    return {
      get on() { return on; },
      toggle() {
        if (!ctx && !init()) return false;
        on = !on;
        if (on) ctx.resume(); else ctx.suspend();
        return on;
      },
      ambience(fireLevel, dt) {
        if (!live()) return;
        fireBed.gain.setTargetAtTime(0.03 + fireLevel * 0.14, ctx.currentTime, 0.3);
        if (Math.random() < dt * (3 + fireLevel * 30)) burst("highpass", 2500 + Math.random() * 3000, 1800, 0.03 + Math.random() * 0.05, 0.05 + fireLevel * 0.25);
      },
      boom() {
        if (!live()) return;
        burst("lowpass", 2400, 60, 1.8, 1.0);
        burst("bandpass", 900, 200, 0.5, 0.6);
        tone("sine", 90, 28, 1.0, 0.9);
        for (let i = 0; i < 6; i++) burst("highpass", 3000, 1500, 0.06, 0.25, 0.3 + Math.random() * 1.2);
      },
      whoosh() {
        if (!live()) return;
        burst("bandpass", 250, 2600, 0.7, 0.5);
        for (let i = 0; i < 5; i++) burst("highpass", 2800, 1600, 0.05, 0.2, 0.2 + Math.random() * 0.8);
      },
      // A scream: a buzzy voice source with shaky pitch, shaped into an "aaah" by three vowel formants.
      scream(delay = 0, strength = 1) {
        if (!live()) return;
        const t0 = ctx.currentTime + delay;
        const high = Math.random() < 0.5;
        const base = (high ? 620 : 380) * (0.9 + Math.random() * 0.25);
        const dur = (1.1 + Math.random() * 0.6) * (0.6 + 0.4 * strength);
        const voice = ctx.createOscillator();
        voice.type = "sawtooth";
        voice.frequency.setValueAtTime(base * 0.75, t0);
        voice.frequency.linearRampToValueAtTime(base * 1.3, t0 + 0.14);
        voice.frequency.linearRampToValueAtTime(base * 1.12, t0 + dur * 0.65);
        voice.frequency.exponentialRampToValueAtTime(base * 0.55, t0 + dur);
        for (const [rate, depth] of [[6 + Math.random() * 2, 0.045], [19 + Math.random() * 8, 0.02]]) {
          const lfo = ctx.createOscillator();
          lfo.frequency.value = rate;
          const amt = ctx.createGain();
          amt.gain.value = base * depth;
          lfo.connect(amt).connect(voice.frequency);
          lfo.start(t0); lfo.stop(t0 + dur + 0.05);
        }
        const rasp = ctx.createWaveShaper();
        const curve = new Float32Array(256);
        for (let i = 0; i < 256; i++) { const x = (i / 128) - 1; curve[i] = Math.tanh(x * 3); }
        rasp.curve = curve;
        const env = ctx.createGain();
        env.gain.setValueAtTime(0.0001, t0);
        env.gain.exponentialRampToValueAtTime(0.32 * strength, t0 + 0.06);
        env.gain.setValueAtTime(0.32 * strength, t0 + dur * 0.6);
        env.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
        const shift = high ? 1.18 : 1;
        voice.connect(rasp);
        for (const [f, q, g] of [[800, 7, 1], [1200, 9, 0.6], [2700, 10, 0.28]]) {
          const bp = ctx.createBiquadFilter();
          bp.type = "bandpass"; bp.frequency.value = f * shift; bp.Q.value = q;
          const fg = ctx.createGain();
          fg.gain.value = g * 2.2;
          rasp.connect(bp).connect(fg).connect(env);
        }
        env.connect(master);
        voice.start(t0); voice.stop(t0 + dur + 0.05);
        burst("bandpass", 2600, 1800, dur * 0.9, 0.06 * strength, delay);
      },
      alarm() {
        if (!live()) return;
        for (let i = 0; i < 4; i++) { tone("sawtooth", 660, 990, 0.32, 0.08, i * 0.36); }
      },
    };
  })();

  const soundBtn = el("sim-sound");
  if (soundBtn) {
    soundBtn.addEventListener("click", () => {
      const on = sound.toggle();
      soundBtn.setAttribute("aria-pressed", String(on));
      soundBtn.setAttribute("aria-label", on ? "Mute sound" : "Turn sound on");
      soundBtn.classList.toggle("on", on);
      soundBtn.querySelector("span").textContent = on ? "Sound on" : "Sound off";
    });
  }

  function showAlert() {
    if (!alertBox) return;
    alertBox.hidden = !danger.on;
    if (!danger.on) return;
    alertBox.classList.toggle("safe", danger.barriers);
    const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
    const tally = `${plural(danger.people, "person", "people")} burned, ${plural(danger.vehicles, "vehicle", "vehicles")} destroyed`;
    if (danger.barriers) {
      el("sim-alert-title").textContent = "Road blocked over the burning seam";
      el("sim-alert-text").textContent = danger.people + danger.vehicles
        ? `Trapped between the barriers: ${tally}. Everyone else waits safely outside.`
        : "Anyone between the barriers is trapped on the burning section. Everyone else waits safely outside.";
    } else {
      el("sim-alert-title").textContent = "Road open over a burning seam";
      el("sim-alert-text").textContent = danger.people + danger.vehicles ? `${tally} on the open road.` : "The ground under the road is burning. Anyone who drives or walks through is in danger.";
    }
  }

  function setBarriers(up) {
    danger.barriers = up;
    tween(fire, { closed: up ? 1 : 0 });
    for (const b of barriers) {
      if (up) { b.visible = true; tween(b.scale, { y: 1, ease: "back.out(2.5)", duration: 0.7 }); }
      else tween(b.scale, { y: 0.001, duration: 0.35, onComplete: () => { if (!danger.barriers) b.visible = false; } });
    }
  }


  function update(fromPreset) {
    inputs.heat = +sliders.heat.value;
    inputs.dist = +sliders.dist.value;
    inputs.speed = +sliders.speed.value;
    inputs.cloud = +sliders.cloud.value;
    inputs.seen = toggles.seen.checked;
    inputs.factory = toggles.factory.checked;
    out.heat.textContent = `+${inputs.heat} °C`;
    out.dist.textContent = inputs.dist >= 1000 ? "1 km" : `${inputs.dist} m`;
    out.speed.textContent = inputs.speed === 0 ? "not moving" : `${inputs.speed} m a day`;
    out.cloud.textContent = `${inputs.cloud}%`;
    if (!fromPreset) document.querySelectorAll(".presets button").forEach((b) => b.classList.remove("on"));
    const r = assess(inputs);
    el("m-risk").textContent = `${r.score.toFixed(2)} ${r.level}`;
    el("m-conf").textContent = `${Math.round(r.conf * 100)}%`;
    tween(el("b-risk"), { width: `${Math.min(100, r.score * 100)}%` });
    tween(el("b-conf"), { width: `${r.conf * 100}%` });
    el("sim-reach").textContent = inputs.factory ? "Factories don't spread, so no prediction is drawn." : inputs.speed > 0
      ? `In 48 days it could spread about ${Math.round(r.reach)} m${r.projected > 0 ? `, to within ${Math.round(r.projected)} m of the road` : ", reaching the road"} (the red dashed area).`
      : "It isn't moving, so its predicted spread stays around it.";
    el("sim-decision").textContent = titles[r.decision];
    el("sim-why").textContent = r.why;
    el("sim-icon").textContent = icons[r.decision];
    banner.className = `sim-banner ${r.decision === "watch" ? "" : r.decision === "hidden" ? "hidden-fire" : r.decision}`;
    if (lastDecision !== r.decision && window.gsap && lastDecision !== null) window.gsap.fromTo(banner, { scale: 0.96 }, { scale: 1, duration: 0.45, ease: "back.out(3)" });
    lastDecision = r.decision;
    // Danger is physical: a hot fire within the closing distance has reached the road, whatever the satellite sees.
    const nowDanger = !inputs.factory && inputs.heat >= 8 && inputs.dist <= RULES.closeWithin;
    if (nowDanger && !danger.on) { danger.people = 0; danger.vehicles = 0; }
    danger.on = nowDanger;
    const wantClosed = r.decision === "close";
    if (!wantClosed) {
      if (danger.barriers || fire.closed > 0) setBarriers(false);
    } else if (!danger.barriers) {
      setBarriers(true);
      if (nowDanger) sound.alarm();
    }
    showAlert();
    const z = fireZ(inputs.dist);
    tween(fire, { z, heat: inputs.heat, reach: inputs.factory ? 0 : r.reach / M_PER_UNIT, cloud: inputs.cloud / 100, review: r.decision === "ask" ? 1 : 0, factory: inputs.factory ? 1 : 0, hazard: nowDanger ? 1 : 0 });
    el("lab-fire").classList.toggle("factory", inputs.factory);
    el("lab-fire").textContent = inputs.factory ? "Coke ovens" : `Fire +${inputs.heat} °C`;
    el("lab-dist").textContent = inputs.dist >= 1000 ? "1 km to road" : `${inputs.dist} m to road`;
    const sr = el("sim-sr");
    if (sr) {
      const what = inputs.factory ? "Coke ovens" : `An underground fire ${inputs.heat} °C hotter than the ground around it`;
      const move = inputs.factory || inputs.speed === 0 ? "not moving" : `moving ${inputs.speed} m a day toward the road`;
      const sky = inputs.cloud >= 90 ? "hidden under cloud" : inputs.cloud > 0 ? `${inputs.cloud}% under cloud` : "under a clear sky";
      const road = r.decision === "close" ? "Barriers are across the road and the trucks stop." : r.decision === "ask" ? "The road is marked yellow for review and stays open." : "The road is open and trucks keep moving.";
      const hazard = nowDanger ? (r.decision === "close" ? " The ground under the road is burning, and the barriers keep people and vehicles out." : " The ground under the open road is burning, and people and vehicles crossing it are caught by the fire.") : "";
      sr.textContent = `3D view. ${what}, ${inputs.dist} m from a haul road, ${move}, ${sky}. ${road}${hazard} Decision: ${titles[r.decision]}. Risk ${r.score.toFixed(2)}, ${Math.round(r.conf * 100)}% sure.`;
    }
  }

  const presets = {
    quiet: { heat: 6, dist: 800, speed: 0.5, cloud: 0, seen: true, factory: false },
    near: { heat: 20, dist: 80, speed: 3.5, cloud: 0, seen: true, factory: false },
    spread: { heat: 16, dist: 420, speed: 5, cloud: 0, seen: true, factory: false },
    cloud: { heat: 18, dist: 200, speed: 2, cloud: 40, seen: true, factory: false },
    factory: { heat: 30, dist: 600, speed: 0, cloud: 0, seen: true, factory: true },
  };
  function apply(p, b) {
    for (const k of ["heat", "dist", "speed", "cloud"]) sliders[k].value = p[k];
    toggles.seen.checked = p.seen;
    toggles.factory.checked = p.factory;
    document.querySelectorAll(".presets button").forEach((x) => x.classList.toggle("on", x === b));
    update(true);
  }
  document.querySelectorAll(".presets button[data-preset]").forEach((b) => b.addEventListener("click", () => apply(presets[b.dataset.preset], b)));

  const realRow = el("real-examples");
  if (realRow && window.fetch) {
    const short = (iso) => new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
    const verb = { close: "closed a road", escalate: "asked a person", log: "kept watching", exclude: "ignored as factory heat" };
    fetch(window.FW_EXAMPLES || "/api/public/examples").then((r) => (r.ok ? r.json() : [])).then((list) => {
      if (!Array.isArray(list) || !list.length) return;
      for (const ex of list) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = `${ex.name}, ${short(ex.date)}`;
        b.title = `On this photo Fire Watch ${verb[ex.decision] || "logged it"}. Load its numbers.`;
        b.setAttribute("aria-label", `Load ${ex.name} from the ${short(ex.date)} photo, where Fire Watch ${verb[ex.decision] || "logged it"}`);
        b.addEventListener("click", () => apply(ex, b));
        realRow.append(b);
      }
      realRow.hidden = false;
    }).catch(() => {});
  }
  for (const s of Object.values(sliders)) s.addEventListener("input", () => update(false));
  for (const t of Object.values(toggles)) t.addEventListener("change", () => update(false));

  let dragging = false, lastX = 0, lastY = 0, idle = 0;
  stage.addEventListener("pointerdown", (e) => { if (e.target.closest("button")) return; dragging = true; lastX = e.clientX; lastY = e.clientY; stage.setPointerCapture(e.pointerId); });
  stage.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    view.yaw -= (e.clientX - lastX) * 0.006;
    view.pitch = Math.max(0.24, Math.min(1.2, view.pitch + (e.clientY - lastY) * 0.004));
    lastX = e.clientX; lastY = e.clientY; idle = 0;
  });
  stage.addEventListener("pointerup", () => { dragging = false; });
  stage.addEventListener("keydown", (e) => {
    const turns = { ArrowLeft: [0.12, 0], ArrowRight: [-0.12, 0], ArrowUp: [0, -0.06], ArrowDown: [0, 0.06] };
    const t = turns[e.key];
    if (!t) return;
    e.preventDefault();
    view.yaw += t[0];
    view.pitch = Math.max(0.24, Math.min(1.2, view.pitch + t[1]));
    idle = 0;
  });
  stage.addEventListener("pointercancel", () => { dragging = false; });

  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    view.radius = w < 520 ? 30 : 24;
    camera.updateProjectionMatrix();
  }
  if ("ResizeObserver" in window) new ResizeObserver(resize).observe(stage);
  window.addEventListener("resize", resize);
  resize();

  const labels = { fire: el("lab-fire"), dist: el("lab-dist"), road: el("lab-road") };
  const v3 = new T.Vector3();
  function place(label, x, y, z) {
    v3.set(x, y, z).project(camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    label.style.transform = `translate(${((v3.x + 1) / 2) * w}px, ${((1 - v3.y) / 2) * h}px) translate(-50%, -100%)`;
    label.style.opacity = v3.z < 1 ? 1 : 0;
  }
  function placeCenter(label, x, y, z) {
    v3.set(x, y, z).project(camera);
    const w = stage.clientWidth, h = stage.clientHeight;
    label.style.transform = `translate(${((v3.x + 1) / 2) * w}px, ${((1 - v3.y) / 2) * h}px) translate(-50%, -50%)`;
  }

  function charTo(mats, k) { for (const m of mats) m.color.copy(m.userData.base).lerp(cBurnt, k); }
  function fadeTo(mats, a) { for (const m of mats) m.opacity = a; }
  function respawn(m) {
    m.state = "ok"; m.timer = 0; m.fall = 0; m.flameLevel = 0;
    m.x = m.dir > 0 ? -EDGE : EDGE;
    m.g.rotation.set(0, m.dir > 0 ? 0 : Math.PI, 0);
    if (m.arms) m.arms.forEach((a) => { a.rotation.x = 0; });
    charTo(m.mats, 0);
    fadeTo(m.mats, 1);
  }
  function ignite(m) {
    if (m.kind === "walker") { m.state = "burning"; m.timer = 0; danger.people++; sound.whoosh(); sound.scream(0.05); sound.scream(1.3 + Math.random() * 0.4, 0.6); }
    else { m.state = "wreck"; m.timer = 0; danger.vehicles++; blast(m.x, 0.25, m.lane); sound.boom(); charTo(m.mats, 0.85); m.g.rotation.x = (rand() - 0.5) * 0.35; m.g.rotation.z = (rand() - 0.5) * 0.4; }
    m.flameLevel = 1;
    showAlert();
  }

  // Stop everyone at the barriers when the road is closed, queueing behind each other.
  function setLimits() {
    for (const m of movers) m.limit = null;
    if (!danger.barriers) return;
    const lanes = new Map();
    for (const m of movers) {
      if (m.state !== "ok") continue;
      const stopAt = m.dir > 0 ? -SPAN - 0.75 : SPAN + 0.75;
      if (m.dir > 0 ? m.x > stopAt + 0.01 : m.x < stopAt - 0.01) continue;
      const key = `${m.lane}`;
      if (!lanes.has(key)) lanes.set(key, []);
      lanes.get(key).push(m);
    }
    for (const list of lanes.values()) {
      const dir = list[0].dir;
      list.sort((a, b) => dir * (b.x - a.x));
      let limit = dir > 0 ? -SPAN - 0.75 : SPAN + 0.75;
      for (const m of list) { m.limit = limit; limit -= dir * m.gap; }
    }
  }

  function moveRoad(dt, t) {
    setLimits();
    const hot = fire.hazard > 0.6 && !reduced;
    for (const m of movers) {
      const prev = m.x;
      if (m.state === "ok") {
        let next = m.x + m.dir * m.speed * dt * (reduced ? 0 : 1);
        if (m.limit !== null && m.limit !== undefined) {
          if (m.dir > 0 && next > m.limit) next = Math.max(m.x, m.limit);
          if (m.dir < 0 && next < m.limit) next = Math.min(m.x, m.limit);
        }
        if (next > EDGE) next = -EDGE;
        if (next < -EDGE) next = EDGE;
        m.x = next;
        const inside = Math.abs(m.x) < SPAN - 0.2;
        if (hot && inside) {
          const entered = Math.abs(prev) >= SPAN - 0.2;
          const chance = danger.barriers ? dt * 4 : entered ? (m.kind === "walker" ? 0.95 : 0.85) * Math.min(1, 0.4 + fire.heat / 20) : dt * (m.kind === "walker" ? 0.9 : 0.6);
          if (rand() < chance) ignite(m);
        }
        if (m.kind === "walker") {
          const moving = Math.abs(m.x - prev) > 1e-4;
          m.phase += dt * (moving ? 9 : 0);
          walkPose(m, m.phase, moving ? 0.55 : 0);
        }
      } else if (m.state === "burning") {
        m.timer += dt;
        m.x += m.dir * m.speed * 2.6 * dt;
        m.phase += dt * 20;
        walkPose(m, m.phase, 0.9);
        m.arms[0].rotation.x = m.arms[1].rotation.x = Math.sin(m.phase * 0.5) * 0.8;
        m.g.rotation.y = (m.dir > 0 ? 0 : Math.PI) + Math.sin(t * 9) * 0.5;
        charTo(m.mats, Math.min(0.9, m.timer / 2));
        if (m.timer > 2.1) { m.state = "down"; m.timer = 0; m.side = rand() < 0.5 ? -1 : 1; }
      } else if (m.state === "down") {
        m.timer += dt;
        m.fall = Math.min(1, m.fall + dt * 3);
        m.g.rotation.x = m.side * m.fall * Math.PI / 2;
        walkPose(m, 0, 0);
        m.flameLevel = Math.max(0.15, 1 - m.timer / 3);
        if (m.timer > 4.5) fadeTo(m.mats, Math.max(0, 1 - (m.timer - 4.5)));
        if (m.timer > 5.5) respawn(m);
      } else if (m.state === "wreck") {
        m.timer += dt;
        m.flameLevel = m.timer < 0.3 ? 1.3 : Math.max(0.25, 1.1 - m.timer / 7);
        if (m.timer > 7) fadeTo(m.mats, Math.max(0, 1 - (m.timer - 7)));
        if (m.timer > 8) respawn(m);
      }
      const lift = m.kind === "walker" && m.state === "down" ? 0.08 : 0.1;
      m.g.position.set(m.x, lift, m.lane);
      if (m.state === "ok") fadeTo(m.mats, 1 - smooth(EDGE - 8, EDGE, Math.abs(m.x)));
      if (m.state === "ok") m.flameLevel = 0;
      m.flames.position.set(m.x, m.kind === "walker" ? 0.12 : 0.35, m.lane);
      animateFlames(m.flames, dt, m.flameLevel);
    }
  }

  function moveStrollers(dt) {
    for (const s of strollers) {
      if (!reduced) s.d += s.dir * s.speed * dt;
      if (s.d > s.total) { s.d = s.total; s.dir = -1; }
      if (s.d < 0) { s.d = 0; s.dir = 1; }
      let d = s.d, i = 0;
      while (i < s.segs.length - 1 && d > s.segs[i]) { d -= s.segs[i]; i++; }
      const [x0, z0] = s.path[i], [x1, z1] = s.path[i + 1];
      const f = Math.min(1, d / s.segs[i]);
      const x = x0 + (x1 - x0) * f, z = z0 + (z1 - z0) * f;
      s.g.position.set(x, height(x, z), z);
      s.g.rotation.y = Math.atan2(-(z1 - z0) * s.dir, (x1 - x0) * s.dir);
      s.phase += dt * (reduced ? 0 : 8);
      walkPose(s, s.phase, reduced ? 0 : 0.5);
    }
  }

  function burnTrees(dt) {
    const active = fire.factory < 0.5 && fire.heat >= 8;
    const R = fireRadius();
    for (const tr of trees) {
      const d = Math.hypot(tr.x - fire.x, tr.z - fire.z);
      const roadside = fire.hazard > 0.6 && Math.abs(tr.z - ROAD_Z) < 2.8 && Math.abs(tr.x) < SPAN + 0.6;
      const inFire = active && (d < R + 0.2 || roadside);
      if (inFire) tr.burn = Math.min(1, tr.burn + dt * (0.18 + Math.max(0, R - d) * 0.08));
      else tr.burn = Math.max(0, tr.burn - dt * 0.045);
      const b = tr.burn;
      if (b < 0.5) tr.leafMat.color.copy(tr.base).lerp(cScorch, b * 2);
      else tr.leafMat.color.copy(cScorch).lerp(cChar, (b - 0.5) * 2);
      tr.coreMat.color.copy(tr.leafMat.color).multiplyScalar(0.42);
      tr.trunkMat.color.copy(tr.bark).lerp(cChar, Math.min(1, b * 1.4));
      const shrink = 1 - 0.75 * smooth(0.45, 1, b);
      for (const c of tr.crowns) c.scale.copy(c.userData.size).multiplyScalar(shrink);
      const level = inFire ? (b < 1 ? 0.35 + 0.75 * Math.sin(Math.min(1, b) * Math.PI) : 0.2) : Math.max(0, b - 0.85) * 2;
      animateFlames(tr.flames, dt, b > 0.04 ? level : 0);
    }
  }

  let visible = true, last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible) return;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = now / 1000;
    idle += dt;
    if (!dragging && idle > 2.5 && !reduced) view.yaw += dt * 0.05;
    camera.position.set(target.x + view.radius * Math.sin(view.yaw) * Math.cos(view.pitch), target.y + view.radius * Math.sin(view.pitch), target.z + view.radius * Math.cos(view.yaw) * Math.cos(view.pitch));
    if (shake > 0.001) {
      camera.position.x += (Math.random() - 0.5) * shake;
      camera.position.y += (Math.random() - 0.5) * shake;
      shake *= Math.max(0, 1 - dt * 5);
    }
    // At low angles, aim a little above the target so the horizon and sky stay in view.
    const lift = 0.19 * smooth(0.9, 0.35, view.pitch);
    const aimDown = view.pitch - lift;
    const flat = view.radius * Math.cos(view.pitch);
    lookAt.set(target.x, camera.position.y - flat * Math.tan(aimDown), target.z);
    camera.lookAt(lookAt);
    sky.position.copy(camera.position);
    skyClouds.position.set(camera.position.x, 120, camera.position.z);
    skyCloudTex.offset.x += dt * 0.0012;
    skyCloudTex.offset.y += dt * 0.0005;

    const gy = height(fire.x, fire.z) + 0.06;
    const pulse = reduced ? 1 : 1 + 0.08 * Math.sin(t * 3);
    const size = (1.4 + fire.heat * 0.11) * pulse;
    glow.position.set(fire.x, gy, fire.z);
    glow.scale.set(size * (1 + fire.factory * 1.2), size, 1);
    glow.material.map = fire.factory > 0.5 ? ovenTex : fireTex;
    glow.material.opacity = Math.min(1, 0.35 + fire.heat / 26) * (1 - fire.cloud * 0.6);
    cracks.position.set(fire.x, gy + 0.02, fire.z);
    cracks.scale.setScalar(fireRadius() * 2.3);
    cracks.material.opacity = (1 - fire.factory) * Math.min(1, 0.25 + fire.heat / 24) * (0.85 + 0.15 * Math.sin(t * 2.3));
    ventFlames.position.set(fire.x, gy, fire.z);
    animateFlames(ventFlames, dt, (1 - fire.factory) * Math.max(0, (fire.heat - 8) / 18));
    fireLight.position.set(fire.x, gy + 0.6, fire.z);
    fireLight.intensity = (0.6 + fire.heat / 10) * pulse * (1 - fire.cloud * 0.5);
    fireLight.color.setHex(fire.factory > 0.5 ? 0xffe39a : 0xff7a2a);
    ovens.visible = fire.factory > 0.5;
    ovens.position.set(fire.x, height(fire.x, fire.z), fire.z);

    const reach = fire.reach;
    const rx = 1.1 + fire.heat * 0.03;
    const rz = Math.max(1, reach / 2 + 0.9);
    const cz = fire.z + reach / 2;
    cone.position.set(fire.x, gy + 0.03, cz);
    cone.scale.set(rx, rz, 1);
    coneEdge.position.set(fire.x, gy + 0.05, cz);
    coneEdge.scale.set(rx, 1, rz);
    cone.visible = coneEdge.visible = fire.factory < 0.5;

    const lp = distLine.geometry.attributes.position;
    lp.setXYZ(0, fire.x + 0.01, gy + 0.15, fire.z);
    lp.setXYZ(1, fire.x + 0.01, 0.2, ROAD_Z - 0.65);
    lp.needsUpdate = true;
    distLine.computeLineDistances();

    span.material.color.setHex(fire.closed > fire.review ? 0xd23f1d : 0xe2b007);
    span.material.opacity = Math.max(fire.closed, fire.review * 0.75) * 0.85 * (1 - fire.hazard * 0.6);
    roadCracks.material.opacity = fire.hazard * (0.8 + 0.2 * Math.sin(t * 3));
    roadVents.forEach((f) => animateFlames(f, dt, fire.hazard * (0.55 + 0.45 * Math.sin(t * 1.7 + f.userData.phase))));

    for (const s of smoke) {
      const u = s.userData;
      u.t = (u.t + dt * (0.12 + fire.heat / 160)) % 1;
      s.position.set(fire.x + u.dx * (1 + u.t), gy + 0.2 + u.t * 3.2, fire.z + u.dz * (1 + u.t));
      s.scale.setScalar(0.6 + u.t * 1.8);
      s.material.opacity = (1 - u.t) * 0.35 * Math.min(1, fire.heat / 14) * (fire.factory > 0.5 ? 1.2 : 1);
      s.visible = !reduced;
    }
    clouds.forEach((g) => {
      const u = g.userData;
      g.position.set(fire.x + u.ox + Math.sin(t * 0.15 + u.phase) * 0.8, 3.6 + (u.phase % 2) * 0.5, fire.z + u.oz);
      g.visible = fire.cloud > 0.02;
      const shown = u.phase / 6 < fire.cloud + 0.05;
      g.scale.setScalar(shown ? 1 : 0.001);
    });
    cloudMat.opacity = Math.min(0.92, fire.cloud * 1.3);

    moveRoad(dt, t);
    sound.ambience((1 - fire.factory) * Math.min(1, fire.heat / 26) * (0.5 + 0.5 * fire.hazard), dt);
    moveStrollers(dt);
    burnTrees(dt);
    const burning = fire.factory < 0.5 && fire.heat >= 8;
    scorchGround(fire.x, fire.z, fireRadius(), burning ? Math.min(0.85, (fire.heat - 6) / 16) * (1 - fire.factory) : 0);
    charGrass(dt, burning, fireRadius());
    animateBlasts(dt);

    renderer.render(scene, camera);
    place(labels.fire, fire.x, gy + 1.9 + fire.factory * 1.2, fire.z);
    placeCenter(labels.dist, fire.x + 0.01, 0.4, (fire.z + ROAD_Z - 0.65) / 2);
    labels.dist.style.opacity = Math.abs(ROAD_Z - fire.z) > 2 ? 1 : 0;
    place(labels.road, -12, 0.6, ROAD_Z);
  }

  if ("IntersectionObserver" in window) new IntersectionObserver((e) => { visible = e[0].isIntersecting; last = performance.now(); }).observe(stage);
  const useRules = (status) => {
    const r = status && status.rules;
    if (!r || !r.close_buffer_m || r.close_buffer_m === RULES.closeWithin) return;
    RULES.closeWithin = r.close_buffer_m;
    update(false);
  };
  window.addEventListener("fw-status", (e) => useRules(e.detail));
  useRules(window.FW_STATUS);
  update(false);
  fire.z = fireZ(inputs.dist); fire.heat = inputs.heat;
  requestAnimationFrame(frame);
})();
