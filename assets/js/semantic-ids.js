/*
 * Home page animation: from products to semantic IDs and back.
 *
 * One cycle tells the generative-recommendation story:
 *   1. the dots form a product (a mug, a bicycle, ...);
 *   2. they dissolve into an embedding space where each dot is one catalog
 *      item and each category forms a lobe;
 *   3. the space is quantized into semantic IDs: three tokens per item
 *      (category, type, variant), so items collapse onto their codes;
 *   4. a user history lights up, linked by a trail;
 *   5. the next item is generated token by token, each token narrowing the
 *      candidates, and the dots fly together into the predicted product.
 * The next cycle starts from that product.
 *
 * Hovering a dot shows the item and its semantic ID. Colors follow the site's
 * CSS custom properties (light and dark). Honors prefers-reduced-motion and
 * pauses when off-screen or when the tab is hidden.
 */

import { CATALOG, buildProducts, sampleProduct, displayCode, productName } from "./semantic-ids-products.js";

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";

const figure = document.querySelector(".home-viz");
if (figure) {
  init(figure).catch((err) => {
    console.warn("semantic-ids animation disabled:", err);
    figure.remove();
  });
}

async function init(figure) {
  const THREE = await import(THREE_URL);
  const canvas = figure.querySelector("canvas");
  const stage = figure.querySelector(".home-viz__stage");
  const tokensEl = figure.querySelector(".home-viz__tokens");
  const tipEl = figure.querySelector(".home-viz__tip");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Colors are passed through untouched (sRGB in, sRGB out).
  THREE.ColorManagement.enabled = false;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 20);
  camera.position.set(0, 0, 4.6);

  const rand = mulberry32(11);
  const gauss = () => {
    const u = Math.max(rand(), 1e-9), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const randUnit = () => {
    const z = rand() * 2 - 1, a = rand() * Math.PI * 2, s = Math.sqrt(1 - z * z);
    return [s * Math.cos(a), z, s * Math.sin(a)];
  };

  // ---------- Catalog: every dot is an item with a 3-token semantic ID ----------

  const small = window.innerWidth < 600;
  const N = small ? 1600 : 2800;
  const K1 = CATALOG.length, K2 = 4, K3 = 4;

  // Category lobes, related categories close to each other.
  const LOBES = [
    [0.85, 0.3, 0.25], [0.5, 0.8, -0.25],    // kitchen
    [0.4, -0.75, 0.3], [-0.2, -0.75, 0.5],   // home
    [-0.8, 0.3, 0.3], [-0.55, 0.7, -0.35],   // electronics
    [-0.35, -0.25, -0.85], [0.3, 0.0, -0.85] // outdoor
  ].map((p) => p.map((x) => x * 0.85));
  const typeOff = [], varOff = [], filament = [];
  for (let c = 0; c < K1; c++) {
    filament.push(randUnit());
    typeOff.push([]); varOff.push([]);
    for (let t = 0; t < K2; t++) {
      typeOff[c].push(randUnit().map((x) => x * 0.24));
      varOff[c].push([]);
      for (let v = 0; v < K3; v++) varOff[c][t].push(randUnit().map((x) => x * 0.09));
    }
  }

  const codes = new Float32Array(N * 3);
  const cont = new Float32Array(N * 4); // xyz + color shade
  const quant = new Float32Array(N * 4);
  const seed = new Float32Array(N);
  const index = new Float32Array(N);
  const byCat = Array.from({ length: K1 }, () => []);
  const catShade = (c) => c / (K1 - 1);

  for (let i = 0; i < N; i++) {
    const c1 = i % K1, c2 = Math.floor(rand() * K2), c3 = Math.floor(rand() * K3);
    codes.set([c1, c2, c3], i * 3);
    byCat[c1].push(i);
    const L = LOBES[c1], T = typeOff[c1][c2], Vo = varOff[c1][c2][c3], F = filament[c1];
    const f = gauss() * 0.11;
    for (let d = 0; d < 3; d++) {
      cont[i * 4 + d] = L[d] + T[d] + 0.4 * Vo[d] + gauss() * 0.075 + F[d] * f;
      quant[i * 4 + d] = L[d] * 1.12 + T[d] * 1.1 + Vo[d] * 1.1 + (rand() - 0.5) * 0.022;
    }
    cont[i * 4 + 3] = quant[i * 4 + 3] = catShade(c1);
    seed[i] = rand();
    index[i] = i;
  }

  // One point cloud per product, each dot keeps its own index across shapes.
  const productDefs = buildProducts(THREE);
  const shapes = productDefs.map((p, c) => {
    const xyz = sampleProduct(THREE, p, N, rand);
    const out = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      out.set(xyz.subarray(i * 3, i * 3 + 3), i * 4);
      out[i * 4 + 3] = Math.min(1, Math.max(0, catShade(c) + xyz[i * 3 + 1] * 0.07));
    }
    return out;
  });

  // ---------- Points ----------

  const geometry = new THREE.BufferGeometry();
  const fromAttr = new THREE.BufferAttribute(new Float32Array(N * 4), 4);
  const toAttr = new THREE.BufferAttribute(new Float32Array(N * 4), 4);
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3)); // unused, required
  geometry.setAttribute("aFrom", fromAttr);
  geometry.setAttribute("aTo", toAttr);
  geometry.setAttribute("aCode", new THREE.BufferAttribute(codes, 3));
  geometry.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
  geometry.setAttribute("aIdx", new THREE.BufferAttribute(index, 1));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 3);

  const uniforms = {
    uMix: { value: 1 },
    uTime: { value: 0 },
    uDrift: { value: 0 },
    uSize: { value: small ? 11 : 10 },
    uPixelRatio: { value: renderer.getPixelRatio() },
    uColA: { value: new THREE.Color() },
    uColB: { value: new THREE.Color() },
    uColC: { value: new THREE.Color() },
    uColHi: { value: new THREE.Color() },
    uOpacity: { value: 0.85 },
    uFocus: { value: new THREE.Vector3() },
    uFocusLevel: { value: 0 },
    uFocusAmt: { value: 0 },
    uHover: { value: new THREE.Vector3(-1, -1, -1) },
    uHoverAmt: { value: 0 },
    uHist: { value: [-1, -1, -1, -1] },
    uHistLit: { value: 0 },
    uHistAmt: { value: 0 },
    uPred: { value: -1 },
    uPredAmt: { value: 0 },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    vertexShader: /* glsl */ `
      attribute vec4 aFrom;
      attribute vec4 aTo;
      attribute vec3 aCode;
      attribute float aSeed;
      attribute float aIdx;
      uniform float uMix, uTime, uDrift, uSize, uPixelRatio;
      uniform vec3 uFocus; uniform float uFocusLevel, uFocusAmt;
      uniform vec3 uHover; uniform float uHoverAmt;
      uniform float uHist[4]; uniform float uHistLit, uHistAmt;
      uniform float uPred, uPredAmt;
      varying float vShade, vAlpha, vHi;

      float same(float a, float b) { return step(abs(a - b), 0.5); }

      void main() {
        // Staggered morph: dots leave and arrive at slightly different times.
        float t = smoothstep(0.0, 1.0, clamp(uMix * 1.5 - aSeed * 0.5, 0.0, 1.0));
        vec3 p = mix(aFrom.xyz, aTo.xyz, t);
        vShade = mix(aFrom.w, aTo.w, t);
        p += uDrift * vec3(sin(uTime * 0.7 + aSeed * 31.0),
                           cos(uTime * 0.6 + aSeed * 17.0),
                           sin(uTime * 0.5 + aSeed * 11.0));

        // Generation: each token narrows the candidates.
        float m = 1.0;
        if (uFocusLevel > 0.5) m *= same(aCode.x, uFocus.x);
        if (uFocusLevel > 1.5) m *= same(aCode.y, uFocus.y);
        if (uFocusLevel > 2.5) m *= same(aCode.z, uFocus.z);
        float alpha = mix(1.0, mix(0.12, 1.0, m), uFocusAmt);

        // User history and the predicted item.
        float hi = 0.0;
        for (int k = 0; k < 4; k++) {
          if (float(k) < uHistLit && abs(aIdx - uHist[k]) < 0.5) hi = uHistAmt;
        }
        float pred = abs(aIdx - uPred) < 0.5 ? uPredAmt : 0.0;
        hi = max(hi, pred);

        // Hover: the item's whole code lights up, its category a little.
        float hov = uHoverAmt * same(aCode.x, uHover.x);
        float hovCode = hov * same(aCode.y, uHover.y) * same(aCode.z, uHover.z);
        alpha = max(alpha, hov);
        hi = max(hi, hovCode * 0.8);

        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        float pulse = 1.0 + pred * 0.35 * sin(uTime * 6.0);
        gl_PointSize = uSize * uPixelRatio * (1.0 + 1.8 * hi * pulse + 0.25 * hov) / -mv.z;

        float depth = clamp((-mv.z - 3.2) / 2.8, 0.0, 1.0);
        vAlpha = alpha * (1.0 - 0.5 * depth);
        vHi = hi;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColA, uColB, uColC, uColHi;
      uniform float uOpacity;
      varying float vShade, vAlpha, vHi;

      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float r = length(c);
        if (r > 0.5) discard;
        float a = smoothstep(0.5, 0.28, r);
        vec3 col = vShade < 0.5 ? mix(uColA, uColB, vShade * 2.0) : mix(uColB, uColC, vShade * 2.0 - 1.0);
        col = mix(col, uColHi, vHi * 0.85);
        gl_FragColor = vec4(col, a * uOpacity * max(vAlpha, vHi));
      }
    `,
  });

  const group = new THREE.Group();
  group.rotation.x = 0.2;
  group.add(new THREE.Points(geometry, material));
  scene.add(group);

  // Trail through the user history, ending on the predicted item.
  const TRAIL_PTS = 160;
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_PTS * 3), 3));
  trailGeo.setDrawRange(0, 0);
  const trailMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
  const trail = new THREE.Line(trailGeo, trailMat);
  group.add(trail);

  // ---------- Theme ----------

  let running = false; // true while the animation loop is active

  function applyTheme() {
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--c-accent").trim() || "#1f4e79";
    const text = css.getPropertyValue("--c-text").trim() || "#1c2129";
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    uniforms.uColA.value.set(accent);
    uniforms.uColB.value.set(accent).offsetHSL(-0.1, dark ? 0 : 0.1, dark ? -0.06 : 0.1);
    uniforms.uColC.value.set(accent).offsetHSL(0.45, dark ? -0.1 : 0.05, dark ? -0.05 : 0.12);
    uniforms.uColHi.value.set(text);
    trailMat.color.set(text);
    uniforms.uOpacity.value = dark ? 0.9 : 0.82;
  }
  applyTheme();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    applyTheme();
    if (!running) render();
  });

  // ---------- Size ----------

  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(() => {
    resize();
    if (!running) render();
  }).observe(stage);
  resize();

  // ---------- Morph state ----------

  function setInstant(target) {
    fromAttr.array.set(target);
    toAttr.array.set(target);
    fromAttr.needsUpdate = toAttr.needsUpdate = true;
    uniforms.uMix.value = 1;
  }
  function morphTo(target) {
    fromAttr.array.set(toAttr.array);
    toAttr.array.set(target);
    fromAttr.needsUpdate = toAttr.needsUpdate = true;
    uniforms.uMix.value = 0;
  }

  // CPU copy of the vertex shader's morph, for hover picking.
  function pointPosition(i, out) {
    const t = smooth(clamp(uniforms.uMix.value * 1.5 - seed[i] * 0.5));
    const f = fromAttr.array, g = toAttr.array, o = i * 4;
    return out.set(f[o] + (g[o] - f[o]) * t, f[o + 1] + (g[o + 1] - f[o + 1]) * t, f[o + 2] + (g[o + 2] - f[o + 2]) * t);
  }

  // ---------- Story ----------

  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const code = (i) => [codes[i * 3], codes[i * 3 + 1], codes[i * 3 + 2]];

  let shapeCat = Math.floor(Math.random() * K1);
  let story = null;

  function newStory() {
    // The next product: often from a related category, sometimes not.
    let pred = Math.random() < 0.6 ? shapeCat ^ 1 : pick([...Array(K1).keys()].filter((c) => c !== shapeCat));
    const hist = [
      pick(byCat[shapeCat]),
      pick(byCat[pick([...Array(K1).keys()].filter((c) => c !== pred && c !== shapeCat))]),
      pick(byCat[shapeCat]),
      pick(byCat[pred]),
    ];
    const predItem = pick(byCat[pred].filter((i) => i !== hist[3]));
    const points = [...hist, predItem].map((i) => new THREE.Vector3(quant[i * 4], quant[i * 4 + 1], quant[i * 4 + 2]));
    const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
    const arr = trailGeo.attributes.position.array;
    curve.getPoints(TRAIL_PTS - 1).forEach((p, k) => p.toArray(arr, k * 3));
    trailGeo.attributes.position.needsUpdate = true;
    uniforms.uHist.value = hist;
    uniforms.uPred.value = predItem;
    return { pred, predItem, predCode: code(predItem), histCodes: hist.map(code) };
  }

  function setTrail(fraction) {
    trailGeo.setDrawRange(0, Math.max(0, Math.round(fraction * (TRAIL_PTS - 1)) + 1));
  }

  // The model's view of the story: the history as a sequence of semantic IDs
  // (one row per visited item, shown as the trail reaches it), then the
  // generated ID filling in token by token.
  let tokensHTML = "";
  function showTokens(histCount, level = 0, withName = false) {
    if (histCount <= 0) {
      tokensEl.classList.remove("is-visible");
      return;
    }
    let html = story.histCodes
      .slice(0, histCount)
      .map((c) => `<span class="home-viz__ids is-hist">⟨${displayCode(...c).join(", ")}⟩</span>`)
      .join("");
    if (level > 0) {
      const shown = displayCode(...story.predCode).map((x, k) => (k < level ? x : "·"));
      html += `<span class="home-viz__ids is-pred">→ ⟨${shown.join(", ")}⟩</span>`;
    }
    if (withName) html += `<span class="home-viz__name">${productName(...story.predCode)}</span>`;
    if (html !== tokensHTML) {
      tokensEl.innerHTML = tokensHTML = html;
    }
    tokensEl.classList.add("is-visible");
  }

  // Phases of one cycle: [name, seconds]
  const PHASES = [
    ["shape", 3.2],
    ["toCont", 2.2],
    ["cont", 1.6],
    ["toQuant", 1.8],
    ["history", 3.6],
    ["generate", 3.0],
    ["reveal", 1.4],
    ["toShape", 2.4],
  ];
  let phase = 0, phaseTime = 0;

  function enterPhase(name) {
    switch (name) {
      case "shape":
        uniforms.uDrift.value = 0;
        break;
      case "toCont":
        story = newStory();
        morphTo(cont);
        break;
      case "toQuant":
        morphTo(quant);
        break;
      case "toShape":
        shapeCat = story.pred;
        morphTo(shapes[shapeCat]);
        break;
    }
  }

  function updatePhase(name, p, dt) {
    const fadeIn = (x) => Math.min(1, x);
    switch (name) {
      case "shape":
        break;
      case "toCont":
      case "toQuant":
        uniforms.uMix.value = ease(p);
        uniforms.uDrift.value = name === "toCont" ? 0.02 * p : 0.02 * (1 - p);
        break;
      case "cont":
        uniforms.uDrift.value = 0.02;
        break;
      case "history":
        uniforms.uHistAmt.value = fadeIn(p * 4);
      {
        // The trail reaches history item k at fraction k/4; light it and
        // show its ID at that moment.
        const f = Math.min(1, p * 1.12) * 0.75;
        const lit = Math.min(4, Math.floor(f * 4 + 1e-6) + 1);
        uniforms.uHistLit.value = lit;
        trailMat.opacity = 0.45 * fadeIn(p * 4);
        setTrail(f);
        showTokens(lit);
        break;
      }
      case "generate": {
        const level = Math.min(3, Math.floor(p * 3.2) + 1);
        uniforms.uFocus.value.set(...story.predCode);
        uniforms.uFocusLevel.value = level;
        uniforms.uFocusAmt.value = fadeIn(p * 6);
        if (level === 3) {
          uniforms.uPredAmt.value = fadeIn((p * 3.2 - 2) * 3);
          setTrail(0.75 + 0.25 * fadeIn((p * 3.2 - 2) * 2));
        }
        showTokens(4, level);
        break;
      }
      case "reveal":
        showTokens(4, 3, true);
        break;
      case "toShape": {
        uniforms.uMix.value = ease(p);
        const out = 1 - fadeIn(p * 2.5);
        uniforms.uFocusAmt.value = out;
        uniforms.uHistAmt.value = out;
        uniforms.uPredAmt.value = out;
        trailMat.opacity = 0.45 * out;
        if (p > 0.15) showTokens(0);
        if (p >= 1) {
          uniforms.uHistLit.value = 0;
          uniforms.uFocusLevel.value = 0;
          setTrail(0);
        }
        break;
      }
    }
  }

  // ---------- Hover: item and its semantic ID ----------

  const hoverable = () => {
    const name = PHASES[phase][0];
    return reducedMotion || (name !== "shape" && name !== "toCont" && name !== "toShape");
  };
  let hoverTarget = 0, hovered = -1;
  const proj = new THREE.Vector3();

  function pickAt(px, py) {
    const w = stage.clientWidth, h = stage.clientHeight;
    group.updateMatrixWorld();
    let best = -1, bestD = 14 * 14;
    for (let i = 0; i < N; i++) {
      pointPosition(i, proj).applyMatrix4(group.matrixWorld).project(camera);
      const sx = (proj.x * 0.5 + 0.5) * w, sy = (-proj.y * 0.5 + 0.5) * h;
      const d = (sx - px) ** 2 + (sy - py) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  stage.addEventListener("pointermove", (e) => {
    const rect = stage.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const i = hoverable() ? pickAt(px, py) : -1;
    if (i < 0) {
      hoverTarget = 0;
      tipEl.classList.remove("is-visible");
    } else {
      hoverTarget = 1;
      if (i !== hovered) {
        hovered = i;
        const c = code(i);
        uniforms.uHover.value.set(...c);
        tipEl.innerHTML = `<span class="home-viz__name">${productName(...c)}</span><span class="home-viz__ids">⟨${displayCode(...c).join(", ")}⟩</span>`;
      }
      tipEl.style.left = `${px}px`;
      tipEl.style.top = `${py}px`;
      tipEl.classList.toggle("is-left", px > rect.width * 0.55);
      tipEl.classList.add("is-visible");
    }
    if (reducedMotion) {
      uniforms.uHoverAmt.value = hoverTarget;
      render();
    }
  });
  stage.addEventListener("pointerleave", () => {
    hoverTarget = 0;
    tipEl.classList.remove("is-visible");
    if (reducedMotion) {
      uniforms.uHoverAmt.value = 0;
      render();
    }
  });

  function render() {
    renderer.render(scene, camera);
  }

  // ---------- Reduced motion: one still frame of the semantic-ID space ----------

  if (reducedMotion) {
    setInstant(quant);
    phase = PHASES.findIndex(([n]) => n === "history");
    group.rotation.y = 0.4;
    render();
    figure.classList.add("is-ready");
    return;
  }

  // ---------- Loop ----------

  setInstant(shapes[shapeCat]);
  enterPhase(PHASES[0][0]);

  let visible = true;
  let last = performance.now();
  let time = 0;

  function frame(now) {
    if (!running) return;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    time += dt;
    phaseTime += dt;

    let [name, dur] = PHASES[phase];
    if (phaseTime >= dur) {
      updatePhase(name, 1, dt); // finish the phase cleanly
      phaseTime -= dur;
      phase = (phase + 1) % PHASES.length;
      [name, dur] = PHASES[phase];
      enterPhase(name);
    }
    updatePhase(name, Math.min(1, phaseTime / dur), dt);

    uniforms.uTime.value = time;
    uniforms.uHoverAmt.value += (hoverTarget - uniforms.uHoverAmt.value) * Math.min(1, dt * 8);
    group.rotation.y = 0.55 * Math.sin(time * 0.28);

    render();
    requestAnimationFrame(frame);
  }

  function setRunning() {
    const should = visible && !document.hidden;
    if (should === running) return;
    running = should;
    if (running) {
      last = performance.now();
      requestAnimationFrame(frame);
    }
  }

  new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
    setRunning();
  }).observe(stage);
  document.addEventListener("visibilitychange", setRunning);

  render();
  figure.classList.add("is-ready");
  setRunning();
}

// ---------- Helpers ----------

function clamp(x) {
  return Math.min(1, Math.max(0, x));
}

function smooth(x) {
  return x * x * (3 - 2 * x);
}

function ease(x) {
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
