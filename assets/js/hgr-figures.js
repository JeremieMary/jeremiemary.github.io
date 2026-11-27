/*
  Figures for the post "Maximal correlation for fairness, seven years later".
  - #hgr-shuffle: positive pairs vs. pairs with y permuted inside the batch.
  - #hgr-spectrum: dependence spectrum, rank-k capture and the HGR upper bound.
  Plain script, no dependencies. Included once per figure, so it guards itself.
  Colors come from CSS custom properties (see _sass/_post-figures.scss).
*/
(function () {
  "use strict";
  if (window.__hgrFigures) return;
  window.__hgrFigures = true;

  var SVG = "http://www.w3.org/2000/svg";
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(name, attrs, parent) {
    var n = document.createElementNS(SVG, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function fmt(v) { return v.toFixed(2); }

  // Seeded RNG so the first batch is the same for every reader.
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
  }

  /* ---------------- Statistics ---------------- */

  function pearson(x, y) {
    var n = x.length, mx = 0, my = 0, sxy = 0, sxx = 0, syy = 0, i;
    for (i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    for (i = 0; i < n; i++) {
      var a = x[i] - mx, b = y[i] - my;
      sxy += a * b; sxx += a * a; syy += b * b;
    }
    return sxy / Math.sqrt(sxx * syy);
  }

  // HGR on quantile bins: second singular value of B_ij = p_ij / sqrt(p_i p_j).
  // The top singular value is 1 (constant functions); we remove it and run
  // power iteration on the centered matrix.
  function binnedHGR(x, y, nb) {
    var n = x.length;
    function ranks(v) {
      var idx = v.map(function (_, i) { return i; }).sort(function (a, b) { return v[a] - v[b]; });
      var r = new Array(n);
      for (var i = 0; i < n; i++) r[idx[i]] = Math.min(nb - 1, Math.floor((i * nb) / n));
      return r;
    }
    var bx = ranks(x), by = ranks(y), i, j;
    var P = [], px = new Array(nb).fill(0), py = new Array(nb).fill(0);
    for (i = 0; i < nb; i++) P.push(new Array(nb).fill(0));
    for (i = 0; i < n; i++) { P[bx[i]][by[i]] += 1 / n; px[bx[i]] += 1 / n; py[by[i]] += 1 / n; }
    var B = [];
    for (i = 0; i < nb; i++) {
      B.push([]);
      for (j = 0; j < nb; j++) B[i].push(P[i][j] / Math.sqrt(px[i] * py[j]) - Math.sqrt(px[i] * py[j]));
    }
    var v = [];
    for (j = 0; j < nb; j++) v.push(1 + 0.1 * j);
    var s = 0;
    for (var it = 0; it < 200; it++) {
      var u = new Array(nb).fill(0), w = new Array(nb).fill(0);
      for (i = 0; i < nb; i++) for (j = 0; j < nb; j++) u[i] += B[i][j] * v[j];
      for (j = 0; j < nb; j++) for (i = 0; i < nb; i++) w[j] += B[i][j] * u[i];
      var norm = Math.sqrt(w.reduce(function (a, b) { return a + b * b; }, 0)) || 1;
      v = w.map(function (a) { return a / norm; });
      s = Math.sqrt(norm);
    }
    return s;
  }

  /* ---------------- Figure 1: shuffle ---------------- */

  function initShuffle(root) {
    var svg = root.querySelector("svg");
    var W = 420, H = 300, pad = 28, N = 300;
    var statEls = {
      state: root.querySelector('[data-k="state"]'),
      pearson: root.querySelector('[data-k="pearson"]'),
      hgr: root.querySelector('[data-k="hgr"]')
    };
    var btn = root.querySelector('[data-k="toggle"]');
    var btnNew = root.querySelector('[data-k="resample"]');

    el("line", { x1: pad, y1: H - pad, x2: W - pad / 2, y2: H - pad, class: "post-fig__axis" }, svg);
    el("line", { x1: pad, y1: pad / 2, x2: pad, y2: H - pad, class: "post-fig__axis" }, svg);
    var tx = el("text", { x: W - pad / 2, y: H - 8, "text-anchor": "end", class: "post-fig__tick" }, svg); tx.textContent = "x";
    var ty = el("text", { x: 10, y: pad / 2 + 8, class: "post-fig__tick" }, svg); ty.textContent = "y";

    var layer = el("g", {}, svg);
    var dots = [];
    for (var i = 0; i < N; i++) dots.push(el("circle", { r: 3, class: "post-fig__dot" }, layer));

    var seed = 7, x, y, perm, shuffled = false, cur, anim = null;

    function sx(v) { return pad + ((v + 1) / 2) * (W - 1.5 * pad); }
    function sy(v) { return H - pad - ((v + 0.25) / 1.5) * (H - 1.5 * pad); }

    function sample() {
      var r = rng(seed);
      x = []; y = [];
      for (var i = 0; i < N; i++) {
        var xi = 2 * r() - 1;
        x.push(xi);
        y.push(xi * xi + 0.07 * gauss(r));
      }
      cur = y.slice();
    }
    function newPerm() {
      var r = rng(seed * 31 + Math.floor(Math.random() * 1e6));
      perm = []; for (var i = 0; i < N; i++) perm.push(i);
      for (i = N - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)); var t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    }
    function target() { return shuffled ? perm.map(function (j) { return y[j]; }) : y.slice(); }

    function draw() {
      for (var i = 0; i < N; i++) {
        dots[i].setAttribute("cx", sx(x[i]).toFixed(1));
        dots[i].setAttribute("cy", sy(cur[i]).toFixed(1));
      }
    }
    function stats() {
      var yy = target();
      statEls.state.innerHTML = shuffled ? "shuffled, (x<sub>i</sub>, y<sub>σ(i)</sub>)" : "joint, (x<sub>i</sub>, y<sub>i</sub>)";
      statEls.pearson.textContent = fmt(pearson(x, yy));
      statEls.hgr.textContent = fmt(binnedHGR(x, yy, 6));
      root.classList.toggle("is-shuffled", shuffled);
      btn.textContent = shuffled ? "Restore the joint pairs" : "Shuffle y within the batch";
    }
    function go() {
      var from = cur.slice(), to = target();
      if (anim) cancelAnimationFrame(anim);
      stats();
      if (reduceMotion) { cur = to; draw(); return; }
      var t0 = null, D = 900;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var t = Math.min(1, (ts - t0) / D);
        for (var i = 0; i < N; i++) {
          // Stagger by x so the eye can follow the shuffle across the batch.
          var d = Math.min(1, Math.max(0, (t * 1.6 - ((x[i] + 1) / 2) * 0.6)));
          var e = d < 0.5 ? 2 * d * d : 1 - Math.pow(-2 * d + 2, 2) / 2;
          cur[i] = from[i] + (to[i] - from[i]) * e;
        }
        draw();
        anim = t < 1 ? requestAnimationFrame(step) : null;
      }
      anim = requestAnimationFrame(step);
    }

    btn.addEventListener("click", function () {
      shuffled = !shuffled;
      if (shuffled) newPerm();
      go();
    });
    btnNew.addEventListener("click", function () {
      seed += 1; shuffled = false; sample(); draw(); stats();
    });

    sample(); newPerm(); draw(); stats();
  }

  /* ---------------- Figure 2: spectrum and bound ---------------- */

  var SPECTRA = {
    fast: (function () { var a = []; for (var i = 0; i < 16; i++) a.push(0.5 * Math.pow(0.55, i)); return a; })(),
    slow: (function () { var a = []; for (var i = 0; i < 16; i++) a.push(0.5 / Math.sqrt(i + 1)); return a; })()
  };

  function initSpectrum(root) {
    var svg = root.querySelector("svg");
    var W = 560, H = 260, left = 44, right = 16, top = 14, barH = 130;
    var slider = root.querySelector('[data-k="k"]');
    var kout = root.querySelector('[data-k="kout"]');
    var out = {
      hgr: root.querySelector('[data-k="hgr"]'),
      cap: root.querySelector('[data-k="cap"]'),
      res: root.querySelector('[data-k="res"]'),
      bound: root.querySelector('[data-k="bound"]')
    };
    var decayBtns = root.querySelectorAll("[data-decay]");
    var decay = "fast";

    var base = top + barH;
    el("line", { x1: left, y1: base, x2: W - right, y2: base, class: "post-fig__axis" }, svg);
    var yl = el("text", { x: left - 8, y: top + 10, "text-anchor": "end", class: "post-fig__tick" }, svg); yl.textContent = "ρᵢ²";
    var bw = (W - left - right) / 16;
    var bars = [], labels = [];
    for (var i = 0; i < 16; i++) {
      bars.push(el("rect", { x: left + i * bw + 3, width: bw - 6, y: base, height: 0, rx: 2, class: "post-fig__bar" }, svg));
      var t = el("text", { x: left + i * bw + bw / 2, y: base + 14, "text-anchor": "middle", class: "post-fig__tick" }, svg);
      t.textContent = i + 1;
      labels.push(t);
    }
    var kLine = el("line", { y1: top - 4, y2: base + 2, class: "post-fig__kline" }, svg);

    // Gauge: [0, max] with HGR and the bound decomposed into its two terms.
    var gy = 214, gx0 = left, gx1 = W - right, gMax = 1.4;
    function gx(v) { return gx0 + (v / gMax) * (gx1 - gx0); }
    el("line", { x1: gx0, y1: gy, x2: gx1, y2: gy, class: "post-fig__axis" }, svg);
    [0, 0.25, 0.5, 0.75, 1, 1.25].forEach(function (v) {
      el("line", { x1: gx(v), y1: gy, x2: gx(v), y2: gy + 4, class: "post-fig__axis" }, svg);
      var tt = el("text", { x: gx(v), y: gy + 16, "text-anchor": "middle", class: "post-fig__tick" }, svg);
      tt.textContent = v.toFixed(2);
    });
    var segFound = el("rect", { y: gy - 16, height: 12, rx: 2, class: "post-fig__seg-found" }, svg);
    var segRes = el("rect", { y: gy - 16, height: 12, rx: 2, class: "post-fig__seg-res" }, svg);
    var chiMark = el("line", { y1: gy - 20, y2: gy + 2, class: "post-fig__mark post-fig__mark--chi" }, svg);
    var chiTxt = el("text", { y: gy + 30, "text-anchor": "middle", class: "post-fig__tick" }, svg);
    chiTxt.textContent = "√χ²";
    var hgrMark = el("line", { y1: gy - 22, y2: gy + 2, class: "post-fig__mark" }, svg);
    var hgrTxt = el("text", { y: gy - 26, "text-anchor": "middle", class: "post-fig__tick post-fig__tick--strong" }, svg);
    hgrTxt.textContent = "HGR";
    var boundTxt = el("text", { y: gy - 26, "text-anchor": "middle", class: "post-fig__tick" }, svg);
    boundTxt.textContent = "bound";

    function update() {
      var k = +slider.value, rho = SPECTRA[decay];
      var maxSq = rho[0] * rho[0];
      var total = 0, cap = 0, smax = 0;
      for (var i = 0; i < 16; i++) {
        var e = rho[i] * rho[i];
        total += e;
        if (i < k) { cap += e; smax = Math.max(smax, rho[i]); }
        var h = (e / maxSq) * (barH - 6);
        bars[i].setAttribute("y", (base - h).toFixed(1));
        bars[i].setAttribute("height", h.toFixed(1));
        bars[i].setAttribute("class", "post-fig__bar " + (i < k ? "is-found" : "is-res"));
      }
      var res = total - cap, bound = smax + Math.sqrt(res);
      var kx = left + k * bw;
      kLine.setAttribute("x1", kx); kLine.setAttribute("x2", kx);
      kout.textContent = k;

      segFound.setAttribute("x", gx(0)); segFound.setAttribute("width", Math.max(0, gx(smax) - gx(0)));
      segRes.setAttribute("x", gx(smax)); segRes.setAttribute("width", Math.max(0, gx(bound) - gx(smax)));
      hgrMark.setAttribute("x1", gx(rho[0])); hgrMark.setAttribute("x2", gx(rho[0]));
      hgrTxt.setAttribute("x", gx(rho[0]));
      chiMark.setAttribute("x1", gx(Math.sqrt(total))); chiMark.setAttribute("x2", gx(Math.sqrt(total)));
      chiTxt.setAttribute("x", gx(Math.sqrt(total)));
      var bx = gx(Math.min(bound, gMax));
      boundTxt.setAttribute("x", Math.abs(bx - gx(rho[0])) < 40 ? bx + 26 : bx);

      out.hgr.textContent = fmt(rho[0]);
      out.cap.textContent = fmt(cap);
      out.res.textContent = fmt(res);
      out.bound.textContent = fmt(smax) + " + √" + fmt(res) + " = " + fmt(bound);
    }

    slider.addEventListener("input", update);
    decayBtns.forEach(function (b) {
      b.addEventListener("click", function () {
        decay = b.getAttribute("data-decay");
        decayBtns.forEach(function (o) {
          var on = o === b;
          o.classList.toggle("is-on", on);
          o.setAttribute("aria-pressed", on ? "true" : "false");
        });
        update();
      });
    });
    update();
  }

  function boot() {
    var a = document.getElementById("hgr-shuffle");
    var b = document.getElementById("hgr-spectrum");
    if (a) initShuffle(a);
    if (b) initSpectrum(b);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
