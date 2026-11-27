/*
  Figure for the post "The telescope distance, and a correction".
  #tel-window: two rotation processes with the same one-dimensional marginal.
  For each window length k the empirical d_{H_k} over halfspaces in R^k is
  computed as the largest Kolmogorov-Smirnov gap between the projections of
  the two sets of windows, over a random search of directions (an approximate
  empirical risk minimizer, as an SVM would be in the paper).
  Plain script, no dependencies. Colors come from CSS custom properties.
*/
(function () {
  "use strict";
  if (window.__telFigures) return;
  window.__telFigures = true;

  var SVG = "http://www.w3.org/2000/svg";
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(name, attrs, parent) {
    var n = document.createElementNS(SVG, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
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

  /* ---------------- Processes and the empirical distance ---------------- */

  var N = 500, KMAX = 6, SD = 0.3;
  var ALPHA_A = (Math.sqrt(5) - 1) / 2 - 0.5; // 0.118...
  var ALPHA_B = Math.PI / 8;                  // 0.392...

  // r_{i+1} = r_i + alpha mod 1; X_i ~ N(0, SD^2) if r_i < 1/2, N(1, SD^2) otherwise.
  function rotation(alpha, n, r) {
    var x = [], ph = r();
    for (var i = 0; i < n; i++) {
      x.push((ph < 0.5 ? 0 : 1) + SD * gauss(r));
      ph = (ph + alpha) % 1;
    }
    return x;
  }

  function project(x, k, d) {
    var out = new Float64Array(x.length - k + 1);
    for (var i = 0; i < out.length; i++) {
      var s = 0;
      for (var j = 0; j < k; j++) s += d[j] * x[i + j];
      out[i] = s;
    }
    return out;
  }

  // Largest gap between the two empirical CDFs, and where it is reached.
  function ks(a, b) {
    a = Array.prototype.slice.call(a).sort(function (p, q) { return p - q; });
    b = Array.prototype.slice.call(b).sort(function (p, q) { return p - q; });
    var i = 0, j = 0, best = 0, at = a[0];
    while (i < a.length && j < b.length) {
      var v = Math.min(a[i], b[j]);
      while (i < a.length && a[i] <= v) i++;
      while (j < b.length && b[j] <= v) j++;
      var g = Math.abs(i / a.length - j / b.length);
      if (g > best) { best = g; at = v; }
    }
    return { d: best, t: at, a: a, b: b };
  }

  function dk(x, y, k, r) {
    if (k === 1) {
      var res1 = ks(x, y);
      res1.dir = [1];
      return res1;
    }
    var best = null;
    function tryDir(d) {
      var res = ks(project(x, k, d), project(y, k, d));
      if (!best || res.d > best.d) { res.dir = d; best = res; }
    }
    var i, j, d;
    for (i = 0; i < 220; i++) {
      d = [];
      for (j = 0; j < k; j++) d.push(gauss(r));
      tryDir(d);
    }
    for (i = 0; i < 140; i++) {
      d = best.dir.map(function (v) { return v + 0.3 * gauss(r); });
      tryDir(d);
    }
    return best;
  }

  /* ---------------- Figure ---------------- */

  function initWindow(root) {
    var svg = root.querySelector("svg");
    var W = 560, H = 344;
    var kBtns = root.querySelectorAll("[data-kwin]");
    var modeBtns = root.querySelectorAll("[data-mode]");
    var btnNew = root.querySelector('[data-k="resample"]');
    var out = {
      dk: root.querySelector('[data-k="dk"]'),
      wk: root.querySelector('[data-k="wk"]'),
      sum: root.querySelector('[data-k="sum"]'),
      klab: root.querySelectorAll('[data-k="klab"]')
    };

    var seed = 3, k = 1, same = false, res = [], X, Y, slide = null;

    /* Strip: the first T steps of each series, with a window of length k. */
    var T = 64, sx0 = 46, sx1 = W - 10, rows = [32, 86], band = 46;
    function stx(i) { return sx0 + (i + 0.5) * (sx1 - sx0) / T; }
    // Values live roughly in [-0.7, 1.7]; map that range onto the row band.
    function sty(row, v) { return rows[row] - (Math.min(1.8, Math.max(-0.8, v)) - 0.5) / 2.6 * band; }
    var winRect = el("rect", { y: 4, height: 108, rx: 4, class: "tel-fig__win" }, svg);
    var rowLab = [];
    var lines = [], dotsLayer = [];
    for (var r0 = 0; r0 < 2; r0++) {
      el("line", { x1: sx0, x2: sx1, y1: rows[r0] + band * 0.55, y2: rows[r0] + band * 0.55, class: "post-fig__axis" }, svg);
      rowLab.push(el("text", { x: 4, y: rows[r0] + 4, class: "post-fig__tick post-fig__tick--strong" }, svg));
      lines.push(el("polyline", { class: "tel-fig__path tel-fig__path--" + (r0 ? "b" : "a") }, svg));
      var g = el("g", {}, svg), arr = [];
      for (var i = 0; i < T; i++) arr.push(el("circle", { r: 2.6, class: "tel-fig__dot tel-fig__dot--" + (r0 ? "b" : "a") }, g));
      dotsLayer.push(arr);
    }

    /* Left panel: empirical CDFs of the projections along the best direction. */
    var L = { x0: 46, x1: 290, y0: 156, y1: 310 };
    el("line", { x1: L.x0, y1: L.y1, x2: L.x1, y2: L.y1, class: "post-fig__axis" }, svg);
    el("line", { x1: L.x0, y1: L.y0, x2: L.x0, y2: L.y1, class: "post-fig__axis" }, svg);
    var t0 = el("text", { x: L.x0 - 6, y: L.y1 + 4, "text-anchor": "end", class: "post-fig__tick" }, svg); t0.textContent = "0";
    var t1 = el("text", { x: L.x0 - 6, y: L.y0 + 4, "text-anchor": "end", class: "post-fig__tick" }, svg); t1.textContent = "1";
    var lTitle = el("text", { x: L.x0, y: L.y0 - 12, class: "post-fig__tick" }, svg);
    var lAxis = el("text", { x: L.x1, y: L.y1 + 16, "text-anchor": "end", class: "post-fig__tick" }, svg);
    var cdfA = el("path", { class: "tel-fig__cdf tel-fig__cdf--a" }, svg);
    var cdfB = el("path", { class: "tel-fig__cdf tel-fig__cdf--b" }, svg);
    var gap = el("line", { class: "tel-fig__gap" }, svg);
    var gapTxt = el("text", { class: "post-fig__tick post-fig__tick--strong" }, svg);

    /* Right panel: d_k for k = 1..KMAX. */
    var R = { x0: 336, x1: W - 10, y0: 156, y1: 310, max: 0.4 };
    el("line", { x1: R.x0, y1: R.y1, x2: R.x1, y2: R.y1, class: "post-fig__axis" }, svg);
    var rTitle = el("text", { x: R.x0, y: R.y0 - 12, class: "post-fig__tick" }, svg);
    rTitle.textContent = "Estimated term for each window length k";
    var bw = (R.x1 - R.x0) / KMAX, bars = [], barVals = [];
    for (var b = 0; b < KMAX; b++) {
      var bar = el("rect", { x: R.x0 + b * bw + 5, width: bw - 10, y: R.y1, height: 0, rx: 2, class: "post-fig__bar tel-fig__bar", tabindex: "-1" }, svg);
      bar.addEventListener("click", (function (kk) { return function () { setK(kk); }; })(b + 1));
      bars.push(bar);
      var lab = el("text", { x: R.x0 + b * bw + bw / 2, y: R.y1 + 14, "text-anchor": "middle", class: "post-fig__tick" }, svg);
      lab.textContent = b + 1;
      barVals.push(el("text", { x: R.x0 + b * bw + bw / 2, "text-anchor": "middle", class: "post-fig__tick" }, svg));
    }
    var kAxis = el("text", { x: R.x1, y: R.y1 + 28, "text-anchor": "end", class: "post-fig__tick" }, svg);
    kAxis.textContent = "window length k";

    function sample() {
      var r = rng(seed * 7919 + 17);
      X = rotation(ALPHA_A, N, r);
      Y = rotation(same ? ALPHA_A : ALPHA_B, N, r);
      res = [];
      for (var kk = 1; kk <= KMAX; kk++) res.push(dk(X, Y, kk, r));
      rowLab[0].textContent = "A";
      rowLab[1].textContent = "B";
      root.querySelector('[data-k="alphaB"]').textContent = same ? "0.118" : "0.393";
    }

    function drawStrip() {
      for (var row = 0; row < 2; row++) {
        var s = row ? Y : X, pts = [];
        for (var i = 0; i < T; i++) {
          var cx = stx(i), cy = sty(row, s[i]);
          dotsLayer[row][i].setAttribute("cx", cx.toFixed(1));
          dotsLayer[row][i].setAttribute("cy", cy.toFixed(1));
          pts.push(cx.toFixed(1) + "," + cy.toFixed(1));
        }
        lines[row].setAttribute("points", pts.join(" "));
      }
    }
    function placeWindow(pos) {
      var step = (sx1 - sx0) / T;
      winRect.setAttribute("x", (sx0 + pos * step).toFixed(1));
      winRect.setAttribute("width", (k * step).toFixed(1));
    }
    function runSlide() {
      if (slide) cancelAnimationFrame(slide);
      if (reduceMotion) { placeWindow(6); return; }
      var start = null, D = 3200, last = T - k;
      function step(ts) {
        if (start === null) start = ts;
        var u = Math.min(1, (ts - start) / D);
        // Step from position to position, as the sum over i in the estimate does.
        placeWindow(Math.round(u * last));
        slide = u < 1 ? requestAnimationFrame(step) : null;
      }
      slide = requestAnimationFrame(step);
    }

    function cdfPath(sorted, lo, hi) {
      var n = sorted.length, d = "", i;
      function px(v) { return L.x0 + (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (L.x1 - L.x0); }
      function py(p) { return L.y1 - p * (L.y1 - L.y0); }
      d = "M" + L.x0 + "," + py(0);
      for (i = 0; i < n; i++) {
        var x = px(sorted[i]).toFixed(1);
        d += "L" + x + "," + py(i / n).toFixed(1) + "L" + x + "," + py((i + 1) / n).toFixed(1);
      }
      return d + "L" + L.x1 + "," + py(1);
    }

    function update() {
      var cur = res[k - 1];
      var lo = Math.min(cur.a[0], cur.b[0]), hi = Math.max(cur.a[cur.a.length - 1], cur.b[cur.b.length - 1]);
      cdfA.setAttribute("d", cdfPath(cur.a, lo, hi));
      cdfB.setAttribute("d", cdfPath(cur.b, lo, hi));

      // Gap at the threshold t*: the best halfspace {a·w <= t*}.
      function frac(sorted, t) { var c = 0; while (c < sorted.length && sorted[c] <= t) c++; return c / sorted.length; }
      var fa = frac(cur.a, cur.t), fb = frac(cur.b, cur.t);
      var gx = L.x0 + (cur.t - lo) / (hi - lo) * (L.x1 - L.x0);
      var ya = L.y1 - fa * (L.y1 - L.y0), yb = L.y1 - fb * (L.y1 - L.y0);
      gap.setAttribute("x1", gx); gap.setAttribute("x2", gx);
      gap.setAttribute("y1", ya); gap.setAttribute("y2", yb);
      gapTxt.setAttribute("x", gx > L.x1 - 50 ? gx - 6 : gx + 6);
      gapTxt.setAttribute("text-anchor", gx > L.x1 - 50 ? "end" : "start");
      gapTxt.setAttribute("y", ((ya + yb) / 2 + 4).toFixed(1));
      gapTxt.textContent = "gap " + cur.d.toFixed(2);
      lTitle.textContent = k === 1
        ? "k = 1: CDFs of single values"
        : "k = " + k + ": CDFs along the best halfspace normal";
      lAxis.textContent = k === 1 ? "xᵢ" : "⟨a, (xᵢ, …, xᵢ₊" + "₀₁₂₃₄₅₆₇₈₉".charAt(k - 1) + ")⟩";

      var total = 0;
      for (var b = 0; b < KMAX; b++) {
        var v = res[b].d, h = Math.min(1, v / R.max) * (R.y1 - R.y0);
        total += v / ((b + 1) * (b + 1));
        bars[b].setAttribute("y", (R.y1 - h).toFixed(1));
        bars[b].setAttribute("height", h.toFixed(1));
        bars[b].classList.toggle("is-on", b === k - 1);
        barVals[b].setAttribute("y", (R.y1 - h - 4).toFixed(1));
        barVals[b].textContent = v.toFixed(2);
      }
      out.dk.textContent = cur.d.toFixed(3);
      out.wk.textContent = "1/" + k * k;
      out.sum.textContent = total.toFixed(3);
      out.klab.forEach(function (n) { n.textContent = k; });
      kBtns.forEach(function (o) {
        var on = +o.getAttribute("data-kwin") === k;
        o.classList.toggle("is-on", on);
        o.setAttribute("aria-pressed", on ? "true" : "false");
      });
    }

    function setK(kk) { k = kk; update(); runSlide(); }

    kBtns.forEach(function (bt) {
      bt.addEventListener("click", function () { setK(+bt.getAttribute("data-kwin")); });
    });
    modeBtns.forEach(function (bt) {
      bt.addEventListener("click", function () {
        same = bt.getAttribute("data-mode") === "same";
        modeBtns.forEach(function (o) {
          var on = o === bt;
          o.classList.toggle("is-on", on);
          o.setAttribute("aria-pressed", on ? "true" : "false");
        });
        sample(); drawStrip(); update();
      });
    });
    btnNew.addEventListener("click", function () { seed += 1; sample(); drawStrip(); update(); runSlide(); });

    sample(); drawStrip(); update(); placeWindow(6);
    // Slide once when the figure first scrolls into view.
    if (!reduceMotion && "IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) {
        if (es[0].isIntersecting) { runSlide(); io.disconnect(); }
      }, { threshold: 0.5 });
      io.observe(root);
    }
  }

  function boot() {
    var a = document.getElementById("tel-window");
    if (a) initWindow(a);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
