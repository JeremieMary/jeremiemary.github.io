/*
  Figures for the post on COSETTE and MARIUS (generative recommendation).
  - #sid-cosette: residual quantization of a toy catalog, with a slider that
    adds the collaborative (co-occurrence) term to the tokenizer objective.
  - #sid-marius: how TIGER and MARIUS read the history while decoding the
    semantic ID of the next item.
  - #sid-results: Amazon Reviews 2023 results (Table 11 of the paper).
  - #sid-timeline: filter for the timeline of semantic IDs in recommendation.
  Plain script, no dependencies. Included once per figure, so it guards itself.
  Colors come from CSS custom properties (see _sass/_post-figures.scss).
*/
(function () {
  "use strict";
  if (window.__sidFigures) return;
  window.__sidFigures = true;

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
  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function pct(v) { return Math.round(100 * v) + "%"; }

  /* ---------------- Figure 1: COSETTE on a toy catalog ---------------- */

  // Four text categories (shape) and four shopping timelines (who buys it).
  var CATS = ["Footwear", "Electronics", "Outdoor gear", "Food"];
  var TIMELINES = [
    { name: "Trail runner", items: [["Trail running shoes", 0], ["GPS sports watch", 1], ["Energy gels", 3], ["Running vest", 2], ["Wireless earbuds", 1], ["Running socks", 0], ["Electrolyte tablets", 3], ["Protein bars", 3]] },
    { name: "Camper", items: [["Hiking boots", 0], ["Two-person tent", 2], ["Headlamp", 1], ["Sleeping bag", 2], ["Freeze-dried meals", 3], ["Camp stove", 2], ["Power bank", 1], ["Wool socks", 0]] },
    { name: "Beach trip", items: [["Flip-flops", 0], ["Cooler box", 2], ["Bluetooth speaker", 1], ["Tortilla chips", 3], ["Beach umbrella", 2], ["Water shoes", 0], ["Waterproof phone pouch", 1], ["Sparkling water", 3]] },
    { name: "Coffee at home", items: [["Espresso beans", 3], ["Coffee grinder", 1], ["Wool slippers", 0], ["Milk frother", 1], ["Dark chocolate", 3], ["Thermos flask", 2], ["House sandals", 0], ["Folding garden chair", 2]] }
  ];
  var CAT_CENTER = [[-0.52, 0.5], [0.52, 0.5], [-0.52, -0.5], [0.52, -0.5]];
  var TL_CENTER = [[0, 0.62], [-0.64, 0], [0.64, 0], [0, -0.62]];
  var K = [4, 3, 3];

  function kmeans(pts, cents, iters) {
    var n = pts.length, k = cents.length, a = new Array(n);
    for (var it = 0; it < iters; it++) {
      for (var i = 0; i < n; i++) {
        var best = 0, bd = Infinity;
        for (var j = 0; j < k; j++) {
          var dx = pts[i][0] - cents[j][0], dy = pts[i][1] - cents[j][1], d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = j; }
        }
        a[i] = best;
      }
      var sum = [], cnt = [];
      for (j = 0; j < k; j++) { sum.push([0, 0]); cnt.push(0); }
      for (i = 0; i < n; i++) { sum[a[i]][0] += pts[i][0]; sum[a[i]][1] += pts[i][1]; cnt[a[i]]++; }
      for (j = 0; j < k; j++) if (cnt[j]) cents[j] = [sum[j][0] / cnt[j], sum[j][1] / cnt[j]];
    }
    return a;
  }

  function inertia(pts, cents, a) {
    var e = 0;
    for (var i = 0; i < pts.length; i++) {
      var dx = pts[i][0] - cents[a[i]][0], dy = pts[i][1] - cents[a[i]][1];
      e += dx * dx + dy * dy;
    }
    return e;
  }
  function permutations(k) {
    if (k === 1) return [[0]];
    var out = [];
    permutations(k - 1).forEach(function (p) {
      for (var i = 0; i <= p.length; i++) { var q = p.slice(); q.splice(i, 0, k - 1); out.push(q); }
    });
    return out;
  }
  var PERM4 = permutations(4);

  // Multi-start k-means for the first level, then relabel the clusters so that
  // each code stays as close as possible to where it was in the previous frame.
  function stableKmeans(pts, prev, inits) {
    var best = null, bestE = Infinity;
    inits.concat([prev]).forEach(function (init) {
      var c = init.map(function (p) { return p.slice(); });
      var a = kmeans(pts, c, 20), e = inertia(pts, c, a);
      if (e < bestE - 1e-9) { bestE = e; best = { c: c, a: a }; }
    });
    var bestP = PERM4[0], bestD = Infinity;
    PERM4.forEach(function (perm) {
      var d = 0;
      for (var j = 0; j < 4; j++) {
        var dx = best.c[perm[j]][0] - prev[j][0], dy = best.c[perm[j]][1] - prev[j][1];
        d += dx * dx + dy * dy;
      }
      if (d < bestD) { bestD = d; bestP = perm; }
    });
    var inv = [];
    bestP.forEach(function (from, to) { inv[from] = to; });
    for (var j = 0; j < 4; j++) prev[j] = best.c[bestP[j]];
    return best.a.map(function (v) { return inv[v]; });
  }

  function initCosette(root) {
    var svg = root.querySelector("svg");
    var W = 340, H = 340, pad = 14;
    var slider = root.querySelector('[data-k="lambda"]');
    var lout = root.querySelector('[data-k="lout"]');
    var btn = root.querySelector('[data-k="train"]');
    var listEl = root.querySelector('[data-k="timelines"]');
    var out = {
      co: root.querySelector('[data-k="co"]'),
      cat: root.querySelector('[data-k="cat"]')
    };

    var r = rng(5);
    var items = [];
    TIMELINES.forEach(function (tl, t) {
      tl.items.forEach(function (it) {
        var c = CAT_CENTER[it[1]], m = TL_CENTER[t];
        var a = 2 * Math.PI * r(), rad = 0.08 + 0.3 * Math.sqrt(r());
        var b = 2 * Math.PI * r(), rad2 = 0.06 + 0.26 * Math.sqrt(r());
        items.push({
          name: it[0], cat: it[1], tl: t,
          p0: [c[0] + rad * Math.cos(a), c[1] + rad * Math.sin(a)],
          p1: [m[0] + rad2 * Math.cos(b), m[1] + rad2 * Math.sin(b)],
          p: [0, 0], code: [0, 0, 0]
        });
      });
    });

    function sx(v) { return pad + ((v + 1) / 2) * (W - 2 * pad); }
    function sy(v) { return H - pad - ((v + 1) / 2) * (H - 2 * pad); }

    var gLinks = el("g", {}, svg);
    var gTrail = el("g", {}, svg);
    var gCent = el("g", {}, svg);
    var gItems = el("g", {}, svg);
    var tip = el("text", { class: "post-fig__tick post-fig__tick--strong", "text-anchor": "middle" }, svg);

    var links = [], marks = [], cents = [];
    items.forEach(function (it, i) {
      links.push(el("line", { class: "sid-fig__link" }, gLinks));
      var m;
      if (it.cat === 0) m = el("circle", { r: 5.5 }, gItems);
      else if (it.cat === 1) m = el("rect", { width: 10, height: 10, rx: 1.5 }, gItems);
      else if (it.cat === 2) m = el("path", { d: "M0,-6.5 L6,4.5 L-6,4.5 Z" }, gItems);
      else m = el("path", { d: "M0,-6.5 L6.5,0 L0,6.5 L-6.5,0 Z" }, gItems);
      m.setAttribute("class", "sid-fig__item");
      m.addEventListener("mouseenter", function () { focusTl(it.tl, i); });
      m.addEventListener("mouseleave", function () { focusTl(-1); });
      marks.push(m);
    });
    for (var c = 0; c < K[0]; c++) {
      cents.push(el("path", { d: "M-7,0 H7 M0,-7 V7", class: "sid-fig__cent sid-c" + c }, gCent));
    }
    var trail = el("polyline", { class: "sid-fig__trail" }, gTrail);

    // Timeline rows (HTML), one chip per item colored by its first token.
    var rows = [];
    TIMELINES.forEach(function (tl, t) {
      var row = document.createElement("div");
      row.className = "sid-fig__row";
      row.tabIndex = 0;
      var name = document.createElement("span");
      name.className = "sid-fig__rowname";
      name.textContent = tl.name;
      var chips = document.createElement("span");
      chips.className = "sid-fig__chips";
      var chipEls = [];
      items.forEach(function (it) {
        if (it.tl !== t) return;
        var ch = document.createElement("span");
        ch.className = "sid-fig__chip";
        chips.appendChild(ch);
        chipEls.push(ch);
      });
      row.appendChild(name);
      row.appendChild(chips);
      row.addEventListener("mouseenter", function () { focusTl(t); });
      row.addEventListener("mouseleave", function () { focusTl(-1); });
      row.addEventListener("focus", function () { focusTl(t); });
      row.addEventListener("blur", function () { focusTl(-1); });
      listEl.appendChild(row);
      rows.push({ el: row, chips: chipEls });
    });

    // Codebooks, warm-started from one frame to the next so code indices stay stable.
    var C1 = CAT_CENTER.map(function (p) { return p.slice(); });
    var C2 = [[0.12, 0], [-0.06, 0.1], [-0.06, -0.1]];
    var C3 = [[0.05, 0], [-0.03, 0.04], [-0.03, -0.04]];
    var lambda = 0, focused = -1, hoverItem = -1;
    var ri = rng(17), INITS = [CAT_CENTER, TL_CENTER];
    for (var s0 = 0; s0 < 6; s0++) {
      var pick = [];
      while (pick.length < 4) { var q = Math.floor(ri() * items.length); if (pick.indexOf(q) < 0) pick.push(q); }
      INITS.push(pick);
    }

    function quantize() {
      items.forEach(function (it) {
        it.p = [it.p0[0] + lambda * (it.p1[0] - it.p0[0]), it.p0[1] + lambda * (it.p1[1] - it.p0[1])];
      });
      var pts = items.map(function (it) { return it.p; });
      // Random-item initializations are resolved against the current positions.
      var inits = INITS.map(function (init) {
        return typeof init[0] === "number" ? init.map(function (q) { return pts[q].slice(); }) : init;
      });
      var a1 = stableKmeans(pts, C1, inits);
      var r1 = items.map(function (it, i) { return [it.p[0] - C1[a1[i]][0], it.p[1] - C1[a1[i]][1]]; });
      var a2 = kmeans(r1, C2, 12);
      var r2 = r1.map(function (v, i) { return [v[0] - C2[a2[i]][0], v[1] - C2[a2[i]][1]]; });
      var a3 = kmeans(r2, C3, 12);
      items.forEach(function (it, i) { it.code = [a1[i], a2[i], a3[i]]; });
    }

    function draw() {
      items.forEach(function (it, i) {
        var x = sx(it.p[0]), y = sy(it.p[1]), cc = C1[it.code[0]];
        var m = marks[i];
        if (it.cat === 0) { m.setAttribute("cx", x.toFixed(1)); m.setAttribute("cy", y.toFixed(1)); }
        else if (it.cat === 1) { m.setAttribute("x", (x - 5).toFixed(1)); m.setAttribute("y", (y - 5).toFixed(1)); }
        else m.setAttribute("transform", "translate(" + x.toFixed(1) + "," + y.toFixed(1) + ")");
        var dim = focused >= 0 && it.tl !== focused;
        m.setAttribute("class", "sid-fig__item sid-c" + it.code[0] + (dim ? " is-dim" : ""));
        var l = links[i];
        l.setAttribute("x1", x.toFixed(1)); l.setAttribute("y1", y.toFixed(1));
        l.setAttribute("x2", sx(cc[0]).toFixed(1)); l.setAttribute("y2", sy(cc[1]).toFixed(1));
        l.setAttribute("class", "sid-fig__link sid-c" + it.code[0] + (dim ? " is-dim" : ""));
      });
      C1.forEach(function (p, j) {
        cents[j].setAttribute("transform", "translate(" + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1) + ")");
      });
      if (focused >= 0) {
        trail.setAttribute("points", items.filter(function (it) { return it.tl === focused; })
          .map(function (it) { return sx(it.p[0]).toFixed(1) + "," + sy(it.p[1]).toFixed(1); }).join(" "));
        trail.style.display = "";
      } else trail.style.display = "none";
      if (hoverItem >= 0) {
        var it = items[hoverItem];
        tip.textContent = it.name + "  ·  " + it.code.join("-");
        var tx = Math.min(W - 90, Math.max(90, sx(it.p[0])));
        tip.setAttribute("x", tx.toFixed(1));
        tip.setAttribute("y", (sy(it.p[1]) > 40 ? sy(it.p[1]) - 12 : sy(it.p[1]) + 22).toFixed(1));
        tip.style.display = "";
      } else tip.style.display = "none";

      // Chips and stats.
      TIMELINES.forEach(function (tl, t) {
        var ch = rows[t].chips, j = 0;
        items.forEach(function (it) {
          if (it.tl !== t) return;
          ch[j].className = "sid-fig__chip sid-c" + it.code[0];
          ch[j].title = it.name + ": " + it.code.join("-");
          j++;
        });
        rows[t].el.classList.toggle("is-on", focused === t);
      });
      var co = 0, coN = 0, ca = 0, caN = 0;
      for (var i = 0; i < items.length; i++) for (var j2 = i + 1; j2 < items.length; j2++) {
        var same = items[i].code[0] === items[j2].code[0];
        if (items[i].tl === items[j2].tl) { coN++; if (same) co++; }
        if (items[i].cat === items[j2].cat) { caN++; if (same) ca++; }
      }
      out.co.textContent = pct(co / coN);
      out.cat.textContent = pct(ca / caN);
      lout.textContent = lambda.toFixed(2);
    }

    function focusTl(t, i) {
      focused = t;
      hoverItem = typeof i === "number" ? i : -1;
      draw();
    }

    function setLambda(v) { lambda = v; slider.value = Math.round(v * 100); quantize(); draw(); }

    var anim = null;
    function animateTo(target) {
      if (anim) cancelAnimationFrame(anim);
      if (reduceMotion) { setLambda(target); return; }
      var from = lambda, t0 = null, D = 1600;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var t = Math.min(1, (ts - t0) / D);
        setLambda(from + (target - from) * ease(t));
        anim = t < 1 ? requestAnimationFrame(step) : null;
      }
      anim = requestAnimationFrame(step);
    }

    slider.addEventListener("input", function () {
      if (anim) { cancelAnimationFrame(anim); anim = null; }
      setLambda(+slider.value / 100);
    });
    btn.addEventListener("click", function () { animateTo(lambda < 0.5 ? 0.7 : 0); });

    setLambda(0);
  }

  /* ---------------- Figure 2: TIGER vs MARIUS while decoding ---------------- */

  function initMarius(root) {
    var svg = root.querySelector("svg");
    var W = 560, L = 4, NMAX = 8;
    var nSlider = root.querySelector('[data-k="n"]');
    var nout = root.querySelector('[data-k="nout"]');
    var btn = root.querySelector('[data-k="decode"]');
    var out = {
      tTok: root.querySelector('[data-k="tTok"]'),
      mTok: root.querySelector('[data-k="mTok"]'),
      tRead: root.querySelector('[data-k="tRead"]'),
      mRead: root.querySelector('[data-k="mRead"]')
    };

    var lanes = [
      { key: "t", y: 24, title: "TIGER: the decoder cross-attends to all N·L history tokens" },
      { key: "m", y: 196, title: "MARIUS: one token per item, the depth decoder only reads h" }
    ];
    var layers = {};
    lanes.forEach(function (ln) {
      var t = el("text", { x: 8, y: ln.y, class: "post-fig__tick post-fig__tick--strong" }, svg);
      t.textContent = ln.title;
      layers[ln.key] = {
        box: el("g", {}, svg), attn: el("g", {}, svg), seq: el("g", {}, svg), dec: el("g", {}, svg)
      };
    });

    var N = +nSlider.value, history = [], seed = 3;
    var r = rng(seed);
    function newItem() { return [0, 1, 2, 3].map(function () { return Math.floor(r() * 4); }); }
    for (var i = 0; i < NMAX + 2; i++) history.push(newItem());
    var pending = newItem();
    var decoded = 0, active = -1, flying = 0;

    function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }

    function draw() {
      var hist = history.slice(history.length - N);
      var lt = layers.t, lm = layers.m;
      [lt, lm].forEach(function (g) { clear(g.box); clear(g.attn); clear(g.seq); clear(g.dec); });

      // ---- TIGER lane ----
      var x0 = 10, sq = 9, gap = 2, igap = 6;
      var ty = 112, tw = N * (L * (sq + gap) + igap) - igap;
      var tokPos = [];
      hist.forEach(function (code, n) {
        for (var l = 0; l < L; l++) {
          var x = x0 + n * (L * (sq + gap) + igap) + l * (sq + gap);
          el("rect", { x: x, y: ty, width: sq, height: sq, rx: 1.5, class: "sid-fig__tok sid-c" + code[l] }, lt.seq);
          tokPos.push(x + sq / 2);
        }
      });
      el("rect", { x: x0 - 4, y: 44, width: tw + 8, height: 46, rx: 6, class: "sid-fig__block" }, lt.box);
      var et = el("text", { x: x0 + tw / 2, y: 64, "text-anchor": "middle", class: "post-fig__tick" }, lt.box);
      et.textContent = "Encoder (bidirectional)";
      var et2 = el("text", { x: x0 + tw / 2, y: 80, "text-anchor": "middle", class: "post-fig__tick" }, lt.box);
      et2.textContent = (N * L) + " tokens, recomputed for each request";
      tokPos.forEach(function (x) { el("line", { x1: x, y1: ty, x2: x, y2: 90, class: "sid-fig__wire" }, lt.box); });

      var dx0 = 440, dsq = 18;
      el("rect", { x: dx0 - 8, y: 44, width: 4 * (dsq + 4) + 12, height: 46, rx: 6, class: "sid-fig__block" }, lt.box);
      var dt = el("text", { x: dx0 + 42, y: 64, "text-anchor": "middle", class: "post-fig__tick" }, lt.box);
      dt.textContent = "Decoder";
      for (var l = 0; l < L; l++) {
        var x = dx0 + l * (dsq + 4);
        var on = l < decoded;
        el("rect", { x: x, y: 104, width: dsq, height: dsq, rx: 2, class: "sid-fig__tok " + (on ? "sid-c" + pending[l] : "sid-fig__tok--empty") + (l === active ? " is-active" : "") }, lt.dec);
        if (l === active) {
          tokPos.forEach(function (tx) {
            el("path", { d: "M" + (x + dsq / 2) + ",122 C" + (x + dsq / 2) + ",172 " + tx + ",172 " + tx + "," + (ty + sq + 1), class: "sid-fig__attn" }, lt.attn);
          });
        }
      }

      // ---- MARIUS lane ----
      var my = 284, fsq = 24, fgap = 10;
      var fPos = [];
      hist.forEach(function (code, n) {
        var x = x0 + n * (fsq + fgap);
        for (var l = 0; l < L; l++) {
          el("rect", { x: x, y: my + l * (fsq / L), width: fsq, height: fsq / L, class: "sid-fig__tok sid-fig__tok--flat sid-c" + code[l] }, lm.seq);
        }
        el("rect", { x: x, y: my, width: fsq, height: fsq, rx: 2, class: "sid-fig__fuse" }, lm.seq);
        fPos.push(x + fsq / 2);
      });
      var mw = N * (fsq + fgap) - fgap;
      el("rect", { x: x0 - 4, y: 214, width: mw + 8, height: 46, rx: 6, class: "sid-fig__block" }, lm.box);
      var mt = el("text", { x: x0 + mw / 2, y: 234, "text-anchor": "middle", class: "post-fig__tick" }, lm.box);
      mt.textContent = "Temporal transformer (causal)";
      var mt2 = el("text", { x: x0 + mw / 2, y: 250, "text-anchor": "middle", class: "post-fig__tick" }, lm.box);
      mt2.textContent = N + " tokens, KV cache";
      fPos.forEach(function (x) { el("line", { x1: x, y1: my, x2: x, y2: 260, class: "sid-fig__wire" }, lm.box); });
      // Context vector h leaving the last position.
      var hx = x0 + mw + 4, hy = 237;
      el("path", { d: "M" + hx + "," + hy + " H" + (dx0 - 26), class: "sid-fig__wire sid-fig__wire--strong" }, lm.box);
      el("circle", { cx: dx0 - 22, cy: hy, r: 5, class: "sid-fig__h" + (active >= 0 ? " is-active" : "") }, lm.box);
      var ht = el("text", { x: dx0 - 22, y: hy - 10, "text-anchor": "middle", class: "post-fig__tick" }, lm.box);
      ht.textContent = "h";
      el("rect", { x: dx0 - 8, y: 214, width: 4 * (dsq + 4) + 12, height: 46, rx: 6, class: "sid-fig__block" }, lm.box);
      var dd = el("text", { x: dx0 + 42, y: 234, "text-anchor": "middle", class: "post-fig__tick" }, lm.box);
      dd.textContent = "Depth";
      for (l = 0; l < L; l++) {
        x = dx0 + l * (dsq + 4);
        on = l < decoded;
        el("rect", { x: x, y: 276, width: dsq, height: dsq, rx: 2, class: "sid-fig__tok " + (on ? "sid-c" + pending[l] : "sid-fig__tok--empty") + (l === active ? " is-active" : "") }, lm.dec);
        if (l === active) {
          el("path", { d: "M" + (x + dsq / 2) + ",276 C" + (x + dsq / 2) + ",250 " + (dx0 - 22) + ",262 " + (dx0 - 22) + "," + (hy + 5), class: "sid-fig__attn sid-fig__attn--one" }, lm.attn);
        }
      }

      // Numbers for the current N (L = 4).
      out.tTok.textContent = N * L;
      out.mTok.textContent = N;
      out.tRead.textContent = N * L;
      out.mRead.textContent = "1 (h)";
      nout.textContent = N;
    }

    var timer = null;
    function decode() {
      if (timer) return;
      btn.disabled = true;
      decoded = 0; active = 0;
      var stepMs = reduceMotion ? 0 : 650;
      function tick() {
        draw();
        if (active < L) {
          timer = setTimeout(function () {
            decoded = active + 1;
            active = decoded < L ? decoded : -1;
            if (active === -1) {
              draw();
              timer = setTimeout(function () {
                history.push(pending);
                pending = newItem();
                decoded = 0;
                timer = null;
                btn.disabled = false;
                draw();
              }, stepMs);
            } else tick();
          }, stepMs);
        }
      }
      tick();
    }

    nSlider.addEventListener("input", function () { N = +nSlider.value; draw(); });
    btn.addEventListener("click", decode);
    draw();

    // Decode one item the first time the figure scrolls into view.
    if (!reduceMotion && "IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) {
        if (es[0].isIntersecting) { io.disconnect(); setTimeout(decode, 400); }
      }, { threshold: 0.6 });
      io.observe(root);
    }
  }

  /* ---------------- Figure 3: Amazon Reviews 2023 results ---------------- */

  // Table 11: R@5, NDCG@5, R@10, NDCG@10 for TIGER, SASRec++, MARIUS.
  var RES = [
    ["Arts, Crafts & Sewing", [1.75, 1.14, 2.77, 1.47], [3.51, 2.42, 5.09, 2.93], [3.49, 2.37, 5.30, 2.95]],
    ["Automotive", [0.69, 0.44, 1.13, 0.58], [2.55, 1.84, 3.53, 2.15], [2.32, 1.58, 3.45, 1.94]],
    ["Beauty & Personal Care", [0.98, 0.64, 1.63, 0.84], [2.68, 1.88, 3.84, 2.25], [2.71, 1.81, 4.04, 2.24]],
    ["Cell Phones & Acc.", [1.85, 1.22, 2.88, 1.55], [3.30, 2.26, 4.76, 2.74], [3.72, 2.52, 5.45, 3.07]],
    ["Grocery & Gourmet", [1.52, 1.00, 2.39, 1.28], [3.02, 2.07, 4.39, 2.51], [3.22, 2.15, 4.83, 2.67]],
    ["Health & Household", [0.96, 0.64, 1.49, 0.81], [2.20, 1.53, 3.19, 1.85], [2.27, 1.54, 3.42, 1.91]],
    ["Office Products", [1.81, 1.25, 2.67, 1.53], [3.25, 2.33, 4.49, 2.73], [3.49, 2.43, 4.98, 2.91]],
    ["Patio, Lawn & Garden", [0.90, 0.58, 1.54, 0.78], [2.37, 1.68, 3.32, 1.99], [2.43, 1.65, 3.58, 2.02]],
    ["Pet Supplies", [1.41, 0.90, 2.31, 1.19], [3.05, 2.08, 4.53, 2.56], [3.16, 2.11, 4.81, 2.64]],
    ["Sports & Outdoors", [0.94, 0.61, 1.50, 0.80], [2.41, 1.69, 3.48, 2.03], [2.47, 1.66, 3.72, 2.06]]
  ];
  var METRICS = { r5: 0, n5: 1, r10: 2, n10: 3 };
  var MODELS = ["TIGER", "SASRec++", "MARIUS"];

  function initResults(root) {
    var svg = root.querySelector("svg");
    var W = 560, left = 150, right = 44, top = 8, rowH = 31, bh = 7;
    var btns = root.querySelectorAll("[data-metric]");
    var metric = "r10";
    var H = top + RES.length * rowH + 22;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    el("line", { x1: left, y1: top, x2: left, y2: top + RES.length * rowH, class: "post-fig__axis" }, svg);
    var rows = RES.map(function (row, i) {
      var y = top + i * rowH;
      var lab = el("text", { x: left - 8, y: y + 15, "text-anchor": "end", class: "post-fig__tick" }, svg);
      lab.textContent = row[0];
      var bars = [0, 1, 2].map(function (m) {
        var b = el("rect", { x: left, y: y + 3 + m * (bh + 1), height: bh, width: 0, rx: 1.5, class: "sid-fig__bar sid-fig__bar--" + m }, svg);
        var t = el("title", {}, b);
        return { rect: b, title: t };
      });
      var val = el("text", { y: y + 3 + 2 * (bh + 1) + bh - 0.5, class: "post-fig__tick sid-fig__val" }, svg);
      return { bars: bars, val: val, y: y, data: row };
    });
    var ticks = el("g", {}, svg);

    function update() {
      var mi = METRICS[metric];
      var max = 0;
      RES.forEach(function (r) { for (var m = 1; m <= 3; m++) max = Math.max(max, r[m][mi]); });
      max = Math.ceil(max);
      var sc = function (v) { return (v / max) * (W - left - right); };
      while (ticks.firstChild) ticks.removeChild(ticks.firstChild);
      for (var v = 0; v <= max; v++) {
        var xx = left + sc(v);
        el("line", { x1: xx, y1: top, x2: xx, y2: top + RES.length * rowH, class: "sid-fig__grid" }, ticks);
        var tt = el("text", { x: xx, y: H - 6, "text-anchor": "middle", class: "post-fig__tick" }, ticks);
        tt.textContent = v + "%";
      }
      rows.forEach(function (row) {
        var vals = [row.data[1][mi], row.data[2][mi], row.data[3][mi]];
        row.bars.forEach(function (b, m) {
          b.rect.setAttribute("width", Math.max(0.5, sc(vals[m])).toFixed(1));
          b.title.textContent = MODELS[m] + ": " + vals[m].toFixed(2) + "%";
        });
        row.val.setAttribute("x", (left + sc(vals[2]) + 4).toFixed(1));
        row.val.textContent = vals[2].toFixed(2);
        var d = vals[2] - vals[1];
        row.val.setAttribute("class", "post-fig__tick sid-fig__val" + (d >= 0 ? " is-win" : ""));
      });
    }
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        metric = b.getAttribute("data-metric");
        btns.forEach(function (o) {
          var on = o === b;
          o.classList.toggle("is-on", on);
          o.setAttribute("aria-pressed", on ? "true" : "false");
        });
        update();
      });
    });
    update();
  }

  /* ---------------- Timeline filter ---------------- */

  function initTimeline(root) {
    var btns = root.querySelectorAll("[data-tl]");
    var items = root.querySelectorAll("li[data-tags]");
    btns.forEach(function (b) {
      b.addEventListener("click", function () {
        var f = b.getAttribute("data-tl");
        btns.forEach(function (o) {
          var on = o === b;
          o.classList.toggle("is-on", on);
          o.setAttribute("aria-pressed", on ? "true" : "false");
        });
        items.forEach(function (li) {
          var tags = li.getAttribute("data-tags").split(" ");
          li.classList.toggle("is-out", f !== "all" && tags.indexOf(f) < 0);
        });
      });
    });
  }

  function boot() {
    var d = document.getElementById("sid-timeline");
    if (d) initTimeline(d);
    var a = document.getElementById("sid-cosette");
    var b = document.getElementById("sid-marius");
    var c = document.getElementById("sid-results");
    if (a) initCosette(a);
    if (b) initMarius(b);
    if (c) initResults(c);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
