/*
 * Product shapes and the synthetic catalog for the home page animation
 * (see semantic-ids.js).
 *
 * Each product is a handful of Three.js primitives. sampleProduct() turns one
 * into a point cloud by area-weighted sampling of its surface; thin parts get
 * a weight boost so they stay legible as dots.
 *
 * Categories are ordered so that related ones sit next to each other
 * (kitchen, home, electronics, outdoor): they share neighbouring colors and
 * neighbouring lobes in the embedding space.
 */

export const CATALOG = [
  {
    key: "mug",
    types: ["coffee mug", "travel mug", "espresso cup", "tea cup"],
    variants: ["Ceramic", "Matte black", "Enamel", "Stoneware"],
  },
  {
    key: "bottle",
    types: ["wine bottle", "water bottle", "olive oil bottle", "glass carafe"],
    variants: ["Glass", "Amber", "Recycled", "Frosted"],
  },
  {
    key: "lamp",
    types: ["desk lamp", "reading lamp", "clip lamp", "architect lamp"],
    variants: ["Brass", "Black", "White", "LED"],
  },
  {
    key: "chair",
    types: ["dining chair", "office chair", "armchair", "bar stool"],
    variants: ["Oak", "Walnut", "Velvet", "Steel"],
  },
  {
    key: "headphones",
    types: ["over-ear headphones", "studio headphones", "gaming headset", "on-ear headphones"],
    variants: ["Wireless", "Noise-cancelling", "Foldable", "Retro"],
  },
  {
    key: "camera",
    types: ["mirrorless camera", "instant camera", "compact camera", "film camera"],
    variants: ["Black", "Silver", "Vintage", "Pro"],
  },
  {
    key: "bicycle",
    types: ["road bike", "city bike", "gravel bike", "folding bike"],
    variants: ["Carbon", "Steel", "Electric", "Aluminium"],
  },
  {
    key: "sunglasses",
    types: ["aviator sunglasses", "round sunglasses", "sport sunglasses", "cat-eye sunglasses"],
    variants: ["Polarized", "Tortoiseshell", "Gold", "Mirrored"],
  },
];

// Codebook entries as shown to the viewer (real semantic IDs are opaque
// indices into codebooks of a few hundred entries).
const LEVEL1 = [17, 203, 88, 141, 56, 230, 9, 172];
export function displayCode(c1, c2, c3) {
  return [LEVEL1[c1], (c1 * 53 + c2 * 29 + 11) % 256, (c1 * 31 + c2 * 17 + c3 * 71 + 3) % 256];
}

export function productName(c1, c2, c3) {
  const cat = CATALOG[c1];
  return `${cat.variants[c3]} ${cat.types[c2]}`;
}

// ---------- Shapes ----------

export function buildProducts(THREE) {
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  function part(geo, { pos = [0, 0, 0], rot = [0, 0, 0], scale = [1, 1, 1], w = 1 } = {}) {
    const m = new THREE.Matrix4().compose(
      V(...pos),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(...rot)),
      V(...scale)
    );
    return { geo, matrix: m, w };
  }

  // A thin cylinder from a to b.
  function tube(a, b, r, w = 3) {
    const va = V(...a), vb = V(...b);
    const dir = vb.clone().sub(va);
    const geo = new THREE.CylinderGeometry(r, r, dir.length(), 10, 1, true);
    const q = new THREE.Quaternion().setFromUnitVectors(V(0, 1, 0), dir.clone().normalize());
    const m = new THREE.Matrix4().compose(va.clone().add(vb).multiplyScalar(0.5), q, V(1, 1, 1));
    return { geo, matrix: m, w };
  }

  const lathe = (pts, w = 1) =>
    part(new THREE.LatheGeometry(pts.map(([x, y]) => new THREE.Vector2(x, y)), 48), { w });

  const products = {
    mug: {
      view: [0.15, -0.55],
      parts: [
        lathe([[0, -0.6], [0.5, -0.6], [0.53, -0.56], [0.55, 0.6], [0.51, 0.6], [0.48, -0.5], [0, -0.5]]),
        part(new THREE.TorusGeometry(0.28, 0.065, 10, 28, Math.PI), { pos: [0.53, 0.03, 0], rot: [0, 0, -Math.PI / 2], w: 1.6 }),
      ],
    },

    bottle: {
      view: [0.12, 0],
      parts: [
        lathe([
          [0, -0.95], [0.3, -0.95], [0.33, -0.9], [0.33, 0.2], [0.29, 0.38], [0.15, 0.55],
          [0.12, 0.62], [0.12, 0.92], [0.14, 0.94], [0.14, 1.0], [0, 1.0],
        ]),
        // label band
        part(new THREE.CylinderGeometry(0.335, 0.335, 0.42, 48, 1, true), { pos: [0, -0.35, 0], w: 0.6 }),
      ],
    },

    lamp: {
      view: [0.1, 0.5],
      parts: [
        part(new THREE.CylinderGeometry(0.42, 0.48, 0.08, 36), { pos: [0, -0.95, 0], w: 0.8 }),
        tube([0, -0.92, 0], [0.32, 0.05, 0], 0.035),
        tube([0.32, 0.05, 0], [-0.25, 0.55, 0], 0.035),
        part(new THREE.SphereGeometry(0.07, 12, 8), { pos: [0.32, 0.05, 0], w: 2 }),
        part(new THREE.SphereGeometry(0.06, 12, 8), { pos: [-0.25, 0.55, 0], w: 2 }),
        part(new THREE.ConeGeometry(0.34, 0.42, 36, 1, true), { pos: [-0.37, 0.38, 0], rot: [0, 0, -0.6], w: 1.2 }),
        part(new THREE.SphereGeometry(0.1, 12, 8), { pos: [-0.46, 0.25, 0], w: 1.5 }),
      ],
    },

    chair: {
      view: [0.25, 0.7],
      parts: [
        part(new THREE.BoxGeometry(1, 0.08, 0.95), { w: 0.8 }),
        part(new THREE.BoxGeometry(1, 0.85, 0.07), { pos: [0, 0.5, -0.44], w: 0.7 }),
        tube([0.44, 0, 0.42], [0.46, -0.9, 0.46], 0.04),
        tube([-0.44, 0, 0.42], [-0.46, -0.9, 0.46], 0.04),
        tube([0.44, 0, -0.42], [0.46, -0.9, -0.46], 0.04),
        tube([-0.44, 0, -0.42], [-0.46, -0.9, -0.46], 0.04),
      ],
    },

    headphones: {
      view: [0.1, 0.35],
      parts: [
        part(new THREE.TorusGeometry(0.7, 0.055, 8, 56, Math.PI), { pos: [0, 0.05, 0], w: 2.2 }),
        part(new THREE.CylinderGeometry(0.27, 0.27, 0.2, 32), { pos: [0.74, -0.2, 0], rot: [0, 0, Math.PI / 2] }),
        part(new THREE.CylinderGeometry(0.27, 0.27, 0.2, 32), { pos: [-0.74, -0.2, 0], rot: [0, 0, Math.PI / 2] }),
        part(new THREE.TorusGeometry(0.21, 0.07, 8, 28), { pos: [0.6, -0.2, 0], rot: [0, Math.PI / 2, 0], w: 1.2 }),
        part(new THREE.TorusGeometry(0.21, 0.07, 8, 28), { pos: [-0.6, -0.2, 0], rot: [0, Math.PI / 2, 0], w: 1.2 }),
        tube([0.7, 0.05, 0], [0.74, -0.05, 0], 0.03),
        tube([-0.7, 0.05, 0], [-0.74, -0.05, 0], 0.03),
      ],
    },

    camera: {
      view: [0.18, -0.5],
      parts: [
        part(new THREE.BoxGeometry(1.5, 0.9, 0.5), { w: 0.7 }),
        part(new THREE.CylinderGeometry(0.32, 0.34, 0.42, 36, 1, true), { pos: [0.1, -0.03, 0.46], rot: [Math.PI / 2, 0, 0], w: 1.3 }),
        part(new THREE.TorusGeometry(0.31, 0.04, 8, 36), { pos: [0.1, -0.03, 0.67], w: 1.6 }),
        part(new THREE.CircleGeometry(0.22, 28), { pos: [0.1, -0.03, 0.6], w: 0.8 }),
        part(new THREE.BoxGeometry(0.45, 0.22, 0.4), { pos: [-0.15, 0.56, 0], w: 0.9 }),
        part(new THREE.CylinderGeometry(0.08, 0.08, 0.08, 16), { pos: [0.5, 0.5, 0.05], w: 1.5 }),
        part(new THREE.BoxGeometry(0.25, 0.8, 0.15), { pos: [-0.6, -0.02, 0.3], w: 0.8 }),
      ],
    },

    bicycle: {
      view: [0.1, 0.3],
      parts: (() => {
        const R = [-0.62, -0.32, 0], BB = [0, -0.32, 0], S = [-0.18, 0.3, 0], H = [0.4, 0.3, 0], F = [0.62, -0.32, 0];
        const ps = [
          part(new THREE.TorusGeometry(0.42, 0.035, 8, 64), { pos: R, w: 2.2 }),
          part(new THREE.TorusGeometry(0.42, 0.035, 8, 64), { pos: F, w: 2.2 }),
          tube(BB, S, 0.03), tube(S, H, 0.03), tube(H, BB, 0.03),
          tube(S, R, 0.025), tube(BB, R, 0.025), tube(H, F, 0.028),
          tube(S, [-0.22, 0.44, 0], 0.025), tube(H, [0.38, 0.46, 0], 0.025),
          tube([0.38, 0.46, -0.22], [0.38, 0.46, 0.22], 0.025),
          part(new THREE.BoxGeometry(0.26, 0.05, 0.11), { pos: [-0.24, 0.47, 0], w: 2 }),
          part(new THREE.CylinderGeometry(0.06, 0.06, 0.04, 16), { pos: BB, rot: [Math.PI / 2, 0, 0], w: 2 }),
        ];
        // a few spokes per wheel
        for (const c of [R, F]) {
          for (let k = 0; k < 8; k++) {
            const a = (k / 8) * Math.PI * 2;
            ps.push(tube(c, [c[0] + 0.4 * Math.cos(a), c[1] + 0.4 * Math.sin(a), 0], 0.006, 1.2));
          }
        }
        return ps;
      })(),
    },

    sunglasses: {
      view: [0.15, 0.55],
      parts: [
        part(new THREE.TorusGeometry(0.28, 0.035, 8, 44), { pos: [0.36, 0, 0], scale: [1, 0.78, 1], w: 2.2 }),
        part(new THREE.TorusGeometry(0.28, 0.035, 8, 44), { pos: [-0.36, 0, 0], scale: [1, 0.78, 1], w: 2.2 }),
        part(new THREE.CircleGeometry(0.26, 32), { pos: [0.36, 0, -0.005], scale: [1, 0.78, 1], w: 0.25 }),
        part(new THREE.CircleGeometry(0.26, 32), { pos: [-0.36, 0, -0.005], scale: [1, 0.78, 1], w: 0.25 }),
        part(new THREE.TorusGeometry(0.09, 0.025, 6, 16, Math.PI), { pos: [0, 0.07, 0], w: 2 }),
        tube([0.64, 0.06, 0], [0.7, 0.06, -0.95], 0.025),
        tube([-0.64, 0.06, 0], [-0.7, 0.06, -0.95], 0.025),
        tube([0.7, 0.06, -0.95], [0.68, -0.12, -1.1], 0.025),
        tube([-0.7, 0.06, -0.95], [-0.68, -0.12, -1.1], 0.025),
      ],
    },
  };

  return CATALOG.map((c) => products[c.key]);
}

// Area-weighted surface sampling of a product, normalized to a bounding
// radius of `radius` and rotated to its preferred viewing angle.
export function sampleProduct(THREE, product, n, rand, radius = 1.1) {
  const tris = []; // [ax, ay, az, bx, ..., cz]
  const cum = [];
  let total = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const ab = new THREE.Vector3(), ac = new THREE.Vector3();

  for (const p of product.parts) {
    const geo = p.geo.index ? p.geo.toNonIndexed() : p.geo;
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i += 3) {
      a.fromBufferAttribute(pos, i).applyMatrix4(p.matrix);
      b.fromBufferAttribute(pos, i + 1).applyMatrix4(p.matrix);
      c.fromBufferAttribute(pos, i + 2).applyMatrix4(p.matrix);
      const area = ab.subVectors(b, a).cross(ac.subVectors(c, a)).length() * 0.5;
      if (area <= 0) continue;
      total += area * p.w;
      cum.push(total);
      tris.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    }
  }

  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = rand() * total;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < r) lo = mid + 1; else hi = mid;
    }
    let u = rand(), v = rand();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const t = lo * 9;
    for (let d = 0; d < 3; d++) {
      out[i * 3 + d] = tris[t + d] + u * (tris[t + 3 + d] - tris[t + d]) + v * (tris[t + 6 + d] - tris[t + d]);
    }
  }

  // Center, apply the viewing angle, scale to the target radius.
  const box = new THREE.Box3().setFromArray(out);
  const center = box.getCenter(new THREE.Vector3());
  const rot = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(product.view[0], product.view[1], 0));
  let maxR = 0;
  const v3 = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    v3.fromArray(out, i * 3).sub(center).applyMatrix4(rot).toArray(out, i * 3);
    maxR = Math.max(maxR, v3.length());
  }
  const s = radius / maxR;
  for (let i = 0; i < out.length; i++) out[i] *= s;
  return out;
}
