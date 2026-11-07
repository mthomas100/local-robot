"""desk_reach.py — the SO-101's reach map for the desk cell (2026-11-03, F6; discipline 1): where can the tool point
straight down? Samples the three pitch joints within their limits, keeps the configurations whose tool axis is within
2 deg of vertical, and plots (left) their tool heights against horizontal radius, with the heights the desk cell
needs (grasp heights of the admitted items, the carry height); (right) the plan view of the cell with the belt, the
three bins and the annulus the tool can reach top-down at grasp and carry height.

    envs/mj/bin/python -I factory/desk/desk_reach.py   -> docs/media/F6-reach-map.png, results/reach.json
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import desk_ik  # noqa: E402
from desk_layout import DEFAULT as L  # noqa: E402
import desk_items as I  # noqa: E402


def sample(n=400_000, seed=0):
    import so101
    s = so101.arm_spec(); s.body("base").pos = L.base; arm = desk_ik.Arm(model=s.compile()); rng = np.random.default_rng(seed); pts = []
    for _ in range(n):
        q = rng.uniform(arm.lo, arm.hi); q[0] = 0.0; q[4] = 0.0
        p, R = arm.fk(q)
        if R[2, 0] < -math.cos(math.radians(2.0)): pts.append((math.hypot(p[0], p[1]), p[2]))
    return np.array(pts)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pts = sample(); b = L.belt
    zr = np.arange(0.0, 0.40, 0.005); band = {}
    # radius range reachable top-down at each tool height (2.5 mm bins)
    for z in np.arange(0.04, 0.24, 0.0025):
        r = pts[np.abs(pts[:, 1] - z) < 0.00125, 0]
        if len(r) > 5: band[round(float(z), 4)] = (float(np.percentile(r, 0.5)), float(np.percentile(r, 99.5)))
    zmax = max(band); fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    a = ax[0]; a.scatter(pts[:, 0], pts[:, 1], s=0.3, c="#4a7bd0", alpha=0.25, rasterized=True)
    a.axhline(L.base[2], color="#888", lw=0.8, ls=":"); a.text(0.005, L.base[2] + 0.003, "arm base (on a 10 cm riser)", fontsize=8, color="#555")
    a.axhline(b.surface_z, color="#333", lw=1); a.text(0.005, b.surface_z + 0.003, "belt surface", fontsize=8)
    for n in I.admitted():
        inf = I.info(n); top = b.surface_z + inf["height"]; g = top - float(np.clip(0.45 * inf["height"], 0.012, 0.03))
        a.plot([0.17, 0.30], [g, g], color="#c0392b", lw=1.5); a.text(0.302, g + 0.006 * (I.admitted().index(n) % 2), f"grasp: {n.replace('standin_', 'stand-in ')}", fontsize=7, va="center")
    a.axvspan(math.hypot(b.x - b.width / 2, 0), math.hypot(b.x + b.width / 2, b.zone[1]), color="#f5b041", alpha=0.18, lw=0); a.text(0.172, 0.005, "belt (radius)", fontsize=8, color="#a35f00")
    a.set_xlim(0, 0.42); a.set_ylim(-0.0, 0.26); a.set_xlabel("horizontal radius from the pan axis (m)"); a.set_ylabel("tool-frame height above the desk (m)")
    a.set_title(f"top-down reach: the tool points straight down only below {zmax:.3f} m (base + {zmax - L.base[2]:.3f})", fontsize=10)
    a = ax[1]; a.set_aspect("equal")
    a.add_patch(plt.Rectangle((b.x - b.width / 2, b.y_min), b.width, b.y_max - b.y_min, color="#222", alpha=0.85)); a.add_patch(plt.Rectangle((b.x - b.width / 2, b.zone[0]), b.width, b.zone[1] - b.zone[0], fill=False, ec="#f1c40f", lw=1.5))
    col = dict(metal="#9aa0a6", plastic="#3b7dd8", cardboard="#c68642")
    for c in L.bins:
        x, y = c.xy; t = math.radians(c.angle_deg); R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * c.inner / 2 @ R.T + [x, y]; a.add_patch(plt.Polygon(corners, color=col[c.material], alpha=0.8)); a.text(x, y, c.material, ha="center", va="center", fontsize=8)
    a.add_patch(plt.Rectangle((-L.riser[0] / 2, -L.riser[1] / 2), L.riser[0], L.riser[1], color="#555")); a.text(0, 0, "SO-101", ha="center", va="center", color="w", fontsize=7)
    for z, cc, lab in ((0.115, "#c0392b", "grasp height (soup can)"), (zmax - 0.005, "#27ae60", "highest top-down height")):
        k = min(band, key=lambda k: abs(k - z)); r0, r1 = band[k]; th = np.linspace(-math.radians(110), math.radians(110), 200)
        for r in (r0, r1): a.plot(r * np.cos(th), r * np.sin(th), color=cc, lw=1)
        a.plot([], [], color=cc, label=f"{lab}: r {r0:.2f}-{r1:.2f} m")
    a.legend(loc="lower left", fontsize=7); a.set_xlim(-0.2, 0.42); a.set_ylim(-0.34, 0.34); a.set_title("plan view: belt section, L0 zone (yellow), bins, top-down annuli (pan +/-110 deg)", fontsize=10)
    fig.tight_layout(); out = os.path.join(ROOT, "docs", "media", "F6-reach-map.png"); fig.savefig(out, dpi=110)
    json.dump(dict(date="2026-11-03", samples=len(pts), max_topdown_tool_z=zmax, band={str(k): v for k, v in band.items()}), open(os.path.join(HERE, "results", "reach.json"), "w"), indent=0)
    print(out, "max top-down tool z", zmax)


if __name__ == "__main__":
    main()
