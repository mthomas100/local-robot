"""reach_map.py — discipline 1 (cell design and layout): the UR5e + 2F-85 reach envelope by sampling joint space
through MuJoCo forward kinematics, drawn over the belt, the reach zone and the chutes; plus the cycle-time budget
(2026-11-02, docs/sorting-cell-spec.md §3 row 1; acceptance: the zone and the chutes lie inside the envelope).

    envs/mj/bin/python -I factory/cell/reach_map.py [--samples 400000] [--out docs/media/F0-reach-map.png]
Writes the plot and factory/cell/results/reach-map.json (coverage numbers and the budget table).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
from layout import Layout, DEFAULT  # noqa: E402
import build_mjcf  # noqa: E402


def sample_fk(lay: Layout, n, seed=0, topdown_deg=25.0):
    """Pinch-site positions for n joint samples within the UR5e limits. Half the samples are uniform in joint space
    (the full envelope, any orientation); half use the structure of the UR5e wrist to point the gripper straight
    down: wrist_2 = -pi/2 and shoulder_lift + elbow + wrist_1 = -pi/2 (the three parallel pitch axes sum to the tool
    pitch), with pan, lift, elbow and wrist_3 uniform. `down` marks samples whose tool axis is within topdown_deg of
    straight down (checked by FK, not assumed)."""
    s = build_mjcf.robot_spec(); s.body("base").pos = lay.robot_base; m = s.compile(); d = mujoco.MjData(m)
    jids = [m.joint(j).id for j in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")]
    lo = np.array([m.jnt_range[j][0] for j in jids]); hi = np.array([m.jnt_range[j][1] for j in jids])
    qadr = np.array([m.jnt_qposadr[j] for j in jids]); site = m.site("2f85/pinch").id
    rng = np.random.default_rng(seed); n_any = n // 2; n_down = n - n_any
    Q = rng.uniform(lo, hi, size=(n, 6))
    Qd = Q[n_any:]; Qd[:, 4] = -math.pi / 2
    w1 = -math.pi / 2 - Qd[:, 1] - Qd[:, 2]; w1 = (w1 + math.pi) % (2 * math.pi) - math.pi   # wrap into the joint range
    Qd[:, 3] = np.clip(w1, lo[3], hi[3])
    P = np.zeros((n, 3)); Zdir = np.zeros((n, 3))
    for i in range(n):
        d.qpos[qadr] = Q[i]; mujoco.mj_kinematics(m, d)
        P[i] = d.site_xpos[site]; Zdir[i] = d.site_xmat[site].reshape(3, 3)[:, 2]
    down = Zdir[:, 2] < -math.cos(math.radians(topdown_deg))
    return Q, P, down, (lo, hi)


class TopDownIK:
    """Exact reachability for a gripper pointing straight down (2026-11-02). With wrist_2 = -pi/2 and
    shoulder_lift + elbow + wrist_1 = -pi/2 the tool points down, so the pinch position depends only on pan, lift and
    elbow: three joints for three coordinates. Solve them by least squares from a few seeds; a point is reachable if
    the residual is under 1 mm with every joint (wrist_1 included) inside its limits."""

    def __init__(self, lay: Layout):
        s = build_mjcf.robot_spec(); s.body("base").pos = lay.robot_base; self.m = m = s.compile(); self.d = mujoco.MjData(m)
        js = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
        self.qadr = np.array([m.joint(j).qposadr[0] for j in js]); self.site = m.site("2f85/pinch").id
        self.lo = np.array([m.jnt_range[m.joint(j).id][0] for j in js]); self.hi = np.array([m.jnt_range[m.joint(j).id][1] for j in js])

    def full(self, x):
        pan, lift, elb = x; return np.array([pan, lift, elb, -math.pi / 2 - lift - elb, -math.pi / 2, 0.0])

    def fk(self, q):
        self.d.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, self.d); return self.d.site_xpos[self.site].copy()

    def solve(self, target, seeds=None):
        from scipy.optimize import least_squares
        target = np.asarray(target, float); best = None
        pan0 = math.atan2(target[1], target[0]) - math.pi / 2 - 0.2   # the arm faces +y at pan = -pi/2
        for x0 in seeds or ([pan0, -1.6, 1.6], [pan0, -2.2, 2.0], [pan0, -1.0, 1.0], [pan0 + 0.4, -1.6, 1.6]):
            r = least_squares(lambda x: self.fk(self.full(x)) - target, x0, bounds=(self.lo[:3], self.hi[:3]), xtol=1e-10, ftol=1e-10)
            q = self.full(r.x); err = float(np.linalg.norm(r.fun)); ok = err < 1e-3 and np.all(q >= self.lo) and np.all(q <= self.hi)
            if ok: return True, q, err
            if best is None or err < best[2]: best = (False, q, err)
        return best

    def grid_coverage(self, rect, z, cell=0.02):
        x0, x1, y0, y1 = rect; xs = np.arange(x0 + cell / 2, x1, cell); ys = np.arange(y0 + cell / 2, y1, cell)
        ok = np.array([[self.solve((x, y, z))[0] for y in ys] for x in xs])
        return float(ok.mean()), ok, xs, ys


def coverage(P, rect, zlo, zhi, cell=0.02):
    """Fraction of 2 cm grid cells of the rectangle (x0, x1, y0, y1) that contain at least one reachable point with
    z in [zlo, zhi]."""
    x0, x1, y0, y1 = rect; sel = (P[:, 2] >= zlo) & (P[:, 2] <= zhi) & (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    nx, ny = max(1, int(round((x1 - x0) / cell))), max(1, int(round((y1 - y0) / cell)))
    if not sel.any(): return 0.0, np.zeros((nx, ny), bool)
    ix = np.clip(((P[sel, 0] - x0) / cell).astype(int), 0, nx - 1); iy = np.clip(((P[sel, 1] - y0) / cell).astype(int), 0, ny - 1)
    hit = np.zeros((nx, ny), bool); hit[ix, iy] = True
    return float(hit.mean()), hit


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--samples", type=int, default=400000); p.add_argument("--out", default=os.path.join(ROOT, "docs", "media", "F0-reach-map.png"))
    a = p.parse_args(); lay = DEFAULT; b = lay.belt
    t0 = time.perf_counter(); Q, P, down, (lo, hi) = sample_fk(lay, a.samples); t_fk = time.perf_counter() - t0
    Pd = P[down]
    pick_z = (b.surface_z + 0.01, b.surface_z + 0.30)                      # pinch heights that reach an item on the belt
    zone_rect = (b.zone[0], b.zone[1], b.y - b.width / 2, b.y + b.width / 2)
    belt_rect = (b.x_min, b.x_max, b.y - b.width / 2, b.y + b.width / 2)
    zone_cov, zone_hit = coverage(Pd, zone_rect, *pick_z)
    belt_cov, _ = coverage(Pd, belt_rect, *pick_z)
    chute_cov = {}
    for c in lay.chutes:
        h = c.inner / 2; chute_cov[c.material], _ = coverage(Pd, (c.x - h, c.x + h, c.y - h, c.y + h), c.rim_z + 0.03, c.rim_z + 0.35)
    ik = TopDownIK(lay); t1 = time.perf_counter(); ik_cov = {}
    for zname, z in (("grasp_low", b.surface_z + 0.10), ("grasp_high", b.surface_z + 0.25)):   # pinch heights for a short and a tall item
        ik_cov[f"zone_{zname}"], _, _, _ = ik.grid_coverage(zone_rect, z)
    for c in lay.chutes:
        h = c.inner / 2; ik_cov[f"chute_{c.material}"], _, _, _ = ik.grid_coverage((c.x - h, c.x + h, c.y - h, c.y + h), c.rim_z + 0.25)
    t_ik = time.perf_counter() - t1
    r_max = float(np.linalg.norm(P[:, :2] - np.array(lay.robot_base[:2]), axis=1).max())
    budget = lay.cycle_budget()
    res = dict(samples=a.samples, fk_seconds=round(t_fk, 1), topdown_fraction=round(float(down.mean()), 4), pick_z=pick_z,
               zone_coverage=round(zone_cov, 4), belt_coverage=round(belt_cov, 4), chute_coverage={k: round(v, 4) for k, v in chute_cov.items()},
               max_horizontal_reach_m=round(r_max, 3), cycle_budget=budget,
               ik_coverage={k: round(v, 4) for k, v in ik_cov.items()}, ik_seconds=round(t_ik, 1),
               ik_note="exact top-down IK on a 2 cm grid: zone at pinch heights surface+0.10 and +0.25 m, chute openings at rim+0.25 m",
               layout=json.loads(lay.to_json()))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "reach-map.json"), "w") as f: json.dump(res, f, indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle, Circle
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5), gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.75]))
    # (a) top view: every top-down pinch position at belt-pick height, the cell drawn over it
    sel = (Pd[:, 2] >= pick_z[0]) & (Pd[:, 2] <= pick_z[1])
    ax[0].hexbin(Pd[sel, 0], Pd[sel, 1], gridsize=70, cmap="Blues", mincnt=1, linewidths=0.1)
    ax[0].add_patch(Rectangle((b.x_min, b.y - b.width / 2), b.x_max - b.x_min, b.width, fill=False, ec="k", lw=1.5, label="belt"))
    ax[0].add_patch(Rectangle((b.zone[0], b.y - b.width / 2), b.zone[1] - b.zone[0], b.width, fill=True, fc=(1, 0.85, 0.1, 0.25), ec="orange", lw=2, label=f"reach zone ({zone_cov*100:.0f}% covered)"))
    for c in lay.chutes:
        h = c.inner / 2; ax[0].add_patch(Rectangle((c.x - h, c.y - h), c.inner, c.inner, fill=True, fc=(0.2, 0.7, 0.3, 0.3), ec="green", lw=1.5))
        ax[0].text(c.x, c.y, f"{c.material}\n{chute_cov[c.material]*100:.0f}%", ha="center", va="center", fontsize=8)
    ax[0].add_patch(Rectangle((-lay.pedestal[0] / 2, -lay.pedestal[1] / 2), lay.pedestal[0], lay.pedestal[1], fc="0.4", ec="k"))
    ax[0].add_patch(Circle(lay.robot_base[:2], r_max, fill=False, ls="--", ec="0.5", label=f"max horizontal reach {r_max:.2f} m"))
    ax[0].set_aspect("equal"); ax[0].set_xlim(-1.4, 1.4); ax[0].set_ylim(-1.1, 1.3); ax[0].set_xlabel("x (m, belt direction)"); ax[0].set_ylabel("y (m)")
    ax[0].set_title(f"Top view: pinch points reachable with the gripper pointing down\n(z between {pick_z[0]:.2f} and {pick_z[1]:.2f} m; {sel.sum():,} of {a.samples:,} samples)")
    ax[0].legend(loc="lower left", fontsize=8)
    # (b) side view along the belt: the slice over the belt centreline
    slab = np.abs(P[:, 1] - b.y) < 0.06
    ax[1].hexbin(P[slab, 0], P[slab, 2], gridsize=60, cmap="Greys", mincnt=1, linewidths=0.1, label="any orientation")
    slab_d = np.abs(Pd[:, 1] - b.y) < 0.06
    ax[1].hexbin(Pd[slab_d, 0], Pd[slab_d, 2], gridsize=60, cmap="Blues", mincnt=1, linewidths=0.1, alpha=0.9)
    ax[1].add_patch(Rectangle((b.x_min, b.surface_z - b.plate_thickness), b.x_max - b.x_min, b.plate_thickness, fc="k"))
    ax[1].add_patch(Rectangle((b.zone[0], b.surface_z), b.zone[1] - b.zone[0], 0.3, fc=(1, 0.85, 0.1, 0.25), ec="orange", lw=2))
    ax[1].add_patch(Rectangle((-lay.pedestal[0] / 2, 0), lay.pedestal[0], lay.pedestal[2], fc="0.4", ec="k"))
    ax[1].set_aspect("equal"); ax[1].set_xlim(-1.4, 1.4); ax[1].set_ylim(0, 1.9); ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("z (m)")
    ax[1].set_title(f"Side view over the belt centreline (y = {b.y} m)\ngrey: any orientation, blue: gripper down; yellow: items on the belt")
    # (c) the budget table
    ax[2].axis("off")
    rows = [[f"{v:.2f}", f"{budget[v]['zone_seconds']:.2f}", f"{budget[v]['items_per_minute_at_0_3m_spacing']:.0f}"] for v in b.speeds]
    tb = ax[2].table(cellText=rows, colLabels=["belt m/s", "s in zone", "items/min\n(0.3 m apart)"], loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.8)
    ax[2].set_title(f"Cycle-time budget ({b.zone[1] - b.zone[0]:.1f} m zone)\nexact top-down IK: zone {ik_cov['zone_grasp_low']*100:.0f}% (low grasp), {ik_cov['zone_grasp_high']*100:.0f}% (high)", fontsize=11)
    ax[2].text(0.0, 0.42, f"UR5e + 2F-85 on a {lay.pedestal[2]:.2f} m pedestal\nbelt surface at z = {b.surface_z} m, {b.width} m wide\n"
               f"{a.samples:,} joint samples ({a.samples//2:,} uniform, {a.samples - a.samples//2:,} wrist-down), {t_fk:.0f} s of FK\n{down.mean()*100:.1f}% point down (within 25 deg)\nsampled coverage: zone {zone_cov*100:.0f}%, chutes " +
               ", ".join(f"{k} {v*100:.0f}%" for k, v in chute_cov.items()) + "\nexact IK (2 cm grid): chutes " +
               ", ".join(f"{c.material} {ik_cov['chute_' + c.material]*100:.0f}%" for c in lay.chutes), fontsize=9, va="top", transform=ax[2].transAxes)
    fig.suptitle("F0 reach map: the sorting cell inside the UR5e envelope (local-robot, 2026-11-02)", fontsize=13)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out, dpi=110)
    print(json.dumps({k: v for k, v in res.items() if k != "layout"}))
    print(a.out)


if __name__ == "__main__":
    main()
