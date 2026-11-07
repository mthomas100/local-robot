"""perceive.py — F4 groundwork (2026-11-02; not yet wired into the line): classical perception from the
overhead D435i, measured against the simulator's ground truth.

    depth image -> height above the belt -> items = connected blobs higher than 8 mm (scipy.ndimage.label)
    per blob: position = centroid of its top-surface points (back-projected through the camera intrinsics),
              yaw = principal axis of the footprint, top height, footprint length and width, mean colour
    identity: nearest class centre in (top height, length, width, mean colour), and k-nearest-neighbours on (geometry,
              4x4x4 colour histogram), learned from synthetic renders labelled by MuJoCo's segmentation image (the "synthetic labelled data" of the plan, no human labelling)

    envs/mj/bin/python -I factory/vision/perceive.py [--scenes 60] [--seed 0]
        -> factory/vision/results/f4-perception.json (position / yaw error, identity accuracy, time per frame)
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
sys.path.insert(0, os.path.join(FACTORY, "cell"))
import mujoco  # noqa: E402
from scipy import ndimage  # noqa: E402

MIN_H = 0.008; MIN_PIX = 60
BOXES = ("cracker_box", "sugar_box", "pudding_box", "gelatin_box")   # where yaw matters for the grasp (cans are round)


class Camera:
    def __init__(self, cell):
        self.cell = cell; cam = cell.lay.camera; self.cam = cam; self.W, self.H = cam.width, cam.height
        self.f = (self.H / 2) / math.tan(math.radians(cam.fovy) / 2); self.r = mujoco.Renderer(cell.model, self.H, self.W)
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5); self.du = (u - self.W / 2) / self.f; self.dv = (v - self.H / 2) / self.f

    def grab(self):
        r = self.r; d = self.cell.data; r.update_scene(d, camera=self.cam.name)
        rgb = r.render().copy(); r.enable_depth_rendering(); r.update_scene(d, camera=self.cam.name); depth = r.render().copy(); r.disable_depth_rendering()
        r.enable_segmentation_rendering(); r.update_scene(d, camera=self.cam.name); seg = r.render()[..., 0].copy(); r.disable_segmentation_rendering()
        return rgb, depth, seg

    def points(self, depth):
        c = self.cam.pos; x = c[0] + self.du * depth; y = c[1] - self.dv * depth; z = c[2] - depth
        return np.stack([x, y, z], -1)


def blobs(cell, cam, rgb, depth):
    b = cell.lay.belt; P = cam.points(depth); h = P[..., 2] - b.surface_z
    on_belt = (np.abs(P[..., 1] - b.y) < b.width / 2 - 0.005) & (np.abs(P[..., 0]) < 0.69)
    lab, n = ndimage.label((h > MIN_H) & on_belt); out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < MIN_PIX: continue
        hk = h[m]; top = float(np.percentile(hk, 95)); pts = P[m]; sel = hk > top - 0.01
        xy = pts[:, :2]; c = pts[sel, :2].mean(0); X = xy - xy.mean(0)
        w, V = np.linalg.eigh(X.T @ X / len(X)); ax = V[:, 1]; yaw = math.atan2(ax[1], ax[0])
        proj = X @ V; L = float(np.ptp(proj[:, 1])); Wd = float(np.ptp(proj[:, 0]))
        yaw_rect = min_area_yaw(pts[sel, :2]) if sel.sum() >= 5 else yaw
        col = rgb[m].mean(0) / 255.0; q = (rgb[m] // 64).astype(int); hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64) / m.sum()
        out.append(dict(pos=c, yaw=yaw, top=top, length=L, width=Wd, color=col, hist=hist, yaw_rect=yaw_rect, mask=m, pixels=int(m.sum())))
    return out


def min_area_yaw(xy):
    """Yaw of the minimum-area rectangle around the top face's points (rotating calipers over the convex hull edges)."""
    from scipy.spatial import ConvexHull
    try: H = xy[ConvexHull(xy).vertices]
    except Exception: return 0.0
    best = (1e9, 0.0)
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]; th = math.atan2(e[1], e[0]); c, sn = math.cos(th), math.sin(th)
        R = H @ np.array([[c, -sn], [sn, c]]); area = float(np.ptp(R[:, 0]) * np.ptp(R[:, 1]))
        if area < best[0]: best = (area, th)
    return best[1]


def feat(bl): return np.r_[bl["top"] / 0.1, bl["length"] / 0.1, bl["width"] / 0.1, bl["color"] * 2.0]


def feat_hist(bl): return np.r_[3 * bl["top"] / 0.1, 2 * bl["length"] / 0.1, 2 * bl["width"] / 0.1, 4 * np.sqrt(bl["hist"])]   # geometry + a 4x4x4 colour histogram (Hellinger)


def scene(cell, rng, names, k=3):
    cell.reset(); cell.belt(0.0, ramp=0); chosen = list(rng.choice(names, size=k, replace=False)); xs = np.sort(rng.uniform(-0.55, 0.55, k))
    while k > 1 and np.min(np.diff(xs)) < 0.22: xs = np.sort(rng.uniform(-0.55, 0.55, k))
    for n, x in zip(chosen, xs): cell.spawn(n, x=float(x), y=cell.lay.belt.y + float(rng.uniform(-0.08, 0.08)), yaw=float(rng.uniform(-math.pi, math.pi)), with_belt_velocity=False)
    cell.step(250); return chosen


def truth(cell, seg, bl):
    """Which item a blob is (the segmentation id under most of its pixels), and its true pose."""
    m = cell.model; geoms = seg[bl["mask"]]; geoms = geoms[geoms >= 0]
    if not len(geoms): return None
    gid = int(np.bincount(geoms).argmax()); body = m.geom_bodyid[gid]; name = cell.item_of_body.get(body)
    if name is None: return None
    s = cell.item_state(name); R = np.zeros(9); mujoco.mju_quat2Mat(R, s["quat"]); R = R.reshape(3, 3)
    return name, s["pos"][:2], math.atan2(R[1, 0], R[0, 0])


def yaw_err(a, b, sym=math.pi):
    d = (a - b) % sym; return min(d, sym - d)


def main():
    import warnings; warnings.filterwarnings("ignore")
    from cell_mujoco import CellMJ
    p = argparse.ArgumentParser(); p.add_argument("--scenes", type=int, default=60); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    cell = CellMJ(); cam = Camera(cell); names = list(cell.items); rng = np.random.default_rng(a.seed); t0 = time.perf_counter()
    rows = {"train": [], "test": []}
    for split, n_sc in (("train", a.scenes), ("test", a.scenes // 2)):
        for _ in range(n_sc):
            scene(cell, rng, names); tg = time.perf_counter(); rgb, depth, seg = cam.grab(); t_render = time.perf_counter() - tg
            tp = time.perf_counter(); bls = blobs(cell, cam, rgb, depth); t_perc = time.perf_counter() - tp
            for bl in bls:
                tr = truth(cell, seg, bl)
                if tr: rows[split].append(dict(name=tr[0], f=feat(bl), fh=feat_hist(bl), yaw_rect=bl["yaw_rect"], pos_err=float(np.linalg.norm(bl["pos"] - tr[1])), yaw=bl["yaw"], yaw_true=tr[2], t_render=t_render, t_perc=t_perc))
    # identity: nearest class centre (per-feature standardised) from the training scenes
    F = np.array([r["f"] for r in rows["train"]]); mu, sd = F.mean(0), F.std(0) + 1e-6; cls = sorted({r["name"] for r in rows["train"]})
    C = {c: ((np.array([r["f"] for r in rows["train"] if r["name"] == c]) - mu) / sd).mean(0) for c in cls}
    ok = []; confusion = {}
    for r in rows["test"]:
        z = (r["f"] - mu) / sd; guess = min(C, key=lambda c: np.linalg.norm(z - C[c])); ok.append(guess == r["name"])
        if guess != r["name"]: confusion[f"{r['name']} -> {guess}"] = confusion.get(f"{r['name']} -> {guess}", 0) + 1
    # identity, second try: k nearest neighbours (k = 3) on geometry + colour histogram
    Fh = np.array([r["fh"] for r in rows["train"]]); yh = [r["name"] for r in rows["train"]]; ok_h = []
    for r in rows["test"]:
        nn = np.argsort(np.linalg.norm(Fh - r["fh"], axis=1))[:3]; votes = [yh[i] for i in nn]; ok_h.append(max(set(votes), key=votes.count) == r["name"])
    T = rows["test"]; pe = np.array([r["pos_err"] for r in T]) * 1000; ye = np.degrees([yaw_err(r["yaw"], r["yaw_true"], sym=math.pi / 2) for r in T])
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), scenes=dict(train=a.scenes, test=a.scenes // 2), detections=dict(train=len(rows["train"]), test=len(T)),
             position_err_mm=dict(mean=round(float(pe.mean()), 1), p95=round(float(np.percentile(pe, 95)), 1), max=round(float(pe.max()), 1)),
             yaw_err_deg_boxes_mod90=dict(pca=round(float(np.mean([yaw_err(r["yaw"], r["yaw_true"], math.pi / 2) for r in T if r["name"] in BOXES])) * 180 / math.pi, 1),
                                           min_area_rect=round(float(np.mean([yaw_err(r["yaw_rect"], r["yaw_true"], math.pi / 2) for r in T if r["name"] in BOXES])) * 180 / math.pi, 1),
                                           min_area_rect_p95=round(float(np.percentile([yaw_err(r["yaw_rect"], r["yaw_true"], math.pi / 2) for r in T if r["name"] in BOXES], 95)) * 180 / math.pi, 1),
                                           n=sum(r["name"] in BOXES for r in T),
                                           per_item={b: round(float(np.mean([yaw_err(r["yaw_rect"], r["yaw_true"], math.pi / 2) for r in T if r["name"] == b])) * 180 / math.pi, 1)
                                                     for b in BOXES if any(r["name"] == b for r in T)}),
             yaw_err_deg_mod90=dict(mean=round(float(np.mean(ye)), 1), p95=round(float(np.percentile(ye, 95)), 1)),
             identity_accuracy=round(float(np.mean(ok)), 3), confusion=confusion, identity_accuracy_knn_hist=round(float(np.mean(ok_h)), 3),
             ms_per_frame=dict(render_rgb_depth_seg=round(1000 * float(np.mean([r["t_render"] for r in T])), 1), perception=round(1000 * float(np.mean([r["t_perc"] for r in T])), 1)),
             note="items standing apart on a stopped belt (no clutter); yaw error modulo 90 degrees (a box's long axis is ambiguous from PCA on round items)",
             wall_s=round(time.perf_counter() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(S, open(os.path.join(HERE, "results", "f4-perception.json"), "w"), indent=1); print(json.dumps(S))


if __name__ == "__main__":
    main()
