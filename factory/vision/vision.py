"""vision.py — F4: the line sees for itself (2026-11-02; disciplines 7, 8; docs/sorting-cell-spec.md §4 F4).

F2 and F3 were told where every item is (tracking.Detector samples the simulator's poses and adds noise) and what it is
(the report carries its name). L4 takes both away. `VisionDetector` has the same interface (`poll(t, names) ->
reports`), but it only looks at the overhead D435i:

    every 1/30 s (sim time):  render RGB + depth at 424 x 240 (a D435i depth mode) and the robot's own silhouette
       robot self-filter:     pixels where the camera sees the arm or gripper are blanked out. The robot's pose is known
                              from its joint encoders, so a real cell renders the same mask from its URDF (ROS
                              robot_self_filter); here MuJoCo's segmentation image provides it, and it is used for
                              nothing else (items behind the arm are simply not seen)
       segmentation:          height above the belt plane > 8 mm, inside the belt's edges -> connected blobs
       per blob:              3-D points (back-projected through the intrinsics), the top face (points within 1 cm of
                              the highest), its centroid, the minimum-area rectangle of the top face and of the footprint,
                              a 4x4x4 colour histogram
       identity:              3 nearest neighbours on (geometry, colour histogram), learned from synthetic renders
                              labelled by the segmentation image (`build_identity`; no human labelling)
       association:           each blob joins the nearest vision track predicted along the belt (gate 6 cm); a track
                              votes on its identity over frames, and reports once 3 frames agree
       pose:                  position = the top face's centre, corrected by the class's known top-face offset;
                              yaw = the rectangle side that matches the class's footprint (mod 180 deg), smoothed
       tipped:                top height far from the class's standing height -> tilt reported as 90 deg
    each report reaches the tracker `latency_s` after the frame was taken (exposure + transfer + this processing)

    from vision import VisionDetector, VisionCfg       # Line(..., perception="vision")
    envs/mj/bin/python -I factory/vision/vision.py build      # -> factory/vision/identity.npz (synthetic labelled set)
    envs/mj/bin/python -I factory/vision/vision.py measure    # -> factory/vision/results/f4-vision-static.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from dataclasses import dataclass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
for p in (HERE, os.path.join(FACTORY, "cell")):
    if p not in sys.path: sys.path.insert(0, p)
import mujoco  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

IDENTITY = os.path.join(HERE, "identity.npz")
ROUND = ("master_chef_can", "tomato_soup_can", "tuna_fish_can")   # yaw does not matter for these
MIN_H = 0.008


@dataclass
class VisionCfg:
    rate_hz: float = 30.0
    latency_s: float = 0.08          # exposure + USB transfer + render-and-perceive (~25 ms measured here) + margin
    width: int = 424
    height: int = 240
    gate_m: float = 0.06
    min_votes: int = 3
    x_view: tuple = (-0.69, 0.69)
    split: bool = True               # L3: split blobs that look like no single item at their top surfaces


def robot_geom_mask(model):
    """True for every geom on the robot (the bodies below 'base': arm, gripper, cup, wrist camera mount)."""
    base = model.body("base").id; robot = np.zeros(model.nbody, bool)
    for b in range(model.nbody):
        a = b
        while a > 0 and a != base: a = model.body_parentid[a]
        robot[b] = a == base
    return robot[model.geom_bodyid]


class Cam:
    def __init__(self, cell, width=424, height=240):
        c = cell.lay.camera; self.cell = cell; self.cam = c; self.W, self.H = width, height
        self.f = (self.H / 2) / math.tan(math.radians(c.fovy) / 2); self.r = mujoco.Renderer(cell.model, self.H, self.W)
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5); self.du = (u - self.W / 2) / self.f; self.dv = (v - self.H / 2) / self.f
        self.robot = robot_geom_mask(cell.model); self.min_pix = max(30, int(60 * (width * height) / (848 * 480)))
        # background: an empty-cell frame marks the static fixtures standing above the belt plane (zone markers, rails)
        items = np.zeros(cell.model.nbody, bool); items[list(cell.item_of_body)] = True; self.item_geom = items[cell.model.geom_bodyid]
        _, depth, seg = self.grab(); h = self.points(depth)[..., 2] - cell.lay.belt.surface_z; v = seg >= 0; fixed = np.zeros_like(v)
        fixed[v] = ~self.robot[seg[v]] & ~self.item_geom[seg[v]]; self.fixture = ndimage.binary_dilation(fixed & (h > MIN_H), iterations=2)

    def grab(self, seg_too=True):
        # the zone markers are a visual overlay for people watching (a bar floating 3 cm over the belt that items pass
        # through), not objects a depth camera would see: hidden while the camera renders (alpha 0 is not drawn)
        m = self.cell.model; ov = [m.geom(n).id for n in ("zone_in", "zone_out")]; a0 = m.geom_rgba[ov, 3].copy(); m.geom_rgba[ov, 3] = 0.0
        try: return self._grab(seg_too)
        finally: m.geom_rgba[ov, 3] = a0

    def _grab(self, seg_too):
        r = self.r; d = self.cell.data; r.update_scene(d, camera=self.cam.name); rgb = r.render().copy()
        r.enable_depth_rendering(); r.update_scene(d, camera=self.cam.name); depth = r.render().copy(); r.disable_depth_rendering()
        seg = None
        if seg_too:
            r.enable_segmentation_rendering(); r.update_scene(d, camera=self.cam.name); seg = r.render()[..., 0].copy(); r.disable_segmentation_rendering()
        return rgb, depth, seg

    def points(self, depth):
        c = self.cam.pos; return np.stack([c[0] + self.du * depth, c[1] - self.dv * depth, c[2] - depth], -1)


def rect(xy):
    """Minimum-area rectangle (rotating calipers): (centre xy, edge angle, length along it, length across)."""
    try: H = xy[ConvexHull(xy).vertices]
    except Exception: return xy.mean(0), 0.0, 0.0, 0.0
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]; th = math.atan2(e[1], e[0]); c, s = math.cos(th), math.sin(th)
        R = H @ np.array([[c, -s], [s, c]]); a, b = float(np.ptp(R[:, 0])), float(np.ptp(R[:, 1]))
        if best is None or a * b < best[0]:
            mid = np.array([(R[:, 0].max() + R[:, 0].min()) / 2, (R[:, 1].max() + R[:, 1].min()) / 2])
            best = (a * b, np.array([[c, -s], [s, c]]) @ mid, th, a, b)
    return best[1], best[2], best[3], best[4]


def describe(P, h, m, rgb, near_robot, partial_extra=False):
    """One blob's measurements: 3-D points, top face, rectangles, colour histogram, whether it is a partial view."""
    xs = P[..., 0][m]; partial = bool(partial_extra or xs.min() < -0.67 or xs.max() > 0.67 or m[:, :3].any() or m[:, -3:].any() or m[:3].any() or m[-3:].any() or (near_robot & m).any())
    hk = h[m]; top = float(np.percentile(hk, 95)); pts = P[m]; sel = hk > top - 0.01
    c_top = pts[sel, :2].mean(0)
    tr = rect(pts[sel, :2]) if sel.sum() >= 5 else (c_top, 0.0, 0.0, 0.0); fr = rect(pts[:, :2])
    q = (rgb[m] // 64).astype(int); hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64) / m.sum()
    return dict(c_top=c_top, top=top, top_rect=tr, foot_rect=fr, hist=hist, mask=m, pixels=int(m.sum()), partial=partial,
                length=max(fr[2], fr[3]), width=min(fr[2], fr[3]))


def split_blob(h, m, min_marker=10):
    """L3 (touching items): split a blob at its separate top surfaces. Markers = connected pixels within 6 mm of the
    highest point in a 9 x 9 window (each item's top face, or a bottle's cap); every other pixel of the blob goes to
    the marker it drains to on the height map (watershed). Returns a list of masks (one if nothing to split)."""
    hm = np.where(m, h, 0.0); top = ndimage.maximum_filter(hm, size=9); plateau = m & (hm >= top - 0.006)
    mk, n = ndimage.label(ndimage.binary_erosion(plateau, iterations=1))
    keep = [k for k in range(1, n + 1) if (mk == k).sum() >= min_marker]
    if len(keep) < 2: return [m]
    markers = np.zeros(h.shape, np.int16); markers[~m] = -1
    for j, k in enumerate(keep): markers[mk == k] = j + 1
    cost = np.where(m, np.clip((hm.max() - hm) / max(hm.max(), 1e-3) * 1000, 0, 1000), 1000).astype(np.uint16)
    seg = ndimage.watershed_ift(cost, markers)
    return [seg == j + 1 for j in range(len(keep)) if (seg == j + 1).sum() >= 30]


def blobs(cam, rgb, depth, seg=None, idm=None):
    """Blobs on the belt. With an identity model, a blob unlike any single item (its nearest training neighbour
    farther than the model's 99th-percentile single-item distance) is split at its top surfaces (L3 clutter), and
    the split is kept if every part looks like an item."""
    cell = cam.cell; b = cell.lay.belt; P = cam.points(depth); h = P[..., 2] - b.surface_z
    ok = (np.abs(P[..., 1] - b.y) < b.width / 2 - 0.005) & (np.abs(P[..., 0]) < 0.69) & (h > MIN_H) & ~cam.fixture
    robot = np.zeros_like(ok)
    if seg is not None:
        g = seg; v = g >= 0; robot[v] = cam.robot[g[v]]; robot = ndimage.binary_dilation(robot, iterations=2); ok &= ~robot
    near_robot = ndimage.binary_dilation(robot, iterations=2)
    lab, n = ndimage.label(ok); out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < cam.min_pix: continue
        bl = describe(P, h, m, rgb, near_robot)
        if idm is not None and idm.nn_dist(feat(bl)) > idm.tau:
            parts = split_blob(h, m)
            if len(parts) > 1:
                pb = [describe(P, h, pm, rgb, near_robot) for pm in parts]
                if max(idm.nn_dist(feat(x)) for x in pb) < idm.nn_dist(feat(bl)):
                    for x in pb: x["split"] = True
                    out.extend(pb); continue
            bl["unknown"] = True   # looks like no single item and does not split into items: tracked, not reported
        out.append(bl)
    return out


def feat(bl): return np.r_[3 * bl["top"] / 0.1, 2 * bl["length"] / 0.1, 2 * bl["width"] / 0.1, 4 * np.sqrt(bl["hist"])]


def wrap_pi(a): return (a + math.pi / 2) % math.pi - math.pi / 2       # to [-90, 90) deg: a box's yaw is mod 180


def axis_of(rec, bl, src="foot"):
    """The item's x axis (mod 180 deg) from a rectangle: the side whose length matches the item's x extent."""
    ex, ey = rec["extents"][0], rec["extents"][1]
    c, th, a, b = bl["top_rect"] if (src == "top" and bl["top_rect"][2] > 0.02) else bl["foot_rect"]
    along = th if a >= b else th + math.pi / 2          # the rectangle's long side
    return wrap_pi(along if ex >= ey else along + math.pi / 2)


def pose_of(name, rec, bl, pm):
    """(xy of the item's origin, yaw, whether the 180-degree flip was resolved) from one blob and the class's pose model
    pm = {src: which rectangle gives the axis, sign: where the top face sits along the item's x axis relative to the
    footprint's centre (resolves the flip of asymmetric items: a sprayer head, a nozzle), offset: the origin relative
    to the footprint's centre in the item frame}."""
    if name in ROUND: return bl["foot_rect"][0], 0.0, True
    a = axis_of(rec, bl, pm["src"]); fc = bl["foot_rect"][0]; resolved = False
    if abs(pm["sign"]) > 0.006:
        d = float(np.dot(bl["c_top"] - fc, [math.cos(a), math.sin(a)]))
        if abs(d) > 0.003: resolved = True; a = a if d * pm["sign"] > 0 else a + math.pi
    c, s = math.cos(a), math.sin(a); o = pm["offset"]
    return fc + np.array([c * o[0] - s * o[1], s * o[0] + c * o[1]]), a, resolved


class IdentityModel:
    def __init__(self, path=IDENTITY):
        if not os.path.exists(path):   # not in git (a learned artefact): rebuilt from synthetic renders in ~25 s
            build()
        z = np.load(path, allow_pickle=True); self.F = z["F"]; self.y = list(z["y"]); self.pose = z["pose"].item(); self.top_h = z["top_h"].item()

    @property
    def tau(self):
        """The 99th percentile of a single item's distance to its nearest other training blob (leave one out)."""
        if not hasattr(self, "_tau"):
            D = np.linalg.norm(self.F[:, None, :] - self.F[None, :, :], axis=2); np.fill_diagonal(D, np.inf); self._tau = float(np.percentile(D.min(1), 99))
        return self._tau

    def nn_dist(self, f): return float(np.min(np.linalg.norm(self.F - f, axis=1)))

    def predict(self, f, k=3):
        nn = np.argsort(np.linalg.norm(self.F - f, axis=1))[:k]; votes = [self.y[i] for i in nn]
        return max(set(votes), key=votes.count)


def item_pose(cell, name):
    s = cell.item_state(name); R = np.zeros(9); mujoco.mju_quat2Mat(R, s["quat"]); R = R.reshape(3, 3)
    return s["pos"], math.atan2(R[1, 0], R[0, 0]), s["tilt_deg"]


def truth(cell, seg, bl):
    geoms = seg[bl["mask"]]; geoms = geoms[geoms >= 0]
    if not len(geoms): return None
    name = cell.item_of_body.get(cell.model.geom_bodyid[int(np.bincount(geoms).argmax())])
    return name


def scene(cell, rng, names, k=3, belt_speed=0.0):
    cell.reset(); cell.belt(belt_speed, ramp=0); chosen = list(rng.choice(names, size=k, replace=False)); xs = np.sort(rng.uniform(-0.58, 0.58, k))
    while k > 1 and np.min(np.diff(xs)) < 0.22: xs = np.sort(rng.uniform(-0.58, 0.58, k))
    for n, x in zip(chosen, xs): cell.spawn(n, x=float(x), y=cell.lay.belt.y + float(rng.uniform(-0.08, 0.08)), yaw=float(rng.uniform(-math.pi, math.pi)), with_belt_velocity=False)
    cell.step(250); return chosen


def collect(cell, cam, rng, n_scenes):
    rows = []; names = list(cell.items)
    for _ in range(n_scenes):
        scene(cell, rng, names); rgb, depth, seg = cam.grab()
        for bl in blobs(cam, rgb, depth, seg):
            n = truth(cell, seg, bl)
            if n is None: continue
            p, yaw, tilt = item_pose(cell, n); rows.append(dict(name=n, f=feat(bl), bl={k: v for k, v in bl.items() if k != "mask"}, pos=p, yaw=yaw, tilt=tilt))
    return rows


def build(n_scenes=240, seed=0, cfg=VisionCfg()):
    """The synthetic labelled set: features of every blob in n_scenes random scenes, labelled by the segmentation image,
    plus per class the top face's offset from the item's origin (item frame) and its standing top height."""
    import warnings; warnings.filterwarnings("ignore")
    from cell_mujoco import CellMJ
    cell = CellMJ(); cam = Cam(cell, cfg.width, cfg.height); rows = [r for r in collect(cell, cam, np.random.default_rng(seed), n_scenes) if r["tilt"] < 10]
    pose, top_h = {}, {}
    ang = lambda a, b, sym: min((a - b) % sym, sym - (a - b) % sym)
    for n in cell.items:
        R = [r for r in rows if r["name"] == n]
        if not R: continue
        rec = cell.items[n]["rec"]; top_h[n] = float(np.median([r["bl"]["top"] for r in R]))
        if n in ROUND: pose[n] = dict(src="foot", sign=0.0, offset=np.zeros(2)); continue
        err = {src: np.mean([ang(axis_of(rec, r["bl"], src), r["yaw"], math.pi) for r in R]) for src in ("top", "foot")}
        src = min(err, key=err.get)
        sign = float(np.mean([np.dot(r["bl"]["c_top"] - r["bl"]["foot_rect"][0], [math.cos(r["yaw"]), math.sin(r["yaw"])]) for r in R]))
        pm = dict(src=src, sign=sign, offset=np.zeros(2)); d = []
        for r in R:   # origin - footprint centre, in the frame of the *estimated* yaw (what the detector will have)
            _, y, _ = pose_of(n, rec, r["bl"], pm); c, s_ = math.cos(y), math.sin(y); w = r["pos"][:2] - r["bl"]["foot_rect"][0]; d.append([c * w[0] + s_ * w[1], -s_ * w[0] + c * w[1]])
        pm["offset"] = np.array(d).mean(0); pm["axis_err_deg"] = {k: round(math.degrees(v), 1) for k, v in err.items()}; pose[n] = pm
    np.savez(IDENTITY, F=np.array([r["f"] for r in rows]), y=np.array([r["name"] for r in rows]), pose=pose, top_h=top_h,
             meta=json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M"), scenes=n_scenes, seed=seed, rows=len(rows), res=[cfg.width, cfg.height])))
    return len(rows)


class VisionDetector:
    """Drop-in for tracking.Detector: poll(t, names) -> the reports arriving at t ({name, t, pos, yaw, tilt}). `names`
    is ignored (vision does not know what is on the belt). The belt encoder's speed predicts tracks between frames."""

    def __init__(self, cell, cfg: VisionCfg = None, rng=None, model: IdentityModel = None, belt_speed=0.0, cam=None):
        self.cell = cell; self.cfg = cfg or VisionCfg(); self.cam = cam or Cam(cell, self.cfg.width, self.cfg.height)
        self.id = model or IdentityModel(); self.v = float(belt_speed); self.next_t = 0.0; self.queue = []
        self.tracks = []; self.next_id = 0; self.frames = 0; self.ms = []; self.keep_view = False; self.view = None

    def _frame(self, t):
        t0 = time.perf_counter(); rgb, depth, seg = self.cam.grab(); bls = blobs(self.cam, rgb, depth, seg, self.id if self.cfg.split else None); dets = []
        for bl in bls:
            if not (self.cfg.x_view[0] <= bl["c_top"][0] <= self.cfg.x_view[1]) or bl["partial"] or bl.get("unknown"): continue
            dets.append(dict(c=bl["c_top"], top=bl["top"], bl=bl, label=self.id.predict(feat(bl))))
        # association: predict each track along the belt, greedy nearest within the gate
        for tr in self.tracks: tr["pred"] = tr["c"] + np.array([self.v * (t - tr["t"]), 0.0])
        pairs = sorted(((float(np.linalg.norm(d["c"] - tr["pred"])), i, j) for i, d in enumerate(dets) for j, tr in enumerate(self.tracks)), key=lambda x: x[0])
        used_d, used_t = set(), set()
        for dist, i, j in pairs:
            if dist > self.cfg.gate_m or i in used_d or j in used_t: continue
            used_d.add(i); used_t.add(j); self._update(self.tracks[j], dets[i], t)
        for i, d in enumerate(dets):
            if i not in used_d: self.tracks.append(self._new(d, t))
        self.tracks = [tr for tr in self.tracks if t - tr["t"] < 0.5]
        # one report per identity: if two tracks claim the same label, the one with the larger share speaks
        best = {}
        for tr in self.tracks:
            if tr["t"] != t: continue
            lab, nv = max(tr["votes"].items(), key=lambda kv: kv[1])
            if nv < self.cfg.min_votes: continue
            share = nv / sum(tr["votes"].values())
            if lab not in best or share > best[lab][0]: best[lab] = (share, tr)
        reps = []
        for lab, (share, tr) in best.items():
            rec = self.cell.items[lab]["rec"]; xy, y = self._pose(tr, lab, rec)
            tipped = abs(tr["top"] - self.id.top_h.get(lab, tr["top"])) > max(0.03, 0.25 * self.id.top_h.get(lab, 0.1))
            reps.append(dict(name=lab, t=t, pos=np.array([xy[0], xy[1], self.cell.lay.belt.surface_z + tr["top"]]), yaw=y, tilt=90.0 if tipped else 0.0, track=tr["id"], share=round(share, 2)))
        self.frames += 1; self.ms.append(1000 * (time.perf_counter() - t0))
        if self.keep_view:   # for the live view and the videos: what the camera saw and made of it
            box = lambda m: (int(np.flatnonzero(m.any(0))[0]), int(np.flatnonzero(m.any(1))[0]), int(np.flatnonzero(m.any(0))[-1]), int(np.flatnonzero(m.any(1))[-1]))
            said = {r["track"]: r for r in reps}
            self.view = dict(t=t, rgb=rgb, blobs=[dict(box=box(bl["mask"]), partial=bl["partial"]) for bl in bls],
                             tracks=[dict(box=box(tr["bl"]["mask"]), id=tr["id"], label=said[tr["id"]]["name"] if tr["id"] in said else None) for tr in self.tracks if tr["t"] == t])
        return reps

    def _new(self, d, t):
        self.next_id += 1; return dict(id=self.next_id, c=d["c"], t=t, top=d["top"], votes={d["label"]: 1}, yaws=[], bl=d["bl"])

    def _update(self, tr, d, t):
        tr.update(c=d["c"], t=t, top=0.7 * tr["top"] + 0.3 * d["top"], bl=d["bl"]); tr["votes"][d["label"]] = tr["votes"].get(d["label"], 0) + 1

    def _pose(self, tr, lab, rec):
        """Pose from the newest blob; the yaw is averaged over the track's last 8 frames (circular mean when the flip is
        resolved, axial mean otherwise) and kept on the side of 180 degrees nearest the previous report."""
        xy, a, resolved = pose_of(lab, rec, tr["bl"], self.id.pose[lab]); tr["yaws"] = (tr["yaws"] + [(a, resolved)])[-8:]
        if all(r for _, r in tr["yaws"]):
            A = np.array([y for y, _ in tr["yaws"]]); y = math.atan2(np.sin(A).mean(), np.cos(A).mean())
        else:
            A = 2 * np.array([y for y, _ in tr["yaws"]]); y = 0.5 * math.atan2(np.sin(A).mean(), np.cos(A).mean())
            prev = tr.get("y_out")
            if prev is not None and abs((y - prev + math.pi) % (2 * math.pi) - math.pi) > math.pi / 2: y += math.pi
        y = (y + math.pi) % (2 * math.pi) - math.pi; tr["y_out"] = y; return xy, y

    def poll(self, t, names=None):
        if t >= self.next_t:
            self.next_t = t + 1.0 / self.cfg.rate_hz
            for rep in self._frame(t): self.queue.append((t + self.cfg.latency_s, rep))
        out = [r for (ta, r) in self.queue if ta <= t]; self.queue = [(ta, r) for (ta, r) in self.queue if ta > t]; return out


def overlay(view, scale=2):
    """The camera's frame with what perception made of it: every blob boxed (grey: partial, not trusted), every
    reporting track labelled with its identity and track number."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(view["rgb"]).resize((view["rgb"].shape[1] * scale, view["rgb"].shape[0] * scale)); d = ImageDraw.Draw(im)
    for b in view["blobs"]:
        if b["partial"]: d.rectangle([v * scale for v in b["box"]], outline=(150, 150, 150), width=1)
    for tr in view["tracks"]:
        x0, y0, x1, y1 = [v * scale for v in tr["box"]]; col = (40, 230, 90) if tr["label"] else (250, 200, 40)
        d.rectangle((x0, y0, x1, y1), outline=col, width=2); txt = f"#{tr['id']} {tr['label'] or '?'}"
        d.rectangle((x0, max(0, y0 - 14), x0 + 7 * len(txt), max(0, y0 - 14) + 13), fill=(0, 0, 0)); d.text((x0 + 2, max(0, y0 - 14)), txt, fill=col)
    d.rectangle((0, 0, 190, 16), fill=(0, 0, 0)); d.text((4, 2), f"overhead D435i 424x240  t {view['t']:.2f} s", fill=(255, 255, 255))
    return im


def yaw_err(a, b, sym=math.pi):
    d = (a - b) % sym; return min(d, sym - d)


def measure(n_scenes=60, seed=7, belt_speed=0.3, cfg=VisionCfg()):
    """Static scenes (a fresh seed, not the training one) and a moving belt: detector reports vs the true poses."""
    import warnings; warnings.filterwarnings("ignore")
    from cell_mujoco import CellMJ
    cell = CellMJ(); rng = np.random.default_rng(seed); names = list(cell.items); det = VisionDetector(cell, cfg, belt_speed=belt_speed)
    rows = []
    for _ in range(n_scenes):
        scene(cell, rng, names, belt_speed=0.0); cell.belt(belt_speed, ramp=0)
        det.tracks = []; det.queue = []; det.next_t = 0.0; t_end = cell.data.time + 0.5
        while cell.data.time < t_end:
            t = cell.data.time
            for rep in det.poll(t):
                n = rep["name"]; truth_n = min(names, key=lambda m: np.linalg.norm(cell.item_state(m)["pos"][:2] - rep["pos"][:2]))
                p, yaw, tilt = item_pose(cell, truth_n)
                # the report is latency old: compare with where the item was when the frame was taken
                p_then = p[:2] - np.array([belt_speed * (t - rep["t"]), 0.0])
                rows.append(dict(name=truth_n, label=n, pos_err=float(np.linalg.norm(rep["pos"][:2] - p_then)), yaw_err=yaw_err(rep["yaw"], yaw), yaw_err360=yaw_err(rep["yaw"], yaw, 2 * math.pi), tilt=tilt))
            cell.step(1)
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), scenes=n_scenes, seed=seed, belt_speed=belt_speed, reports=len(rows), resolution=[cfg.width, cfg.height],
               identity_accuracy=round(float(np.mean([r["name"] == r["label"] for r in rows])), 4),
               position_err_mm=dict(mean=round(1000 * float(np.mean([r["pos_err"] for r in rows])), 1), p95=round(1000 * float(np.percentile([r["pos_err"] for r in rows], 95)), 1)),
               per_item={n: dict(pos_mm=round(1000 * float(np.mean([r["pos_err"] for r in rows if r["name"] == n])), 1),
                                 yaw_deg=round(math.degrees(float(np.mean([r["yaw_err"] for r in rows if r["name"] == n]))), 1) if n not in ROUND else None,
                                 yaw360_deg=round(math.degrees(float(np.mean([r["yaw_err360"] for r in rows if r["name"] == n]))), 1) if n not in ROUND else None,
                                 identity=round(float(np.mean([r["label"] == n for r in rows if r["name"] == n])), 3), reports=sum(r["name"] == n for r in rows))
                         for n in names if any(r["name"] == n for r in rows)},
               ms_per_frame=round(float(np.mean(det.ms)), 1), frames=det.frames)
    return out


def measure_clutter(n_scenes=60, seed=11):
    """L3: two items placed touching on a stopped belt. Are both found and named? (split on vs off)"""
    import warnings; warnings.filterwarnings("ignore")
    from cell_mujoco import CellMJ
    cell = CellMJ(); cam = Cam(cell); idm = IdentityModel(); rng = np.random.default_rng(seed); names = list(cell.items); rows = []
    for _ in range(n_scenes):
        cell.reset(); cell.belt(0.0, ramp=0); a, b = rng.choice(names, 2, replace=False); x = float(rng.uniform(-0.4, 0.4)); y = cell.lay.belt.y
        ya, yb = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-math.pi, math.pi))
        hx = lambda n, yaw: 0.5 * (abs(cell.items[n]["rec"]["extents"][0] * math.cos(yaw)) + abs(cell.items[n]["rec"]["extents"][1] * math.sin(yaw)))
        cell.spawn(a, x=x, y=y, yaw=ya, with_belt_velocity=False); cell.spawn(b, x=x + hx(a, ya) + hx(b, yb) + 0.002, y=y + float(rng.uniform(-0.02, 0.02)), yaw=yb, with_belt_velocity=False)
        cell.step(300); rgb, d, seg = cam.grab(); truth_ = sorted([str(a), str(b)]); r = dict(pair=truth_)
        for mode, model in (("split", idm), ("no_split", None)):
            bls = blobs(cam, rgb, d, seg, model); labs = sorted(str(idm.predict(feat(bl))) for bl in bls if not bl.get("unknown"))
            r[mode] = dict(both=labs == truth_, wrong=len(set(labs) - set(truth_)), blobs=len(bls), unknown=sum(bool(bl.get("unknown")) for bl in bls))
        rows.append(r)
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), scenes=n_scenes, seed=seed, note="two items touching side by side on a stopped belt",
             split=dict(both_named=sum(r["split"]["both"] for r in rows), wrong_names=sum(r["split"]["wrong"] for r in rows), unknown=sum(r["split"]["unknown"] for r in rows)),
             no_split=dict(both_named=sum(r["no_split"]["both"] for r in rows), wrong_names=sum(r["no_split"]["wrong"] for r in rows)),
             failures=[r for r in rows if not r["split"]["both"]])
    json.dump(S, open(os.path.join(HERE, "results", "f5-clutter-perception.json"), "w"), indent=1); return S


def main():
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=("build", "measure", "clutter")); p.add_argument("--scenes", type=int, default=None); p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--out", default="f4-vision-static.json", help="file name under factory/vision/results/ (the lab writes f4-vision-lab.json)")
    a = p.parse_args()
    if a.cmd == "build": print("rows", build(a.scenes or 240))
    elif a.cmd == "clutter": S = measure_clutter(a.scenes or 60); print(json.dumps({k: v for k, v in S.items() if k != "failures"}))
    else:
        S = measure(a.scenes or 60, belt_speed=a.speed); os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        json.dump(S, open(os.path.join(HERE, "results", os.path.basename(a.out)), "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
