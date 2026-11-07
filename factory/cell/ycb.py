"""ycb.py — the sorting cell's object catalogue and asset pipeline (2026-11-02, docs/sorting-cell-spec.md §2, discipline 2).

Turns the downloaded YCB "google_16k" meshes (scripts/fetch_ycb.sh -> assets/ycb/_dl/<id>/google_16k/) into what both
engines load, all in the item's canonical frame (upright stable pose, origin at the bottom centre; see
canonical_frame): assets/ycb/<id>/visual.obj + texture_map.png (the textured visual mesh) and assets/ycb/<id>/hull_NN.obj
(CoACD convex pieces for collision, computed once on the scan and cached in _dl/<id>/coacd/; MuJoCo collides on the
convex hull of each mesh geom, Genesis reads the same compiled geoms), plus assets/ycb/index.json with mass, inertia
(from the hulls, uniform density scaled to the real mass), extents and the material class that decides the chute.

    envs/genesis/bin/python -I factory/cell/ycb.py            # process every item (idempotent, ~1 min)
    envs/genesis/bin/python -I factory/cell/ycb.py --force    # redo the decomposition
"""
from __future__ import annotations
import argparse, json, math, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "ycb")
DL = os.path.join(ASSETS, "_dl")

# id, short name, material class (= the chute), mass in grams as listed by the YCB object table (ASSETS.md says where
# the numbers come from and what was verified).
ITEMS = [
    ("002_master_chef_can", "master_chef_can", "metal", 414),
    ("003_cracker_box", "cracker_box", "cardboard", 411),
    ("004_sugar_box", "sugar_box", "cardboard", 514),
    ("005_tomato_soup_can", "tomato_soup_can", "metal", 349),
    ("006_mustard_bottle", "mustard_bottle", "plastic", 603),
    ("007_tuna_fish_can", "tuna_fish_can", "metal", 171),
    ("008_pudding_box", "pudding_box", "cardboard", 187),
    ("009_gelatin_box", "gelatin_box", "cardboard", 97),
    ("010_potted_meat_can", "potted_meat_can", "metal", 370),
    ("021_bleach_cleanser", "bleach_cleanser", "plastic", 1131),
    ("022_windex_bottle", "windex_bottle", "plastic", 1022),
]
MATERIALS = ("plastic", "metal", "cardboard")  # chute order, left to right along the belt
FRICTION = {"plastic": 0.6, "metal": 0.5, "cardboard": 0.7}  # sliding friction vs the belt and the pads (plausible, untuned)


def load_index() -> dict:
    with open(os.path.join(ASSETS, "index.json")) as f:
        return json.load(f)


def _inertia_of_hulls(hulls, mass):
    """Mass, centre of mass and the 3x3 inertia about the body origin for a set of convex trimeshes sharing one
    density, scaled so the total is `mass` (kg)."""
    vols = np.array([h.volume for h in hulls]); rho = mass / vols.sum()
    com = sum(h.center_mass * h.volume for h in hulls) * rho / mass
    I = np.zeros((3, 3))
    for h in hulls:
        m = rho * h.volume; d = h.center_mass - com
        I += h.moment_inertia * rho + m * (d @ d * np.eye(3) - np.outer(d, d))  # parallel axis to the body COM
    return float(mass), com, I


def _rotation_between(u, v):
    """Smallest rotation matrix taking unit vector u to unit vector v (Rodrigues)."""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); c = float(u @ v); w = np.cross(u, v); s = np.linalg.norm(w)
    if s < 1e-9: return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer([1, 0, 0], [1, 0, 0])
    k = w / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def canonical_frame(hulls, com):
    """The item's canonical frame (2026-11-02): scanned YCB meshes are slightly tilted and their origins are off the
    object (the soup can's by 8 cm), so a scan placed 'upright' may balance on a bump and tip over (the Windex bottle
    did). Pick the stable resting pose of the union convex hull (with the real centre of mass) whose up direction is
    closest to the scan's +z, rotate by the smallest rotation that makes it upright (no spin), then put the origin at
    the bottom centre of the bounding box. Returns the 4x4 transform from scan to canonical coordinates, the tilt it
    removed in degrees and the pose's quasi-static probability."""
    import trimesh
    union = trimesh.util.concatenate(hulls).convex_hull
    Ts, probs = trimesh.poses.compute_stable_poses(union, center_mass=com, sigma=0.0, n_samples=1)
    tilts = [math.degrees(math.acos(np.clip(T[2, 2], -1, 1))) for T in Ts]   # angle between the scan's +z and world up in that pose
    i = int(np.argmin(tilts)); R = Ts[i][:3, :3]
    up_in_scan = R.T @ np.array([0.0, 0.0, 1.0])                             # the scan direction that points up when resting
    Rmin = _rotation_between(up_in_scan, np.array([0.0, 0.0, 1.0]))
    v = union.vertices @ Rmin.T; lo, hi = v.min(0), v.max(0)
    T = np.eye(4); T[:3, :3] = Rmin; T[:3, 3] = (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])
    return T, float(tilts[i]), float(probs[i])


BASE_BAND = 0.010    # m: the bottom slice whose outline becomes the base pad
# per-item overrides: the Windex scan (a transparent bottle) is broken: its trigger head floats above the neck and
# its bottom is cut off diagonally, so its bottom 1 cm outlines only a corner (20 cm2); its lower body is prismatic,
# so the outline of the bottom 4.5 cm is the bottle's real footprint (rendered and checked 2026-11-02).
BASE_BAND_ITEM = {"022_windex_bottle": 0.045}
BASE_PAD = 0.003     # m: pad thickness


def base_pad(hulls, band=BASE_BAND):
    """A flat convex pad under the item (2026-11-02): the 2D convex hull of every collision vertex in the bottom
    BASE_BAND, extruded from z = 0 to BASE_PAD. A turntable scan captures an object's underside worst (it stood on the
    turntable), and the Windex scan's bottom touched a flat surface over 14 of its ~48 cm2, leaving its centre of mass
    2.4 mm from the support edge so it tipped over on the belt; real containers stand on their rim, whose outline is
    what this pad is. Harmless for items whose bottom is already flat (cans, boxes)."""
    import trimesh
    from scipy.spatial import ConvexHull
    v = np.vstack([h.vertices for h in hulls]); pts = v[v[:, 2] < v[:, 2].min() + band][:, :2]
    if len(pts) < 3: return None
    ring = pts[ConvexHull(pts).vertices]
    z0 = v[:, 2].min()
    return trimesh.Trimesh(np.vstack([np.c_[ring, np.full(len(ring), z0)], np.c_[ring, np.full(len(ring), z0 + BASE_PAD)]])).convex_hull


def process(item, force=False, log=print):
    import trimesh, coacd
    oid, name, material, grams = item
    src = os.path.join(DL, oid, "google_16k"); raw = os.path.join(DL, oid, "coacd"); dst = os.path.join(ASSETS, oid)
    os.makedirs(dst, exist_ok=True); os.makedirs(raw, exist_ok=True)
    if not os.path.exists(os.path.join(src, "textured.obj")):
        raise SystemExit(f"{oid}: missing {src}/textured.obj; run scripts/fetch_ycb.sh")
    mesh = trimesh.load(os.path.join(src, "textured.obj"), force="mesh", process=False)
    raw_files = sorted(f for f in os.listdir(raw) if f.startswith("hull_"))
    t0 = time.perf_counter()
    if not raw_files or force:   # CoACD on the scan as downloaded; cached under _dl/<id>/coacd/
        for f in raw_files: os.remove(os.path.join(raw, f))
        coacd.set_log_level("error")
        parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), threshold=0.04,
                                max_convex_hull=12, resolution=2000, seed=0)
        raw_files = []
        for k, (v, fc) in enumerate(parts):
            h = trimesh.Trimesh(v, fc, process=True).convex_hull
            fn = f"hull_{k:02d}.obj"; h.export(os.path.join(raw, fn)); raw_files.append(fn)
    t_dec = time.perf_counter() - t0
    raw_hulls = [trimesh.load(os.path.join(raw, f), force="mesh") for f in raw_files]
    _, com_scan, _ = _inertia_of_hulls(raw_hulls, grams / 1000.0)
    T, tilt, prob = canonical_frame(raw_hulls, com_scan)
    for f in os.listdir(dst):
        if f.startswith("hull_") or f.startswith("visual") or f.startswith("material"): os.remove(os.path.join(dst, f))
    vis = mesh.copy(); vis.apply_transform(T); vis.export(os.path.join(dst, "visual.obj"))   # keeps the UVs (checked: MuJoCo reads 1 texcoord per vertex)
    shutil.copy(os.path.join(src, "texture_map.png"), os.path.join(dst, "texture_map.png"))
    hulls, hull_files = [], []
    for f, h in zip(raw_files, raw_hulls):
        h = h.copy(); h.apply_transform(T); h.export(os.path.join(dst, f)); hulls.append(h); hull_files.append(f)
    pad = base_pad(hulls, BASE_BAND_ITEM.get(oid, BASE_BAND))
    if pad is not None:
        pad.export(os.path.join(dst, "hull_base.obj")); hulls.append(pad); hull_files.append("hull_base.obj")
    mass, com, I = _inertia_of_hulls(hulls, grams / 1000.0)
    allv = np.vstack([h.vertices for h in hulls]); lo, hi = allv.min(0), allv.max(0)
    rec = dict(id=oid, name=name, material=material, mass_kg=round(mass, 4), friction=FRICTION[material],
               extents=[round(float(x), 4) for x in (hi - lo)], bounds=[[round(float(x), 4) for x in lo], [round(float(x), 4) for x in hi]],
               com=[round(float(x), 5) for x in com], inertia_full=[round(float(x), 7) for x in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])],
               hulls=hull_files, hull_volume_cm3=round(sum(h.volume for h in hulls) * 1e6, 1),
               convex_hull_volume_cm3=round(mesh.convex_hull.volume * 1e6, 1), faces=int(len(mesh.faces)),
               visual="visual.obj", texture="texture_map.png", decomposition_seconds=round(t_dec, 1),
               scan_to_item=[[round(float(x), 6) for x in row] for row in T], scan_tilt_removed_deg=round(tilt, 2), stable_pose_probability=round(prob, 3))
    log(f"{oid:22s} {material:9s} {mass*1000:5.0f} g  {len(hull_files):2d} hulls  hullvol {rec['hull_volume_cm3']:7.1f} cm3  "
        f"(convex hull {rec['convex_hull_volume_cm3']:7.1f})  extents {rec['extents']}  tilt removed {tilt:4.1f} deg  p(upright) {prob:.2f}  {rec['decomposition_seconds']} s")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--force", action="store_true"); p.add_argument("--only", nargs="*")
    a = p.parse_args()
    index = {}
    if os.path.exists(os.path.join(ASSETS, "index.json")) and not a.force:
        index = load_index()
    for item in ITEMS:
        if a.only and item[0] not in a.only and item[1] not in a.only: continue
        index[item[0]] = process(item, force=a.force)
    with open(os.path.join(ASSETS, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote {os.path.join(ASSETS, 'index.json')} ({len(index)} items)")


if __name__ == "__main__":
    main()
