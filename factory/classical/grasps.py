"""grasps.py — discipline 4 (2026-11-02, F1): a ranked library of antipodal top-down grasps per YCB item for the
Robotiq 2F-85, computed from the item's scanned surface in its canonical frame (factory/cell/ycb.py).

A top-down grasp: the gripper points straight down, its fingers close along a horizontal axis at angle `yaw` (item
frame) through the point `pos` (the pinch point, item frame). Candidates: for each closing direction (every 10°), each
lateral offset and each depth below the item's top, cast rays at the pinch height from both sides toward the centre
line; the two surface hits are the contact points. Keep a candidate if

  * the width between the contacts fits the 2F-85 with margin (opening 85 mm; we require width <= 78 mm),
  * both surface normals lie inside the friction cone of the closing axis (antipodal: half-angle atan(mu), mu = 0.5),
  * the pinch point is deep enough for the pads to hold (>= 8 mm below the top) but the palm clears the top (the pads
    reach about 35 mm below the palm), and the fingertips stay >= 15 mm above the belt.

Score (higher is better): antipodality (smaller normal angles), lateral offset from the centre of mass (less torque),
width margin, grip depth. The library is written to factory/classical/grasps.json; `best(item, yaw_world)` returns
grasps in the item frame for the pick pipeline.

    envs/mj/bin/python -I factory/classical/grasps.py        # compute and print the library summary
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
import ycb  # noqa: E402

MAX_WIDTH = 0.078; MU = 0.5; PAD_REACH = 0.035; MIN_DEPTH = 0.008; MIN_TIP_CLEAR = 0.015
OUT = os.path.join(HERE, "grasps.json")


def candidates(rec, yaw_step_deg=10, offsets=(-0.015, 0.0, 0.015), depths=(0.012, 0.02, 0.028)):
    import trimesh
    mesh = trimesh.load(os.path.join(ycb.ASSETS, rec["id"], "visual.obj"), force="mesh", process=False)
    lo, hi = np.array(rec["bounds"][0]), np.array(rec["bounds"][1]); top = hi[2]; com = np.array(rec["com"])
    cone = math.atan(MU); out = []
    for d in depths:
        z = top - d
        if z - 0.0 < MIN_TIP_CLEAR + 0.01: continue   # fingertips (about 1 cm below the pinch) must clear the belt
        for yd in range(0, 180, yaw_step_deg):
            yaw = math.radians(yd); a = np.array([math.cos(yaw), math.sin(yaw), 0.0]); perp = np.array([-a[1], a[0], 0.0])
            for off in offsets:
                c0 = np.array([com[0], com[1], z]) + off * perp
                far = 0.3
                origins = np.array([c0 - far * a, c0 + far * a]); dirs = np.array([a, -a])
                locs, idx_ray, idx_tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
                if len(set(idx_ray.tolist())) < 2: continue
                p1 = locs[list(idx_ray).index(0)]; p2 = locs[list(idx_ray).index(1)]
                n1 = mesh.face_normals[idx_tri[list(idx_ray).index(0)]]; n2 = mesh.face_normals[idx_tri[list(idx_ray).index(1)]]
                width = float(np.linalg.norm(p2 - p1))
                if width > MAX_WIDTH or width < 0.01: continue
                ang1 = math.acos(np.clip(np.dot(n1, -a), -1, 1)); ang2 = math.acos(np.clip(np.dot(n2, a), -1, 1))   # outward normals vs the push directions
                if ang1 > cone or ang2 > cone: continue
                pinch = (p1 + p2) / 2
                lateral = float(np.linalg.norm((pinch - com)[:2]))
                score = (1.0 - (ang1 + ang2) / (2 * cone)) * 1.0 + (1.0 - lateral / 0.05) * 0.6 + (MAX_WIDTH - width) / MAX_WIDTH * 0.3 + (d / 0.028) * 0.3
                out.append(dict(pos=[round(float(x), 4) for x in pinch], yaw=round(yaw, 4), width=round(width, 4), depth=d,
                                normal_angles_deg=[round(math.degrees(ang1), 1), round(math.degrees(ang2), 1)], lateral=round(lateral, 4), score=round(float(score), 4)))
    out.sort(key=lambda g: -g["score"])
    return out


def build():
    index = ycb.load_index(); lib = {}
    for iid, rec in index.items():
        c = candidates(rec); lib[rec["name"]] = dict(id=iid, n=len(c), grasps=c[:60])
        g = c[0] if c else None
        print(f"{rec['name']:18s} {len(c):4d} grasps; best: yaw {math.degrees(g['yaw']) if g else float('nan'):5.0f} deg width {g['width']*1000 if g else 0:4.0f} mm "
              f"depth {g['depth']*1000 if g else 0:3.0f} mm normals {g['normal_angles_deg'] if g else '-'} score {g['score'] if g else '-'}")
    with open(OUT, "w") as f: json.dump(lib, f, indent=1)
    print("wrote", OUT); return lib


def load():
    with open(OUT) as f: return json.load(f)


def world_grasps(lib, name, item_pos, item_quat):
    """The item's grasps in the world: (pinch position, closing-axis yaw in the world, width, score), best first. The
    item must stand upright (its yaw is read from the quaternion)."""
    w, x, y, z = item_quat; item_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(item_yaw), math.sin(item_yaw); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    out = []
    for g in lib[name]["grasps"]:
        p = np.asarray(item_pos) + Rz @ np.array(g["pos"]); yaw = (g["yaw"] + item_yaw + math.pi / 2) % math.pi - math.pi / 2   # a closing axis is symmetric under 180 deg
        out.append(dict(pos=p, yaw=yaw, width=g["width"], depth=g["depth"], score=g["score"]))
    return out


if __name__ == "__main__":
    build()
