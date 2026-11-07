"""chute_test.py — drop an item into each chute and check it ends inside: its centre of mass within the opening and
below the rim, no penetration (both engines must agree that the chutes are open bins, not solid blocks; 2026-11-02, F0
cross-check). The drop pose is checked for collisions first: a drop that starts inside the belt edge tests nothing.

    envs/mj/bin/python -I factory/cell/chute_test.py --engine mujoco
    envs/genesis/bin/python -I factory/cell/chute_test.py --engine genesis
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def _rot(q):
    import numpy as np
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _corners(rec):
    (x0, y0, z0), (x1, y1, z1) = rec["bounds"]
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--engine", default="mujoco"); p.add_argument("--item", default="tomato_soup_can"); a = p.parse_args()
    if a.engine == "mujoco":
        from cell_mujoco import CellMJ; cell = CellMJ()
    else:
        from cell_genesis import CellGS; cell = CellGS(backend="cpu", cameras=False, items=[a.item])
    out = []
    import numpy as np, math
    rec = cell.items[a.item]["rec"]
    yaw = 0.0 if rec["extents"][0] >= rec["extents"][1] else math.pi / 2   # long horizontal side along the belt (x), clear of the belt edge
    quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    for ch in cell.lay.chutes:
        cell.reset(); it = cell.items[a.item]
        drop_z = ch.rim_z + 0.05   # item frame origin = its bottom centre (ycb.canonical_frame)
        if a.engine == "mujoco":
            import mujoco
            q = it["qpos"]; cell.data.qpos[q:q + 3] = (ch.x, ch.y, drop_z); cell.data.qpos[q + 3:q + 7] = quat; mujoco.mj_forward(cell.model, cell.data)
            clash = cell.contacts_with(a.item)
            if clash: raise SystemExit(f"drop pose collides with {sorted(set(b for x, y, _ in clash for b in (x, y) if b != a.item))}: fix the test, not the chute")
        else:
            it["ent"].set_pos(np.array([ch.x, ch.y, drop_z]), zero_velocity=True); it["ent"].set_quat(quat, zero_velocity=True)
        cell.step(1000)
        s = cell.item_state(a.item); floor_top = ch.rim_z - ch.depth
        R = _rot(s["quat"]); lowest = min((R @ np.array(v) + s["pos"])[2] for v in _corners(rec))
        com_world = R @ np.array(rec["com"]) + s["pos"]
        inside = abs(com_world[0] - ch.x) < ch.inner / 2 and abs(com_world[1] - ch.y) < ch.inner / 2 and com_world[2] < ch.rim_z
        pen = cell.penetration(a.item)
        out.append(dict(chute=ch.material, item=a.item, com=[round(float(x), 3) for x in com_world], lowest_point_z=round(float(lowest), 4), chute_floor_top_z=round(floor_top, 4),
                        rim_z=ch.rim_z, tilt_deg=round(float(s["tilt_deg"]), 1), penetration_mm=round(pen * 1000, 2), inside=bool(inside and pen < 0.003)))
        print(a.engine, json.dumps(out[-1]))
    print(a.engine, "all inside:", all(o["inside"] for o in out))
    with open(os.path.join(HERE, "results", f"chute-test-{a.engine}.json"), "w") as f: json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
