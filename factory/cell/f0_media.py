"""f0_media.py — the F0 stills and videos for the README and the lab, reproducibly (2026-11-02).

    envs/mj/bin/python -I factory/cell/f0_media.py            # MuJoCo media
    envs/genesis/bin/python -I factory/cell/f0_media.py --genesis   # the Genesis still
Writes docs/media/F0-*.png (in git) and F0-*.mp4 (on disk, gitignored).
"""
import argparse, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(ROOT, "docs", "media")
SCENE_ITEMS = [("mustard_bottle", -0.75, 0.4), ("tomato_soup_can", -0.45, 0.0), ("cracker_box", -0.12, 1.2), ("bleach_cleanser", 0.2, 2.0), ("tuna_fish_can", 0.45, 0.0), ("pudding_box", 0.75, 0.6)]


def mujoco_media():
    import mujoco, imageio.v3 as iio
    from cell_mujoco import CellMJ
    from layout import Layout
    c = CellMJ()
    for n, x, yaw in SCENE_ITEMS: c.spawn(n, x=x, yaw=yaw)
    c.step(100)
    c.still(os.path.join(OUT, "F0-cell.png"), "cell"); c.still(os.path.join(OUT, "F0-overhead.png"), "overhead", 848, 480); c.still(os.path.join(OUT, "F0-end.png"), "end")
    print("stills")
    # the belt videos
    import subprocess
    for cam in ("cell", "overhead"):
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "belt_video.py"), "--speed", "0.3", "--seconds", "12", "--cams", cam, "--out", OUT], check=True, stdout=subprocess.DEVNULL)
        os.replace(os.path.join(OUT, f"belt-0.30-{cam}.mp4"), os.path.join(OUT, f"F0-belt-0.30-{cam}.mp4"))
    print("belt videos")
    # instructive failure: the Menagerie home pose leaves the gripper over the belt; the cracker box walks into it
    old = Layout(); old.home_qpos = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    f = CellMJ(old); f.belt(0.3); f.step(500); f.spawn("cracker_box", x=-0.75, yaw=0.0)
    frames = []
    for k in range(int(5.0 / 0.002)):
        f.step()
        if k % 17 == 0: frames.append(f.render("cell").copy())
    iio.imwrite(os.path.join(OUT, "F0-fail-gripper-in-the-way.mp4"), np.stack(frames), fps=30, codec="libx264", plugin="pyav")
    print("failure clip")
    # all items, textured (top row) and as collision pieces (bottom row), and the Windex scan close up
    import build_mjcf, ycb
    from layout import DEFAULT
    idx = ycb.load_index()
    s = mujoco.MjSpec(); build_mjcf._options(s, DEFAULT); s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(3, 3, 0.1), rgba=(0.85, 0.85, 0.87, 1))
    s.worldbody.add_light(pos=(0, -2, 3), dir=(0, 0.6, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.8, 0.8, 0.8))
    for k, rec in enumerate(idx.values()): build_mjcf.add_item(s, rec, pos=(-2.5 + 0.5 * k, 0, 0))
    m = s.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 300, 220); rows = []
    for grp in (2, 3):   # one tile per item: textured scan (top row), collision pieces + base pad (bottom row)
        tiles = []
        for k, rec in enumerate(idx.values()):
            opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = 1; opt.geomgroup[grp] = 1
            cam = mujoco.MjvCamera(); cam.lookat[:] = (-2.5 + 0.5 * k, 0, rec["extents"][2] / 2); cam.distance = 0.55; cam.azimuth = 70; cam.elevation = -18
            r.update_scene(d, camera=cam, scene_option=opt); tiles.append(r.render().copy())
        rows.append(np.concatenate(tiles, 1))
    iio.imwrite(os.path.join(OUT, "F0-items.png"), np.concatenate(rows, 0))
    rec = idx["022_windex_bottle"]; s = mujoco.MjSpec(); build_mjcf._options(s, DEFAULT); s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1), rgba=(0.85, 0.85, 0.87, 1))
    s.worldbody.add_light(pos=(0, -1, 2), dir=(0, 0.5, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL); build_mjcf.add_item(s, rec)
    m = s.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 480, 480); ims = []
    for az, grp in ((0, 2), (90, 2), (0, 3), (90, 3)):
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = 1; opt.geomgroup[grp] = 1
        cam = mujoco.MjvCamera(); cam.lookat[:] = (0, 0, 0.13); cam.distance = 0.6; cam.azimuth = az; cam.elevation = -5
        r.update_scene(d, camera=cam, scene_option=opt); ims.append(r.render().copy())
    iio.imwrite(os.path.join(OUT, "F0-windex-scan.png"), np.concatenate(ims, 1))
    print("items and windex")


def genesis_media():
    from cell_genesis import CellGS
    c = CellGS(backend="cpu", items=[n for n, _, _ in SCENE_ITEMS])
    for n, x, yaw in SCENE_ITEMS: c.spawn(n, x=x, yaw=yaw, with_belt_velocity=False)
    c.step(100)
    c.still(os.path.join(OUT, "F0-cell-genesis.png"), "cell"); c.still(os.path.join(OUT, "F0-overhead-genesis.png"), "overhead")
    print("genesis stills")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--genesis", action="store_true"); a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    genesis_media() if a.genesis else mujoco_media()
