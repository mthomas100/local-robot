"""belt_video.py — record items riding the running belt in MuJoCo, from one or more cameras (2026-11-02, F0 visuals).

    envs/mj/bin/python -I factory/cell/belt_video.py [--speed 0.3] [--seconds 10] [--cams cell,overhead] [--out DIR]
Spawns a mixed stream of items at the upstream end every `--gap` seconds while the belt runs; writes one mp4 per camera.
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cell_mujoco import CellMJ  # noqa: E402

ORDER = ["mustard_bottle", "tomato_soup_can", "cracker_box", "bleach_cleanser", "tuna_fish_can", "sugar_box",
         "windex_bottle", "master_chef_can", "pudding_box", "potted_meat_can", "gelatin_box"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--speed", type=float, default=0.3); p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--gap", type=float, default=1.2); p.add_argument("--cams", default="cell"); p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--fps", type=int, default=30); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); rng = np.random.default_rng(a.seed); cams = a.cams.split(",")
    import imageio.v3 as iio
    c = CellMJ(); c.belt(a.speed); c.step(int(1.0 / c.model.opt.timestep))   # belt up to speed first
    frames = {k: [] for k in cams}; dt = c.model.opt.timestep; every = max(1, round(1 / (dt * a.fps)))
    next_spawn, k_item, live = 0.0, 0, []
    t0 = time.perf_counter()
    for k in range(int(a.seconds / dt)):
        t = k * dt
        if t >= next_spawn:
            name = ORDER[k_item % len(ORDER)]; k_item += 1
            if name in live: live.remove(name)
            c.spawn(name, rng=rng); live.append(name); next_spawn += a.gap
        for n in list(live):   # past the end of the belt: park it (it would fall off; F2 counts that as a miss)
            if c.item_state(n)["pos"][0] > c.lay.belt.x_max + 0.05: c.park(n, ORDER.index(n)); live.remove(n)
        c.step()
        if k % every == 0:
            for cam in cams: frames[cam].append(c.render(cam).copy())
    outs = []
    for cam in cams:
        path = os.path.join(a.out, f"belt-{a.speed:.2f}-{cam}.mp4"); iio.imwrite(path, np.stack(frames[cam]), fps=a.fps, codec="libx264", plugin="pyav"); outs.append(path)
    print(f"{len(frames[cams[0]])} frames per camera in {time.perf_counter() - t0:.1f} s"); [print(o) for o in outs]


if __name__ == "__main__":
    main()
