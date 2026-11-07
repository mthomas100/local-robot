"""live_view.py — watch the sorting cell run live in an interactive MuJoCo window (2026-11-02).

    envs/mj/bin/mjpython factory/cell/live_view.py [--speed 0.3] [--gap 1.2]
The belt runs in real time and a mixed stream of YCB items rides through the reach zone; items that reach the end are
recycled to the upstream end. Mouse: left-drag orbits, right-drag pans, scroll zooms, double-click selects a body
(ctrl + right-drag pushes it). Keys: up/down arrow = belt speed +/- 0.05 m/s, B = stop/start the belt, N = drop the
next item now, C = toggle collision hulls (what the physics sees), R = reset. Close the window to stop.
macOS needs mjpython (the window owns the main thread); `bin/rl cell live` runs it in the background.
"""
import argparse, os, sys, time
import numpy as np
import mujoco, mujoco.viewer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cell_mujoco import CellMJ  # noqa: E402
from belt_video import ORDER  # noqa: E402

KEY_UP, KEY_DOWN = 265, 264


def main():
    p = argparse.ArgumentParser(); p.add_argument("--speed", type=float, default=0.3); p.add_argument("--gap", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    c = CellMJ(); rng = np.random.default_rng(a.seed)
    st = dict(speed=a.speed, running=True, drop=False, reset=False, hulls=False)

    def on_key(k):
        if k == KEY_UP: st["speed"] = min(0.8, round(st["speed"] + 0.05, 2))
        elif k == KEY_DOWN: st["speed"] = max(0.0, round(st["speed"] - 0.05, 2))
        elif k == ord("B"): st["running"] = not st["running"]
        elif k == ord("N"): st["drop"] = True
        elif k == ord("C"): st["hulls"] = not st["hulls"]
        elif k == ord("R"): st["reset"] = True
        print(f"belt {st['speed'] if st['running'] else 0.0:.2f} m/s", flush=True)

    with mujoco.viewer.launch_passive(c.model, c.data, key_callback=on_key, show_left_ui=False, show_right_ui=False) as v:
        with v.lock():
            v.cam.lookat[:] = (0.0, 0.42, 0.85); v.cam.distance = 2.9; v.cam.azimuth = 125; v.cam.elevation = -28
        dt = c.model.opt.timestep; live = []; next_spawn = 0.5; k_item = 0; wall0 = time.perf_counter(); sim0 = c.data.time
        c.belt(st["speed"])
        while v.is_running():
            if st["reset"]:
                c.reset(); live.clear(); k_item = 0; st["reset"] = False; wall0 = time.perf_counter(); sim0 = c.data.time; next_spawn = sim0 + 0.5
            c.belt(st["speed"] if st["running"] else 0.0)
            if st["drop"] or (st["running"] and c.data.time >= next_spawn):
                name = ORDER[k_item % len(ORDER)]; k_item += 1
                if name in live: live.remove(name)
                c.spawn(name, rng=rng); live.append(name); st["drop"] = False
                next_spawn = c.data.time + (a.gap * 0.3 / max(st["speed"], 0.05))   # keep ~the same spacing on the belt at any speed
            for n in list(live):
                if c.item_state(n)["pos"][0] > c.lay.belt.x_max + 0.05: c.park(n, ORDER.index(n)); live.remove(n)
            # step until the simulation catches up with the wall clock (real time), then draw
            target = sim0 + (time.perf_counter() - wall0)
            n = 0
            while c.data.time < target and n < 40: c.step(); n += 1
            with v.lock():
                v.opt.geomgroup[3] = 1 if st["hulls"] else 0
            v.sync()
            lag = c.data.time - (sim0 + (time.perf_counter() - wall0))
            if lag > 0: time.sleep(min(lag, 0.02))


if __name__ == "__main__":
    main()
