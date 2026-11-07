"""tracking_study.py — discipline 8's acceptance (2026-11-02, F2): intercept prediction error versus belt speed and
camera latency, with and without latency compensation. An item rides the belt; the detector reports it at 30 Hz with
3 mm noise and the given latency; the Kalman track, at the moment the item reaches x = -0.45 (about where the planner
commits to a pick), predicts where the item will be `horizon` seconds later; the simulation then runs on and the
prediction is compared with the truth.

    envs/mj/bin/python -I factory/line/tracking_study.py   -> factory/line/results/f2-tracking.json
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
from tracking import Detector, DetectorCfg, Tracker  # noqa: E402

ITEMS = ("mustard_bottle", "tomato_soup_can", "cracker_box", "tuna_fish_can", "bleach_cleanser", "sugar_box")


def one(cell, name, speed, latency, compensate, horizon=1.5, seed=0, x_commit=-0.45):
    rng = np.random.default_rng(seed)
    cell.reset(); cell.belt(speed); cell.step(int(0.6 / cell.model.opt.timestep))   # the belt is up to speed before items arrive
    cell.spawn(name, x=-0.95, yaw=float(rng.uniform(-math.pi, math.pi)), rng=rng)
    det = Detector(cell, DetectorCfg(latency_s=latency, compensate_latency=compensate), rng=rng); trk = Tracker(v_prior=(0.0, 0.0))
    pred = None; t_commit = None
    while True:
        cell.step(); t = cell.data.time; trk.feed(det.poll(t, [name]), t)
        x = cell.item_state(name)["pos"][0]
        if pred is None and x >= x_commit and name in trk.confirmed():
            t_commit = t; pred, v = trk.confirmed()[name].predict(t + horizon)
        if pred is not None and t >= t_commit + horizon:
            truth = cell.item_state(name)["pos"]; return float(np.linalg.norm(pred[:2] - truth[:2])), float(v[0])
        if t > 20: return None, None


def table(rows):
    """The rows keyed for the lab's tables: "0.30 m/s · 60 ms · stamped" -> errors."""
    return {f"{r['speed']:.2f} m/s · {1000 * r['latency_s']:.0f} ms · {'stamped' if r['compensated'] else 'as fresh'}": {k: r[k] for k in ("err_mean_mm", "err_max_mm", "v_est_err_pct")} for r in rows}


def main():
    import warnings; warnings.filterwarnings("ignore")
    from cell_mujoco import CellMJ
    cell = CellMJ(); rows = []; t0 = time.perf_counter()
    for speed in (0.15, 0.3, 0.5):
        for latency in (0.0, 0.06, 0.12):
            for comp in ((True, False) if latency else (True,)):
                errs, vs = [], []
                for k, name in enumerate(ITEMS):
                    e, v = one(cell, name, speed, latency, comp, seed=k)
                    if e is not None: errs.append(e); vs.append(v)
                rows.append(dict(speed=speed, latency_s=latency, compensated=comp, n=len(errs), err_mean_mm=round(1000 * float(np.mean(errs)), 2),
                                 err_max_mm=round(1000 * float(np.max(errs)), 2), v_est_err_pct=round(100 * float(np.mean(np.abs(np.array(vs) - speed))) / speed, 2)))
                print(rows[-1], flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = dict(date=time.strftime("%Y-%m-%d"), horizon_s=1.5, commit_x=-0.45, detector=DetectorCfg().__dict__, rows=rows, wall_s=round(time.perf_counter() - t0, 1))
    out["table"] = table(rows)
    json.dump(out, open(os.path.join(HERE, "results", "f2-tracking.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
