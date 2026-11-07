"""eval_f2.py — the F2 acceptance evaluation (2026-11-02, docs/sorting-cell-spec.md §4 F2): the line at belt speeds 0.15 and
0.3 m/s, items arriving as a random stream at several rates, each scheduler, a few seeds; throughput (correctly sorted
items per minute) and miss rate per configuration. The best scheduler's score is the baseline F3 must beat.

    envs/mj/bin/python -I factory/line/eval_f2.py [--seconds 180] [--seeds 2] [--workers 6]   -> factory/line/results/f2-eval.json, runs/f2-eval/
    envs/mj/bin/python -I factory/line/eval_f2.py --video 60 --speed 0.3 --rate 8 [--scheduler lookahead] [--cam pick]
"""
import argparse, itertools, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)


def _init():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell")); from procs import die_with_parent; die_with_parent()


def _one(cfg):
    import warnings; warnings.filterwarnings("ignore")
    from line import Line
    ln = Line(speed=cfg["speed"], rate_per_min=cfg["rate"], scheduler=cfg["scheduler"], seed=cfg["seed"])
    out = ln.run(seconds=cfg["seconds"]); out["config"] = cfg; return out


def video(speed, rate, scheduler, seconds, path, cam="pick", seed=0, width=960, height=540, fps=25):
    import imageio.v3 as iio, mujoco
    from PIL import Image, ImageDraw
    from line import Line
    ln = Line(speed=speed, rate_per_min=rate, scheduler=scheduler, seed=seed); frames = []; r = mujoco.Renderer(ln.c.model, height, width)
    every = max(1, round(1.0 / (ln.c.model.opt.timestep * fps)))

    def grab(L):
        r.update_scene(L.c.data, camera=cam); im = Image.fromarray(r.render()); d = ImageDraw.Draw(im)
        done = L.log_items; ok = sum(i["outcome"] == "sorted" for i in done); cur = L.cur.name if (L.cur and not L.recovering) else ("recovering" if L.cur else "waiting")
        d.rectangle((0, 0, width, 24), fill=(0, 0, 0))
        d.text((8, 5), f"t {L.c.data.time:5.1f} s   belt {speed:.2f} m/s   {rate:g} items/min   {scheduler}   |   sorted {ok}/{len(done)}   |   robot: {cur}", fill=(255, 255, 255))
        frames.append(np.asarray(im))
    S = ln.run(seconds=seconds, on_frame=grab, frame_every=every)
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", plugin="pyav"); return path, S


def add_rows(S):
    """The table keyed by configuration (for the lab's tables)."""
    S["rows"] = {f"{r['speed']:.2f} m/s · {r['rate']:g}/min · {r['scheduler']}": {k: r[k] for k in ("sorted_per_min", "sorted_per_min_sd", "miss_rate", "attempt_success", "cycle_s", "grasp_err_mm", "collisions", "items")} for r in S["table"]}
    S["best_rows"] = {f"{float(v):.2f} m/s": dict(rate=r["rate"], scheduler=r["scheduler"], sorted_per_min=r["sorted_per_min"], miss_rate=r["miss_rate"]) for v, r in S["best_per_speed"].items()}
    return S


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=180.0); p.add_argument("--seeds", type=int, default=2); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--speeds", default="0.15,0.3"); p.add_argument("--rates", default="4,8,12"); p.add_argument("--schedulers", default="greedy,edf,lookahead")
    p.add_argument("--video", type=float, default=0.0, help="instead: render this many seconds of one configuration"); p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--rate", type=float, default=8.0); p.add_argument("--scheduler", default="lookahead"); p.add_argument("--cam", default="pick"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="f2-eval"); p.add_argument("--out", default=None)
    a = p.parse_args()
    run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True); res = os.path.join(HERE, "results"); os.makedirs(res, exist_ok=True)
    if a.video:
        path = a.out or os.path.join(run, f"line-{a.speed:.2f}-{a.rate:g}-{a.scheduler}-{a.cam}.mp4")
        vp, S = video(a.speed, a.rate, a.scheduler, a.video, path, a.cam, a.seed); print(json.dumps(S["summary"])); print("video:", vp)
        os.system(f"{ROOT}/bin/rl-sheet {vp} {vp[:-4]}-sheet.png >/dev/null 2>&1"); print("sheet:", vp[:-4] + "-sheet.png"); return
    cfgs = [dict(speed=float(v), rate=float(r), scheduler=s, seed=k, seconds=a.seconds) for v, r, s, k in
            itertools.product(a.speeds.split(","), a.rates.split(","), a.schedulers.split(","), range(a.seeds))]
    import multiprocessing as mp
    t0 = time.perf_counter(); rows = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool, open(os.path.join(run, "episodes.jsonl"), "w") as f:
        for out in pool.imap_unordered(_one, cfgs):
            rows.append(out); f.write(json.dumps(out, default=float) + "\n"); f.flush(); s = out["summary"]
            print(f"v {s['speed']:.2f} rate {s['rate_per_min']:4.1f} {s['scheduler']:9s} seed {out['config']['seed']}: sorted/min {s['sorted_per_min']:5.2f}  miss {s['miss_rate']:.2f}  {s['outcomes']}  attempts {s['attempts']}", flush=True)
    # aggregate over seeds
    agg = {}
    for o in rows:
        c = o["config"]; k = (c["speed"], c["rate"], c["scheduler"]); agg.setdefault(k, []).append(o["summary"])
    table = []
    for (v, r, sch), ss in sorted(agg.items()):
        from collections import Counter
        oc = Counter(); [oc.update(s["outcomes"]) for s in ss]; at = Counter(); [at.update(s["attempts"]) for s in ss]
        n = sum(s["items_done"] for s in ss); spm = [s["sorted_per_min"] for s in ss]
        table.append(dict(speed=v, rate=r, scheduler=sch, seeds=len(ss), sorted_per_min=round(float(np.mean(spm)), 2), sorted_per_min_sd=round(float(np.std(spm)), 2),
                          miss_rate=round(1 - oc["sorted"] / max(n, 1), 3), items=n, outcomes=dict(oc), attempts=dict(at),
                          attempt_success=round(at["ok"] / max(sum(at.values()), 1), 3), cycle_s=round(float(np.mean([s["cycle_s"] for s in ss if s["cycle_s"]])), 2),
                          grasp_err_mm=round(float(np.mean([s["grasp_err_mm"]["mean"] for s in ss if s["grasp_err_mm"]])), 1), collisions=sum(s["collisions"] for s in ss)))
    best = {}
    for row in table:
        k = row["speed"]
        if k not in best or row["sorted_per_min"] > best[k]["sorted_per_min"]: best[k] = row
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), seconds=a.seconds, seeds=a.seeds, table=table, best_per_speed={str(k): v for k, v in best.items()}, wall_s=round(time.perf_counter() - t0, 1))
    add_rows(S)
    json.dump(S, open(os.path.join(res, f"{a.name}.json"), "w"), indent=1)
    for row in table: print({k: row[k] for k in ("speed", "rate", "scheduler", "sorted_per_min", "sorted_per_min_sd", "miss_rate", "attempt_success", "cycle_s", "grasp_err_mm", "collisions")})
    print("summary:", os.path.join(res, f"{a.name}.json"))


if __name__ == "__main__":
    main()
