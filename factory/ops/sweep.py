"""sweep.py — F5: the full cell's operating envelope (2026-11-02; discipline 15).

The whole cell (vision only, the student carry, the hybrid gripper) over belt speed (0.15, 0.3 and the fast belt
0.5 m/s), arrival rate and clutter (share of arrivals that come as two touching items). Each run is 180 s on its own
seed; the operations report (report.py) turns the table into throughput-vs-reliability trade-offs.

    envs/mj/bin/python -I factory/ops/sweep.py [--speeds 0.15,0.3,0.5] [--rates 8,12,16] [--clutter 0,0.25] [--seeds 2] [--workers 8]
        [--v2 fast,early,deadline --name f7-sweep-v2]   (F7: the v2 cell)
        -> factory/ops/results/<name>.json, runs/<name>/episodes.jsonl (resumable)
"""
import argparse, itertools, json, os, sys, time
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
for p in (os.path.join(FACTORY, "learned"), os.path.join(FACTORY, "line"), os.path.join(FACTORY, "cell")): sys.path.insert(0, p)


def _one(cfg):
    import warnings; warnings.filterwarnings("ignore")
    from line_learned import LearnedLine
    ln = LearnedLine(policy=os.path.join(FACTORY, "learned", "student.json"), privileged_carry=False, speed=cfg["speed"], rate_per_min=cfg["rate"],
                     scheduler="lookahead", seed=cfg["seed"], perception=cfg["perception"], clutter=cfg["clutter"], v2=tuple(cfg.get("v2", ())))
    out = ln.run(seconds=cfg["seconds"]); S = out["summary"]
    S["clutter_items"] = dict(Counter(i["outcome"] for i in out["items"] if i.get("clutter")))
    return dict(config=cfg, summary=S)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--speeds", default="0.15,0.3,0.5"); p.add_argument("--rates", default="8,12,16"); p.add_argument("--clutter", default="0,0.25")
    p.add_argument("--seeds", type=int, default=2); p.add_argument("--seed0", type=int, default=500); p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--perception", default="vision"); p.add_argument("--workers", type=int, default=8); p.add_argument("--name", default="f5-sweep")
    p.add_argument("--v2", default="", help="F7: comma-separated line.V2_FLAGS (empty: the v1 cell)")
    a = p.parse_args()
    v2 = [f for f in a.v2.split(",") if f]
    cfgs = [dict(speed=float(s), rate=float(r), clutter=float(c), seed=k, seconds=a.seconds, perception=a.perception, **({"v2": v2} if v2 else {}))
            for s, r, c, k in itertools.product(a.speeds.split(","), a.rates.split(","), a.clutter.split(","), range(a.seed0, a.seed0 + a.seeds))]
    import multiprocessing as mp
    from procs import die_with_parent
    run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True); path = os.path.join(run, "episodes.jsonl"); t0 = time.perf_counter(); eps = []
    if os.path.exists(path):
        want = {json.dumps(c, sort_keys=True) for c in cfgs}; eps = [e for e in map(json.loads, open(path)) if json.dumps(e["config"], sort_keys=True) in want]
    done = {json.dumps(e["config"], sort_keys=True) for e in eps}; todo = [c for c in cfgs if json.dumps(c, sort_keys=True) not in done]
    print(f"{len(eps)} on disk, {len(todo)} to go", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=die_with_parent, maxtasksperchild=3) as pool, open(path, "a") as f:
        for e in pool.imap_unordered(_one, todo):
            f.write(json.dumps(e, default=float) + "\n"); f.flush(); eps.append(e); c = e["config"]; print(c["speed"], c["rate"], c["clutter"], c["seed"], e["summary"]["sort_rate"], flush=True)
    rows = []
    for s, r, c in sorted({(e["config"]["speed"], e["config"]["rate"], e["config"]["clutter"]) for e in eps}):
        ss = [e["summary"] for e in eps if (e["config"]["speed"], e["config"]["rate"], e["config"]["clutter"]) == (s, r, c)]
        m = lambda k: round(float(np.mean([x[k] for x in ss if x.get(k) is not None])), 3) if any(x.get(k) is not None for x in ss) else None
        rows.append(dict(speed=s, rate=r, clutter=c, runs=len(ss), sorted_per_min=m("sorted_per_min"), sort_rate=m("sort_rate"), arrivals_per_min=m("arrivals_per_min"),
                         attempt_success=m("attempt_success"), cycle_s=m("cycle_s"), protective_stops=int(sum(x["collisions"] for x in ss)),
                         outcomes=dict(sum((Counter(x["outcomes"]) for x in ss), Counter())), clutter_items=dict(sum((Counter(x["clutter_items"]) for x in ss), Counter()))))
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), v2=v2, seconds=a.seconds, seeds=a.seeds, seed0=a.seed0, perception=a.perception, carry="student", rows=rows,
             table={f"{r['speed']:.2f} m/s · {r['rate']:g}/min · clutter {r['clutter']:.0%}": {k: r[k] for k in ("sorted_per_min", "sort_rate", "arrivals_per_min", "attempt_success", "cycle_s", "protective_stops")} for r in rows},
             wall_s=round(time.perf_counter() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(S, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
    for k, v in S["table"].items(): print(k, v)


if __name__ == "__main__":
    main()
