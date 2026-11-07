"""f7.py — F7: the fast belt (Factory v2, 2026-11-03; docs/sorting-cell-spec.md §4 row F7; disciplines 6, 8, 9,
12, 15).

The F5 cell (vision only, the student carry, the hybrid gripper, the lookahead scheduler) is the v1 cell. v2 is the
same cell with the changes of `line.V2_FLAGS` (fast, early, deadline; see line.py), each measured on its own here
before they are combined.

    envs/mj/bin/python -I factory/ops/f7.py ablate [--speeds 0.3,0.5] [--seeds 6] [--seed0 7100] [--workers 8]
        each change alone and all together against v1, on the F5 stream -> factory/ops/results/f7-ablation.json
    envs/mj/bin/python -I factory/ops/f7.py eval --seeds N --seed0 S [--workers 8]
        the acceptance: v1 vs v2 paired on the same seeds at 0.5 and 0.3 m/s -> factory/ops/results/f7-eval.json
        (protocol: factory/ops/results/f7-eval-protocol.md, committed before the run)
Episodes go to runs/<name>/episodes.jsonl (resumable).
"""
import argparse, itertools, json, math, os, sys, time
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
for p in (os.path.join(FACTORY, "learned"), os.path.join(FACTORY, "line"), os.path.join(FACTORY, "cell")): sys.path.insert(0, p)

STREAM = dict(rate=12.0, clutter=0.25)    # the F5 shift's stream
SECONDS = 180.0
V2 = ("fast", "early", "deadline")


def _one(cfg):
    import warnings; warnings.filterwarnings("ignore")
    from line_learned import LearnedLine
    ln = LearnedLine(policy=os.path.join(FACTORY, "learned", "student.json"), privileged_carry=False, speed=cfg["speed"], rate_per_min=cfg["rate"],
                     scheduler="lookahead", seed=cfg["seed"], perception="vision", clutter=cfg["clutter"], v2=tuple(cfg["v2"]))
    out = ln.run(seconds=cfg["seconds"]); S = out["summary"]
    S["clutter_items"] = dict(Counter(i["outcome"] for i in out["items"] if i.get("clutter")))
    S["collision_details"] = [a.get("detail") for a in out["attempts"] if a.get("outcome") == "collision"]
    return dict(config=cfg, summary=S)


def run(cfgs, name, workers):
    import multiprocessing as mp
    from procs import die_with_parent
    d = os.path.join(ROOT, "runs", name); os.makedirs(d, exist_ok=True); path = os.path.join(d, "episodes.jsonl"); eps = []
    key = lambda c: json.dumps(c, sort_keys=True)
    if os.path.exists(path):
        want = {key(c) for c in cfgs}; eps = [e for e in map(json.loads, open(path)) if key(e["config"]) in want]
    done = {key(e["config"]) for e in eps}; todo = [c for c in cfgs if key(c) not in done]
    print(f"{len(eps)} on disk, {len(todo)} to go", flush=True)
    if todo:
        with mp.get_context("spawn").Pool(workers, initializer=die_with_parent, maxtasksperchild=3) as pool, open(path, "a") as f:
            for e in pool.imap_unordered(_one, todo):
                f.write(json.dumps(e, default=float) + "\n"); f.flush(); eps.append(e); c = e["config"]
                print(c["speed"], "+".join(c["v2"]) or "v1", c["seed"], e["summary"]["sorted_per_min"], e["summary"]["outcomes"], flush=True)
    return eps


def label(v2): return "+".join(v2) if v2 else "v1"


def ci_diff(a, b, z=1.96):
    """Paired difference b - a: mean, 95% CI (normal on the per-seed differences), n."""
    d = np.array(b) - np.array(a); n = len(d); m = float(d.mean()); s = float(d.std(ddof=1)) if n > 1 else 0.0
    h = z * s / math.sqrt(n) if n > 1 else float("nan"); return dict(mean=round(m, 3), lo=round(m - h, 3), hi=round(m + h, 3), sd=round(s, 3), n=n)


def table(eps, variants, speeds):
    rows = {}
    for sp in speeds:
        for v in variants:
            ss = [e["summary"] for e in eps if e["config"]["speed"] == sp and tuple(e["config"]["v2"]) == tuple(v)]
            if not ss: continue
            m = lambda k: round(float(np.mean([x[k] for x in ss if x.get(k) is not None])), 3) if any(x.get(k) is not None for x in ss) else None
            rows[f"{sp:.2f} m/s · {label(v)}"] = dict(speed=sp, v2=list(v), runs=len(ss), sorted_per_min=m("sorted_per_min"), sort_rate=m("sort_rate"),
                                                     attempt_success=m("attempt_success"), cycle_s=m("cycle_s"), protective_stops=int(sum(x["collisions"] for x in ss)),
                                                     wrong_chute=int(sum(x["outcomes"].get("wrong_chute", 0) for x in ss)),
                                                     outcomes=dict(sum((Counter(x["outcomes"]) for x in ss), Counter())),
                                                     attempts=dict(sum((Counter(x["attempts"]) for x in ss), Counter())),
                                                     stop_details=sorted({str(dd) for x in ss for dd in x.get("collision_details", [])}))
    return rows


def paired(eps, speeds, base, other):
    out = {}
    for sp in speeds:
        get = lambda v: {e["config"]["seed"]: e["summary"]["sorted_per_min"] for e in eps if e["config"]["speed"] == sp and tuple(e["config"]["v2"]) == tuple(v)}
        a, b = get(base), get(other); seeds = sorted(set(a) & set(b))
        if seeds: out[f"{sp:.2f}"] = ci_diff([a[s] for s in seeds], [b[s] for s in seeds])
    return out


def ablate(a):
    speeds = [float(x) for x in a.speeds.split(",")]; variants = [(), ("fast",), ("early",), ("deadline",), V2]
    cfgs = [dict(speed=sp, v2=list(v), seed=s, seconds=a.seconds, **STREAM) for sp, v, s in itertools.product(speeds, variants, range(a.seed0, a.seed0 + a.seeds))]
    t0 = time.time(); eps = run(cfgs, a.name or "f7-ablation", a.workers)
    R = dict(date=time.strftime("%Y-%m-%d %H:%M"), stream=STREAM, seconds=a.seconds, seeds=[a.seed0, a.seed0 + a.seeds - 1], perception="vision", carry="student",
             table=table(eps, variants, speeds), vs_v1={label(v): paired(eps, speeds, (), v) for v in variants[1:]}, wall_s=round(time.time() - t0, 1))
    json.dump(R, open(os.path.join(HERE, "results", f"{a.name or 'f7-ablation'}.json"), "w"), indent=1)
    for k, v in R["table"].items(): print(k, {x: v[x] for x in ("sorted_per_min", "sort_rate", "protective_stops", "wrong_chute", "outcomes")})
    print(json.dumps(R["vs_v1"], indent=1))


def evaluate(a):
    speeds = [0.5, 0.3]; v2 = tuple(a.v2.split(",")) if a.v2 else V2
    cfgs = [dict(speed=sp, v2=list(v), seed=s, seconds=a.seconds, **STREAM) for sp, v, s in itertools.product(speeds, [(), v2], range(a.seed0, a.seed0 + a.seeds))]
    t0 = time.time(); eps = run(cfgs, "f7-eval", a.workers)
    T = table(eps, [(), v2], speeds); P = paired(eps, speeds, (), v2)
    v1_stops = set().union(*[set(T[f"{sp:.2f} m/s · v1"]["stop_details"]) for sp in speeds]); v2_stops = set().union(*[set(T[f"{sp:.2f} m/s · {label(v2)}"]["stop_details"]) for sp in speeds])
    stop_class = lambda d: d.split(" ")[0] if d else d
    new_classes = sorted({stop_class(d) for d in v2_stops} - {stop_class(d) for d in v1_stops})
    wrong = sum(T[f"{sp:.2f} m/s · {label(v2)}"]["wrong_chute"] for sp in speeds)
    verdict = dict(fast_beats_v1=bool(P["0.50"]["lo"] > 0), not_worse_at_0_3=bool(P["0.30"]["lo"] > -a.margin), wrong_chutes_v2=int(wrong), new_stop_classes=new_classes)
    verdict["accepted"] = bool(verdict["fast_beats_v1"] and verdict["not_worse_at_0_3"] and wrong == 0 and not new_classes)
    R = dict(date=time.strftime("%Y-%m-%d %H:%M"), stream=STREAM, seconds=a.seconds, seeds=[a.seed0, a.seed0 + a.seeds - 1], v2=list(v2), margin=a.margin,
             rule="0.5 m/s: 95% CI of (v2 - v1) sorted/min above 0; 0.3 m/s: its lower bound above -margin; 0 wrong chutes and no new protective-stop class for v2",
             table=T, paired=P, verdict=verdict, wall_s=round(time.time() - t0, 1))
    json.dump(R, open(os.path.join(HERE, "results", "f7-eval.json"), "w"), indent=1)
    print(json.dumps(dict(paired=P, verdict=verdict), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0]); p.add_argument("what", choices=["ablate", "eval"])
    p.add_argument("--speeds", default="0.3,0.5"); p.add_argument("--seeds", type=int, default=6); p.add_argument("--seed0", type=int, default=7100)
    p.add_argument("--seconds", type=float, default=SECONDS); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--v2", default=""); p.add_argument("--name", default="", help="ablate: output name (default f7-ablation)"); p.add_argument("--margin", type=float, default=0.5)
    a = p.parse_args()
    ablate(a) if a.what == "ablate" else evaluate(a)
