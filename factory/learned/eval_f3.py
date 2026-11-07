"""eval_f3.py — the F3 acceptance evaluation (2026-11-02, docs/sorting-cell-spec.md §4 F3): does the learned carry beat F2
on L2 (random spacing, mixed materials) with objects it never trained on, beyond the noise floor?

Three carries on the same line and the same seeds: F2 (the classical carry), the prior (the hand-written PD + release
rule the policy starts from, i.e. what F3 gets without learning) and the learned policy. Two streams:
  held-out   only the 3 held-out items (carry.HELD_OUT) arrive
  mixed      all 11 items; the held-out items' share is also counted on its own
Belt 0.15 and 0.3 m/s, 180 s per run, --seeds seeds from --seed0 (default 6 from 0; the recorded attempt 2 used
10 from 0; the F3 start states came from seeds 100+). The noise floor is the
spread of the paired difference (same speed, rate, stream, seed) across seeds: a carry wins when the 95% interval of
its mean paired gain over F2 lies above zero.

    envs/mj/bin/python -I factory/learned/eval_f3.py [--seeds 6] [--seconds 180] [--workers 14] [--policy factory/learned/policy.json]
        -> factory/learned/results/f3-eval.json, runs/f3-eval/episodes.jsonl
"""
import argparse, itertools, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import carry  # noqa: E402

T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 12: 2.20, 16: 2.13, 20: 2.09}


def _one(cfg):
    import warnings; warnings.filterwarnings("ignore")
    from line import Line
    from line_learned import LearnedLine
    items = list(carry.HELD_OUT) if cfg["stream"] == "held-out" else None
    kw = dict(speed=cfg["speed"], rate_per_min=cfg["rate"], scheduler="lookahead", seed=cfg["seed"], items=items)
    ln = Line(**kw) if cfg["carry"] == "f2" else LearnedLine(policy="prior" if cfg["carry"] == "prior" else cfg["policy"], **kw)
    out = ln.run(seconds=cfg["seconds"]); S = out["summary"]; mins = cfg["seconds"] / 60.0
    ho = [i for i in out["items"] if i["name"] in carry.HELD_OUT]
    S["held_out_sorted_per_min"] = round(sum(i["outcome"] == "sorted" for i in ho) / mins, 3)
    S["held_out_items"] = len(ho); S["held_out_sort_rate"] = round(sum(i["outcome"] == "sorted" for i in ho) / max(len(ho), 1), 3)
    return dict(config=cfg, summary=S, attempts=out["attempts"])


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return dict(mean=round(float(x.mean()), 3) if n else None, lo=None, hi=None, n=n)
    h = T95.get(n, 2.0) * x.std(ddof=1) / math.sqrt(n); return dict(mean=round(float(x.mean()), 3), lo=round(float(x.mean() - h), 3), hi=round(float(x.mean() + h), 3), n=n)


def analyse(eps, carries=("f2", "prior", "learned")):
    key = lambda c: (c["stream"], c["speed"], c["rate"])
    groups = sorted({key(e["config"]) for e in eps}); table = {}; paired = {}
    for g in groups:
        for cr in carries:
            ss = [e["summary"] for e in eps if key(e["config"]) == g and e["config"]["carry"] == cr]
            if not ss: continue
            m = lambda k: round(float(np.mean([s[k] for s in ss if s[k] is not None])), 3)
            table[f"{g[0]} · {g[1]:.2f} m/s · {g[2]:g}/min · {cr}"] = dict(
                sorted_per_min=m("sorted_per_min"), sorted_per_min_sd=round(float(np.std([s["sorted_per_min"] for s in ss])), 3),
                held_out_sorted_per_min=m("held_out_sorted_per_min"), held_out_sort_rate=m("held_out_sort_rate"), miss_rate=m("miss_rate"),
                attempt_success=m("attempt_success"), cycle_s=m("cycle_s"), protective_stops=int(sum(s["collisions"] for s in ss)),
                dropped=int(sum(s["outcomes"].get("dropped", 0) for s in ss)), wrong_chute=int(sum(s["outcomes"].get("wrong_chute", 0) for s in ss)), runs=len(ss))
        for cr in carries[1:]:
            d_all, d_ho = [], []
            for e in eps:
                c = e["config"]
                if key(c) != g or c["carry"] != cr: continue
                base = next((b for b in eps if key(b["config"]) == g and b["config"]["carry"] == "f2" and b["config"]["seed"] == c["seed"]), None)
                if base: d_all.append(e["summary"]["sorted_per_min"] - base["summary"]["sorted_per_min"]); d_ho.append(e["summary"]["held_out_sorted_per_min"] - base["summary"]["held_out_sorted_per_min"])
            if d_all:
                paired[f"{g[0]} · {g[1]:.2f} m/s · {g[2]:g}/min · {cr} − f2"] = dict(sorted_per_min=ci(d_all), held_out_sorted_per_min=ci(d_ho))
    return table, paired


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, default=6); p.add_argument("--seed0", type=int, default=0, help="first seed (a fresh evaluation after a change uses new seeds)"); p.add_argument("--seconds", type=float, default=180.0); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--policy", default=os.path.join(HERE, "policy.json")); p.add_argument("--speeds", default="0.15,0.3")
    p.add_argument("--streams", default="held-out,mixed"); p.add_argument("--carries", default="f2,prior,learned"); p.add_argument("--name", default="f3-eval")
    a = p.parse_args()
    rate = {"held-out": 8.0, "mixed": 16.0}
    cfgs = [dict(stream=st, speed=float(v), rate=rate[st], seed=k, carry=cr, seconds=a.seconds, policy=os.path.abspath(a.policy))
            for st, v, k, cr in itertools.product(a.streams.split(","), a.speeds.split(","), range(a.seed0, a.seed0 + a.seeds), a.carries.split(","))]
    import multiprocessing as mp
    run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True); t0 = time.perf_counter()
    # each run is appended as it finishes; a rerun skips the configurations already on disk (memguard may stop a long eval)
    path = os.path.join(run, "episodes.jsonl"); eps = []
    if os.path.exists(path):
        want = {json.dumps(c, sort_keys=True) for c in cfgs}
        eps = [e for e in map(json.loads, open(path)) if json.dumps(e["config"], sort_keys=True) in want]
    done = {json.dumps(e["config"], sort_keys=True) for e in eps}; todo = [c for c in cfgs if json.dumps(c, sort_keys=True) not in done]
    print(f"{len(eps)} runs on disk, {len(todo)} to go", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=carry.die_with_parent, maxtasksperchild=3) as pool, open(path, "a") as f:   # workers grow ~0.5 GB per long run: recycle them
        for e in pool.imap_unordered(_one, todo):
            f.write(json.dumps(e, default=float) + "\n"); f.flush(); eps.append(e)
    table, paired = analyse(eps, tuple(a.carries.split(",")))
    pol = json.load(open(a.policy)) if os.path.exists(a.policy) else {}
    verdict = {}
    for k, v in paired.items():
        if not k.endswith("learned − f2"): continue
        metric = v["held_out_sorted_per_min"]; verdict[k] = dict(held_out_gain=metric["mean"], ci95=[metric["lo"], metric["hi"]], beats_f2=bool(metric["lo"] is not None and metric["lo"] > 0))
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), seconds=a.seconds, seeds=a.seeds, seed0=a.seed0, held_out=list(carry.HELD_OUT), policy_iter=pol.get("iter"),
             rates=rate, table=table, paired=paired, verdict=verdict, accepted=bool(verdict) and all(v["beats_f2"] for v in verdict.values()),
             wall_s=round(time.perf_counter() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(S, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
    for k, v in table.items(): print(k, v)
    for k, v in paired.items(): print(k, v)
    print("accepted:", S["accepted"])


if __name__ == "__main__":
    main()
