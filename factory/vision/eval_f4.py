"""eval_f4.py — the F4 acceptance evaluation (2026-11-02; docs/sorting-cell-spec.md §4 F4): with the camera as
its only source of where items are and what they are (L4), does the cell sort within 10 points of the same cell told
the true poses (L2)?

Four cells on the same L2 stream (random spacing, all 11 items, 16 arrivals/min), same seeds, belt 0.15 and 0.3 m/s:
  L2           privileged detections (tracking.Detector: true pose + noise, true identity) + the F3 carry (teacher,
               reads the held item's true pose)
  L4           vision.VisionDetector (overhead D435i: depth blobs, kNN identity, tracks) + the F4 student carry
               (CarryCtl(privileged=False), distilled from the teacher: factory/learned/student.json)
  L2-f2, L4-f2 the same two perceptions with F2's classical carry (isolates what perception alone costs)
Success = sort rate (items sorted / items that arrived and left). Rule, fixed before the run: L4 passes at a belt
speed when the 95% t-interval of the paired difference L2 - L4 (same seed) has its upper end below 0.10 (a
non-inferiority test: "within 10 points" must hold beyond the noise, not just on average). Accepted when it passes at
both speeds.

    bin/memguard 56 envs/mj/bin/python -I factory/vision/eval_f4.py [--seeds 16] [--seconds 180] [--workers 8]
        -> factory/vision/results/<name>.json, runs/<name>/episodes.jsonl  (resumable)
Rendering: each vision run renders 30 frames per simulated second with OpenGL (light GPU use).
"""
import argparse, itertools, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
for p in (HERE, os.path.join(FACTORY, "learned"), os.path.join(FACTORY, "line")): sys.path.insert(0, p)

CELLS = {"L2": ("privileged", "teacher"), "L4": ("vision", "student"), "L2-f2": ("privileged", "f2"), "L4-f2": ("vision", "f2")}
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 12: 2.20, 16: 2.13, 20: 2.09}


def _one(cfg):
    import warnings; warnings.filterwarnings("ignore")
    from line import Line
    from line_learned import LearnedLine
    per, cr = CELLS[cfg["cell"]]
    kw = dict(speed=cfg["speed"], rate_per_min=cfg["rate"], scheduler="lookahead", seed=cfg["seed"], perception=per)
    if cr == "f2": ln = Line(**kw)
    elif cr == "teacher": ln = LearnedLine(policy=cfg["teacher"], **kw)
    else: ln = LearnedLine(policy=cfg["student"], privileged_carry=False, **kw)
    out = ln.run(seconds=cfg["seconds"]); S = out["summary"]
    if per == "vision": S["vision_ms_per_frame"] = round(float(np.mean(ln.vision.ms)), 1); S["vision_frames"] = ln.vision.frames
    return dict(config=cfg, summary=S, items=[{k: i.get(k) for k in ("name", "outcome")} for i in out["items"]])


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return dict(mean=round(float(x.mean()), 4) if n else None, lo=None, hi=None, n=n)
    h = T95.get(n, 2.0) * x.std(ddof=1) / math.sqrt(n); return dict(mean=round(float(x.mean()), 4), lo=round(float(x.mean() - h), 4), hi=round(float(x.mean() + h), 4), n=n)


def analyse(eps):
    speeds = sorted({e["config"]["speed"] for e in eps}); table, paired, verdict = {}, {}, {}
    get = lambda cell, sp: {e["config"]["seed"]: e["summary"] for e in eps if e["config"]["cell"] == cell and e["config"]["speed"] == sp}
    for sp in speeds:
        for cell in CELLS:
            ss = list(get(cell, sp).values())
            if not ss: continue
            table[f"{sp:.2f} m/s · {cell}"] = dict(sort_rate=round(float(np.mean([s["sort_rate"] for s in ss])), 4), sorted_per_min=round(float(np.mean([s["sorted_per_min"] for s in ss])), 3),
                                                   attempt_success=round(float(np.mean([s["attempt_success"] for s in ss])), 3), wrong_chute=int(sum(s["outcomes"].get("wrong_chute", 0) for s in ss)),
                                                   dropped=int(sum(s["outcomes"].get("dropped", 0) for s in ss)), knocked=int(sum(s["attempts"].get("knocked", 0) for s in ss)),
                                                   grasp_err_mm=round(float(np.mean([s["grasp_err_mm"]["mean"] for s in ss if s.get("grasp_err_mm")])), 1) if any(s.get("grasp_err_mm") for s in ss) else None,
                                                   ms_per_frame=round(float(np.mean([s["vision_ms_per_frame"] for s in ss])), 1) if "vision_ms_per_frame" in ss[0] else None, runs=len(ss))
        for a, b in (("L2", "L4"), ("L2-f2", "L4-f2")):
            A, B = get(a, sp), get(b, sp); common = sorted(set(A) & set(B))
            if common: paired[f"{sp:.2f} m/s · {a} − {b}"] = ci([A[k]["sort_rate"] - B[k]["sort_rate"] for k in common])
        d = paired.get(f"{sp:.2f} m/s · L2 − L4")
        if d and d["hi"] is not None: verdict[f"{sp:.2f} m/s"] = dict(gap=d["mean"], ci95=[d["lo"], d["hi"]], within_10_points=bool(d["hi"] < 0.10))
    return table, paired, verdict


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, default=16); p.add_argument("--seed0", type=int, default=300); p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--speeds", default="0.15,0.3"); p.add_argument("--cells", default=",".join(CELLS)); p.add_argument("--name", default="f4-eval")
    p.add_argument("--teacher", default=os.path.join(FACTORY, "learned", "policy.json")); p.add_argument("--student", default=os.path.join(FACTORY, "learned", "student.json"))
    a = p.parse_args()
    cfgs = [dict(cell=c, speed=float(v), rate=16.0, seed=k, seconds=a.seconds, teacher=os.path.abspath(a.teacher), student=os.path.abspath(a.student))
            for v, k, c in itertools.product(a.speeds.split(","), range(a.seed0, a.seed0 + a.seeds), a.cells.split(","))]
    cfgs.sort(key=lambda c: CELLS[c["cell"]][0] != "vision")   # the slow (rendering) runs first
    import multiprocessing as mp
    sys.path.insert(0, os.path.join(FACTORY, "cell")); from procs import die_with_parent
    run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True); t0 = time.perf_counter(); path = os.path.join(run, "episodes.jsonl"); eps = []
    if os.path.exists(path):
        want = {json.dumps(c, sort_keys=True) for c in cfgs}; eps = [e for e in map(json.loads, open(path)) if json.dumps(e["config"], sort_keys=True) in want]
    done = {json.dumps(e["config"], sort_keys=True) for e in eps}; todo = [c for c in cfgs if json.dumps(c, sort_keys=True) not in done]
    print(f"{len(eps)} runs on disk, {len(todo)} to go", flush=True)
    with mp.get_context("spawn").Pool(a.workers, initializer=die_with_parent, maxtasksperchild=3) as pool, open(path, "a") as f:
        for e in pool.imap_unordered(_one, todo):
            f.write(json.dumps(e, default=float) + "\n"); f.flush(); eps.append(e); print(e["config"]["cell"], e["config"]["speed"], e["config"]["seed"], e["summary"]["sort_rate"], flush=True)
    table, paired, verdict = analyse(eps)
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), seconds=a.seconds, seeds=a.seeds, seed0=a.seed0, rate_per_min=16.0, cells={k: dict(perception=v[0], carry=v[1]) for k, v in CELLS.items()},
             table=table, paired=paired, verdict=verdict, accepted=bool(verdict) and len(verdict) == len(a.speeds.split(",")) and all(v["within_10_points"] for v in verdict.values()),
             rule="L2 - L4 paired sort-rate difference: upper end of the 95% interval below 0.10 at every belt speed", wall_s=round(time.perf_counter() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(S, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
    for k, v in table.items(): print(k, v)
    for k, v in paired.items(): print(k, v)
    print("verdict", verdict, "accepted:", S["accepted"])


if __name__ == "__main__":
    main()
