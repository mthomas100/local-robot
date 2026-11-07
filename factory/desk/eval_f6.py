"""eval_f6.py — the F6 acceptance evaluation (2026-11-03; protocol: factory/desk/results/f6-eval-protocol.md, committed
before the run). With the real-device kit on (servos: PD loop, 12-bit encoder, deadband, bus delay, back-EMF, horn
play; camera: depth and RGB noise, latency and jitter, a planted hand-eye error that the cell's calibration removes):

  calib   the calibration routine on several planted hand-eye errors: recovered within 2 mm and 1 deg?
  picks   200 seeded L0 trials of the admitted items, the classical pipeline and the ONNX policy on the same seeds
  rate    the ONNX loop paced against the wall clock: does it hold 30 Hz?

    envs/mj/bin/python -I factory/desk/eval_f6.py calib|picks|rate|all [--workers 10]
Writes factory/desk/results/f6-eval.json (merged, one key per part).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import desk_items as I  # noqa: E402

OUT = os.path.join(HERE, "results", "f6-eval.json")
# Attempt 2 (protocol f6-eval-protocol.md, 2026-11-03): fresh seeds. Attempt 1 used picks 20000-20199,
# camera 4242, calibration 4242-4246, rate 20200-20219 (results/f6-eval-attempt1.json); those seeds then served
# development of the calibration fix, so they are spent.
EVAL_SEED0 = 21_000; N_TRIALS = 200
CAM_SEED = 4300                       # the planted hand-eye error of the evaluation session
CALIB_SEEDS = [4300, 4301, 4302, 4303, 4304]
RATE_SEED0 = 21_200; RATE_TRIALS = 20


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _session(cam_seed):
    import camera as C
    from desk_sim import DeskSim
    from servo import DeviceKit
    sk = DeviceKit.real_kit(); sim = DeskSim(kit=sk); cam = C.Camera(sim, C.CameraKit.real_kit(seed=cam_seed), rng=np.random.default_rng(cam_seed))
    rep = C.calibrate(sim, cam, sk.backlash_rad); return sim, cam, rep, sk


def calib():
    rows = []
    for s in CALIB_SEEDS:
        t0 = time.time(); sim, cam, rep, sk = _session(s); rep["seed"] = s; rep["seconds"] = round(time.time() - t0, 1)
        rep["ok_2mm_1deg"] = bool(rep["ok"] and rep["after_mm"] <= 2.0 and rep["after_deg"] <= 1.0); rows.append(rep); cam.close()
        print(json.dumps(rep), flush=True)
    table = {f"seed {r['seed']}": dict(before_mm=r["before_mm"], before_deg=r["before_deg"], after_mm=r.get("after_mm"), after_deg=r.get("after_deg"), poses=r.get("n"), passed=r["ok_2mm_1deg"]) for r in rows}
    return dict(planted="8 mm, 2.5 deg in seeded random directions (CameraKit.real_kit)", rows=rows, table=table, passed=all(r["ok_2mm_1deg"] for r in rows))


def _pick_worker(seeds):
    import desk_pick as P, bc
    sim, cam, rep, sk = _session(CAM_SEED); pol = bc.OnnxPolicy(); out = []
    items = I.admitted()
    for seed in seeds:
        name = items[seed % len(items)]
        c = P.trial(sim, name, seed, cam=cam); o = bc.policy_trial(sim, pol, name, seed, cam)
        out.append(dict(seed=seed, item=name, classical=c, onnx={k: v for k, v in o.items() if k != "timing"}, infer_ms_p99=o.get("timing", {}).get("infer_ms_p99")))
    cam.close(); return dict(calibration=rep, trials=out)


def picks(workers):
    import multiprocessing as mp
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + N_TRIALS)); chunks = [seeds[i::workers] for i in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool: res = pool.map(_pick_worker, chunks)
    trials = sorted([t for r in res for t in r["trials"]], key=lambda t: t["seed"])
    summ = {}
    for key in ("classical", "onnx"):
        oc = Counter(t[key]["outcome"] for t in trials); k = oc.get("ok", 0)
        per = {}
        for n in I.admitted():
            tt = [t for t in trials if t["item"] == n]; per[n] = f"{sum(t[key]['outcome'] == 'ok' for t in tt)}/{len(tt)}"
        summ[key] = dict(ok=k, n=len(trials), rate=round(k / len(trials), 4), wilson95=wilson(k, len(trials)), outcomes=dict(oc), per_item=per,
                         wrong_class=sum(t[key].get("seen") not in (None, t["item"]) for t in trials))
    table = {k: dict(sorted=f"{summ[k]['ok']}/{summ[k]['n']}", rate=summ[k]["rate"], wilson95=summ[k]["wilson95"], bar=0.9 if k == "classical" else 0.8, **summ[k]["per_item"],
                     failures={o: c for o, c in summ[k]["outcomes"].items() if o != "ok"}) for k in ("classical", "onnx")}
    return dict(seeds=[seeds[0], seeds[-1]], cam_seed=CAM_SEED, table=table, calibration=res[0]["calibration"], seconds=round(time.time() - t0, 1),
                classical=summ["classical"], onnx=summ["onnx"], passed=bool(summ["classical"]["rate"] >= 0.90 and summ["onnx"]["rate"] >= 0.80),
                infer_ms_p99_max=max(t["infer_ms_p99"] or 0 for t in trials), trials=trials)


def rate():
    import bc
    sim, cam, rep, sk = _session(CAM_SEED); pol = bc.OnnxPolicy(); rows = []
    items = I.admitted()
    for s in range(RATE_SEED0, RATE_SEED0 + RATE_TRIALS):
        r = bc.policy_trial(sim, pol, items[s % len(items)], s, cam, pace=True); rows.append(dict(seed=s, outcome=r["outcome"], **r.get("timing", {})))
        print(json.dumps(rows[-1]), flush=True)
    per = [r["period_ms_mean"] for r in rows if "period_ms_mean" in r]
    agg = dict(period_ms_mean=round(float(np.mean(per)), 3), period_ms_p99_max=round(max(r["period_ms_p99"] for r in rows), 3),
               period_ms_max=round(max(r["period_ms_max"] for r in rows), 3), late_ticks=int(sum(r["late_ticks"] for r in rows)),
               infer_ms_p99_max=round(max(r["infer_ms_p99"] for r in rows), 3), ticks=int(len(rows) * round(bc.T_RUN * 30)))
    agg["passed"] = bool(abs(agg["period_ms_mean"] - 1000 / 30) <= 1000 / 30 * 0.01 and agg["late_ticks"] == 0)
    cam.close(); return dict(rule="mean tick period within 1% of 33.33 ms and no tick later than 1.5 periods", summary=agg, rows=rows)


def save(part, data):
    allr = json.load(open(OUT)) if os.path.exists(OUT) else {}
    allr[part] = data; allr["date"] = "2026-11-03"; json.dump(allr, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["calib", "picks", "rate", "all"]); ap.add_argument("--workers", type=int, default=10); a = ap.parse_args()
    if a.what in ("calib", "all"): r = calib(); save("calib", r); print("calib passed:", r["passed"])
    if a.what in ("picks", "all"):
        r = picks(a.workers); save("picks", r)
        print("picks:", json.dumps({k: r[k] for k in ("classical", "onnx", "passed", "infer_ms_p99_max", "seconds")}, indent=1))
    if a.what in ("rate", "all"): r = rate(); save("rate", r); print("rate:", json.dumps(r["summary"]))
