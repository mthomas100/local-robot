"""eval_f1.py — the F1 acceptance evaluation (2026-11-02, docs/sorting-cell-spec.md §4 F1): N L0 trials (belt stopped, one
item standing on it), items cycled over all 11, x uniform in the reach zone within +-0.4 m, y within +-0.08 m of the
belt centre, yaw uniform; every trial seeded so it replays exactly. Trials run in parallel worker processes (each with
its own MuJoCo cell), CPU only.

Writes:
  runs/<name>/trials.jsonl                  one line per trial (pose, tool, grasp, outcome, failure class, cycle time)
  runs/<name>/config.json                   the arguments and the layout
  factory/classical/results/<name>.json     the summary: success rate with a 95% Wilson interval, per item, per
                                            failure class, cycle and planning times, max tracking error
  runs/<name>/eval.mp4 + eval-sheet.png     a few trials rendered back to back from the 'pick' camera (--video K)

    envs/mj/bin/python -I factory/classical/eval_f1.py [--trials 200] [--workers 8] [--video 6] [--name f1-eval]
    envs/mj/bin/python -I factory/classical/eval_f1.py --replay <index> [--name f1-eval]   # rerun one trial, verbose
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))

ITEMS = ("master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "tuna_fish_can", "pudding_box",
         "gelatin_box", "potted_meat_can", "bleach_cleanser", "windex_bottle")


def trial_specs(n, seed=0, x_half=0.4, y_half=0.08):
    """The trial list: item k mod 11, pose from a per-trial generator seeded with (seed, k)."""
    from layout import DEFAULT as L
    out = []
    for k in range(n):
        r = np.random.default_rng([seed, k])
        out.append(dict(index=k, item=ITEMS[k % len(ITEMS)], x=float(r.uniform(-x_half, x_half)), y=float(L.belt.y + r.uniform(-y_half, y_half)),
                        yaw=float(r.uniform(-math.pi, math.pi)), seed=k))
    return out


_W = {}


def _worker_init():
    import warnings; warnings.filterwarnings("ignore")
    from procs import die_with_parent; die_with_parent()   # memguard kills only the parent (factory/cell/procs.py)
    from cell_mujoco import CellMJ
    import pick
    _W["cell"] = CellMJ(); _W["planner"] = pick.Planner(_W["cell"]); _W["pick"] = pick


def _run(spec):
    t0 = time.perf_counter()
    r = _W["pick"].trial(_W["cell"], _W["planner"], spec["item"], spec["x"], spec["y"], spec["yaw"], seed=spec["seed"])
    r["index"] = spec["index"]; r["wall_s"] = round(time.perf_counter() - t0, 2)
    return r


def _plan_check(spec):
    """Plan one pick (no dynamics) and sweep the planned path kinematically: at every waypoint (joint-space segments
    subdivided to <= 0.05 rad) the robot must not touch a fixture, and while carried the item (rigidly attached to the
    tool where the grasp put it) must not penetrate a fixture by more than 2 mm. Returns planning time and path time."""
    import mujoco
    pick = _W["pick"]; c = _W["cell"]; pl = _W["planner"]; m, d = c.model, c.data; name = spec["item"]
    c.reset(); c.spawn(name, x=spec["x"], y=spec["y"], yaw=spec["yaw"], with_belt_velocity=False, lift=0.0); mujoco.mj_forward(m, d)
    st = c.item_state(name); rec = c.items[name]["rec"]
    plan, why = pl.plan(name, st["pos"], st["quat"], c.lay.chute(rec["material"]))
    out = dict(index=spec["index"], item=name, planned=plan is not None, failure=why)
    if plan is None: return out
    if "_ex" not in _W: _W["_ex"] = pick.Executor(c)
    ex = _W["_ex"]; it = c.items[name]; qa = it["qpos"]; item_geoms = np.flatnonzero(m.geom_bodyid == it["body"])
    fixture = ex.fixture_geom; held = None; worst_item = 0.0; hits = []
    for seg in plan.segments:
        P = seg.path; pts = [P[0]]
        for a_, b_ in zip(P[:-1], P[1:]):
            n = max(1, int(np.ceil(np.max(np.abs(b_ - a_)) / 0.05))); pts += [a_ + (b_ - a_) * (k / n) for k in range(1, n + 1)]
        for q in pts:
            d.qpos[c.robot_qpos] = q
            if held is not None:
                mujoco.mj_kinematics(m, d); Rt = d.site_xmat[c.pinch_site].reshape(3, 3); pt = d.site_xpos[c.pinch_site]
                d.qpos[qa:qa + 3] = pt + Rt @ held[0]; qq = np.zeros(4); mujoco.mju_mat2Quat(qq, (Rt @ held[1]).reshape(-1)); d.qpos[qa + 3:qa + 7] = qq
            mujoco.mj_forward(m, d)
            hit = ex.robot_fixture_contact()
            if hit: hits.append((seg.name, hit)); break
            if held is not None and seg.name in ("transfer", "lower"):
                for i in range(d.ncon):
                    cc = d.contact[i]
                    if (cc.geom1 in item_geoms and fixture[cc.geom2]) or (cc.geom2 in item_geoms and fixture[cc.geom1]): worst_item = max(worst_item, -cc.dist)
        if seg.name in ("close", "suck"):   # from here the item rides with the tool
            mujoco.mj_kinematics(m, d); Rt = d.site_xmat[c.pinch_site].reshape(3, 3); pt = d.site_xpos[c.pinch_site].copy()
            Ri = np.zeros(9); mujoco.mju_quat2Mat(Ri, d.qpos[qa + 3:qa + 7]); Ri = Ri.reshape(3, 3)
            held = (Rt.T @ (d.qpos[qa:qa + 3] - pt), Rt.T @ Ri)
        if seg.name == "release": held = None
    out.update(plan_ms=round(plan.plan_seconds * 1000, 2), path_s=round(sum(sg.T + sg.dwell for sg in plan.segments), 2), robot_hits=hits[:1],
               item_penetration_mm=round(worst_item * 1000, 2), item_hit=worst_item > 0.002, tool=plan.tool, tilt=round(math.degrees(plan.grasp.get("tilt", 0)), 1))
    return out


def plan_sweep(n, seed=7, workers=6, x_half=0.45, y_half=0.12):
    """Discipline 5's acceptance: plan n picks over the zone (wider than the eval: +-0.45 m, +-0.12 m) and sweep each."""
    import multiprocessing as mp
    specs = trial_specs(n, seed, x_half, y_half)
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init) as pool: rows = pool.map(_plan_check, specs, chunksize=8)
    planned = [r for r in rows if r["planned"]]
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), plans=n, planned=len(planned), unplanned=dict(Counter(r["failure"] for r in rows if not r["planned"])),
             robot_collisions=sum(bool(r["robot_hits"]) for r in planned), item_hits=sum(r["item_hit"] for r in planned),
             worst_item_penetration_mm=max([r["item_penetration_mm"] for r in planned] or [0]),
             plan_ms=dict(mean=round(float(np.mean([r["plan_ms"] for r in planned])), 2), p95=round(float(np.percentile([r["plan_ms"] for r in planned], 95)), 2)),
             path_s=dict(mean=round(float(np.mean([r["path_s"] for r in planned])), 2), max=round(float(np.max([r["path_s"] for r in planned])), 2)),
             tilted=sum(r["tilt"] != 0 for r in planned), examples=[r for r in rows if not r["planned"] or r["robot_hits"] or r["item_hit"]][:20])
    return S


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def summarize(rows, args):
    n = len(rows); ok = sum(r["outcome"] == "success" for r in rows); lo, hi = wilson(ok, n)
    per_item = defaultdict(lambda: Counter())
    for r in rows: per_item[r["item"]][r["outcome"]] += 1
    cyc = [r["cycle_s"] for r in rows if r["outcome"] == "success" and r.get("cycle_s")]
    plan = [r["plan_ms"] for r in rows if r.get("plan_ms") is not None]
    trk = [r["max_track_err_rad"] for r in rows if r.get("max_track_err_rad") is not None]
    tools = Counter(r.get("tool") for r in rows)
    return dict(
        date=time.strftime("%Y-%m-%d %H:%M"), name=args.name, trials=n, successes=ok, success_rate=round(ok / max(n, 1), 4), wilson95=[lo, hi],
        acceptance=dict(threshold=0.95, passed=ok / max(n, 1) >= 0.95),
        failures=dict(Counter(r["outcome"] for r in rows if r["outcome"] != "success")),
        per_item={k: dict(v, n=sum(v.values()), rate=round(v["success"] / sum(v.values()), 3)) for k, v in sorted(per_item.items())},
        tools={str(k): v for k, v in tools.items()},
        cycle_s=dict(mean=round(float(np.mean(cyc)), 2), p50=round(float(np.median(cyc)), 2), p95=round(float(np.percentile(cyc, 95)), 2), max=round(float(np.max(cyc)), 2)) if cyc else None,
        plan_ms=dict(mean=round(float(np.mean(plan)), 1), p95=round(float(np.percentile(plan, 95)), 1), max=round(float(np.max(plan)), 1)) if plan else None,
        max_track_err_rad=round(float(np.max(trk)), 4) if trk else None,
        collisions=sum(r["outcome"] == "collision" for r in rows),
        sampling=dict(x_half=args.x_half, y_half=args.y_half, seed=args.seed, items="all 11, cycled"),
        failed_trials=[dict(index=r["index"], item=r["item"], outcome=r["outcome"], x=r["x"], y=r["y"], yaw=r["yaw"], detail=r.get("detail")) for r in rows if r["outcome"] != "success"][:60],
    )


def video(specs, path, width=960, height=540, fps=30, camera="pick"):
    """Render the given trials back to back into one mp4, a caption burned into each frame."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw
    _worker_init(); cell = _W["cell"]; frames = []; every = max(1, round(1.0 / (cell.model.opt.timestep * fps)))
    for s in specs:
        cap = f"#{s['index']} {s['item']}  x {s['x']:+.2f}  yaw {math.degrees(s['yaw']):+.0f} deg"

        def grab(c, cap=cap):
            im = Image.fromarray(c.render(camera, width, height)); d = ImageDraw.Draw(im); d.rectangle((0, 0, width, 26), fill=(0, 0, 0))
            d.text((8, 6), cap, fill=(255, 255, 255)); frames.append(np.asarray(im))
        r = _W["pick"].trial(cell, _W["planner"], s["item"], s["x"], s["y"], s["yaw"], seed=s["seed"], on_frame=grab, frame_every=every)
        for _ in range(int(0.5 * fps)): grab(cell, cap + f"  -> {r['outcome']}")
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", plugin="pyav")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trials", type=int, default=200); p.add_argument("--workers", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x-half", type=float, default=0.4); p.add_argument("--y-half", type=float, default=0.08)
    p.add_argument("--video", type=int, default=6, help="render this many trials (failures first) into eval.mp4; 0 = none")
    p.add_argument("--name", default="f1-eval"); p.add_argument("--replay", type=int, default=None)
    p.add_argument("--only", default=None, help="comma-separated items to evaluate (default: all, cycled)")
    p.add_argument("--plans", type=int, default=0, help="instead of trials: plan this many picks and sweep the paths for collisions (discipline 5)")
    a = p.parse_args()
    if a.plans:
        t0 = time.perf_counter(); S = plan_sweep(a.plans, workers=a.workers); S["wall_s"] = round(time.perf_counter() - t0, 1)
        out = os.path.join(HERE, "results", "f1-plan-sweep.json"); json.dump(S, open(out, "w"), indent=1)
        print(json.dumps({k: v for k, v in S.items() if k != "examples"})); print("summary:", out); return
    specs = trial_specs(a.trials, a.seed, a.x_half, a.y_half)
    if a.only: specs = [s for s in specs if s["item"] in a.only.split(",")]
    if a.replay is not None:
        s = trial_specs(a.replay + 1, a.seed, a.x_half, a.y_half)[a.replay]; print(s)
        if a.video:   # render the replay from two cameras and open nothing (the caller looks at the sheet)
            run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True)
            for cam in ("pick", "belt"):
                vp = video([s], os.path.join(run, f"replay-{a.replay}-{cam}.mp4"), camera=cam)
                os.system(f"{ROOT}/bin/rl-sheet {vp} {vp[:-4]}-sheet.png 5 4 >/dev/null 2>&1"); print("sheet:", vp[:-4] + "-sheet.png")
            return
        _worker_init(); print(_run(s)); return
    run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True); res_dir = os.path.join(HERE, "results"); os.makedirs(res_dir, exist_ok=True)
    from layout import DEFAULT as L
    with open(os.path.join(run, "config.json"), "w") as f: json.dump(dict(vars(a), layout=json.loads(L.to_json())), f, indent=1)
    t0 = time.perf_counter(); rows = []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_worker_init) as pool, open(os.path.join(run, "trials.jsonl"), "w") as f:
        for r in pool.imap_unordered(_run, specs):
            rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
            if len(rows) % 20 == 0 or r["outcome"] != "success":
                print(f"[{len(rows):4d}/{len(specs)}] #{r['index']:4d} {r['item']:16s} {r['outcome']:12s} {'' if r['outcome'] == 'success' else str(r.get('detail', ''))[:90]}", flush=True)
    rows.sort(key=lambda r: r["index"])
    S = summarize(rows, a); S["wall_s"] = round(time.perf_counter() - t0, 1)
    out = os.path.join(res_dir, f"{a.name}.json")
    with open(out, "w") as f: json.dump(S, f, indent=1)
    print(json.dumps({k: S[k] for k in ("trials", "successes", "success_rate", "wilson95", "failures", "cycle_s", "plan_ms", "max_track_err_rad", "wall_s")}))
    print("per item:", {k: v["rate"] for k, v in S["per_item"].items()})
    print("summary:", out)
    if a.video:
        fails = [s for s in specs if rows[s["index"]]["outcome"] != "success"] if len(rows) == len(specs) else []
        pick_specs = (fails[: max(1, a.video // 3)] + [s for s in specs if s not in fails])[: a.video]   # an instructive failure, then the first items in order
        vp = video(pick_specs, os.path.join(run, "eval.mp4")); print("video:", vp)
        os.system(f"{ROOT}/bin/rl-sheet {vp} {os.path.join(run, 'eval-sheet.png')} >/dev/null 2>&1"); print("sheet:", os.path.join(run, "eval-sheet.png"))


if __name__ == "__main__":
    main()
