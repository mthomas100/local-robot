"""train_f3.py — F3: train the carry policy with Augmented Random Search (2026-11-02).

ARS ("Simple random search provides a competitive approach to reinforcement learning", NeurIPS 2018):
perturb the policy's parameters in N random directions, run episodes with +delta and -delta on the same start states,
and step along the directions whose two sides differed most, scaled by the spread of the returns. No gradients, no
value function; every episode is independent, so 14 CPU processes run them side by side (no GPU).

    envs/mj/bin/python -I factory/learned/train_f3.py [--iters 120] [--workers 14] [--name f3-ars]
    envs/mj/bin/python -I factory/learned/train_f3.py --eval runs/f3-ars/policy.json      # one policy on every start state

Episode: a snapshot of a real F2 grasp (snapshots.py), then the carry policy until the robot is free and the item has
landed. Return = 25 if the item is sorted (carry.SUCCESS_R), minus the seconds until the robot is free (a failure costs
the full 4 s budget): one lost item is worth about 25 s of carrying (run 1 used 15 and traded reliability for speed). Domain randomisation: the held item's mass x0.85-1.2 and its
friction x0.7-1.1. Curriculum: light items (< 0.45 kg) for the first --light fraction of iterations (default 0.1), then all 8 training items.
The 3 held-out items (carry.HELD_OUT) are never trained on; their start states only appear in the evaluations.
Writes runs/<name>/metrics.jsonl, config.json, ckpt-XXXX.json, policy.json (the best by training-set return), and
factory/learned/results/f3-train.json.
"""
import argparse, json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import carry  # noqa: E402

LIGHT = 0.45
_ENV = None; _SNAPS = None


_PRIV = True


def _init(path, privileged=True):
    global _ENV, _SNAPS, _PRIV
    carry.die_with_parent(); _PRIV = privileged
    import warnings; warnings.filterwarnings("ignore")
    _SNAPS = pickle.load(open(path, "rb")); _ENV = carry.Env()


def _run(job):
    theta, i, dr, seed = job
    r = _ENV.episode(_SNAPS[i], carry.Policy(theta), dr=dr, seed=seed, privileged=_PRIV); r["i"] = i; return r


def summarize(rs, snaps):
    out = {}
    for split, keep in (("train", lambda n: n not in carry.HELD_OUT), ("held_out", lambda n: n in carry.HELD_OUT)):
        sub = [r for r in rs if keep(r["item"])]
        if not sub: continue
        ok = [r for r in sub if r["ok"]]; f2 = [snaps[r["i"]]["f2_free_s"] for r in sub if snaps[r["i"]]["f2_free_s"] is not None]
        out[split] = dict(n=len(sub), success=round(len(ok) / len(sub), 4), t_free=round(float(np.mean([r["t_free"] for r in ok])), 3) if ok else None,
                          reward=round(float(np.mean([r["reward"] for r in sub])), 3), f2_t_free=round(float(np.mean(f2)), 3) if f2 else None,
                          f2_success=round(sum(snaps[r["i"]]["f2_item"] == "sorted" for r in sub) / len(sub), 4),
                          failures=dict(__import__("collections").Counter(r["outcome"] for r in sub if not r["ok"])))
    per = {}
    for r in rs:
        p = per.setdefault(r["item"], dict(n=0, ok=0, t=[]))
        p["n"] += 1; p["ok"] += r["ok"]
        if r["ok"]: p["t"].append(r["t_free"])
    out["per_item"] = {k: dict(n=v["n"], success=round(v["ok"] / v["n"], 3), t_free=round(float(np.mean(v["t"])), 2) if v["t"] else None,
                               f2_t_free=round(float(np.mean([s["f2_free_s"] for s in snaps if s["item"] == k and s["f2_free_s"] is not None])), 2),
                               held_out=k in carry.HELD_OUT) for k, v in sorted(per.items())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=80); p.add_argument("--workers", type=int, default=14); p.add_argument("--name", default="f3-ars")
    p.add_argument("--dirs", type=int, default=16); p.add_argument("--top", type=int, default=8); p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--step", type=float, default=0.03); p.add_argument("--per-dir", type=int, default=10); p.add_argument("--light", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10); p.add_argument("--snapshots", default=os.path.join(ROOT, "runs", "f3-snapshots", "snapshots.pkl"))
    p.add_argument("--init", default=None, help="start from this policy.json instead of the prior"); p.add_argument("--eval", default=None, help="evaluate this policy.json on every start state and exit"); p.add_argument("--dr", action="store_true", help="with --eval: randomise")
    p.add_argument("--student", action="store_true", help="F4: train/evaluate the student carry (CarryCtl(privileged=False): no true item state)")
    p.add_argument("--out", default=None, help="where the best policy goes (default factory/learned/policy.json; the student: factory/learned/student.json)")
    a = p.parse_args()
    snaps = pickle.load(open(a.snapshots, "rb"))
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.snapshots, not a.student))
    if a.eval:
        th = carry.Policy(json.load(open(a.eval))["theta"]).theta; t0 = time.perf_counter()
        rs = pool.map(_run, [(th, i, a.dr, 7 + i) for i in range(len(snaps))]); S = summarize(rs, snaps); S["wall_s"] = round(time.perf_counter() - t0, 1)
        print(json.dumps(S)); pool.close(); return
    rng = np.random.default_rng(a.seed); run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True)
    json.dump(vars(a) | dict(held_out=carry.HELD_OUT, features=carry.FEATURES, n_params=carry.NP, date=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(run, "config.json"), "w"), indent=1)
    sys.path.insert(0, os.path.join(ROOT, "factory", "cell")); import ycb
    mass = {r["name"]: r["mass_kg"] for r in ycb.load_index().values()}
    train_all = [i for i, s in enumerate(snaps) if s["item"] not in carry.HELD_OUT]
    train_light = [i for i in train_all if mass[snaps[i]["item"]] < LIGHT]
    eval_train = list(train_all); eval_held = [i for i, s in enumerate(snaps) if s["item"] in carry.HELD_OUT]
    theta = carry.Policy(json.load(open(a.init))["theta"]).theta if a.init else np.zeros(carry.NP); best = (-1e9, theta.copy(), -1); log = open(os.path.join(run, "metrics.jsonl"), "w"); t0 = time.perf_counter(); hist = []

    def evaluate(th, it):
        rs = pool.map(_run, [(th, i, False, 1000 + i) for i in eval_train + eval_held]); S = summarize(rs, snaps)
        json.dump(dict(iter=it, theta=[float(x) for x in th], eval=S), open(os.path.join(run, f"ckpt-{it:04d}.json"), "w"))
        return S

    S0 = evaluate(theta, 0); hist.append(dict(iter=0, eval={k: S0[k] for k in ("train", "held_out")})); print("iter 0", json.dumps(hist[-1]), flush=True)
    log.write(json.dumps(dict(iter=0, wall_s=0.0, eval=hist[-1]["eval"])) + "\n"); log.flush()
    for it in range(1, a.iters + 1):
        pool_idx = train_light if it <= a.light * a.iters else train_all
        batch = rng.choice(pool_idx, size=a.per_dir, replace=False); seeds = rng.integers(1 << 30, size=a.per_dir)
        deltas = rng.standard_normal((a.dirs, carry.NP))
        jobs = []
        for d in deltas:
            for sgn in (1, -1):
                for i, sd in zip(batch, seeds): jobs.append((theta + sgn * a.noise * d, int(i), True, int(sd)))
        rs = pool.map(_run, jobs); R = np.array([r["reward"] for r in rs]).reshape(a.dirs, 2, a.per_dir).mean(axis=2)
        order = np.argsort(-R.max(axis=1))[:a.top]; sd = R[order].std() + 1e-6
        theta = theta + a.step / (a.top * sd) * ((R[order, 0] - R[order, 1])[:, None] * deltas[order]).sum(axis=0)
        ok = np.mean([r["ok"] for r in rs]); tf = np.mean([r["t_free"] for r in rs if r["ok"]]) if ok else None
        rec = dict(iter=it, wall_s=round(time.perf_counter() - t0, 1), reward=round(float(R.mean()), 3), reward_top=round(float(R.max()), 3), success=round(float(ok), 3),
                   t_free=round(float(tf), 3) if tf is not None else None, stage="light" if it <= a.light * a.iters else "all", gain=round(carry.Policy(theta).k, 3), a_scale=round(float(np.exp(np.clip(theta[-1], -1, 1.5))), 3))
        if it % a.eval_every == 0 or it == a.iters:
            S = evaluate(theta, it); rec["eval"] = {k: S[k] for k in ("train", "held_out")}; hist.append(dict(iter=it, eval=rec["eval"]))
            if S["train"]["reward"] > best[0]: best = (S["train"]["reward"], theta.copy(), it)
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
    if S0["train"]["reward"] >= best[0]: best = (S0["train"]["reward"], (carry.Policy(json.load(open(a.init))["theta"]).theta if a.init else np.zeros(carry.NP)), 0)
    json.dump(dict(iter=best[2], theta=[float(x) for x in best[1]], features=carry.FEATURES, chosen_by="training-set evaluation return (no held-out items)"),
              open(os.path.join(run, "policy.json"), "w"), indent=1)
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), name=a.name, iters=a.iters, best_iter=best[2], episodes=a.iters * a.dirs * 2 * a.per_dir,
               wall_s=round(time.perf_counter() - t0, 1), history=hist, config=dict(dirs=a.dirs, top=a.top, noise=a.noise, step=a.step, per_dir=a.per_dir, workers=a.workers))
    dest = a.out or os.path.join(HERE, "student.json" if a.student else "policy.json"); out["student"] = a.student; out["init"] = a.init
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "results", "f4-student-train.json" if a.student else "f3-train.json"), "w"), indent=1)
    import shutil; shutil.copy(os.path.join(run, "policy.json"), dest)
    print("best iter", best[2], "->", dest); pool.close()


if __name__ == "__main__":
    main()
