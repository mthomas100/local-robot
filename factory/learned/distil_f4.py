"""distil_f4.py — F4's teacher-student step (2026-11-02; docs/sorting-cell-spec.md §4 F4, discipline 10c).

The F3 carry policy (the *teacher*) reads the held item's true pose every 20 ms: how far it has swung in the hand,
how fast, and where it would land if let go. A vision-only cell does not have that. The *student* runs the same
controller with `CarryCtl(privileged=False)`: the item is assumed rigid where the grasp put it, so the swing features
read 0 and the landing point is the rigid estimate's; it counts as clear of the fingers after the time it takes to
fall clear. Distillation (DAgger, Ross et al.: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011): run the *student* on the F2 grasp snapshots, record at every tick
its features and what the teacher's residual would output in that same state (from the privileged features), and fit
the student's residual matrix to the teacher's outputs by ridge regression; repeat on the new student's own states.

    envs/mj/bin/python -I factory/learned/distil_f4.py [--rounds 3] [--workers 8]
        -> factory/learned/student.json, factory/learned/results/f4-distil.json
"""
import argparse, json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import carry  # noqa: E402
from train_f3 import summarize  # noqa: E402

_ENV = None; _SNAPS = None; _TEACHER = None


def _init(path, teacher):
    global _ENV, _SNAPS, _TEACHER
    carry.die_with_parent()
    import warnings; warnings.filterwarnings("ignore")
    _SNAPS = pickle.load(open(path, "rb")); _ENV = carry.Env(); _TEACHER = carry.Policy(teacher)


def _run(job):
    theta, i, mode, seed = job
    priv = mode == "teacher"
    r = _ENV.episode(_SNAPS[i], carry.Policy(theta), dr=False, seed=seed, privileged=priv, teacher=_TEACHER if mode == "probe" else None)
    r["i"] = i
    if "probe" in r: r["probe"] = [(f.tolist(), u.tolist()) for f, u in r["probe"]]
    return r


def flat(v):
    """One table row: success and seconds until the robot is free, training and held-out items."""
    return dict(train_sorted=v["train"]["success"], train_free_s=v["train"]["t_free"], held_out_sorted=v["held_out"]["success"], held_out_free_s=v["held_out"]["t_free"],
                failures_train=sum(v["train"]["failures"].values()), failures_held_out=sum(v["held_out"]["failures"].values()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rounds", type=int, default=3); p.add_argument("--workers", type=int, default=8); p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--teacher", default=os.path.join(HERE, "policy.json")); p.add_argument("--snapshots", default=os.path.join(ROOT, "runs", "f3-snapshots", "snapshots.pkl"))
    p.add_argument("--out", default=os.path.join(HERE, "student.json"))
    a = p.parse_args(); t0 = time.perf_counter()
    snaps = pickle.load(open(a.snapshots, "rb")); T = carry.Policy(json.load(open(a.teacher))["theta"])
    train = [i for i, s in enumerate(snaps) if s["item"] not in carry.HELD_OUT]; every = list(range(len(snaps)))
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.snapshots, T.theta), maxtasksperchild=60)
    ev = lambda th, mode: summarize(pool.map(_run, [(th, i, mode, 1000 + i) for i in every]), snaps)
    rows = dict(teacher=ev(T.theta, "teacher"), student_zero_shot=ev(T.theta, "student")); print("teacher", rows["teacher"]["train"], rows["teacher"]["held_out"], flush=True)
    print("zero-shot", rows["student_zero_shot"]["train"], rows["student_zero_shot"]["held_out"], flush=True)
    S = carry.Policy(T.theta); F, U = [], []; best = ("zero-shot", S.theta.copy(), rows["student_zero_shot"]["train"]["reward"])
    for k in range(1, a.rounds + 1):
        rs = pool.map(_run, [(S.theta, i, "probe", 2000 * k + i) for i in train])
        for r in rs:
            for f, u in r["probe"]: F.append(f); U.append(u)
        X = np.array(F); Y = np.array(U)   # aggregate every round's data (DAgger)
        W = np.linalg.solve(X.T @ X + a.ridge * np.eye(carry.NF), X.T @ Y).T
        th = S.theta.copy(); th[:carry.NA * carry.NF] = W.reshape(-1); S = carry.Policy(th)
        rows[f"student_round{k}"] = ev(S.theta, "student"); R = rows[f"student_round{k}"]
        print(f"round {k}: {len(X)} samples, fit rms {float(np.sqrt(np.mean((X @ W.T - Y) ** 2))):.3f}", R["train"], R["held_out"], flush=True)
        if R["train"]["reward"] > best[2]: best = (f"round{k}", S.theta.copy(), R["train"]["reward"])
    pool.close()
    json.dump(dict(theta=[round(float(x), 6) for x in best[1]], features=carry.FEATURES, chosen=best[0], teacher=os.path.relpath(a.teacher, ROOT),
                   note="F4 student carry: run with CarryCtl(privileged=False)"), open(a.out, "w"), indent=1)
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), rounds=a.rounds, ridge=a.ridge, chosen=best[0], chosen_by="training-set mean return",
               results={k: {s: v[s] for s in ("train", "held_out")} for k, v in rows.items()}, per_item={k: v["per_item"] for k, v in rows.items()},
               table={k: flat(v) for k, v in rows.items()},
               wall_s=round(time.perf_counter() - t0, 1))
    json.dump(out, open(os.path.join(HERE, "results", "f4-distil.json"), "w"), indent=1); print("chosen", best[0])


if __name__ == "__main__":
    main()
