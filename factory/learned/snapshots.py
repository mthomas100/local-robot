"""snapshots.py — F3 start states (2026-11-02): run the F2 line and save the complete simulator state at the end of
every lift (the item held, the tool braked to rest above the belt), with what F2's own carry did next from there (its
outcome and how long until the robot was free again). These are the learned carry's training episodes and its paired
baseline: the same state, F2's carry vs the policy's.

    envs/mj/bin/python -I factory/learned/snapshots.py [--runs 24] [--seconds 180] [--workers 12]
        -> runs/f3-snapshots/snapshots.pkl (not in git), factory/learned/results/f3-snapshots.json (summary)
"""
import argparse, json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import carry  # noqa: E402,F401  (sets the paths)
import mujoco  # noqa: E402


def collect(cfg):
    import warnings; warnings.filterwarnings("ignore")
    import line as LN

    class SnapLine(LN.Line):
        def _apply(self, tr):
            if (not self.recovering and self.cur is not None and self.k_traj >= 1 and self.cur.trajs[self.k_traj - 1].name == "lift"
                    and tr.name in ("rise", "transfer") and self.held is not None):
                c = self.c; m, d = c.model, c.data
                st = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
                lower = next(x for x in self.cur.trajs if x.name == "lower"); R_drop = self.planner.arm.fk(lower.q[-1])[1]
                self.snaps.append(dict(state=st, eq={c.vac_item: m.eq_data[c.vac_eq[c.vac_item]].copy()} if c.vac_item else {}, vac_cmd=c.vac_cmd,
                                       vac_item=c.vac_item, belt_target=c.belt_target, belt_ramp=c.belt_ramp, item=self.cur.name, tool=self.cur.tool,
                                       q_ref=self.cur.trajs[self.k_traj - 1].q[-1].copy(), R_drop=R_drop, carry_z=self.cur.meta["carry_z"],
                                       a_carry=self.cur.meta["a_carry"], held=(self.held[0].copy(), self.held[1].copy()), t=float(d.time),
                                       attempt_t_start=self.attempt["t_start"], speed=self.speed, seed=cfg["seed"], rate=self.rate))
            return super()._apply(tr)

    ln = SnapLine(speed=cfg["speed"], rate_per_min=cfg["rate"], scheduler="lookahead", seed=cfg["seed"]); ln.snaps = []
    out = ln.run(seconds=cfg["seconds"])
    for s in ln.snaps:   # what F2's carry did from here
        ev = next((e for e in out["attempts"] if e["item"] == s["item"] and e["t_start"] == s["attempt_t_start"]), None)
        it = next((i for i in out["items"] if i["name"] == s["item"] and i["end_t"] > s["t"]), None)
        s["f2_attempt"] = ev["outcome"] if ev else None
        s["f2_free_s"] = round(ev["t_end"] - s["t"], 3) if ev and ev["outcome"] == "ok" else None
        s["f2_item"] = it["outcome"] if it else None
    return ln.snaps


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=24); p.add_argument("--seconds", type=float, default=180.0); p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    cfgs = [dict(speed=(0.15, 0.3)[k % 2], rate=(8.0, 12.0, 16.0)[(k // 2) % 3], seed=100 + k, seconds=a.seconds) for k in range(a.runs)]   # seeds 100+: disjoint from the eval seeds
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers, initializer=carry.die_with_parent) as pool: parts = pool.map(collect, cfgs)
    snaps = [s for part in parts for s in part]
    out = os.path.join(ROOT, "runs", "f3-snapshots"); os.makedirs(out, exist_ok=True)
    pickle.dump(snaps, open(os.path.join(out, "snapshots.pkl"), "wb"))
    from collections import Counter
    per = Counter(s["item"] for s in snaps); held_out = sum(per[n] for n in carry.HELD_OUT)
    free = [s["f2_free_s"] for s in snaps if s["f2_free_s"] is not None]
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), runs=a.runs, seconds=a.seconds, snapshots=len(snaps), held_out_snapshots=held_out, per_item=dict(per),
             f2_carry_free_s=dict(mean=round(float(np.mean(free)), 3), p95=round(float(np.percentile(free, 95)), 3)) if free else None,
             f2_item_outcomes=dict(Counter(s["f2_item"] for s in snaps)), wall_s=round(time.perf_counter() - t0, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(S, open(os.path.join(HERE, "results", "f3-snapshots.json"), "w"), indent=1)
    print(json.dumps(S))


if __name__ == "__main__":
    main()
