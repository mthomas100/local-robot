"""f1_media.py — the F1 stills, videos and figure for the README and the lab, reproducibly (2026-11-02).

    envs/mj/bin/python -I factory/classical/f1_media.py
Writes docs/media/F1-*.png (in git) and F1-*.mp4 (on disk, gitignored). Needs factory/classical/results/
f1-eval.json, f1-control.json and runs/f1-eval/ (rl pick eval, rl pick control) for the figure and the eval video.
"""
import json, math, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(ROOT, "docs", "media"); RES = os.path.join(HERE, "results")


def closeup(cell, path, lookat, azimuth=140, elevation=-22, distance=0.75, size=(960, 640)):
    import mujoco, imageio.v3 as iio
    cam = mujoco.MjvCamera(); cam.lookat[:] = lookat; cam.azimuth = azimuth; cam.elevation = elevation; cam.distance = distance
    r = mujoco.Renderer(cell.model, size[1], size[0]); r.update_scene(cell.data, camera=cam); iio.imwrite(path, r.render()); r.close(); return path


def clips_and_stills():
    import eval_f1, pick
    eval_f1._worker_init(); cell = eval_f1._W["cell"]; planner = eval_f1._W["planner"]
    # a still of each tool at the moment it holds the item (stop the plan after the grasp closes)
    for item, x, yaw, tag in (("cracker_box", 0.15, 0.5, "fingers"), ("master_chef_can", -0.1, 0.0, "suction")):
        cell.reset(); cell.belt(0.0, ramp=0); cell.spawn(item, x=x, yaw=yaw, with_belt_velocity=False); cell.step(150)
        s = cell.item_state(item); plan, _ = planner.plan(item, s["pos"], s["quat"], cell.lay.chute(cell.items[item]["rec"]["material"]))
        k = [sg.name for sg in plan.segments].index("lift"); plan.segments = plan.segments[:k + 1]; pick.Executor(cell).run(plan, item)
        print(closeup(cell, os.path.join(OUT, f"F1-grasp-{tag}.png"), cell.pinch_pos() + np.array([0, 0, -0.05])))
    # first-success clips, one per tool, from the pick camera
    for item, x, y, yaw, tag in (("cracker_box", 0.15, 0.55, 0.5, "fingers"), ("master_chef_can", -0.1, 0.55, 0.0, "suction"), ("mustard_bottle", 0.35, 0.6, 2.0, "bottle")):
        print(eval_f1.video([dict(index=0, item=item, x=x, y=y, yaw=yaw, seed=0)], os.path.join(OUT, f"F1-pick-{tag}.mp4")))
    # the residual failure: a tall bottle on a tilted grasp swings into the chute wall (eval trial 32)
    s = eval_f1.trial_specs(33)[32]; print(eval_f1.video([s], os.path.join(OUT, "F1-fail-tilted-swing.mp4")))
    # failures fixed on the way, rendered before the fix (runs/f1-eval/replay-*, if still on disk)
    for src, dst in (("replay-106-cell.mp4", "F1-fail-adhesion-flip.mp4"), ("replay-53-cell.mp4", "F1-fail-bleach-swing.mp4"), ("eval.mp4", "F1-eval.mp4"), ("eval-sheet.png", "F1-eval-sheet.png")):
        p = os.path.join(ROOT, "runs", "f1-eval", src)
        if os.path.exists(p): shutil.copy(p, os.path.join(OUT, dst)); print(os.path.join(OUT, dst))


def figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    E = json.load(open(os.path.join(RES, "f1-eval.json"))); C = json.load(open(os.path.join(RES, "f1-control.json")))
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "f1-eval", "trials.jsonl"))]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5)); fig.suptitle(f"F1: classical pick, belt stopped (L0) — {E['successes']}/{E['trials']} = {E['success_rate']:.1%} "
                                                              f"(95% CI {E['wilson95'][0]:.1%}–{E['wilson95'][1]:.1%}), {E['date']}", fontsize=13)
    items = list(E["per_item"]); rates = [E["per_item"][i]["rate"] * 100 for i in items]
    tools = {}
    for r in rows: tools.setdefault(r["item"], r.get("tool"))
    colors = ["#2a6fdb" if tools.get(i) == "fingers" else "#d9822b" for i in items]
    a = ax[0, 0]; a.barh(items, rates, color=colors); a.axvline(95, color="k", ls="--", lw=1); a.set_xlim(80, 100.5); a.set_xlabel("success %"); a.invert_yaxis()
    a.set_title("per item (blue: fingers, orange: suction cup); dashed: the 95% bar")
    a = ax[0, 1]; cyc = [r["cycle_s"] for r in rows if r["outcome"] == "success"]
    a.hist(cyc, bins=24, color="#555"); a.set_xlabel("cycle time, s (approach to home)"); a.set_ylabel("trials")
    a.axvline(3.33, color="#c33", lw=1.5); a.text(3.4, a.get_ylim()[1] * 0.85, "3.3 s: the time an item spends\nin the zone at 0.3 m/s (F2's problem)", color="#c33", fontsize=9)
    a.set_title(f"cycle time: mean {E['cycle_s']['mean']} s, p95 {E['cycle_s']['p95']} s")
    a = ax[1, 0]; fails = E["failures"] or {"none": 0}
    a.bar(list(fails), list(fails.values()), color="#c33"); a.set_title("failures by class (the taxonomy in pick.py)"); a.set_ylabel("trials")
    import control
    from cell_mujoco import CellMJ
    cell = CellMJ(); a = ax[1, 1]
    for j in (0, 3):
        m, d = cell.model, cell.data; cell.reset(); q0 = np.array(cell.lay.home_qpos)
        for _ in range(200): d.ctrl[cell.robot_act] = q0; d.qfrc_applied[cell.robot_dof] = d.qfrc_bias[cell.robot_dof]; cell.step()
        tgt = q0.copy(); tgt[j] += 0.1; y0 = d.qpos[cell.robot_qpos[j]]; ys = []
        for _ in range(750): d.ctrl[cell.robot_act] = tgt; d.qfrc_applied[cell.robot_dof] = d.qfrc_bias[cell.robot_dof]; cell.step(); ys.append(d.qpos[cell.robot_qpos[j]] - y0)
        a.plot(np.arange(750) * m.opt.timestep, np.array(ys) / 0.1, label=f"{cell.robot_joints[j]} (kp {C['kp'][j]:.0f}, kv {C['kv'][j]:.0f})")
    a.axhline(1.0, color="k", lw=0.8); a.axhspan(0.98, 1.02, color="#ccc", alpha=0.5); a.set_xlabel("s"); a.set_ylabel("fraction of a 0.1 rad step"); a.legend(fontsize=9)
    a.set_title(f"servo step response: settles in {C['steps'][0]['settling_s']} s, no overshoot (overdamped)")
    fig.tight_layout(); p = os.path.join(OUT, "F1-results.png"); fig.savefig(p, dpi=110); print(p)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    figure(); clips_and_stills()
