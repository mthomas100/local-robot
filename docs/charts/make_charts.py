"""make_charts.py — every chart in the README, drawn from the result files in this repo (CPU only, no simulation).

    uv run --with matplotlib python docs/charts/make_charts.py        # -> docs/media/charts/*.png

Inputs are the files the runs wrote: factory/*/results/*.json (in git from the start) and docs/results/ (copies of
the run directories' metrics.jsonl, eval.json and timeline.json, which live in the gitignored runs/). Nothing is
smoothed, filtered or re-scored; where a chart needs a number the files do not hold, it is not drawn. Each chart
prints the numbers it plots, so the README can be checked against them.
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "media", "charts"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(ROOT, "docs", "results")

# palette: categorical slots in fixed order (validated: adjacent CVD dE >= 9.1 on the light surface), neutral grey
# for baselines; text never wears a series colour
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
GREY, GREY_L = "#8a8984", "#c9c8c2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GREY_L, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def js(*parts):
    return json.load(open(os.path.join(ROOT, *parts)))


def finish(fig, name, title, source):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Data: " + source, ha="left", va="bottom", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = os.path.join(OUT, name); fig.savefig(p, dpi=min(150, 1600 / fig.get_figwidth()), metadata={"Software": None}); plt.close(fig); print("wrote", os.path.relpath(p, ROOT))


def label_bars(ax, bars, fmt="{:.0f}", dy=0.0, size=8.5):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + dy), ha="center", va="bottom", fontsize=size, color=INK)


# ---------------------------------------------------------------- locomotion: Go1 on the CPU (MuJoCo batch + PPO)
def go1():
    mps = jl(os.path.join(RES, "locomotion", "go1-clean.metrics.jsonl"))
    cpu = jl(os.path.join(RES, "locomotion", "go1-clean-cpu.metrics.jsonl"))
    tl = json.load(open(os.path.join(RES, "locomotion", "go1-show.timeline.json")))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for runs, colour, lab in ((mps, BLUE, "learner on MPS"), (cpu, ORANGE, "learner on the CPU")):
        t = [r["elapsed"] for r in runs]; ax[0].plot(t, [r["reward"] for r in runs], color=colour, lw=1.6, label=f"{lab} ({t[-1]:.0f} s)")
    ax[0].set(title="Reward per step vs wall-clock (same seed, 1,024 envs)", xlabel="seconds of training", ylabel="mean reward per step")
    ax[0].legend(loc="lower right")
    it = [r["it"] for r in mps]
    ax[1].plot(it, [r["falls"] for r in mps], color=BLUE, lw=1.4, label="share of episodes ending in a fall")
    ax[1].plot(it, [r["raw_track"] for r in mps], color=AQUA, lw=1.4, label="velocity-tracking score (0-1)")
    ax[1].set(title="What the reward terms say (learner on MPS)", xlabel="PPO iteration", ylabel="fraction")
    ax[1].legend(loc="center right")
    ax[2].plot(tl["checkpoints"], tl["walk_speed"], color=BLUE, marker="o", ms=5, label="measured forward speed")
    ax[2].axhline(1.0, color=GREY, lw=1, ls="--"); ax[2].text(600, 1.02, "commanded 1 m/s", ha="right", va="bottom", fontsize=8.5, color=INK2)
    first = next(c for c, v in zip(tl["checkpoints"], tl["walk_speed"]) if v >= 0.9)
    ax[2].annotate(f"first checkpoint >= 0.9 m/s: iteration {first}", (first, 0.94), (300, 0.3), fontsize=8.5, color=INK,
                   arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax[2].set(title="Each checkpoint driven at 1 m/s", xlabel="checkpoint (PPO iteration)", ylabel="forward speed, m/s", ylim=(-0.1, 1.15))
    ax[2].text(0.98, 0.04, "fall counter: 0 at all 24 checkpoints; the video shows\nit 575 and 600 collapsing in the final standing segment", transform=ax[2].transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    at = {r["it"]: r["elapsed"] for r in mps}
    print(f"go1: MPS {mps[-1]['elapsed']} s, CPU {cpu[-1]['elapsed']} s, final reward {mps[-1]['reward']} / {cpu[-1]['reward']}, "
          f"steps/s {mps[-1]['steps']/mps[-1]['elapsed']:.0f} / {cpu[-1]['steps']/cpu[-1]['elapsed']:.0f}; first >=0.9 m/s at it {first} = {at[first]} s into the MPS run; falls {tl['falls']}")
    finish(fig, "go1-learning.png", "Engine one: a Unitree Go1 learns to trot on the CPU (MuJoCo 3.17 mujoco.batch, PPO)",
           "docs/results/locomotion/go1-clean*.metrics.jsonl (runs of 2026-11-01, idle machine), go1-show.timeline.json (same seed, 2026-11-01)")


# ---------------------------------------------------------------- locomotion: Go2 on Metal (Genesis + rsl_rl)
def go2():
    m = jl(os.path.join(RES, "locomotion", "go2-base.metrics.jsonl"))
    ev = json.load(open(os.path.join(RES, "locomotion", "go2-base.eval.json")))
    done = json.load(open(os.path.join(RES, "locomotion", "done.json")))["go2-base"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    it = [r["it"] for r in m]
    ax[0].plot(it, [r["Train_mean_reward"] for r in m], color=BLUE); ax[0].set(title="Mean episode return", xlabel="rsl_rl iteration", ylabel="return")
    ax[1].plot(it, [r["Train_mean_episode_length"] for r in m], color=ORANGE)
    ax[1].set(title="Episode length (1,001 steps = never fell)", xlabel="rsl_rl iteration", ylabel="steps")
    segs = ev["segments"]; x = np.arange(len(segs))
    b = ax[2].bar(x, [s["err_vx"] for s in segs], color=[BLUE if abs(s["cmd"][0] - 0.5) < 1e-6 else GREY for s in segs], width=0.6)
    label_bars(ax[2], b, "{:.2f}", 0.004)
    ax[2].set_xticks(x, [f"{s['cmd'][0]:.1f} m/s\n{s['seconds']:.0f} s" for s in segs])
    ax[2].set(title="Eval (CPU backend): speed error", ylabel="mean |vx error|, m/s", ylim=(0, 0.27), xlabel="command (blue: the trained 0.5 m/s)")
    gap = m[-1]["wall"] - m[0]["wall"]
    print(f"go2: {done['seconds']} s total incl. compile, {gap:.1f} s between first and last iteration, final reward {m[-1]['Train_mean_reward']:.2f}, "
          f"tracking_lin_vel {m[-1]['Episode_rew_tracking_lin_vel']:.3f}, fps {m[-1]['Perf_total_fps']:.0f}, eval falls {ev['falls']}, err {[s['err_vx'] for s in segs]}")
    finish(fig, "go2-learning.png", "Engine two: a Unitree Go2 learns to walk on the Metal GPU (Genesis 1.4.5, 4,096 envs, rsl_rl PPO)",
           "docs/results/locomotion/go2-base.metrics.jsonl and go2-base.eval.json (run of 2026-11-01)")


# ---------------------------------------------------------------- simulator throughput benchmarks
def bench():
    rows = jl(os.path.join(RES, "bench", "bench-2026-11-01.jsonl"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    mj = [r for r in rows if r["bench"] == "mujoco_rollout_go2"]
    ax[0].plot([r["nthread"] for r in mj], [r["env_steps_per_sec"] / 1e3 for r in mj], color=BLUE, marker="o")
    for r in mj:
        if r["nthread"] in (1, 8, 18): ax[0].annotate(f"{r['env_steps_per_sec']/1e3:.0f}k", (r["nthread"], r["env_steps_per_sec"] / 1e3), (4, -12), textcoords="offset points", fontsize=8.5, color=INK)
    ax[0].set_xticks([1, 2, 4, 8, 12, 16, 18])
    ax[0].set(title="MuJoCo 3.17 rollout, Go2 scene: thread scaling", xlabel="CPU threads (16 sims per thread)", ylabel="env-steps per second, thousands")
    gen = [r for r in rows if r["bench"] == "genesis_go2"]
    by = defaultdict(dict)
    for r in gen: by[r["backend"]][r["n_envs"]] = r["env_steps_per_sec"]
    ns = sorted({n for d in by.values() for n in d}); x = np.arange(len(ns)); w = 0.38
    for k, (be, colour) in enumerate((("cpu", ORANGE), ("metal", BLUE))):
        vals = [by[be].get(n, 0) / 1e3 for n in ns]
        bars = ax[1].bar(x + (k - 0.5) * w, vals, w * 0.92, color=colour, label={"cpu": "CPU backend", "metal": "Metal GPU"}[be])
        for bb, v in zip(bars, vals):
            if v: ax[1].annotate(f"{v:.0f}k", (bb.get_x() + bb.get_width() / 2, v), (0, 2), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax[1].set_xticks(x, [f"{n:,} envs" for n in ns]); ax[1].legend(loc="upper left")
    ax[1].set(title="Genesis 1.4.5, Go2: CPU vs Metal", ylabel="env-steps per second, thousands")
    tor = [r for r in rows if r["bench"] == "torch_ppo_update" and "updates_per_sec" in r]
    labels = [f"CPU, run {i+1}" for i, r in enumerate(t for t in tor if t["backend"] == "cpu")] + ["MPS"]
    vals = [r["updates_per_sec"] for r in tor if r["backend"] == "cpu"] + [r["updates_per_sec"] for r in tor if r["backend"] == "mps"]
    bars = ax[2].bar(range(len(vals)), vals, color=[ORANGE] * (len(vals) - 1) + [BLUE], width=0.6); label_bars(ax[2], bars, "{:.0f}", 5)
    ax[2].set_xticks(range(len(vals)), labels); ax[2].set(title="torch 2.15 PPO update (MLP, batch 4,096)", ylabel="updates per second")
    print("bench mujoco", [(r["nthread"], r["env_steps_per_sec"]) for r in mj], "genesis", dict(by), "torch", list(zip(labels, vals)))
    finish(fig, "sim-throughput.png", "What this Mac can simulate: measured throughput (M5 Max, 18 CPU cores, 40-core GPU)",
           "docs/results/bench/bench-2026-11-01.jsonl (rl bench, 2026-11-01; other jobs shared the CPU during some runs, as logged)")


# ---------------------------------------------------------------- F0: the belt test
def f0():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    speeds = ("0.15", "0.3", "0.5"); x = np.arange(3); w = 0.38
    for k, (eng, colour) in enumerate((("mujoco", BLUE), ("genesis", ORANGE))):
        d = js("factory", "cell", "results", f"belt-test-{eng}.json"); ps = d["per_speed"]; crit = d["criteria"]
        lab = {"mujoco": "MuJoCo 3.17", "genesis": "Genesis 1.4.5 (CPU)"}[eng] + f": {d['passed']}/{d['rides']} rides pass"
        for a, key, scale, bar in ((ax[0], "worst_tilt_deg", 1, crit["max_tilt_deg"]), (ax[1], "worst_slip_frac", 100, crit["slip_frac"] * 100),
                                   (ax[2], "worst_penetration_mm", 1, crit["penetration_m"] * 1000)):
            vals = [ps[s][key] * scale for s in speeds]
            bb = a.bar(x + (k - 0.5) * w, vals, w * 0.92, color=colour, label=lab)
            label_bars(a, bb, "{:.2f}", 0, 8)
            a.axhline(bar, color=RED, lw=1.2, ls="--")
        print(f"f0 {eng}: passed {d['passed']}/{d['rides']}", {s: ps[s] for s in speeds})
    for a, t, yl, bar in ((ax[0], "Worst tilt of any ride", "degrees", "pass bar 15°"), (ax[1], "Worst slip against the belt", "% of belt speed", "pass bar 2%"),
                          (ax[2], "Worst penetration into the belt", "mm", "pass bar 3 mm")):
        a.set_xticks(x, [f"{s} m/s" for s in speeds]); a.set(title=t, ylabel=yl)
        a.text(0.99, 0.97, f"dashed: {bar}", transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax[0].legend(loc="center left", fontsize=8.5)
    for a in ax: a.set_ylim(0, a.get_ylim()[1] * 1.12)
    finish(fig, "f0-belt-test.png", "F0: 11 scanned objects x 4 orientations x 3 belt speeds, in two physics engines (worst case of 44 rides per bar)",
           "factory/cell/results/belt-test-mujoco.json, belt-test-genesis.json (2026-11-02)")


# ---------------------------------------------------------------- F1: classical pick from a stopped belt
def f1():
    ev = js("factory", "classical", "results", "f1-eval.json"); tr = jl(os.path.join(RES, "f1", "f1-eval.trials.jsonl"))
    budget = js("factory", "cell", "results", "reach-map.json")["cycle_budget"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    items = sorted(ev["per_item"]); ok = [ev["per_item"][i]["success"] for i in items]; n = [ev["per_item"][i]["n"] for i in items]
    fail = [a - b for a, b in zip(n, ok)]; x = np.arange(len(items))
    ax[0].bar(x, ok, color=BLUE, width=0.7, label="picked into the right chute")
    ax[0].bar(x, fail, bottom=ok, color=RED, width=0.7, label="dropped on the way (2 tall bottles)")
    for i, (a, b) in enumerate(zip(ok, n)): ax[0].text(i, b + 0.3, f"{a}/{b}", ha="center", fontsize=8, color=INK)
    ax[0].set_xticks(x, [i.replace("_", " ") for i in items], rotation=35, ha="right"); ax[0].set_ylim(0, 23)
    ax[0].set(title=f"{ev['successes']}/{ev['trials']} trials ({ev['success_rate']*100:.1f}%, Wilson 95% {ev['wilson95'][0]*100:.1f}-{ev['wilson95'][1]*100:.1f}%)", ylabel="trials")
    ax[0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1, 1.0), ncol=2)
    cyc = [t["cycle_s"] for t in tr if t["outcome"] == "success"]
    ax[1].hist(cyc, bins=np.arange(5.5, 10.25, 0.25), color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    for sp, colour in (("0.15", GREY), ("0.3", ORANGE), ("0.5", RED)):
        z = budget[sp]["zone_seconds"]
        ax[1].axvline(z, color=colour, ls="--", lw=1.3); ax[1].text(z + (0.05 if sp != "0.15" else -0.05), ax[1].get_ylim()[1] * 0.92, f"{sp} m/s:\nitem in zone\n{z:.1f} s", fontsize=8, color=INK2, va="top", ha="left" if sp != "0.15" else "right")
    ax[1].set_xlim(1.5, 10.2)
    ax[1].set(title=f"Cycle time of the successful picks: mean {np.mean(cyc):.1f} s, p95 {np.percentile(cyc, 95):.1f} s", xlabel="seconds from approach to home", ylabel="picks")
    print(f"f1: {ev['successes']}/{ev['trials']}, wilson {ev['wilson95']}, cycle mean {np.mean(cyc):.2f} p95 {np.percentile(cyc,95):.2f} max {max(cyc):.2f} (file: {ev['cycle_s']}), tools {ev['tools']}")
    finish(fig, "f1-pick.png", "F1: a classical pick (IK, grasp library, planner, joint control) from a stopped belt, 200 seeded trials",
           "factory/classical/results/f1-eval.json and docs/results/f1/f1-eval.trials.jsonl (2026-11-02 19:45); zone times from factory/cell/results/reach-map.json")


# ---------------------------------------------------------------- F2: the moving belt
def f2():
    ev = js("factory", "line", "results", "f2-eval.json"); tk = js("factory", "line", "results", "f2-tracking.json")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    tab = ev["table"]
    for sp, colour in ((0.15, BLUE), (0.3, ORANGE)):
        for sch, ls, mk in (("lookahead", "-", "o"), ("greedy", ":", "s"), ("edf", "--", "^")):
            rows = sorted([r for r in tab if r["speed"] == sp and r["scheduler"] == sch], key=lambda r: r["rate"])
            kw = dict(color=colour, ls=ls, marker=mk, ms=5, lw=2 if sch == "lookahead" else 1.1, alpha=1 if sch == "lookahead" else 0.75)
            ax[0].plot([r["rate"] for r in rows], [r["sorted_per_min"] for r in rows], label=f"{sp} m/s, {sch}", **kw)
            if sch == "lookahead":
                ax[1].plot([r["rate"] for r in rows], [100 * r["miss_rate"] for r in rows], label=f"{sp} m/s (lookahead)", color=colour, marker=mk)
                print(f"f2 {sp} lookahead", [(r["rate"], r["sorted_per_min"], r["miss_rate"], r["attempt_success"], r["cycle_s"], r["grasp_err_mm"]) for r in rows])
    ax[0].plot([4, 16], [4, 16], color=GREY_L, lw=1, ls="-"); ax[0].text(15.5, 14.2, "every item sorted", fontsize=8, color=INK2, ha="right", rotation=0)
    ax[0].set(title="Sorted per minute, 3 schedulers (2 seeds x 180 s each)", xlabel="items arriving per minute", ylabel="sorted per minute", ylim=(0, 16.5))
    ax[0].legend(fontsize=7.5, ncol=2, loc="upper left")
    ax[1].set(title="Share of items not sorted", xlabel="items arriving per minute", ylabel="% not sorted", ylim=(0, 60)); ax[1].legend(loc="upper left")
    rows = tk["rows"]
    for sp, colour in ((0.15, BLUE), (0.3, ORANGE), (0.5, RED)):
        for comp, ls, lab in ((True, "-", "time-stamped"), (False, "--", "treated as fresh")):
            rr = sorted([r for r in rows if r["speed"] == sp and (r["compensated"] == comp or r["latency_s"] == 0)], key=lambda r: r["latency_s"])
            ax[2].plot([1000 * r["latency_s"] for r in rr], [r["err_mean_mm"] for r in rr], color=colour, ls=ls, marker="o", ms=4, label=f"{sp} m/s, {lab}")
    ax[2].set(title="Kalman tracking: error predicting 1.5 s ahead", xlabel="camera latency, ms", ylabel="mean position error, mm"); ax[2].legend(fontsize=7.5, ncol=1)
    print("f2 tracking", [(r["speed"], r["latency_s"], r["compensated"], r["err_mean_mm"]) for r in rows])
    finish(fig, "f2-line.png", "F2: the belt never stops (Poisson arrivals of all 11 items, intercept planning, Kalman tracks)",
           "factory/line/results/f2-eval.json (2026-11-02 20:24), f2-tracking.json (2026-11-02)")


# ---------------------------------------------------------------- F3: the learned carry (ARS)
def f3():
    r1 = [r for r in jl(os.path.join(RES, "f3", "ars-run1.metrics.jsonl")) if "eval" in r]
    r2 = [r for r in jl(os.path.join(RES, "f3", "ars-run2.metrics.jsonl")) if "eval" in r]
    tr = js("factory", "learned", "results", "f3-train.json"); ev = js("factory", "learned", "results", "f3-eval.json")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for runs, colour, name in ((r1, ORANGE, "run 1"), (r2, BLUE, "run 2 (shipped)")):
        it = [r["iter"] for r in runs]
        ax[0].plot(it, [100 * r["eval"]["train"]["success"] for r in runs], color=colour, marker="o", ms=4, label=f"{name}: training items")
        ax[0].plot(it, [100 * r["eval"]["held_out"]["success"] for r in runs], color=colour, marker="s", ms=4, ls="--", label=f"{name}: held-out items")
        ax[1].plot(it, [r["eval"]["train"]["t_free"] for r in runs], color=colour, marker="o", ms=4, label=f"{name}: training items")
    f2t = r2[0]["eval"]["train"]["f2_t_free"]
    ax[1].axhline(f2t, color=GREY, ls="--", lw=1.2); ax[1].text(118, f2t - 0.06, f"F2's planned carry from the same grasps: {f2t:.2f} s", ha="right", va="top", fontsize=8.5, color=INK2)
    ax[0].axvline(tr["best_iter"], color=GREY_L, lw=1); ax[0].text(tr["best_iter"] + 1, 85.4, f"shipped: run 2,\niteration {tr['best_iter']}", fontsize=8, color=INK2)
    ax[0].set(title="Carries that end in the right chute (483 real F2 grasps)", xlabel="ARS iteration", ylabel="% sorted", ylim=(84, 101)); ax[0].legend(fontsize=7.8, loc="lower left")
    ax[1].set(title="Seconds until the robot is free again", xlabel="ARS iteration", ylabel="seconds", ylim=(0, 2.9)); ax[1].legend(fontsize=8, loc="center right")
    conds = [("held-out · 0.15 m/s · 8/min", "held-out stream\n0.15 m/s"), ("held-out · 0.30 m/s · 8/min", "held-out stream\n0.30 m/s"),
             ("mixed · 0.15 m/s · 16/min", "mixed L2\n0.15 m/s"), ("mixed · 0.30 m/s · 16/min", "mixed L2\n0.30 m/s")]
    x = np.arange(len(conds)); w = 0.36
    for k, (who, colour, lab) in enumerate((("prior", GREY, "hand-written prior − F2"), ("learned", BLUE, "learned − F2"))):
        m = [ev["paired"][f"{c} · {who} − f2"]["held_out_sorted_per_min"] for c, _ in conds]
        mean = np.array([p["mean"] for p in m]); lo = mean - np.array([p["lo"] for p in m]); hi = np.array([p["hi"] for p in m]) - mean
        ax[2].bar(x + (k - 0.5) * w, mean, w * 0.9, color=colour, label=lab, yerr=[lo, hi], capsize=3, error_kw=dict(ecolor=INK2, lw=1))
        for xi, v in zip(x + (k - 0.5) * w, mean): ax[2].text(xi, 0.03, f"+{v:.2f}", ha="center", va="bottom", fontsize=8, color=SURFACE if colour == BLUE else INK)
        print(f"f3 line {who}-f2 held-out gains", [(c, round(p["mean"], 3), round(p["lo"], 3), round(p["hi"], 3)) for (c, _), p in zip(conds, m)])
    ax[2].axhline(0, color=INK2, lw=0.8); ax[2].set_xticks(x, [l for _, l in conds])
    ax[2].set(title="Line acceptance, 40 fresh seeds x 360 s: gain over F2", ylabel="held-out items sorted/min, gain (95% CI)"); ax[2].legend(loc="upper right", fontsize=8.5)
    print("f3 run2 evals", [(r["iter"], r["eval"]["train"]["success"], r["eval"]["held_out"]["success"], r["eval"]["train"]["t_free"]) for r in r2])
    print("f3 run1 evals", [(r["iter"], r["eval"]["train"]["success"], r["eval"]["held_out"]["success"], r["eval"]["train"]["t_free"]) for r in r1])
    finish(fig, "f3-carry.png", "F3: a carry policy learned by Augmented Random Search (66 numbers on a hand-written prior, 14 CPU processes, no GPU)",
           "docs/results/f3/ars-run1.metrics.jsonl, ars-run2.metrics.jsonl (2026-11-02); factory/learned/results/f3-eval.json (attempt 3, 2026-11-02 23:48)")


# ---------------------------------------------------------------- F4: seeing
def f4():
    st = js("factory", "vision", "results", "f4-vision-static.json"); ev = js("factory", "vision", "results", "f4-eval.json")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1, 0.8]))
    items = sorted(st["per_item"], key=lambda k: st["per_item"][k]["pos_mm"]); x = np.arange(len(items))
    b = ax[0].bar(x, [st["per_item"][i]["pos_mm"] for i in items], color=BLUE, width=0.68); label_bars(ax[0], b, "{:.1f}", 0.1, 8)
    ax[0].set_xticks(x, [i.replace("_", " ") for i in items], rotation=35, ha="right")
    ax[0].set(title=f"Position error vs ground truth ({st['reports']:,} reports, identity {st['identity_accuracy']*100:.0f}% correct)", ylabel="mean error, mm")
    cells = (("L2", BLUE, "privileged poses + teacher carry"), ("L4", ORANGE, "camera only + student carry"), ("L2-f2", GREY, "privileged + F2 carry"), ("L4-f2", GREY_L, "camera only + F2 carry"))
    sp = ("0.15 m/s", "0.30 m/s"); x = np.arange(2); w = 0.2
    for k, (c, colour, lab) in enumerate(cells):
        v = [100 * ev["table"][f"{s} · {c}"]["sort_rate"] for s in sp]
        bb = ax[1].bar(x + (k - 1.5) * w, v, w * 0.92, color=colour, label=f"{c}: {lab}"); label_bars(ax[1], bb, "{:.0f}", 0.5, 8)
    ax[1].set_xticks(x, sp); ax[1].set(title="Sort rate on the L2 stream (16 seeds x 180 s)", ylabel="% of arriving items sorted", ylim=(0, 95)); ax[1].legend(fontsize=7.6, loc="upper right")
    g = [ev["paired"][f"{s} · L2 − L4"] for s in sp]; mean = np.array([100 * p["mean"] for p in g])
    ax[2].errorbar(x, mean, yerr=[mean - 100 * np.array([p["lo"] for p in g]), 100 * np.array([p["hi"] for p in g]) - mean], fmt="o", color=BLUE, ms=8, capsize=5, lw=1.6)
    for xi, p in zip(x, g): ax[2].text(xi + 0.08, 100 * p["mean"], f"+{100*p['mean']:.1f}\n[{100*p['lo']:.1f}, {100*p['hi']:.1f}]", va="center", fontsize=8.5, color=INK)
    ax[2].axhline(10, color=RED, ls="--", lw=1.2); ax[2].text(0.5, 10.3, "acceptance bar: upper end below 10 points", ha="center", fontsize=8.5, color=INK2)
    ax[2].axhline(0, color=INK2, lw=0.8); ax[2].set_xticks(x, sp); ax[2].set_xlim(-0.5, 1.7)
    ax[2].set(title="Paired gap L2 − L4 (95% CI)", ylabel="percentage points", ylim=(-2, 12))
    print("f4 static", st["position_err_mm"], {i: st["per_item"][i]["pos_mm"] for i in items}, "ms/frame", st["ms_per_frame"])
    print("f4 eval", {k: (v["sort_rate"], v["sorted_per_min"]) for k, v in ev["table"].items()}, "paired", ev["paired"])
    finish(fig, "f4-vision.png", "F4: the cell sees for itself (overhead depth camera, 424x240, classical perception, no learned detector)",
           "factory/vision/results/f4-vision-static.json (60 fresh scenes, 2026-11-02 22:52), f4-eval.json (protocol fixed before the run, 2026-11-02 23:29)")


# ---------------------------------------------------------------- F5: the 10-minute shift
def f5():
    k = jl(os.path.join(RES, "f5", "kpis.jsonl")); fin = js("factory", "ops", "results", "f5-shift.json")["final"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.9]))
    t = np.array([r["t"] for r in k]) / 60
    ax[0].plot(t, [r["arrived"] for r in k], color=GREY, label="arrived"); ax[0].plot(t, [r["sorted"] for r in k], color=BLUE, label="sorted")
    ax[0].plot(t, [r["losses"]["missed"] for r in k], color=ORANGE, label="missed (never reached in time)")
    for r_, lab in ((k[-1]["arrived"], "arrived"), (k[-1]["sorted"], "sorted"), (k[-1]["losses"]["missed"], "missed")):
        ax[0].text(10.1, r_, f"{r_}", va="center", fontsize=8.5, color=INK)
    ax[0].set(title="Items over the shift", xlabel="simulated minutes", ylabel="items (cumulative)", xlim=(0, 10.8)); ax[0].legend(loc="upper left")
    ax[1].plot(t, [r["sorted_last_min"] for r in k], color=BLUE, lw=1.4)
    ax[1].set(title="Sorted in the last minute (the dashboard's live KPI)", xlabel="simulated minutes", ylabel="items per minute", ylim=(0, 12))
    names = ("availability", "performance", "quality", "oee"); vals = [100 * fin[n] for n in names]
    b = ax[2].bar(range(4), vals, color=[GREY, ORANGE, GREY, BLUE], width=0.62); label_bars(ax[2], b, "{:.0f}%", 1)
    ax[2].set_xticks(range(4), ["availability", "performance", "quality", "OEE\n(= A x P x Q)"]); ax[2].set(title="Overall equipment effectiveness", ylabel="%", ylim=(0, 110))
    print("f5 final", {n: fin[n] for n in ("arrived", "sorted", "sorted_per_min", "availability", "performance", "quality", "oee", "cycle_s", "protective_stops")}, fin["losses"])
    finish(fig, "f5-shift.png", "F5: a 10-minute shift (belt 0.3 m/s, 12 arrivals/min, 25% as touching pairs, camera only, student carry)",
           "docs/results/f5/kpis.jsonl (one row per simulated second) and factory/ops/results/f5-shift.json (2026-11-02 23:16)")


# ---------------------------------------------------------------- F6: the desk arm with the real-device kit
def f6():
    a1 = js("factory", "desk", "results", "f6-eval-attempt1.json"); a2 = js("factory", "desk", "results", "f6-eval.json")
    s1 = js("factory", "desk", "results", "f6-sysid-attempt1.json"); s2 = js("factory", "desk", "results", "f6-sysid-attempt2.json"); s3 = js("factory", "desk", "results", "f6-sysid.json")
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1, 1, 1]))
    for k, (ev, colour, lab) in enumerate(((a1, ORANGE, "attempt 1 (play modelled centred)"), (a2, BLUE, "attempt 2 (play read where the links hang)"))):
        rows = ev["calib"]["rows"]; x = np.arange(len(rows))
        bb = ax[0].bar(x + (k - 0.5) * 0.38, [r["after_mm"] for r in rows], 0.35, color=colour, label=lab); label_bars(ax[0], bb, "{:.2f}", 0.02, 7.5)
    ax[0].axhline(2.0, color=RED, ls="--", lw=1.2); ax[0].text(-0.45, 2.05, "bar 2 mm", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax[0].set_xticks(range(5), [f"error {i+1}" for i in range(5)]); ax[0].legend(fontsize=7.6, loc="upper left")
    ax[0].set(title="Hand-eye calibration (planted 8 mm, 2.5°)", ylabel="camera position error left, mm", ylim=(0, 3.1))
    t = a2["picks"]["table"]; items = ["tomato_soup_can", "potted_meat_can", "standin_pet_bottle", "standin_tea_box"]; x = np.arange(4)
    for k, (who, colour, lab) in enumerate((("classical", GREY, "classical pick"), ("onnx", BLUE, "behaviour-cloned ONNX policy"))):
        v = [int(t[who][i].split("/")[0]) for i in items]
        bb = ax[1].bar(x + (k - 0.5) * 0.38, v, 0.35, color=colour, label=f"{lab}: {t[who]['sorted']}"); label_bars(ax[1], bb, "{:.0f}", 0.3, 8)
    ax[1].set_xticks(x, ["tomato soup", "potted meat", "PET bottle\n(stand-in)", "tea box\n(stand-in)"]); ax[1].legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax[1].set(title="200 seeded picks, full kit on", ylabel="sorted of 50", ylim=(0, 58))
    for k, (s, colour, lab) in enumerate(((s1, GREY, "attempt 1: seeds 1-3"), (s2, ORANGE, "attempt 2: fresh seeds 11-13"), (s3, BLUE, "attempt 3: fresh seeds 21-23"))):
        worst = [100 * max(abs(r["rel_err"]) for r in sd["rows"]) for sd in s["seeds"]]
        bb = ax[2].bar(np.arange(3) + (k - 1) * 0.27, worst, 0.25, color=colour, label=lab); label_bars(ax[2], bb, "{:.1f}", 0.4, 7.5)
        print("f6 sysid", lab, [round(w, 2) for w in worst])
    ax[2].axhline(10, color=RED, ls="--", lw=1.2); ax[2].text(-0.45, 10.6, "bar 10%", ha="left", fontsize=8.5, color=INK2)
    ax[2].set_xticks(range(3), ["1st seed", "2nd seed", "3rd seed"]); ax[2].legend(fontsize=7.6, loc="upper left")
    ax[2].set(title="System ID: worst of 20 parameters", ylabel="worst recovery error, %", ylim=(0, 72))
    rr = a2["rate"]["rows"]; x = np.arange(len(rr))
    ax[3].plot(x, [r["period_ms_max"] for r in rr], "o", color=ORANGE, ms=6, label="slowest tick of the trial")
    ax[3].plot(x, [r["period_ms_mean"] for r in rr], "o", color=BLUE, ms=6, label="mean tick")
    ax[3].axhline(50, color=RED, ls="--", lw=1.2); ax[3].text(len(rr) - 1, 49, "late = 1.5 periods (50 ms)", ha="right", va="top", fontsize=8.5, color=INK2)
    ax[3].set(title=f"30 Hz loop: {a2['rate']['summary']['late_ticks']} late of {a2['rate']['summary']['ticks']:,} ticks", xlabel="trial", ylabel="tick period, ms", ylim=(30, 52))
    ax[3].legend(fontsize=8, loc="center right")
    print("f6 calib a1", [r["after_mm"] for r in a1["calib"]["rows"]], "a2", [(r["after_mm"], r["after_deg"]) for r in a2["calib"]["rows"]], "picks", t["classical"]["sorted"], t["onnx"]["sorted"], t["onnx"]["failures"], "rate", a2["rate"]["summary"])
    finish(fig, "f6-desk.png", "F6: a hobby desk arm (SO-101) with every imperfection of a real desk modelled, and the routines that remove them",
           "factory/desk/results/f6-eval-attempt1.json, f6-eval.json, f6-sysid-attempt1.json, f6-sysid-attempt2.json, f6-sysid.json (2026-11-03)")


# ---------------------------------------------------------------- F7: the fast belt
def f7():
    v1 = js("factory", "ops", "results", "f7-sweep-v1.json"); v2 = js("factory", "ops", "results", "f7-sweep-v2.json")
    ab = js("factory", "ops", "results", "f7-ablation.json"); ev = js("factory", "ops", "results", "f7-eval.json")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for sw, ls, mk, tag in ((v1, "--", "o", "v1"), (v2, "-", "s", "v2")):
        for sp, colour in ((0.15, AQUA), (0.3, BLUE), (0.5, RED)):
            rows = sorted([r for r in sw["rows"] if r["speed"] == sp and r["clutter"] == 0.25], key=lambda r: r["rate"])
            ax[0].plot([r["rate"] for r in rows], [r["sorted_per_min"] for r in rows], color=colour, ls=ls, marker=mk, ms=5, label=f"{tag} {sp} m/s")
            print(f"f7 {tag} {sp} clutter .25", [(r["rate"], r["sorted_per_min"]) for r in rows])
    ax[0].set(title="Envelope (25% touching pairs, 2 seeds x 180 s each)", xlabel="arrivals per minute", ylabel="sorted per minute"); ax[0].legend(fontsize=7.6, ncol=2, loc="upper left")
    flags = ("fast", "early", "deadline", "fast+early+deadline"); x = np.arange(len(flags)); w = 0.38
    for k, (sp, colour) in enumerate((("0.30", BLUE), ("0.50", RED))):
        d = [ab["vs_v1"][f][sp] for f in flags]; m = np.array([p["mean"] for p in d])
        ax[1].bar(x + (k - 0.5) * w, m, w * 0.9, color=colour, label=f"{sp} m/s", yerr=[m - np.array([p["lo"] for p in d]), np.array([p["hi"] for p in d]) - m], capsize=3, error_kw=dict(ecolor=INK2, lw=1))
        for xi, v, p in zip(x + (k - 0.5) * w, m, d): ax[1].text(xi, max(p["hi"], v, 0) + 0.08, f"{v:+.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        print(f"f7 ablation {sp}", [(f, p["mean"], p["lo"], p["hi"]) for f, p in zip(flags, d)])
    ax[1].axhline(0, color=INK2, lw=0.8); ax[1].set_xticks(x, ["fast\nalone", "early\nalone", "deadline\nalone", "all three"]); ax[1].legend(loc="upper left")
    ax[1].set(title="Each change alone and together (6 seeds, paired vs v1)", ylabel="sorted/min gained over v1 (95% CI)", ylim=(-0.5, 5.2))
    tab = ev["table"]; sp = ("0.50", "0.30"); x = np.arange(2)
    for k, (cell, colour) in enumerate((("v1", GREY), ("fast+early+deadline", BLUE))):
        v = [tab[f"{s} m/s · {cell}"]["sorted_per_min"] for s in sp]
        bb = ax[2].bar(x + (k - 0.5) * 0.38, v, 0.35, color=colour, label={"v1": "v1 (the F5 cell)", "fast+early+deadline": "v2 (all three changes)"}[cell]); label_bars(ax[2], bb, "{:.2f}", 0.08, 8.5)
    for xi, s in zip(x, sp):
        p = ev["paired"][s]; ax[2].text(xi, 8.4, f"v2 − v1: +{p['mean']:.2f}\n95% CI [{p['lo']:.2f}, {p['hi']:.2f}]", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax[2].set_xticks(x, [f"{s} m/s" for s in sp]); ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
    ax[2].set(title="Acceptance: 12 fresh seeds x 180 s, paired", ylabel="sorted per minute", ylim=(0, 10.2))
    print("f7 eval", {k: (v["sorted_per_min"], v["sort_rate"], v["wrong_chute"], v["protective_stops"]) for k, v in tab.items()}, ev["paired"])
    finish(fig, "f7-fast-belt.png", "F7: the operating envelope, and what made a 0.5 m/s belt work (v1 sorted nothing there)",
           "factory/ops/results/f7-sweep-v1.json, f7-sweep-v2.json, f7-ablation.json (round 2), f7-eval.json (protocol committed before the run; 2026-11-03)")


if __name__ == "__main__":
    for f in (go1, go2, bench, f0, f1, f2, f3, f4, f5, f6, f7): f()
