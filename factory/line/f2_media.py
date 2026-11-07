"""f2_media.py — the F2 stills, videos and figure for the README and the lab, reproducibly (2026-11-02).

    envs/mj/bin/python -I factory/line/f2_media.py [--figure-only]
Writes docs/media/F2-*.png (git) and F2-*.mp4 (on disk). Needs factory/line/results/f2-eval.json and
f2-tracking.json (rl line eval, rl line tracking) for the figure.
"""
import argparse, json, math, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "docs", "media"); RES = os.path.join(HERE, "results")


def figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    E = json.load(open(os.path.join(RES, "f2-eval.json"))); T = json.load(open(os.path.join(RES, "f2-tracking.json")))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5)); fig.suptitle(f"F2: the classical pipeline on a moving belt — {E['date']} ({E['seconds']:.0f} s per run, {E['seeds']} seeds)", fontsize=13)
    styles = {"greedy": ":", "edf": "--", "lookahead": "-"}; colors = {0.15: "#2a6fdb", 0.3: "#c33"}
    for r_sched in ("greedy", "edf", "lookahead"):
        for v in (0.15, 0.3):
            rows = sorted([r for r in E["table"] if r["speed"] == v and r["scheduler"] == r_sched], key=lambda r: r["rate"])
            if not rows: continue
            x = [r["rate"] for r in rows]
            ax[0, 0].errorbar(x, [r["sorted_per_min"] for r in rows], yerr=[r["sorted_per_min_sd"] for r in rows], ls=styles[r_sched], color=colors[v], marker="o", ms=4, capsize=3, label=f"{v} m/s, {r_sched}")
            ax[0, 1].plot(x, [100 * r["miss_rate"] for r in rows], ls=styles[r_sched], color=colors[v], marker="o", ms=4, label=f"{v} m/s, {r_sched}")
    mx = max(r["rate"] for r in E["table"]); ax[0, 0].plot([0, mx], [0, mx], color="#999", lw=0.8); ax[0, 0].text(mx * 0.55, mx * 0.62, "every item sorted", color="#777", fontsize=9, rotation=30)
    ax[0, 0].set_xlabel("items arriving per minute"); ax[0, 0].set_ylabel("items sorted per minute"); ax[0, 0].set_title("throughput: the robot saturates"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_xlabel("items arriving per minute"); ax[0, 1].set_ylabel("% of items not sorted"); ax[0, 1].set_title("miss rate (missed, late, dropped, wrong chute)")
    a = ax[1, 0]
    for v in (0.15, 0.3, 0.5):
        for comp, ls in ((True, "-"), (False, "--")):
            rows = sorted([r for r in T["rows"] if r["speed"] == v and (r["compensated"] == comp or r["latency_s"] == 0)], key=lambda r: r["latency_s"])
            a.plot([1000 * r["latency_s"] for r in rows], [r["err_mean_mm"] for r in rows], ls=ls, marker="o", ms=4, color={0.15: "#2a6fdb", 0.3: "#c33", 0.5: "#555"}[v],
                   label=f"{v} m/s, {'time-stamped' if comp else 'treated as fresh'}")
    a.set_xlabel("camera latency, ms"); a.set_ylabel("error predicting 1.5 s ahead, mm"); a.set_title("tracking: compensating latency keeps the intercept within ~1 cm"); a.legend(fontsize=8)
    a = ax[1, 1]; best = [r for r in E["table"] if r["scheduler"] == "lookahead"]; best.sort(key=lambda r: (r["speed"], r["rate"]))
    labels = [f"{r['speed']}\n{r['rate']:g}/min" for r in best]; bottom = np.zeros(len(best))
    for k, col in (("sorted", "#2a8a3a"), ("missed", "#bbb"), ("late", "#e0a030"), ("dropped", "#c33"), ("wrong_chute", "#803090"), ("knocked_off", "#333")):
        vals = np.array([r["outcomes"].get(k, 0) / max(r["items"], 1) * 100 for r in best]); a.bar(labels, vals, bottom=bottom, color=col, label=k); bottom += vals
    a.set_ylabel("% of items"); a.set_title("what happened to each item (lookahead scheduler)"); a.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); p = os.path.join(OUT, "F2-results.png"); fig.savefig(p, dpi=110); print(p)


def overhead_tracking(path, seconds=30, speed=0.3, rate=8):
    """The overhead camera's view with the tracker drawn on it: each detection (late, noisy) as a dot, each track's
    estimate of the item *now* as a ring, and the planner's intercept point as a cross."""
    import imageio.v3 as iio, mujoco
    from PIL import Image, ImageDraw
    from line import Line
    ln = Line(speed=speed, rate_per_min=rate, scheduler="lookahead", seed=3); cam = ln.lay.camera; W, H = cam.width, cam.height
    r = mujoco.Renderer(ln.c.model, H, W); f = (H / 2) / math.tan(math.radians(cam.fovy) / 2); frames = []; dets = []
    orig = ln.det.poll if hasattr(ln, "det") else None

    def px(p): z = cam.pos[2] - p[2]; return W / 2 + f * (p[0] - cam.pos[0]) / z, H / 2 - f * (p[1] - cam.pos[1]) / z

    def grab(L):
        r.update_scene(L.c.data, camera=cam.name); im = Image.fromarray(r.render()); d = ImageDraw.Draw(im); t = L.c.data.time
        for n, tr in L.trk.tracks.items():
            p, _ = tr.predict(t); u, v = px(p + np.array([0, 0, L.c.items[n]["rec"]["extents"][2]])); d.ellipse((u - 9, v - 9, u + 9, v + 9), outline=(0, 255, 120), width=2)
            d.text((u + 11, v - 6), n.split("_")[0], fill=(0, 255, 120))
        if L.cur is not None and not L.recovering and L.cur.name:
            u, v = px(L.cur.p_pred + np.array([0, 0, 0.0])); d.line((u - 8, v - 8, u + 8, v + 8), fill=(255, 60, 60), width=2); d.line((u - 8, v + 8, u + 8, v - 8), fill=(255, 60, 60), width=2)
        d.rectangle((0, 0, W, 22), fill=(0, 0, 0)); d.text((6, 4), f"overhead D435i  t {t:5.1f} s   green ring: track estimate now (30 Hz, 3 mm noise, 60 ms late, compensated)   red x: planned grasp point", fill=(255, 255, 255))
        frames.append(np.asarray(im))
    ln.run(seconds=seconds, on_frame=grab, frame_every=20)
    iio.imwrite(path, np.stack(frames), fps=25, codec="libx264", plugin="pyav"); print(path)


def main():
    import warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(); p.add_argument("--figure-only", action="store_true"); a = p.parse_args()
    figure()
    if a.figure_only: return
    import eval_f2
    for v, rate in ((0.3, 8), (0.15, 8)):
        vp, S = eval_f2.video(v, rate, "lookahead", 90, os.path.join(OUT, f"F2-line-{v:.2f}.mp4")); print(vp, S["summary"]["outcomes"])
        os.system(f"{ROOT}/bin/rl-sheet {vp} {vp[:-4]}-sheet.png >/dev/null 2>&1")
    overhead_tracking(os.path.join(OUT, "F2-tracking-overhead.mp4"))
    # a planted fault: the gripper lets go mid-carry on the first attempt; the line detects the drop and recovers
    import imageio.v3 as iio, mujoco
    from PIL import Image, ImageDraw
    from line import Line
    ln = Line(speed=0.15, rate_per_min=4, scheduler="lookahead", seed=1, inject={1: "drop"}); r = mujoco.Renderer(ln.c.model, 540, 960); frames = []

    def grab(L):
        r.update_scene(L.c.data, camera="pick"); im = Image.fromarray(r.render()); d = ImageDraw.Draw(im); last = L.events[-1] if L.events else {}
        d.rectangle((0, 0, 960, 24), fill=(0, 0, 0)); d.text((8, 5), f"t {L.c.data.time:5.1f}  planted fault: the gripper lets go mid-carry   |   last attempt: {last.get('item', '-')} {last.get('outcome', '')}   robot: {'recovering' if L.recovering else (L.cur.name if L.cur else 'waiting')}", fill=(255, 255, 255))
        frames.append(np.asarray(im))
    ln.run(seconds=40, on_frame=grab, frame_every=20); vp = os.path.join(OUT, "F2-fault-drop.mp4"); iio.imwrite(vp, np.stack(frames), fps=25, codec="libx264", plugin="pyav"); print(vp)


if __name__ == "__main__":
    main()
