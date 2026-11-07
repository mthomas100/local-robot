"""f3_media.py — the F3 figure, the learning timeline and the line videos (2026-11-02).

    envs/mj/bin/python -I factory/learned/f3_media.py [--figure-only] [--run runs/f3-ars]
Writes docs/media/F3-results.png and F3-carry-compare.png (git), and on disk: F3-timeline.mp4 (the same carries
by checkpoints of the training, side by side), F3-line-learned-0.30.mp4 and F3-line-f2-0.30.mp4 (the same seed, the
same 90 s of line, F2's carry vs the learned one).
"""
import argparse, glob, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import carry  # noqa: E402
OUT = os.path.join(ROOT, "docs", "media"); RES = os.path.join(HERE, "results")


def figure(run):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]; T = json.load(open(os.path.join(RES, "f3-train.json")))
    E = json.load(open(os.path.join(RES, "f3-eval.json"))) if os.path.exists(os.path.join(RES, "f3-eval.json")) else None
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5)); fig.suptitle(f"F3: a learned carry (ARS, {T['episodes']} episodes, {T['wall_s'] / 60:.0f} min on 14 CPU cores) — {T['date']}", fontsize=13)
    a = ax[0, 0]; it = [m["iter"] for m in M if "reward" in m]
    a.plot(it, [m["reward"] for m in M if "reward" in m], color="#999", lw=1, label="training batch (randomised mass, friction)")
    ev = [m for m in M if "eval" in m]
    a.plot([m["iter"] for m in ev], [m["eval"]["train"]["reward"] for m in ev], "o-", color="#2a6fdb", label="training items, fixed start states")
    a.plot([m["iter"] for m in ev], [m["eval"]["held_out"]["reward"] for m in ev], "s-", color="#c33", label="held-out items (never trained on)")
    a.axvline(T["best_iter"], color="#2a8a3a", ls=":", lw=1); a.text(T["best_iter"], a.get_ylim()[0], " chosen", color="#2a8a3a", fontsize=8, va="bottom")
    a.set_xlabel("ARS iteration"); a.set_ylabel("return per carry (25 if sorted − seconds until free)"); a.set_title("learning curve"); a.legend(fontsize=8)
    a = ax[0, 1]
    for split, col, mk in (("train", "#2a6fdb", "o"), ("held_out", "#c33", "s")):
        a.plot([m["iter"] for m in ev], [100 * m["eval"][split]["success"] for m in ev], mk + "-", color=col, label=f"{split.replace('_', '-')}: learned")
        a.axhline(100 * ev[0]["eval"][split]["f2_success"], color=col, ls="--", lw=1, label=f"{split.replace('_', '-')}: F2's carry, same states")
    a.set_xlabel("ARS iteration"); a.set_ylabel("% of carries that end in the right chute"); a.set_title("reliability"); a.legend(fontsize=8)
    a2 = a.twinx(); a2.plot([m["iter"] for m in ev], [m["eval"]["held_out"]["t_free"] for m in ev], ":", color="#555"); a2.set_ylabel("held-out: s until the robot is free (dotted)", color="#555")
    a = ax[1, 0]
    best = json.load(open(os.path.join(run, f"ckpt-{T['best_iter']:04d}.json")))["eval"]["per_item"] if os.path.exists(os.path.join(run, f"ckpt-{T['best_iter']:04d}.json")) else None
    if best:
        names = list(best); x = np.arange(len(names))
        a.bar(x - 0.2, [best[n]["f2_t_free"] for n in names], 0.4, color="#999", label="F2 carry")
        a.bar(x + 0.2, [best[n]["t_free"] or 0 for n in names], 0.4, color=["#c33" if best[n]["held_out"] else "#2a6fdb" for n in names], label="learned (red: held out)")
        a.set_xticks(x, [n.replace("_", " ").replace(" can", "").replace(" box", "") + ("*" if best[n]["held_out"] else "") for n in names], rotation=40, fontsize=8)
        a.set_ylabel("seconds from lift to robot free"); a.set_title("the carry per item (* = held out)"); a.legend(fontsize=8)
    a = ax[1, 1]
    if E:
        keys = [k for k in E["table"]]; groups = sorted({k.rsplit(" · ", 1)[0] for k in keys}, key=lambda g: (g.split(" · ")[0], g))
        cols = {"f2": "#999", "prior": "#e0a030", "learned": "#2a8a3a"}; x = np.arange(len(groups)); w = 0.27
        for j, cr in enumerate(("f2", "prior", "learned")):
            vals = [E["table"].get(f"{g} · {cr}", {}).get("held_out_sorted_per_min", 0) for g in groups]
            sds = [E["table"].get(f"{g} · {cr}", {}).get("sorted_per_min_sd", 0) if g.startswith("held-out") else 0 for g in groups]
            a.bar(x + (j - 1) * w, vals, w, yerr=sds, color=cols[cr], capsize=2, label={"f2": "F2 carry", "prior": "prior (no learning)", "learned": "learned"}[cr])
        a.set_xticks(x, [g.replace(" · ", "\n") for g in groups], fontsize=8); a.set_ylabel("held-out items sorted per minute")
        a.set_title(f"on the line: held-out items ({E['seeds']} seeds x {E['seconds']:.0f} s)"); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(OUT, "F3-results.png"); fig.savefig(p, dpi=110); print(p)


def timeline(run, iters=None, picks=("windex_bottle", "sugar_box", "tuna_fish_can", "cracker_box", "mustard_bottle"), width=480, height=300):
    """Each checkpoint carries the same start states; panels side by side, one row per start state."""
    import mujoco, imageio.v3 as iio
    from PIL import Image, ImageDraw
    T = json.load(open(os.path.join(RES, "f3-train.json"))); cks = sorted(glob.glob(os.path.join(run, "ckpt-*.json")))
    all_it = [int(os.path.basename(c)[5:9]) for c in cks]
    iters = iters or sorted({0, all_it[len(all_it) // 4], all_it[len(all_it) // 2], T["best_iter"]})
    snaps = pickle.load(open(os.path.join(ROOT, "runs", "f3-snapshots", "snapshots.pkl"), "rb")); rng = np.random.default_rng(5)
    chosen = []
    for n in picks:
        cand = [i for i, s in enumerate(snaps) if s["item"] == n and s["speed"] == 0.3]; chosen.append(int(rng.choice(cand)))
    env = carry.Env(); r = mujoco.Renderer(env.c.model, height, width); rows = []
    for i in chosen:
        cols = []
        for k in iters:
            th = np.array(json.load(open(os.path.join(run, f"ckpt-{k:04d}.json")))["theta"]); frames = []

            def grab(c, ctl, t):
                r.update_scene(c.data, camera="pick"); frames.append(r.render())
            res = env.episode(snaps[i], carry.Policy(th), on_frame=grab, frame_every=20)
            lab = f"iteration {k}{' (prior)' if k == 0 else ''}: {res['outcome']}, free after {res['t_free']:.2f} s"
            F = []
            for f in frames:
                im = Image.fromarray(f); d = ImageDraw.Draw(im); d.rectangle((0, 0, width, 20), fill=(0, 0, 0)); d.text((6, 4), lab, fill=(120, 255, 120) if res["ok"] else (255, 120, 120)); F.append(np.asarray(im))
            cols.append(F)
        n = max(len(c) for c in cols) + 12
        cols = [c + [c[-1]] * (n - len(c)) for c in cols]
        rows.append([np.concatenate([c[j] for c in cols], axis=1) for j in range(n)])
        rows[-1] = [np.concatenate([np.zeros((22, fr.shape[1], 3), np.uint8), fr]) for fr in rows[-1]]
        im = Image.fromarray(rows[-1][0]); ImageDraw.Draw(im).text((6, 4), f"{snaps[i]['item'].replace('_', ' ')}{' (held out, never trained on)' if snaps[i]['item'] in carry.HELD_OUT else ' (training item)'}   F2's carry: robot free after {snaps[i]['f2_free_s']} s", fill=(255, 255, 255))
        hdr = np.asarray(im)[:22]; rows[-1] = [np.concatenate([hdr, fr[22:]]) for fr in rows[-1]]
    video = [fr for row in rows for fr in row]
    p = os.path.join(OUT, "F3-timeline.mp4"); iio.imwrite(p, np.stack(video), fps=25, codec="libx264", plugin="pyav"); print(p)
    # a still: the middle frame of each row
    still = np.concatenate([row[len(row) // 2] for row in rows[:3]]); Image.fromarray(still).save(os.path.join(OUT, "F3-carry-compare.png")); print(os.path.join(OUT, "F3-carry-compare.png"))


def line_videos(seconds=90):
    sys.path.insert(0, os.path.join(ROOT, "factory", "line"))
    import imageio.v3 as iio, mujoco
    from PIL import Image, ImageDraw
    from line import Line
    from line_learned import LearnedLine
    r = None   # one renderer for both runs: a second mujoco.Renderer in the same process rendered black frames here
    for tag, make in (("f2", lambda: Line(speed=0.3, rate_per_min=16, seed=3)), ("learned", lambda: LearnedLine(speed=0.3, rate_per_min=16, seed=3))):
        ln = make(); r = r or mujoco.Renderer(ln.c.model, 540, 960); p = os.path.join(OUT, f"F3-line-{tag}-0.30.mp4")
        import imageio.v2 as iio2
        out = iio2.get_writer(p, fps=25, codec="libx264", macro_block_size=4)   # streamed: 90 s of frames held in memory was 3.5 GB

        def grab(L):
            r.update_scene(L.c.data, camera="pick"); im = Image.fromarray(r.render()); d = ImageDraw.Draw(im); done = L.log_items; ok = sum(i["outcome"] == "sorted" for i in done)
            d.rectangle((0, 0, 960, 24), fill=(0, 0, 0)); d.text((8, 5), f"t {L.c.data.time:5.1f} s   belt 0.30 m/s   16 items/min   carry: {'F2 (classical)' if tag == 'f2' else 'F3 (learned)'}   |   sorted {ok}/{len(done)}", fill=(255, 255, 255))
            out.append_data(np.asarray(im))
        S = ln.run(seconds=seconds, on_frame=grab, frame_every=20); out.close(); print(p, S["summary"]["outcomes"], S["summary"]["cycle_s"])
        os.system(f"{ROOT}/bin/rl-sheet {p} {p[:-4]}-sheet.png >/dev/null 2>&1")


def main():
    import warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(); p.add_argument("--figure-only", action="store_true"); p.add_argument("--run", default=os.path.join(ROOT, "runs", "f3-ars"))
    p.add_argument("--skip-line", action="store_true"); a = p.parse_args()
    figure(a.run)
    if a.figure_only: return
    timeline(a.run)
    if not a.skip_line: line_videos()


if __name__ == "__main__":
    main()
