"""f4_media.py — F4's stills and videos for the README and the lab (2026-11-02).

    envs/mj/bin/python -I factory/vision/f4_media.py [--seconds 90] [--figure-only]
  docs/media/F4-vision-line-0.30.mp4 (+ -sheet.png)  the L4 line: pick camera | what the overhead camera sees,
                                                              every blob boxed, every track labelled (vision + student carry)
  docs/media/F4-camera-view.png                       one overhead frame with perception's boxes and labels
  docs/media/F4-results.png                           L2 vs L4 sort rate per belt speed; perception error per item
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
for p in (HERE, os.path.join(FACTORY, "learned"), os.path.join(FACTORY, "line"), os.path.join(FACTORY, "cell")): sys.path.insert(0, p)
OUT = os.path.join(ROOT, "docs", "media")


def line_video(seconds=90, speed=0.3, seed=5):
    import mujoco, imageio.v2 as iio2
    from PIL import Image, ImageDraw
    from line_learned import LearnedLine
    from vision import overlay
    ln = LearnedLine(policy=os.path.join(FACTORY, "learned", "student.json"), privileged_carry=False, speed=speed, rate_per_min=16, seed=seed, perception="vision")
    r = mujoco.Renderer(ln.c.model, 540, 960); p = os.path.join(OUT, f"F4-vision-line-{speed:.2f}.mp4")
    out = iio2.get_writer(p, fps=25, codec="libx264", macro_block_size=4); still = {}

    def grab(L):
        L.vision.keep_view = True
        r.update_scene(L.c.data, camera="pick"); im = Image.fromarray(r.render()); d = ImageDraw.Draw(im); done = L.log_items; ok = sum(i["outcome"] == "sorted" for i in done)
        d.rectangle((0, 0, 960, 24), fill=(0, 0, 0)); d.text((8, 5), f"t {L.c.data.time:5.1f} s   belt {speed:.2f} m/s   16 items/min   L4: vision only + student carry   |   sorted {ok}/{len(done)}", fill=(255, 255, 255))
        canvas = Image.new("RGB", (960 + 848, 540), (0, 0, 0)); canvas.paste(im, (0, 0))
        if L.vision.view is not None:
            ov = overlay(L.vision.view, scale=2); canvas.paste(ov, (960, 30))
            if len(L.vision.view["tracks"]) >= 2 and "frame" not in still: still["frame"] = ov
        out.append_data(np.asarray(canvas))
    S = ln.run(seconds=seconds, on_frame=grab, frame_every=20); out.close()
    if "frame" in still: still["frame"].save(os.path.join(OUT, "F4-camera-view.png"))
    os.system(f"{ROOT}/bin/rl-sheet {p} {p[:-4]}-sheet.png >/dev/null 2>&1"); print(p, S["summary"]["outcomes"], S["summary"]["sort_rate"])


def figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    E = json.load(open(os.path.join(HERE, "results", "f4-eval.json"))); V = json.load(open(os.path.join(HERE, "results", "f4-vision-static.json")))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cells = ["L2-f2", "L4-f2", "L2", "L4"]; cols = {"L2-f2": "#9db4d6", "L4-f2": "#e3b27c", "L2": "#2f6db5", "L4": "#d9822b"}
    for k, sp in enumerate((0.15, 0.3)):
        for j, c in enumerate(cells):
            row = E["table"].get(f"{sp:.2f} m/s · {c}")
            if row: ax[0].bar(k * 5 + j, row["sort_rate"], color=cols[c], label=c if k == 0 else None)
    ax[0].set_xticks([1.5, 6.5]); ax[0].set_xticklabels(["belt 0.15 m/s", "belt 0.30 m/s"]); ax[0].set_ylabel("sort rate (sorted / arrived)"); ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8, ncol=2); ax[0].set_title("L2 (privileged poses) vs L4 (vision only), 16 seeds x 180 s")
    names = list(E["verdict"]); gaps = [E["verdict"][n]["gap"] for n in names]; lo = [E["verdict"][n]["ci95"][0] for n in names]; hi = [E["verdict"][n]["ci95"][1] for n in names]
    ax[1].errorbar(range(len(names)), gaps, yerr=[np.subtract(gaps, lo), np.subtract(hi, gaps)], fmt="o", capsize=6, color="#333")
    ax[1].axhline(0.10, color="#c33", ls="--", label="the bar: 10 points"); ax[1].axhline(0, color="#999", lw=0.8)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names); ax[1].set_ylabel("L2 − L4 sort rate (95% interval)"); ax[1].legend(fontsize=8); ax[1].set_title("The verdict: the gap's upper end must stay below 0.10")
    it = V["per_item"]; nm = list(it); ax[2].barh(range(len(nm)), [it[n]["pos_mm"] for n in nm], color="#5a9")
    ax[2].set_yticks(range(len(nm))); ax[2].set_yticklabels([n.replace("_", " ") for n in nm], fontsize=8); ax[2].set_xlabel("position error, mm (mean)")
    ax[2].set_title(f"Perception vs ground truth: identity {100 * V['identity_accuracy']:.0f}%, {V['ms_per_frame']:.0f} ms/frame")
    fig.tight_layout(); p = os.path.join(OUT, "F4-results.png"); fig.savefig(p, dpi=110); print(p)


def main():
    import warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(); p.add_argument("--seconds", type=float, default=90); p.add_argument("--figure-only", action="store_true"); p.add_argument("--video-only", action="store_true")
    a = p.parse_args()
    if not a.video_only and os.path.exists(os.path.join(HERE, "results", "f4-eval.json")): figure()
    if not a.figure_only: line_video(a.seconds)


if __name__ == "__main__":
    main()
