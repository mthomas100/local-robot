"""f6_media.py — stills and videos of the desk cell (2026-11-03, F6). Stills go to docs/media (git),
videos too (on disk, ignored by git), contact sheets next to them.

    envs/mj/bin/python -I factory/desk/f6_media.py stills | pick [--item NAME --seed N --kit real|ideal] | onnx | calib | results | all
"""
from __future__ import annotations
import argparse, math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import imageio.v2 as iio  # noqa: E402
from desk_sim import DeskSim  # noqa: E402
from servo import DeviceKit  # noqa: E402
import camera as C  # noqa: E402
import desk_pick as P  # noqa: E402
import desk_items as I  # noqa: E402

MEDIA = os.path.join(ROOT, "docs", "media")


def sheet(video, png):
    subprocess.run([os.path.join(ROOT, "bin", "rl-sheet"), video, png], check=False, capture_output=True); return png


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 + 7 * len(text), 20], fill=(0, 0, 0)); d.text((5, 4), text, fill=(255, 255, 255)); return np.asarray(im)


def stills():
    sim = DeskSim(items=I.admitted()); out = []
    for k, (n, y) in enumerate(zip(sim.items, (-0.075, -0.025, 0.025, 0.075))):
        sim.spawn(n, sim.lay.belt.x + (0.015 if k % 2 else -0.015), y, 0.4 * k)
    sim.step(150)
    for cam in ("desk", "front", "top"):
        out.append(sim.still(os.path.join(MEDIA, f"F6-desk-{cam}.png"), cam, 1280, 720))
    cam = C.Camera(sim, C.CameraKit.real_kit(seed=1), rng=np.random.default_rng(1)); f = cam.capture()
    d = f["depth"]; dep = (np.clip((0.50 - d) / 0.12, 0, 1) * 255).astype(np.uint8); dep[d == 0] = 0
    both = np.concatenate([f["rgb"], np.repeat(dep[..., None], 3, 2)], axis=1)
    p = os.path.join(MEDIA, "F6-camera-rgbd.png"); iio.imwrite(p, label(both, "overhead camera, real-device kit: RGB (noise) | depth (z^2 noise, holes)")); out.append(p)
    cam.close(); return out


def pick(item="tomato_soup_can", seed=7, kit="real", out=None, cam_view=True):
    real = kit == "real"; sk = DeviceKit.real_kit() if real else DeviceKit.ideal()
    sim = DeskSim(items=I.admitted(), kit=sk)
    cam = C.Camera(sim, C.CameraKit.real_kit(seed=seed) if real else C.CameraKit(), rng=np.random.default_rng(seed))
    rep = C.calibrate(sim, cam, sk.backlash_rad) if real else None
    frames = []

    def rec(sim, seg, goal):
        a = sim.render("desk", 848, 480); b = sim.render("side", 424, 240); c = cam.capture()["rgb"]
        a = label(a, f"{item}  seed {seed}  kit {kit}  t {sim.t:4.1f}s  {seg.name}")
        right = np.concatenate([label(b, "side"), label(c, "overhead camera (what the cell sees)")], axis=0)
        frames.append(np.concatenate([a, right], axis=1))
    r = P.trial(sim, item, seed, cam=cam if real else None, record=rec)
    for _ in range(20): frames.append(frames[-1])
    out = out or os.path.join(MEDIA, f"F6-pick-{item}-{kit}.mp4")
    iio.mimsave(out, frames, fps=30, macro_block_size=1); sheet(out, out.replace(".mp4", "-sheet.png")); cam.close()
    return out, r, rep


def onnx_pick(items=("tomato_soup_can", "standin_tea_box", "standin_pet_bottle", "potted_meat_can"), seed0=7100, out=None):
    """The behaviour-cloned ONNX policy at 30 Hz picking each admitted item once, whole kit on."""
    import bc
    sk = DeviceKit.real_kit(); sim = DeskSim(items=I.admitted(), kit=sk)
    cam = C.Camera(sim, C.CameraKit.real_kit(seed=seed0), rng=np.random.default_rng(seed0)); C.calibrate(sim, cam, sk.backlash_rad)
    pol = bc.OnnxPolicy(); out = out or os.path.join(MEDIA, "F6-onnx-pick.mp4"); w = iio.get_writer(out, fps=30, macro_block_size=1); res = []
    for k, item in enumerate(items):
        def rec(sim, seg, goal, item=item):
            a = label(sim.render("desk", 848, 480), f"ONNX policy (behaviour-cloned, 30 Hz): {item}  t {sim.t:4.1f}s  jaw goal {goal[5]:+.2f}")
            right = np.concatenate([label(sim.render("side", 424, 240), "side"), label(cam.capture()["rgb"], "overhead camera")], axis=0)
            w.append_data(np.concatenate([a, right], axis=1))
        res.append(bc.policy_trial(sim, pol, item, seed0 + k, cam, record=rec)["outcome"])
    w.close(); sheet(out, out.replace(".mp4", "-sheet.png")); cam.close(); return out, res


def calib_video(seed=4242, out=None):
    """The calibration routine as a viewer sees it: the arm shows the marker to the camera at every pose."""
    sk = DeviceKit.real_kit(); sim = DeskSim(items=I.admitted(), kit=sk)
    cam = C.Camera(sim, C.CameraKit.real_kit(seed=seed), rng=np.random.default_rng(seed))
    out = out or os.path.join(MEDIA, "F6-calibration.mp4"); w = iio.get_writer(out, fps=30, macro_block_size=1)
    steps = int(round(1 / 30 / sim.model.opt.timestep)); orig = sim.step; n = [0]

    def show(k=1):
        for _ in range(k):
            orig(1); n[0] += 1
            if n[0] % steps == 0:
                a = label(sim.render("desk", 848, 480), "hand-eye calibration: the arm shows the magenta marker to the overhead camera")
                right = np.concatenate([label(sim.render("top", 424, 240), "top"), label(cam.capture()["rgb"], "what the camera sees")], axis=0)
                w.append_data(np.concatenate([a, right], axis=1))
    sim.step = show; rep = C.calibrate(sim, cam, sk.backlash_rad); sim.step = orig
    w.close(); sheet(out, out.replace(".mp4", "-sheet.png")); cam.close(); return out, rep


def results(out=None):
    """The acceptance figure from the recorded results: picks per item (classical vs ONNX, the bars), the calibration
    before/after per planted error (2 mm, 1 deg bars), the 30 Hz loop's tick periods and inference time."""
    import json
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = json.load(open(os.path.join(HERE, "results", "f6-eval.json"))); out = out or os.path.join(MEDIA, "F6-results.png")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6)); fig.suptitle("F6 acceptance with the real-device kit on (attempt 2, fresh seeds)", fontsize=12)
    p = R["picks"]; items = I.admitted(); xs = np.arange(len(items) + 1); w = 0.38
    for k, (key, col) in enumerate((("classical", "#2b6cb0"), ("onnx", "#dd6b20"))):
        per = [int(p[key]["per_item"][n].split("/")[0]) / int(p[key]["per_item"][n].split("/")[1]) for n in items] + [p[key]["rate"]]
        ax[0].bar(xs + (k - 0.5) * w, np.array(per) * 100, w, color=col, label=f"{key}: {p[key]['ok']}/{p[key]['n']}")
    ax[0].axhline(90, color="#2b6cb0", ls="--", lw=1); ax[0].axhline(80, color="#dd6b20", ls="--", lw=1)
    short = {"tomato_soup_can": "tomato\nsoup can", "potted_meat_can": "potted\nmeat can", "standin_pet_bottle": "stand-in\nPET bottle", "standin_tea_box": "stand-in\ntea box"}
    ax[0].set_xticks(xs, [short.get(n, n) for n in items] + ["all 200"], fontsize=8)
    ax[0].set_ylim(50, 102); ax[0].set_ylabel("% in the right bin"); ax[0].legend(loc="lower left", fontsize=8); ax[0].set_title("200 L0 picks (bars: 90% classical, 80% ONNX)", fontsize=10)
    rows = R["calib"]["rows"]; xs = np.arange(len(rows))
    ax[1].bar(xs - 0.2, [r["before_mm"] for r in rows], 0.4, color="#a0aec0", label="before (planted), mm")
    ax[1].bar(xs + 0.2, [r["after_mm"] for r in rows], 0.4, color="#2f855a", label="after calibration, mm")
    ax[1].axhline(2.0, color="#2f855a", ls="--", lw=1)
    for i, r in enumerate(rows): ax[1].text(i + 0.2, r["after_mm"] + 0.15, f"{r['after_mm']:.2f} mm\n{r['after_deg']:.2f}°", ha="center", fontsize=7)
    ax[1].set_xticks(xs, [f"seed {r['seed']}" for r in rows], fontsize=8); ax[1].set_ylabel("camera position error, mm"); ax[1].set_ylim(0, 10.5)
    ax[1].set_title("hand-eye calibration (bar: 2 mm and 1°; planted 8 mm, 2.5°)", fontsize=10); ax[1].legend(fontsize=8, ncol=2, loc="upper center")
    if "rate" in R:
        rr = R["rate"]["rows"]; xs = np.arange(len(rr))
        ax[2].plot(xs, [r["period_ms_mean"] for r in rr], "o-", color="#2b6cb0", label="tick period, mean")
        ax[2].plot(xs, [r["period_ms_max"] for r in rr], "^", color="#c53030", label="tick period, worst")
        ax[2].plot(xs, [r["infer_ms_p99"] for r in rr], "s", color="#dd6b20", label="ONNX inference, p99")
        ax[2].axhline(1000 / 30, color="k", lw=0.8); ax[2].axhline(50, color="#c53030", ls="--", lw=1)
        sm = R["rate"]["summary"]; ax[2].set_title(f"30 Hz loop vs the wall clock: mean {sm['period_ms_mean']:.2f} ms, {sm['late_ticks']} late of {sm['ticks']}", fontsize=10)
        ax[2].set_xlabel("trial"); ax[2].set_ylabel("ms"); ax[2].set_ylim(0, 55); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what"); ap.add_argument("--item", default="tomato_soup_can"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kit", default="real"); a = ap.parse_args()
    if a.what in ("stills", "all"): print(stills())
    if a.what in ("pick", "all"): print(pick(a.item, a.seed, a.kit))
    if a.what in ("onnx", "all"): print(onnx_pick())
    if a.what in ("calib", "all"): print(calib_video())
    if a.what == "results": print(results())
