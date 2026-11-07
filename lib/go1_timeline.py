"""go1_timeline.py <run> [--max 24] [--cols 6] [--out <run>/timeline.mp4] — the learning timeline: every saved checkpoint
of a Go1 run drives its own copy of the robot on the same command script, side by side in one video, labelled by
iteration (2026-11-01). One mujoco.batch of N sims, sim i driven by checkpoint i. Also writes timeline.json (falls and
mean forward speed per checkpoint during the walk segment)."""
import argparse, glob, json, os, re, sys
import numpy as np, torch, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go1_batch317 as g  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run"); p.add_argument("--max", type=int, default=24); p.add_argument("--cols", type=int, default=6)
p.add_argument("--out", default=None); p.add_argument("--fps", type=int, default=25); p.add_argument("--threads", type=int, default=0)
a = p.parse_args()
out = a.out or os.path.join(a.run, "timeline.mp4")
files = sorted(glob.glob(os.path.join(a.run, "ckpt", "policy-*.pt")))
if not files: sys.exit(f"no checkpoints under {a.run}/ckpt (train with the current lib/go1_batch317.py)")
if len(files) > a.max: files = [files[round(i * (len(files) - 1) / (a.max - 1))] for i in range(a.max)]
its = [int(re.search(r"policy-(\d+)\.pt", f).group(1)) for f in files]
N = len(files)
g.THREADS = a.threads
env = g.Go1(N, seed=g.SEED + 1)
nets = []
for f in files:
  net = g.ActorCritic(); net.load_state_dict(torch.load(f, map_location="cpu")); net.eval(); nets.append(net)
W, H = 320, 240
renderer = mujoco.Renderer(env.model, H, W); data = mujoco.MjData(env.model)
cam = mujoco.MjvCamera(); cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.model.body("trunk").id
cam.distance, cam.azimuth, cam.elevation = 1.8, 135.0, -15.0
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20) if os.path.exists("/System/Library/Fonts/Helvetica.ttc") else ImageFont.load_default()
SEGMENTS = ((1.0, 0.0, 0.0, 0.0), (4.0, 1.0, 0.0, 0.0), (2.5, 1.0, 0.0, 1.0), (1.5, 0.0, 0.0, 0.0))
every = max(1, round(1.0 / (g.CTRL_DT * a.fps)))
rows = (N + a.cols - 1) // a.cols
frames, falls, speed = [], np.zeros(N, int), np.zeros(N)
with torch.no_grad():
  for seconds, vx, vy, wz in SEGMENTS:
    steps = int(seconds / g.CTRL_DT)
    for k in range(steps):
      env.command[:], env.until[:], env.steps[:] = (vx, vy, wz), 2, 1
      obs = env.obs(); act = np.zeros((N, g.ACT_DIM), np.float32)
      for i, net in enumerate(nets): act[i] = net(torch.as_tensor(obs[i : i + 1]))[0].numpy()[0]
      _, _, fell, _ = env.step(act)
      if vx == 1.0 and wz == 0.0: speed += env.vel[:, 0] / steps
      ids = np.flatnonzero(fell)
      if ids.size: falls[ids] += 1; env.reset(ids)
      if k % every == 0:
        sheet = Image.new("RGB", (a.cols * W, rows * H), (10, 10, 10)); d = ImageDraw.Draw(sheet)
        for i in range(N):
          env.batch.get_data(i, data); renderer.update_scene(data, cam)
          r, c = divmod(i, a.cols); sheet.paste(Image.fromarray(renderer.render()), (c * W, r * H))
          label = f"it {its[i]}" + (f"  falls {falls[i]}" if falls[i] else "")
          d.rectangle([c * W, r * H, c * W + 10 + len(label) * 11, r * H + 26], fill=(0, 0, 0)); d.text((c * W + 5, r * H + 3), label, fill=(255, 255, 255), font=font)
        frames.append(np.asarray(sheet))
renderer.close()
import imageio.v3 as iio
iio.imwrite(out, np.stack(frames), fps=a.fps, codec="libx264", plugin="pyav")
summary = dict(video=out, checkpoints=its, falls=falls.tolist(), walk_speed=[round(float(x), 2) for x in speed], frames=len(frames))
json.dump(summary, open(os.path.join(a.run, "timeline.json"), "w"), indent=1); print(json.dumps(summary))
