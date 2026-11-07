"""go1_live.py <run> [--port 8765] [--n 16] [--fps 25] — a live view that does not slow training (2026-11-01): its own
batch of n Go1 robots with random joystick commands, driven by the run's latest policy.pt (reloaded whenever the trainer
saves it, every 25 iterations), stepped in real time and streamed as MJPEG on http://127.0.0.1:<port>/ via the Grid class
of go1_batch317. The tiles are therefore the current policy at training-batch-like conditions, not the training sims.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go1_batch317 as g  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run"); p.add_argument("--port", type=int, default=8765); p.add_argument("--n", type=int, default=16)
p.add_argument("--fps", type=int, default=25); p.add_argument("--threads", type=int, default=4)
p.add_argument("--tile", default="256x192", help="tile size WxH; smaller is smoother")
a = p.parse_args()
policy = os.path.join(a.run, "policy.pt"); metrics = os.path.join(a.run, "metrics.jsonl")
g.THREADS = a.threads
env = g.Go1(a.n, seed=g.SEED + 7)
net, saved = g.ActorCritic(), 0.0
tw, th = (int(x) for x in a.tile.split("x"))
grid = g.Grid(env, a.n, os.path.join(a.run, "live-latest.png"), cols=4, w=tw, h=th)
grid.every = max(1, round(1.0 / (g.CTRL_DT * a.fps)))
grid.start_live(a.port, every=grid.every)
with open(os.path.join(a.run, "live.json"), "w") as f:
  json.dump(dict(url=f"http://127.0.0.1:{a.port}/", port=a.port, viewer="go1_live.py", pid=os.getpid()), f)
print(f"live view of {a.run} at http://127.0.0.1:{a.port}/ ({a.n} robots, {a.fps} fps)", flush=True)

def reload():
  global saved
  try:
    m = os.path.getmtime(policy)
  except FileNotFoundError:
    return
  if m > saved:
    try:
      net.load_state_dict(torch.load(policy, map_location="cpu")); net.eval(); saved = m
    except Exception:  # the trainer may be mid-write; try again next time
      return
    try:
      with open(metrics) as f:
        last = f.readlines()[-1]
      row = json.loads(last); grid.it, grid.reward = row["it"], row["reward"]
    except Exception:
      pass

reload()
step, t0 = 0, time.perf_counter()
with torch.no_grad():
  while True:
    if step % 50 == 0:
      reload()
    act = net(torch.as_tensor(env.obs()))[0].numpy() if saved else np.zeros((a.n, g.ACT_DIM), np.float32)
    _, done, fell, _ = env.step(act)
    ids = np.flatnonzero(done)
    if ids.size:
      env.reset(ids)
    grid.tick()
    step += 1
    lag = t0 + step * g.CTRL_DT - time.perf_counter()
    if lag > 0:
      time.sleep(lag)
