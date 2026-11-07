"""bc.py — a sim-trained pick for the SO-101: behaviour cloning from the classical picks, exported to ONNX and run as
a fixed-rate loop (2026-11-03, F6; disciplines 10a, 14).

Behaviour cloning is the simplest kind of robot learning: record an expert doing the task, then train a network to
output what the expert did in each situation (supervised regression, no reward). The expert here is the classical
pipeline (desk_pick.py) with the real-device kit on; the network is what LeRobot-style policies are on the real
SO-101: a function from what the robot senses to the next goal positions for its six servos, called at 30 Hz.

  observation (9)   the camera's item estimate (x, y on the belt, normalized; for a box the axis as cos 2a, sin 2a, an
                    axis having no head or tail; zeros for a round item), the item class (4, one-hot, the camera's
                    call), time since the start / 10 s
  action (6)        goal positions for the six servos (absolute, rad), written over the bus at 30 Hz
  network           MLP 9 -> 256 -> 256 -> 256 -> 6 (SiLU), inputs and outputs normalized inside the graph

Why no servo positions in the observation (found 2026-11-03): the first policy also read the six servo positions.
It fitted the demonstrations to 0.1 deg (teacher-forced) and failed 34 of 40 closed-loop trials: it had learned
"goal = reading + a small offset", the reading follows the goal, so a 1 deg offset on the wrist roll fed back on
itself and the jaw goal ran away to 6 rad (the copycat problem of behaviour cloning). On this arm the servos
already close the position loop at the physics rate; the policy's job is the task (where, which, when), so it sees
the task. The recorded data keep the readings (`features` returns them first) for anyone who wants to try again.

Why time is an input: nothing else says where in the pick the arm is. Half the demonstrations were recorded with
noise on the goals actually sent (DART, sigma `DART_SIGMA`, label = the expert's clean goal), meant for a policy that
reads the servos; with the task-only observation it only widens what the camera saw (the noisy runs knock more items,
and only successful demonstrations are kept).

Commands (CPU; workers in parallel processes):
  envs/mj/bin/python -I factory/desk/bc.py collect --demos 3000 --workers 12   -> runs/f6-bc/data/*.npz
  envs/mj/bin/python -I factory/desk/bc.py train [--epochs 40]                 -> factory/desk/models/pick.onnx (+ .json)
  envs/mj/bin/python -I factory/desk/bc.py check --trials 40                   -> a quick success rate, kit on
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import desk_items as I  # noqa: E402
import desk_pick as P  # noqa: E402

RUN = os.path.join(ROOT, "runs", "f6-bc"); DATA = os.path.join(RUN, "data")
MODELS = os.path.join(HERE, "models"); ONNX = os.path.join(MODELS, "pick.onnx")
CLASSES = ["tomato_soup_can", "potted_meat_can", "standin_pet_bottle", "standin_tea_box"]
ROUND = {"tomato_soup_can", "standin_pet_bottle"}
T_RUN = 10.5          # s the policy runs per trial (the classical plans take 6.4-9.7 s)
DART_SIGMA = 0.015    # rad, noise on the executed arm goals in half the demonstrations
TRAIN_SEED0 = 100_000 # training scenes use seeds from here (the acceptance evaluation uses fresh seeds below it)


PROPRIO = 6          # the first 6 entries of `features` are the servo readings; the policy input drops them


def obs_of(f): return f[PROPRIO:]


def features(q_read, est, cls, t):
    """The recorded feature vector (float32, 15): 6 servo readings, then the policy's 9 inputs (obs_of)."""
    x, y, yaw = est; b = (0.235, 0.0)
    if cls in ROUND: c2 = s2 = 0.0
    else:
        ex, ey = I.info(cls)["extents"][:2]; a = yaw if ex <= ey else yaw + math.pi / 2; c2, s2 = math.cos(2 * a), math.sin(2 * a)
    oh = [1.0 if cls == c else 0.0 for c in CLASSES]
    return np.array([*q_read, (x - b[0]) / 0.05, (y - b[1]) / 0.1, c2, s2, *oh, t / 10.0], np.float32)


# ---------------------------------------------------------------- data
def _worker(args):
    wid, seeds, cam_seed = args
    import camera as C
    from desk_sim import DeskSim
    from servo import DeviceKit
    sk = DeviceKit.real_kit(); sim = DeskSim(kit=sk); cam = C.Camera(sim, C.CameraKit.real_kit(seed=cam_seed), rng=np.random.default_rng(cam_seed))
    rep = C.calibrate(sim, cam, sk.backlash_rad); rng = np.random.default_rng(wid + 7)
    X, Y, E, kept, outcomes = [], [], [], 0, {}
    for k, seed in enumerate(seeds):
        name = CLASSES[seed % 4]; _, truth = P.scene(sim, name, seed)
        cam.flush(); e = cam.estimate()
        if e is None: outcomes["unseen"] = outcomes.get("unseen", 0) + 1; continue
        est = (e["x"], e["y"], e["yaw"]); cls = e["item"]; plan = P.Planner(sim).plan(cls, est)
        if plan is None: outcomes["ik_fail"] = outcomes.get("ik_fail", 0) + 1; continue
        noisy = (k % 2 == 1); t0 = sim.t; xs, ys = [], []

        def on_tick(sim, seg, goal):
            xs.append(features(sim.bus.read(sim.t), est, cls, sim.t - t0)); ys.append(goal.astype(np.float32).copy())
            if noisy:
                g = goal.copy(); g[:5] += rng.normal(size=5) * DART_SIGMA; sim.bus.write(g, sim.t)
        ev = P.Executor(sim).run(plan, name, on_tick=on_tick)
        # pad to T_RUN with the final (home) goal so the policy learns to stay put when the pick is done
        last = ys[-1]; n_pad = int(round((T_RUN - (sim.t - t0)) * P.HOST_HZ))
        for i in range(max(0, n_pad)):
            tt = sim.t - t0 + i / P.HOST_HZ; xs.append(features(sim.bus.read(sim.t), est, cls, tt)); ys.append(last)
        out, _ = P.classify(sim, plan, name, ev); outcomes[out] = outcomes.get(out, 0) + 1
        if out == "ok" and cls == name:
            X.append(np.array(xs)); Y.append(np.array(ys)); E.append(np.full(len(xs), seed)); kept += 1
    if X:
        np.savez_compressed(os.path.join(DATA, f"shard-{wid:02d}.npz"), X=np.concatenate(X), Y=np.concatenate(Y), seed=np.concatenate(E))
    cam.close()
    return dict(worker=wid, demos=kept, outcomes=outcomes, calibration=rep)


def collect(demos, workers):
    import multiprocessing as mp
    os.makedirs(DATA, exist_ok=True)
    for f in glob.glob(os.path.join(DATA, "shard-*.npz")): os.remove(f)
    per = int(math.ceil(demos / workers))
    jobs = [(w, list(range(TRAIN_SEED0 + w * per, TRAIN_SEED0 + (w + 1) * per)), 200 + w) for w in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool: res = pool.map(_worker, jobs)
    summary = dict(demos_requested=demos, kept=sum(r["demos"] for r in res), seconds=round(time.time() - t0, 1), workers=res, dart_sigma=DART_SIGMA)
    json.dump(summary, open(os.path.join(RUN, "collect.json"), "w"), indent=1); return summary


# ---------------------------------------------------------------- training
def train(epochs=40, hidden=256, lr=2e-3, batch=4096, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    sh = [np.load(f) for f in sorted(glob.glob(os.path.join(DATA, "shard-*.npz")))]
    X = np.concatenate([s["X"] for s in sh])[:, PROPRIO:]; Y = np.concatenate([s["Y"] for s in sh]); S = np.concatenate([s["seed"] for s in sh])
    # hold out 5% of the demonstrations (whole episodes) to watch for overfitting
    eps = np.unique(S); rng = np.random.default_rng(seed); val_eps = set(rng.choice(eps, max(1, len(eps) // 20), replace=False).tolist())
    val = np.array([s in val_eps for s in S])
    xm, xs = X.mean(0), X.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0) + 1e-6
    xs = np.maximum(xs, 0.2)   # one-hot and scaled inputs: keep their scale near 1
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32); Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
    w = torch.tensor(1.0 / (ys / ys[:5].mean()) ** 0 , dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    tr_idx = np.nonzero(~val)[0]; va_idx = np.nonzero(val)[0]; steps = epochs * int(math.ceil(len(tr_idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    log = []; t0 = time.time(); noise = torch.tensor(np.r_[np.full(4, 0.004) / xs[:4], np.zeros(X.shape[1] - 4)], dtype=torch.float32)   # ~0.2-0.4 mm / 0.2 deg of camera jitter
    for ep in range(epochs):
        perm = np.random.permutation(tr_idx); net.train(); tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]; xb = Xt[b] + torch.randn(len(b), X.shape[1]) * noise; yb = Yt[b]
            loss = (((net(xb) - yb) ** 2) * w).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            pv = net(Xt[va_idx]).numpy() * ys + ym; err = np.degrees(np.abs(pv - Y[va_idx])).mean(0)
        log.append(dict(epoch=ep, train_mse=tot / len(tr_idx), val_mae_deg=err.round(3).tolist())); print(json.dumps(log[-1]), flush=True)
    # export: normalization inside the graph, so the deployed model takes raw features and returns raw goals
    class Deploy(nn.Module):
        def __init__(s):
            super().__init__(); s.net = net
            s.register_buffer("xm", torch.tensor(xm, dtype=torch.float32)); s.register_buffer("xs", torch.tensor(xs, dtype=torch.float32))
            s.register_buffer("ym", torch.tensor(ym, dtype=torch.float32)); s.register_buffer("ys", torch.tensor(ys, dtype=torch.float32))
        def forward(s, x): return s.net((x - s.xm) / s.xs) * s.ys + s.ym
    os.makedirs(MODELS, exist_ok=True); dep = Deploy().eval()
    torch.onnx.export(dep, torch.zeros(1, X.shape[1]), ONNX, input_names=["obs"], output_names=["goal"], dynamic_axes={"obs": {0: "n"}, "goal": {0: "n"}}, opset_version=17, dynamo=False)
    meta = dict(date="2026-11-03", samples=int(len(X)), episodes=int(len(eps)), val_episodes=len(val_eps), epochs=epochs, hidden=hidden, lr=lr, batch=batch,
                train_seconds=round(time.time() - t0, 1), features=int(X.shape[1]), observation="task (camera estimate, class, time)", classes=CLASSES, t_run=T_RUN, host_hz=P.HOST_HZ, final=log[-1], log=log)
    json.dump(meta, open(ONNX.replace(".onnx", ".json"), "w"), indent=1); json.dump(meta, open(os.path.join(RUN, "train.json"), "w"), indent=1)
    return meta


# ---------------------------------------------------------------- the deployed loop
class OnnxPolicy:
    def __init__(self, path=ONNX):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, obs):
        return self.sess.run(None, {"obs": obs[None, :].astype(np.float32)})[0][0]


def run_policy(sim, policy, name, est, cls, pace=False, record=None):
    """The fixed-rate loop: every 1/30 s read the servos, build the observation, run the network, write the goals.
    With pace=True the loop runs against the wall clock (sleeping to the next tick), so its rate can be measured.
    Returns events for classify() and the loop's timing."""
    dt = 1.0 / P.HOST_HZ; steps = max(1, int(round(dt / sim.model.opt.timestep))); n = int(round(T_RUN * P.HOST_HZ))
    p0 = sim.item_pose(name)[0]; ev = dict(collision=None, knocked=False, drop=False, lifted_after_lift=False)
    t0 = sim.t; infer = []; ticks = []; wall0 = time.perf_counter(); closed_at = None
    for k in range(n):
        if pace:
            target = wall0 + k * dt
            while time.perf_counter() < target: time.sleep(max(0.0, min(0.002, target - time.perf_counter())))
            ticks.append(time.perf_counter())
        a = time.perf_counter(); obs = obs_of(features(sim.bus.read(sim.t), est, cls, sim.t - t0)); goal = policy(obs); infer.append(time.perf_counter() - a)
        sim.bus.write(goal, sim.t)
        if record: record(sim, None, goal)
        for _ in range(steps):
            sim.step(1)
            c = sim.robot_fixture_contact()
            if c and ev["collision"] is None: ev["collision"] = dict(t=round(sim.t - t0, 3), pair=c)
        p = sim.item_pose(name)[0]
        if goal[5] < 0.3 and closed_at is None: closed_at = k
        if closed_at is None and (np.linalg.norm(p[:2] - p0[:2]) > 0.02 or sim.tilt_deg(name) > 20): ev["knocked"] = True
        if p[2] > p0[2] + 0.01: ev["lifted_after_lift"] = True
    sim.step(int(0.8 / sim.model.opt.timestep))
    timing = dict(infer_ms_mean=1000 * float(np.mean(infer)), infer_ms_p99=1000 * float(np.percentile(infer, 99)), infer_ms_max=1000 * float(np.max(infer)))
    if pace and len(ticks) > 2:
        per = np.diff(ticks) * 1000; timing.update(period_ms_mean=float(per.mean()), period_ms_p99=float(np.percentile(per, 99)), period_ms_max=float(per.max()),
                                                     late_ticks=int(np.sum(per > 1000 * dt * 1.5)))
    return ev, timing


def policy_trial(sim, policy, name, seed, cam, pace=False, record=None):
    _, (x, y, yaw) = P.scene(sim, name, seed)
    cam.flush(); e = cam.estimate()
    if e is None: return dict(item=name, seed=seed, outcome="unseen")
    est = (e["x"], e["y"], e["yaw"]); ev, timing = run_policy(sim, policy, name, est, e["item"], pace=pace, record=record)
    out, where = P.classify(sim, None, name, ev)
    return dict(item=name, seen=e["item"], seed=seed, outcome=out, bin=where, x=round(x, 4), y=round(y, 4), yaw=round(yaw, 3), collision=ev["collision"], timing=timing)


def check(trials=40, seed0=90_000, cam_seed=77):
    import camera as C
    from desk_sim import DeskSim
    from servo import DeviceKit
    sk = DeviceKit.real_kit(); sim = DeskSim(kit=sk); cam = C.Camera(sim, C.CameraKit.real_kit(seed=cam_seed), rng=np.random.default_rng(cam_seed))
    rep = C.calibrate(sim, cam, sk.backlash_rad); pol = OnnxPolicy(); res = []
    for s in range(trials):
        r = policy_trial(sim, pol, CLASSES[s % 4], seed0 + s, cam); res.append(r)
        if r["outcome"] != "ok": print("  ", json.dumps({k: v for k, v in r.items() if k != "timing"}))
    from collections import Counter
    c = Counter(r["outcome"] for r in res); print(dict(c), "calibration", rep); cam.close(); return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("what", choices=["collect", "train", "check"])
    ap.add_argument("--demos", type=int, default=3000); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--trials", type=int, default=40); ap.add_argument("--hidden", type=int, default=256)
    a = ap.parse_args()
    if a.what == "collect": print(json.dumps(collect(a.demos, a.workers), indent=1)[:3000])
    elif a.what == "train": m = train(a.epochs, a.hidden); print(json.dumps(m["final"]))
    else: check(a.trials)
