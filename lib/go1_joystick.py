# SPDX-License-Identifier: Apache-2.0
# local-robot/lib/go1_joystick.py — a copy of kevinzakka/mjbatch examples/go1_joystick.py (archived 2026-10-21,
# Apache-2.0) patched on 2026-11-01 for agent-driven runs. Physics, rewards and PPO are the upstream code, unchanged.
# What changed: no glfw window (headless), a run directory (--out) with config.json, metrics.jsonl (one line per
# iteration with every reward term, falls, steps/s) and policy.pt, flags for envs/iters/seed/threads/device, and
# --render which plays a scripted command sequence, writes an mp4 with mujoco.Renderer and eval.json with metrics.

"""PPO training a Unitree Go1 to follow a joystick on mjbatch envs (local-robot patched copy)."""

import argparse
import json
import os
import platform
import sys
import time

import mujoco
import mujoco_menagerie as mm
import numpy as np
import torch
import torch.nn as nn

from mjbatch import Batch


def clock(seconds):
  return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def progress(head, done, elapsed, tail, end=False):
  bar = "━" * round(20 * done) + "─" * (20 - round(20 * done))
  left = clock(elapsed / done - elapsed) if done else "-:--"
  print(f"\r{head} {bar} {clock(elapsed)}, {left} left  {tail}\x1b[K", end="\n" if end else "", flush=True)


NUM_ENVS, HORIZON, ITERS, EPISODE = 1024, 24, 600, 500
TIMESTEP, DECIMATION, KP, KD = 0.004, 5, 35.0, 0.5
ROTOR_INERTIA = 0.000111842  # kg m^2, the Go1 motor's rotor
CTRL_DT, ACTION_SCALE = TIMESTEP * DECIMATION, 0.5
FEET, FOOT_RADIUS = ("FR", "FL", "RR", "RL"), 0.023
STAND = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)  # hip, thigh, calf per leg
POSE_W = np.array([1.0, 1.0, 0.1] * 4)
COMMAND_RANGE = np.array([1.5, 0.8, 1.2])  # forward m/s, sideways m/s, yaw rad/s
COMMAND_ON = np.array([0.9, 0.25, 0.5])  # chance each axis is nonzero when redrawn
COMMAND_SECONDS, FRICTION = 5.0, (0.4, 1.0)
GAIT_HZ, SWING, CEILING, WIDTH = 2.0, 0.06, 0.075, 0.035  # trot Hz, swing m, ceiling m, width m
PHASE = np.array([0.0, 0.5, 0.5, 0.0])  # a trot: FR with RL, FL with RR
SIGMA = 0.25
FLOOR = 0.0
REWARD = dict(track=1.0, turn=0.5, gait=2.0, pose=0.5)
REWARD.update(orient=-5.0, bounce=-0.5, wobble=-0.05, limits=-1.0, rate=-0.01, land=-1.0)
LEGS = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
LEG_SIGN = np.array([-1, 1, 1] * 4, np.float32)
MIRROR = np.concatenate([np.arange(9), 9 + LEGS, 21 + LEGS, 33 + LEGS, np.arange(45, 50)])
SIGN = np.r_[[1, -1, 1, -1, 1, -1, 1, -1, 1], np.tile(LEG_SIGN, 3), [1, -1, -1, -1, -1]].astype(np.float32)
SEED = 0
GAMMA, LAMBDA, CLIP = 0.99, 0.95, 0.2
LR, LR_END, LR_TO = 1e-3, 5e-4, 400
EPOCHS, MINIBATCHES, ENT_COEF, LOG_STD, HIDDEN = 5, 4, 0.005, np.log(0.5), 128
OBS_DIM, ACT_DIM = 50, 12
JOINTS, DOFS = slice(7, 7 + ACT_DIM), slice(6, 6 + ACT_DIM)
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "mps" if torch.mps.is_available() else "cpu"
ACTOR = "cuda" if CUDA else "cpu"  # MPS is slower than the CPU for this net
ASSETS = os.path.join(os.path.dirname(__file__), "..", "vendor", "mjbatch", "examples", "assets")
THEME = os.path.join(ASSETS, "theme.xml")
OUT = "runs/go1"  # set from --out
POLICY = os.path.join(OUT, "policy.pt")
THREADS = 0  # 0: all CPUs


def build_model():
  spec, robot = mujoco.MjSpec.from_file(THEME), mm.get("unitree_go1").spec("go1")
  robot.option.cone, robot.option.impratio = mujoco.mjtCone.mjCONE_PYRAMIDAL, 1.0
  spec.attach(robot, frame=spec.worldbody.add_frame(), prefix="")
  spec.delete(spec.light("spotlight"))
  spec.light("sun").mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
  spec.visual.global_.fovy = 38.0
  spec.visual.global_.bvactive = 0  # on by default in MuJoCo 3.11 and 3x slower
  spec.option.timestep = TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.impratio = 1.0
  spec.option.solver = mujoco.mjtSolver.mjSOL_PGS
  spec.option.iterations = 10
  for geom in spec.geoms:
    if geom.name in FEET:
      geom.condim, geom.solimp = 3, [0.9, 0.95, 0.001, 0.5, 2.0]
      geom.margin = 0.0  # the stock 1 mm margin puts contacts outside the touch sensor
    elif geom.name != "floor":
      geom.contype = geom.conaffinity = 0
  for joint in spec.joints[1:]:
    gear = 9 if joint.name.endswith("calf_joint") else 6
    joint.damping, joint.frictionloss = [0.0, 0.0, 0.0], 0.0
    joint.armature = ROTOR_INERTIA * gear**2
  for actuator in spec.actuators:
    actuator.set_to_position(kp=KP, kv=KD)
    actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
  kind, site, body = mujoco.mjtSensor, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_XBODY
  spec.add_sensor(name="gyro", type=kind.mjSENS_GYRO, objtype=site, objname="imu")
  spec.add_sensor(name="vel", type=kind.mjSENS_VELOCIMETER, objtype=site, objname="imu")
  up = kind.mjSENS_FRAMEZAXIS  # the world's z axis in the trunk frame
  spec.add_sensor(name="up", type=up, objtype=body, objname="world", reftype=body, refname="trunk")
  for foot in FEET:
    spec.add_sensor(name=foot, type=kind.mjSENS_FRAMELINVEL, objtype=site, objname=foot)
    spec.add_sensor(name=foot + "_touch", type=kind.mjSENS_TOUCH, objtype=site, objname=foot)
  model = spec.compile()
  sun = model.light("sun").id
  model.light_poscom0[sun] = -6.0 * model.light_dir0[sun]
  return model


def log_density(z, log_std):
  return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi)


class Go1:
  def __init__(self, num_envs, seed=0):
    self.batch = batch = Batch(build_model(), num_envs, THREADS)
    self.qpos, self.qvel, self.ctrl = (batch.bind(f) for f in ("qpos", "qvel", "ctrl"))
    self.gyro, self.vel, self.up = (batch.sensor(s) for s in ("gyro", "vel", "up"))
    self.foot_vel = [batch.sensor(f) for f in FEET]
    self.foot_force = [batch.sensor(f + "_touch") for f in FEET]
    self.torque = batch.bind("actuator_force")
    self.foot_pos = [batch.site(f).xpos for f in FEET]
    self.trunk = batch.body("trunk").xpos
    self.friction = batch.expand("geom_friction")  # per env, redrawn at reset
    self.feet = [batch.model.geom(f).id for f in FEET]
    limits = batch.model.jnt_range[1 : 1 + ACT_DIM]
    self.lower, self.upper = 0.95 * limits[:, 0], 0.95 * limits[:, 1]
    self.rng = np.random.default_rng(seed)
    self.steps, self.clock = np.zeros(num_envs, np.int64), np.zeros(num_envs)
    self.action = np.zeros((num_envs, ACT_DIM), np.float32)
    self.down = np.ones((num_envs, 4), bool)
    self.command, self.until = np.zeros((num_envs, 3)), np.zeros(num_envs)
    self.reset(np.arange(num_envs))

  def reset(self, ids):
    n = ids.size
    self.batch.reset(ids, keyframe=0)  # before the writes, which it would discard
    self.qpos[ids, JOINTS] = STAND + self.rng.uniform(-0.1, 0.1, (n, ACT_DIM))
    self.qvel[ids, : DOFS.stop] = self.rng.uniform(-0.2, 0.2, (n, DOFS.stop))
    self.friction[ids[:, None], self.feet, 0] = self.rng.uniform(*FRICTION, (n, 1))
    self.batch.forward(ids)  # sensordata is stale until forward runs
    # The keyframe stands 2.4 cm lower than this pose needs, which buries the feet.
    self.qpos[ids, 2] -= np.stack(self.foot_pos, 1)[ids, :, 2].min(1) - FOOT_RADIUS - 0.001
    self.batch.forward(ids)
    self.clock[ids] = self.rng.uniform(0.0, 1.0, n)
    self.resample(ids, keep=0.0)
    self.steps[ids], self.action[ids], self.down[ids] = 0, 0.0, True

  def resample(self, ids, keep=0.5):
    n = ids.size
    fresh = self.rng.uniform(-1.0, 1.0, (n, 3)) * COMMAND_RANGE
    fresh *= self.rng.random((n, 3)) < COMMAND_ON
    kept = self.rng.random((n, 3)) < keep
    self.command[ids] = np.where(kept, self.command[ids], fresh)
    self.until[ids] = self.rng.exponential(COMMAND_SECONDS / CTRL_DT, n)

  def moving(self):
    return (np.linalg.norm(self.command, axis=1) > 0.01)[:, None]

  def obs(self):
    angle, moving = 2 * np.pi * self.clock[:, None], self.moving()
    cols = (
      self.vel,
      self.gyro,
      self.up,
      self.qpos[:, JOINTS] - STAND,
      self.qvel[:, DOFS],
      self.action,
      self.command,
      np.sin(angle) * moving,  # hidden at rest, or it marches waiting for a command
      np.cos(angle) * moving,
    )
    return np.concatenate(cols, 1, dtype=np.float32)

  def step(self, action):
    self.ctrl[:] = STAND + ACTION_SCALE * action
    flying = np.stack(self.foot_vel, 1).copy()
    self.batch.step(nstep=DECIMATION)
    self.steps += 1
    self.until -= 1
    self.resample(np.flatnonzero(self.until <= 0))
    q, up, gyro, command = self.qpos[:, JOINTS], self.up, self.gyro, self.command
    force = np.concatenate(self.foot_force, 1)
    down = force > 0.0
    contact = down | self.down  # a bounce does not end the stance
    height = np.stack(self.foot_pos, 1)[:, :, 2] - FOOT_RADIUS
    slide = np.sum(np.stack(self.foot_vel, 1)[:, :, :2] ** 2, 2)
    moving = self.moving()
    phase = np.sin(2 * np.pi * (self.clock[:, None] + PHASE)) * moving
    low = np.minimum(height - SWING * phase, 0.0)
    high = np.maximum(height - CEILING, 0.0)
    swing = np.exp(-force / 10.0) * np.exp(-((low**2 + high**2) / WIDTH**2))
    blend = np.clip(0.5 + phase, 0.0, 1.0) * moving  # a ramp, not a switch
    over = np.maximum(self.lower - q, 0.0) + np.maximum(q - self.upper, 0.0)
    terms = dict(
      track=np.exp(-np.sum((self.vel[:, :2] - command[:, :2]) ** 2, 1) / SIGMA),
      turn=np.exp(-((gyro[:, 2] - command[:, 2]) ** 2) / SIGMA),
      gait=(blend * swing + (1.0 - blend) * contact * np.exp(-slide / 0.05)).mean(1),
      pose=np.exp(-np.sum((q - STAND) ** 2 * POSE_W, 1)),
      orient=up[:, 0] ** 2 + up[:, 1] ** 2,
      bounce=self.qvel[:, 2] ** 2,
      wobble=gyro[:, 0] ** 2 + gyro[:, 1] ** 2,
      limits=np.sum(over, 1),
      rate=np.sum((action - self.action) ** 2, 1),
      land=np.sum((down & ~self.down) * np.sum(flying**2, 2), 1),
    )
    reward = np.maximum(np.asarray(sum(REWARD[k] * v for k, v in terms.items()), np.float32), FLOOR)
    self.clock = (self.clock + GAIT_HZ * CTRL_DT) % 1.0
    self.action, self.down = action.astype(np.float32), down
    fell = up[:, 2] < 0.0
    return reward, fell | (self.steps >= EPISODE), fell, terms


def mlp(out_dim):
  hidden = (nn.Linear(OBS_DIM, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU())
  return nn.Sequential(*hidden, nn.Linear(HIDDEN, out_dim))


class ActorCritic(nn.Module):
  mean: torch.Tensor
  var: torch.Tensor
  count: torch.Tensor

  def __init__(self):
    super().__init__()
    self.actor, self.critic = mlp(ACT_DIM), mlp(1)
    self.log_std = nn.Parameter(torch.full((ACT_DIM,), LOG_STD))
    self.register_buffer("mean", torch.zeros(OBS_DIM))
    self.register_buffer("var", torch.ones(OBS_DIM))
    self.register_buffer("count", torch.full((), 1e-4))
    for name, x in (("mirror", MIRROR), ("legs", LEGS), ("sign", SIGN), ("leg_sign", LEG_SIGN)):
      self.register_buffer(name, torch.as_tensor(x), persistent=False)

  @torch.no_grad()
  def absorb(self, obs):  # Chan's parallel update of the running statistics
    n, delta = obs.shape[0], obs.mean(0) - self.mean
    total = self.count + n
    var = self.var * self.count + obs.var(0, correction=0) * n
    self.var.copy_((var + delta**2 * self.count * n / total) / total)
    self.mean.add_(delta * n / total)
    self.count.add_(n)

  def forward(self, obs):
    """Average the net's answer with its answer on the mirrored observation mirrored back, so
    the policy commutes with the mirror exactly and the gait cannot limp."""
    flip = (obs[:, self.mirror] * self.sign - self.mean) / (self.var.sqrt() + 1e-5)
    obs = (obs - self.mean) / (self.var.sqrt() + 1e-5)
    mean = self.actor(obs) + self.actor(flip)[:, self.legs] * self.leg_sign
    return 0.5 * mean, self.critic(obs).squeeze(-1)


@torch.no_grad()
def rollout(actor, env):
  def policy(obs):
    return tuple(x.cpu().numpy() for x in actor(torch.as_tensor(obs).to(ACTOR)))

  shapes = dict(obs=(OBS_DIM,), act=(ACT_DIM,), logp=(), val=(), rew=(), alive=())
  buf = {k: np.empty((HORIZON, NUM_ENVS, *v), np.float32) for k, v in shapes.items()}
  means, falls, episodes = [], 0, 0
  obs = env.obs()
  for t in range(HORIZON):
    mean, val = policy(obs)
    log_std = actor.log_std.cpu().numpy()
    noise = env.rng.standard_normal(mean.shape, np.float32)
    act, logp = mean + np.exp(log_std) * noise, log_density(noise, log_std)
    reward, done, fell, terms = env.step(act)
    next_obs = env.obs()
    timeout = done & ~fell
    if timeout.any():  # bootstrap
      reward[timeout] += GAMMA * policy(next_obs[timeout])[1]
    for k, v in dict(obs=obs, act=act, logp=logp, val=val, rew=reward, alive=~done).items():
      buf[k][t] = v
    means.append([terms[k].mean() for k in REWARD])
    ids = np.flatnonzero(done)
    if ids.size:
      episodes, falls = episodes + ids.size, falls + int(fell[ids].sum())
      env.reset(ids)
      next_obs = env.obs()
    obs = next_obs
  batch = {k: torch.as_tensor(v).to(DEVICE) for k, v in buf.items()}
  batch["last_val"] = torch.as_tensor(policy(obs)[1]).to(DEVICE)
  stats = dict(zip(REWARD, np.mean(means, 0), strict=True))
  stats["falls"] = falls / max(episodes, 1)
  return batch, stats


def gae(batch):
  vals = torch.cat([batch["val"], batch["last_val"][None]])
  adv, carry = torch.zeros_like(batch["rew"]), 0.0
  for t in reversed(range(HORIZON)):
    alive = batch["alive"][t]
    delta = batch["rew"][t] + GAMMA * alive * vals[t + 1] - vals[t]
    adv[t] = carry = delta + GAMMA * LAMBDA * alive * carry
  return adv, adv + batch["val"]


def update(net, opt, batch, adv, ret):
  obs, act = batch["obs"].reshape(-1, OBS_DIM), batch["act"].reshape(-1, ACT_DIM)
  logp_old, adv, ret = batch["logp"].reshape(-1), adv.reshape(-1), ret.reshape(-1)
  adv = (adv - adv.mean()) / (adv.std() + 1e-8)
  for _ in range(EPOCHS):
    for i in torch.randperm(obs.shape[0], device=DEVICE).chunk(MINIBATCHES):
      mean, val = net(obs[i])
      logp = log_density((act[i] - mean) / net.log_std.exp(), net.log_std)
      ratio = (logp - logp_old[i]).exp()
      surrogate = torch.min(ratio * adv[i], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[i])
      loss = -surrogate.mean() + 0.5 * (val - ret[i]).pow(2).mean() - ENT_COEF * net.log_std.sum()
      opt.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(net.parameters(), 1.0)
      opt.step()
  net.absorb(obs)  # after the epochs: the batch was collected under the old statistics


def config():
  import mjbatch, mujoco_menagerie
  keys = ("NUM_ENVS", "HORIZON", "ITERS", "EPISODE", "TIMESTEP", "DECIMATION", "KP", "KD", "SEED", "GAMMA", "LAMBDA", "CLIP",
          "LR", "LR_END", "LR_TO", "EPOCHS", "MINIBATCHES", "ENT_COEF", "HIDDEN", "DEVICE", "ACTOR", "THREADS")
  cfg = {k: globals()[k] for k in keys}
  cfg["REWARD"] = REWARD
  cfg["versions"] = dict(mujoco=mujoco.__version__, mjbatch=getattr(mjbatch, "__version__", "0.1.4"), torch=torch.__version__,
                         menagerie=getattr(mujoco_menagerie, "__version__", "?"), python=platform.python_version(), machine=platform.machine())
  return cfg


def train():
  torch.manual_seed(SEED)
  os.makedirs(OUT, exist_ok=True)
  with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config(), f, indent=1)
  metrics = open(os.path.join(OUT, "metrics.jsonl"), "w")
  env = Go1(NUM_ENVS, seed=SEED)
  actor, learner = ActorCritic().to(ACTOR), ActorCritic().to(DEVICE)
  actor.load_state_dict(learner.state_dict())
  opt = torch.optim.Adam(learner.parameters(), LR, fused=True)
  start = time.perf_counter()

  def save():
    torch.save(actor.state_dict(), POLICY + "~")
    os.replace(POLICY + "~", POLICY)

  try:
    for it in range(ITERS):
      opt.param_groups[0]["lr"] = float(np.interp(it, (0, LR_TO), (LR, LR_END)))
      batch, stats = rollout(actor, env)
      update(learner, opt, batch, *gae(batch))
      actor.load_state_dict(learner.state_dict())
      dt, milestone = time.perf_counter() - start, (it + 1) % 100 == 0 or it + 1 == ITERS
      reward = sum(REWARD[k] * stats[k] for k in REWARD)
      tail = f"{(it + 1) * HORIZON * NUM_ENVS / dt / 1e3:3.0f}k steps/s   reward {reward:5.2f}  "
      tail += "  ".join(f"{k} {stats[k]:.2f}" for k in ("track", "turn", "falls"))
      progress(f"{it + 1:4d}/{ITERS}", (it + 1) / ITERS, dt, tail, end=milestone)
      row = dict(it=it + 1, elapsed=round(dt, 2), steps=(it + 1) * HORIZON * NUM_ENVS, steps_per_sec=int((it + 1) * HORIZON * NUM_ENVS / dt),
                 reward=round(float(reward), 4), falls=round(float(stats["falls"]), 4), lr=opt.param_groups[0]["lr"])
      row.update({f"term_{k}": round(float(REWARD[k] * stats[k]), 5) for k in REWARD})
      row.update({f"raw_{k}": round(float(stats[k]), 5) for k in REWARD})
      metrics.write(json.dumps(row) + "\n"); metrics.flush()
      if milestone:
        print(" " * 9 + "  ".join(f"{k} {REWARD[k] * stats[k]:.2f}" for k in REWARD))
      if (it + 1) % 25 == 0 or milestone:  # often, so --play can follow along
        save()
  except KeyboardInterrupt:
    print()
  save()
  metrics.close()
  total = time.perf_counter() - start
  with open(os.path.join(OUT, "done.json"), "w") as f:
    json.dump(dict(iters=ITERS, envs=NUM_ENVS, seconds=round(total, 1), policy=POLICY), f)
  print(f"saved {POLICY} after {clock(total)}")


SEGMENTS = (  # (seconds, forward m/s, sideways m/s, yaw rad/s): the scripted joystick for --render
  (1.0, 0.0, 0.0, 0.0), (3.0, 1.0, 0.0, 0.0), (2.5, 1.0, 0.0, 1.0), (2.0, 0.0, 0.6, 0.0),
  (2.5, 1.5, 0.0, 0.0), (2.0, -0.6, 0.0, 0.0), (1.5, 0.0, 0.0, 0.0),
)


def render(path, fps=50, width=640, height=480):
  """Play the saved policy on one env through SEGMENTS, write an mp4 and eval.json."""
  import imageio.v3 as iio
  env, net = Go1(1, seed=SEED + 1), ActorCritic()
  net.load_state_dict(torch.load(POLICY, map_location="cpu"))
  model, data = env.batch.model, mujoco.MjData(env.batch.model)
  trunk = model.body("trunk").id
  renderer = mujoco.Renderer(model, height, width)
  cam = mujoco.MjvCamera()
  cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, trunk, 2.1, 135.0, -12.0
  frames, report, falls, every = [], [], 0, max(1, round(1.0 / (CTRL_DT * fps)))
  one = np.zeros(1, np.int64)
  with torch.no_grad():
    for seconds, vx, vy, wz in SEGMENTS:
      errs, steps = [], int(seconds / CTRL_DT)
      for k in range(steps):
        env.command[:], env.until[:], env.steps[:] = (vx, vy, wz), 2, 1
        act = net(torch.as_tensor(env.obs()))[0].numpy()
        _, _, fell, _ = env.step(act)
        if fell[0]:
          falls += 1
          env.reset(one)
        errs.append([abs(env.vel[0, 0] - vx), abs(env.vel[0, 1] - vy), abs(env.gyro[0, 2] - wz), env.trunk[0, 2]])
        if k % every == 0:
          data.qpos[:] = env.qpos[0]
          mujoco.mj_forward(model, data)
          renderer.update_scene(data, cam)
          frames.append(renderer.render().copy())
      e = np.mean(errs, 0)
      report.append(dict(seconds=seconds, cmd=[vx, vy, wz], err_vx=round(float(e[0]), 3), err_vy=round(float(e[1]), 3),
                         err_wz=round(float(e[2]), 3), height=round(float(e[3]), 3)))
  renderer.close()
  iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", plugin="pyav")
  summary = dict(video=path, frames=len(frames), fps=fps, falls=falls, segments=report,
                 mean_err_vx=round(float(np.mean([r["err_vx"] for r in report])), 3))
  with open(os.path.join(OUT, "eval.json"), "w") as f:
    json.dump(summary, f, indent=1)
  print(json.dumps(summary))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", default="runs/go1", help="run directory (config.json, metrics.jsonl, policy.pt, eval.json)")
  parser.add_argument("--envs", type=int, default=NUM_ENVS)
  parser.add_argument("--iters", type=int, default=ITERS)
  parser.add_argument("--seed", type=int, default=SEED)
  parser.add_argument("--threads", type=int, default=0, help="sim threads (0: all CPUs)")
  parser.add_argument("--device", default=None, help="learner device: cpu or mps (default: upstream choice)")
  parser.add_argument("--render", metavar="MP4", help="play the saved policy through a command script and write MP4 + eval.json")
  args = parser.parse_args()
  OUT, POLICY, NUM_ENVS, ITERS, SEED, THREADS = args.out, os.path.join(args.out, "policy.pt"), args.envs, args.iters, args.seed, args.threads
  if args.device:
    DEVICE = args.device
  render(args.render) if args.render else train()
