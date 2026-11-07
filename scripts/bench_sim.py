#!/usr/bin/env python
"""bench_sim.py — measure simulation throughput on this Mac, one backend per invocation (2026-11-01).
Each line of output is JSON: {"bench":..., "backend":..., "n_envs":..., "env_steps_per_sec":..., ...}.
Run with the matching venv:
  envs/mj/bin/python scripts/bench_sim.py mj          # native MuJoCo C, multithreaded rollout, Go2 from Menagerie
  envs/mj/bin/python scripts/bench_sim.py gym         # Gymnasium Humanoid-v5 vector envs (sync + async)
  envs/mj/bin/python scripts/bench_sim.py mjx         # MJX (JAX, CPU) batched Go1 joystick env from mujoco_playground
  envs/mj/bin/python scripts/bench_sim.py torch       # PPO-style MLP update micro-benchmark, CPU vs MPS
  envs/genesis/bin/python scripts/bench_sim.py genesis cpu|metal   # Genesis batched Go2
The GPU variants (genesis metal, torch mps) measure best while nothing else uses the GPU.
"""
import json, os, sys, time, pathlib
os.environ.setdefault("MUJOCO_GL", "glfw")
HERE = pathlib.Path(__file__).resolve().parent.parent
MENAGERIE = HERE / "vendor" / "mujoco_menagerie"
which = sys.argv[1] if len(sys.argv) > 1 else "mj"
arg2 = sys.argv[2] if len(sys.argv) > 2 else None
out = lambda **kw: print(json.dumps(kw), flush=True)

def bench_mj():
    import mujoco, numpy as np
    from mujoco import rollout
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE / "unitree_go2" / "scene.xml"))
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else None; mujoco.mj_forward(m, d)
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    s0 = np.empty(nstate); mujoco.mj_getState(m, d, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nstep = 1000
    for th in (1, 2, 4, 8, 12, 16, 18):
        nb = th * 16
        init = np.tile(s0, (nb, 1)); ctrl = np.random.uniform(-0.3, 0.3, (nb, nstep, m.nu)).astype(np.float64)
        datas = [mujoco.MjData(m) for _ in range(th)]
        with rollout.Rollout(nthread=th) as r:
            r.rollout(m, datas, init[:th], ctrl[:th, :10])  # warm up
            t = time.time(); r.rollout(m, datas, init, ctrl); el = time.time() - t
        out(bench="mujoco_rollout_go2", backend="cpu", nthread=th, n_envs=nb, nstep=nstep, env_steps_per_sec=int(nb * nstep / el), secs=round(el, 2), dt=m.opt.timestep)

def bench_gym():
    import gymnasium as gym, numpy as np
    for mode in ("sync", "async"):
        for n in (8, 16):
            ctor = gym.vector.SyncVectorEnv if mode == "sync" else gym.vector.AsyncVectorEnv
            envs = ctor([lambda: gym.make("Humanoid-v5") for _ in range(n)], **({"context": "spawn"} if mode == "async" else {}))
            envs.reset(seed=0); steps = 2000 // n * n
            t = time.time()
            for _ in range(steps // n): envs.step(envs.action_space.sample())
            el = time.time() - t; envs.close()
            out(bench="gym_humanoid_vector", backend="cpu", mode=mode, n_envs=n, env_steps_per_sec=int(steps / el), secs=round(el, 2))

def bench_mjx():
    import jax, jax.numpy as jnp
    from mujoco_playground import registry
    env = registry.load("Go1JoystickFlatTerrain")
    reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
    for n in (64, 256, 1024):
        rng = jax.random.split(jax.random.PRNGKey(0), n)
        t = time.time(); s = reset(rng); act = jnp.zeros((n, env.action_size)); s = step(s, act); jax.block_until_ready(s.obs); comp = time.time() - t
        t = time.time(); k = 20
        for _ in range(k): s = step(s, act)
        jax.block_until_ready(s.obs); el = time.time() - t
        out(bench="mjx_go1_joystick", backend=str(jax.devices()[0]), n_envs=n, env_steps_per_sec=int(n * k / el), compile_secs=round(comp, 1), secs=round(el, 2), note="env step = ctrl step (physics substeps inside)")

def bench_torch():
    import torch, torch.nn as nn
    for dev in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
        if dev == "mps" and arg2 != "mps": out(bench="torch_ppo_update", backend="mps", skipped="pass 'mps' as 2nd arg when the GPU is free"); continue
        net = nn.Sequential(nn.Linear(48, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3); x = torch.randn(4096, 48, device=dev); y = torch.randn(4096, 12, device=dev)
        for _ in range(5): loss = ((net(x) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if dev == "mps": torch.mps.synchronize()
        t = time.time(); k = 100
        for _ in range(k): loss = ((net(x) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if dev == "mps": torch.mps.synchronize()
        el = time.time() - t
        out(bench="torch_ppo_update", backend=dev, batch=4096, updates_per_sec=round(k / el, 1), threads=torch.get_num_threads())

def bench_genesis():
    import genesis as gs
    backend = {"cpu": gs.cpu, "metal": gs.metal}[arg2 or "cpu"]
    gs.init(backend=backend, logging_level="warning", precision="32")
    for n in ((256, 1024, 4096) if arg2 == "metal" else (64, 256)):
        scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.02, substeps=2), rigid_options=gs.options.RigidOptions(enable_joint_limit=True))
        scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)))
        t = time.time(); scene.build(n_envs=n); build = time.time() - t
        for _ in range(5): scene.step()
        t = time.time(); k = 100
        for _ in range(k): scene.step()
        el = time.time() - t
        out(bench="genesis_go2", backend=arg2 or "cpu", n_envs=n, env_steps_per_sec=int(n * k / el), build_secs=round(build, 1), secs=round(el, 2), genesis=gs.__version__)
        gs.destroy() if hasattr(gs, "destroy") else None
        gs.init(backend=backend, logging_level="warning", precision="32")

if __name__ == "__main__":  # the main guard matters: gymnasium AsyncVectorEnv spawns workers that re-import this file on macOS
    {"mj": bench_mj, "gym": bench_gym, "mjx": bench_mjx, "torch": bench_torch, "genesis": bench_genesis}[which]()
