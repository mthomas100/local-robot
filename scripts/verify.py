#!/usr/bin/env python
"""verify.py <env> [--gpu] — smoke-test one venv (2026-11-01). CPU-only unless --gpu is given.
Run as: envs/<env>/bin/python scripts/verify.py <env>. Prints one JSON line per check: {"check":..,"ok":..,..}.
--gpu runs the Metal/MPS checks; run it only while nothing else uses the GPU.
"""
import json, os, sys, time, traceback

GPU = "--gpu" in sys.argv
ENV = next((a for a in sys.argv[1:] if not a.startswith("--")), "mj")
os.environ.setdefault("MUJOCO_GL", "glfw")  # macOS: egl/osmesa are unavailable; glfw hidden window works on the main thread

def check(name, fn):
    t = time.time()
    try:
        info = fn() or {}
        print(json.dumps({"check": name, "ok": True, "secs": round(time.time() - t, 2), **info}), flush=True)
    except Exception as e:  # noqa
        print(json.dumps({"check": name, "ok": False, "secs": round(time.time() - t, 2), "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()[-800:]}), flush=True)

def mj_checks():
    def mujoco_step():
        import mujoco, numpy as np
        xml = "<mujoco><worldbody><light pos='0 0 3'/><geom type='plane' size='5 5 .1'/><body pos='0 0 1'><freejoint/><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        t = time.time(); n = 20000
        for _ in range(n): mujoco.mj_step(m, d)
        return {"mujoco": mujoco.__version__, "steps_per_sec": int(n / (time.time() - t)), "z": round(float(d.qpos[2]), 3)}
    def mujoco_rollout():
        import mujoco, numpy as np
        from mujoco import rollout
        xml = "<mujoco><worldbody><geom type='plane' size='5 5 .1'/><body pos='0 0 1'><freejoint/><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        nb, ns = 64, 500
        init = np.tile(np.concatenate([[0.0], d.qpos, d.qvel, d.act]), (nb, 1))  # time, qpos, qvel, act
        for th in (1, 4):  # mujoco 3.17 API: a Rollout object with nthread and one MjData per thread
            datas = [mujoco.MjData(m) for _ in range(th)]
            with rollout.Rollout(nthread=th) as r:
                t = time.time(); r.rollout(m, datas, init, nstep=ns); el = time.time() - t
            print(json.dumps({"check": f"rollout_nthread{th}", "ok": True, "steps_per_sec": int(nb * ns / el)}), flush=True)
        return {}
    def mujoco_render():
        import mujoco, numpy as np
        xml = "<mujoco><worldbody><light pos='0 0 3'/><geom type='plane' size='5 5 .1'/><body pos='0 0 .5'><geom type='sphere' size='.2' rgba='1 0 0 1'/></body></worldbody></mujoco>"
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 240, 320); r.update_scene(d); img = r.render(); r.close()
        return {"shape": list(img.shape), "nonblack": bool(img.max() > 0)}
    def gym_humanoid():
        import gymnasium as gym
        env = gym.make("Humanoid-v5"); obs, _ = env.reset(seed=0); t = time.time(); n = 2000
        for _ in range(n):
            obs, rew, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc: env.reset()
        env.close()
        return {"gymnasium": gym.__version__, "obs_dim": int(obs.shape[0]), "env_steps_per_sec": int(n / (time.time() - t))}
    def gym_render():
        import gymnasium as gym
        env = gym.make("Humanoid-v5", render_mode="rgb_array"); env.reset(seed=0); f = env.render(); env.close()
        return {"frame": list(f.shape)}
    def torch_check():
        import torch
        return {"torch": torch.__version__, "mps_available": bool(torch.backends.mps.is_available()), "threads": torch.get_num_threads()}
    def sb3_check():
        import stable_baselines3 as sb3, gymnasium as gym
        from stable_baselines3 import PPO
        env = gym.make("InvertedPendulum-v5"); model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, verbose=0, device="cpu")
        t = time.time(); model.learn(2048); return {"sb3": sb3.__version__, "ppo_2048_steps_secs": round(time.time() - t, 1)}
    def jax_check():
        import jax, jax.numpy as jnp
        x = jnp.ones((1024, 1024)); y = (x @ x).block_until_ready()
        return {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()], "sum": float(y[0, 0])}
    def brax_check():
        import brax; return {"brax": brax.__version__}
    def playground_check():
        import mujoco_playground as mp
        from mujoco_playground import registry
        names = list(registry.locomotion.ALL_ENVS)[:40]
        env = registry.load("Go1JoystickFlatTerrain")
        import jax
        rng = jax.random.PRNGKey(0); t = time.time(); reset = jax.jit(env.reset); step = jax.jit(env.step)
        s = reset(rng); comp = time.time() - t; t = time.time()
        for _ in range(20): s = step(s, jax.numpy.zeros(env.action_size))
        s.obs["state"].block_until_ready() if isinstance(s.obs, dict) else s.obs.block_until_ready()
        return {"playground": getattr(mp, "__version__", "?"), "n_locomotion_envs": len(names), "jit_secs": round(comp, 1), "steps_per_sec_1env": int(20 / (time.time() - t))}
    for n, f in [("mujoco_step", mujoco_step), ("mujoco_rollout", mujoco_rollout), ("mujoco_render", mujoco_render), ("gym_humanoid", gym_humanoid),
                 ("gym_render", gym_render), ("torch", torch_check), ("sb3_ppo", sb3_check), ("jax_cpu", jax_check), ("brax", brax_check), ("playground_go1", playground_check)]:
        check(n, f)

def genesis_checks():
    def gen(backend_name):
        def f():
            import genesis as gs
            backend = getattr(gs, backend_name)
            gs.init(backend=backend, logging_level="warning", precision="32")
            scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
            scene.add_entity(gs.morphs.Plane())
            robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
            n_envs = 256 if backend_name == "metal" else 8
            scene.build(n_envs=n_envs)
            for _ in range(10): scene.step()
            t = time.time(); n = 200
            for _ in range(n): scene.step()
            el = time.time() - t
            out = {"genesis": gs.__version__, "backend": backend_name, "n_envs": n_envs, "env_steps_per_sec": int(n * n_envs / el)}
            gs.destroy()
            return out
        return f
    check("genesis_cpu", gen("cpu"))
    if GPU: check("genesis_metal", gen("metal"))
    def rsl(): import rsl_rl, importlib.metadata as md; return {"rsl_rl": md.version("rsl-rl-lib")}
    check("rsl_rl", rsl)
    def torch_check():
        import torch; return {"torch": torch.__version__, "mps_available": bool(torch.backends.mps.is_available())}
    check("torch", torch_check)

def jaxmetal_checks():
    def f():
        import jax, jax.numpy as jnp
        devs = [str(d) for d in jax.devices()]
        x = jnp.ones((2048, 2048)); y = (x @ x).block_until_ready()
        return {"jax": jax.__version__, "devices": devs, "val": float(y[0, 0])}
    if GPU: check("jax_metal", f)
    else: print(json.dumps({"check": "jax_metal", "ok": None, "note": "skipped: needs --gpu"}))

{"mj": mj_checks, "genesis": genesis_checks, "jaxmetal": jaxmetal_checks}[ENV]()
