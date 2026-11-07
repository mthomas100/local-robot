# SPDX-License-Identifier: Apache-2.0
# local-robot/lib/genesis_go2/record.py — play a trained Go2 policy headless on the CPU backend and record an mp4
# plus eval.json (2026-11-01). Based on Genesis examples/locomotion/go2_eval.py (Apache-2.0), which opens a viewer instead.
import argparse, glob, json, os, pickle, re, sys, time
import torch
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_env import Go2Env  # noqa: E402

SEGMENTS = ((4.0, 0.5, 0.0, 0.0), (3.0, 0.3, 0.0, 0.0), (3.0, 0.8, 0.0, 0.0), (3.0, 0.5, 0.0, 0.0))  # seconds, vx, vy, wz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory written by train.py")
    p.add_argument("--ckpt", type=int, default=None, help="model_<ckpt>.pt (default: highest)")
    p.add_argument("--mp4", default=None, help="output video (default <run>/eval.mp4)")
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()
    run = args.run; mp4 = args.mp4 or os.path.join(run, "eval.mp4")
    if args.ckpt is None:
        cks = [int(re.search(r"model_(\d+).pt", f).group(1)) for f in glob.glob(os.path.join(run, "model_*.pt"))]
        args.ckpt = max(cks)
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    with open(os.path.join(run, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, camera=True)
    runner = OnPolicyRunner(env, train_cfg, run, device=gs.device)
    runner.load(os.path.join(run, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    cam = env.cam
    every = max(1, round(1.0 / (env.dt * args.fps)))
    cam.start_recording(save_to_filename=mp4, fps=args.fps)
    obs = env.reset(); falls, report, frames, t0 = 0, [], 0, time.perf_counter()
    with torch.no_grad():
        for seconds, vx, vy, wz in SEGMENTS:
            errs = []
            for k in range(int(seconds / env.dt)):
                env.commands[:] = torch.tensor([vx, vy, wz], dtype=gs.tc_float, device=gs.device)
                env._update_observation(); obs = env.get_observations()
                obs, rews, dones, infos = env.step(policy(obs))
                falls += int(dones.sum().item()) if env.episode_length_buf[0] < env.max_episode_length else 0
                v = env.base_lin_vel[0]; w = env.base_ang_vel[0]; z = env.base_pos[0, 2]
                errs.append([abs(v[0].item() - vx), abs(v[1].item() - vy), abs(w[2].item() - wz), z.item()])
                if k % every == 0:
                    b = env.base_pos[0].tolist()
                    cam.set_pose(pos=(b[0] + 1.6, b[1] - 1.8, b[2] + 0.9), lookat=(b[0], b[1], b[2]))
                    cam.render(); frames += 1
            e = torch.tensor(errs).mean(0).tolist()
            report.append(dict(seconds=seconds, cmd=[vx, vy, wz], err_vx=round(e[0], 3), err_vy=round(e[1], 3), err_wz=round(e[2], 3), height=round(e[3], 3)))
    cam.stop_recording()
    summary = dict(video=mp4, ckpt=args.ckpt, frames=frames, fps=args.fps, falls=falls, segments=report,
                   mean_err_vx=round(sum(r["err_vx"] for r in report) / len(report), 3), render_seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(run, "eval.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
