# SPDX-License-Identifier: Apache-2.0
# local-robot/lib/genesis_go2/train.py — copy of Genesis examples/locomotion/go2_train.py (Apache-2.0) patched 2026-11-01:
# --out run directory, --backend metal|cpu|gpu, done.json with the wall-clock, metrics.jsonl from rsl_rl's TensorBoard log.
import argparse
import json
import sys
import time
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import torch
import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_env import Go2Env  # noqa: E402


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=101, help="Number of learning iterations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--out", default=None, help="run directory (default logs/<exp-name>)")
    parser.add_argument("--backend", default="gpu", choices=["gpu", "metal", "cpu"], help="Genesis backend (gpu resolves to metal here)")
    parser.add_argument("--force", action="store_true", help="replace an existing --out directory that does not look like a run")
    args = parser.parse_args()

    log_dir = args.out or f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    # local-robot: never delete blindly. bin/rl creates the run directory and tees train.log into it before this starts,
    # so train.log is kept; anything else is replaced only if it looks like an earlier run of this trainer (cfgs.pkl or
    # rsl_rl checkpoints), or with --force. (Upstream rmtree'd the whole directory, which also unlinked the live log.)
    old = set(os.listdir(log_dir)) - {"train.log"} if os.path.isdir(log_dir) else set()
    if old:
        looks_like_run = "cfgs.pkl" in old or any(n.startswith("model_") and n.endswith(".pt") for n in old)
        if not (looks_like_run or args.force):
            sys.exit(f"refusing to replace {log_dir}: it does not look like a run directory (pass --force)")
        for n in old:
            q = os.path.join(log_dir, n)
            shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Training throughput comes from the 4096 parallel environments, which need a GPU.
    backend = {"gpu": gs.gpu, "metal": gs.metal, "cpu": gs.cpu}[args.backend]
    gs.init(backend=backend, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(dict(num_envs=args.num_envs, max_iterations=args.max_iterations, seed=args.seed, backend=args.backend,
                       genesis=gs.__version__, torch=torch.__version__, reward_scales=reward_cfg["reward_scales"], command_cfg=command_cfg), f, indent=1)
    t0 = time.perf_counter()

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    seconds = time.perf_counter() - t0
    with open(f"{log_dir}/done.json", "w") as f:
        json.dump(dict(iters=args.max_iterations, envs=args.num_envs, backend=args.backend, seconds=round(seconds, 1), device=str(gs.device)), f)
    print(f"trained {args.max_iterations} iterations x {args.num_envs} envs on {args.backend} in {seconds:.0f} s")


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
