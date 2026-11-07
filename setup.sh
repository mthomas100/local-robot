#!/bin/zsh
# setup.sh — build the Python environments for local-robot (idempotent). Versions are pinned to the ones every result in
# this repository was produced with (November 2026).
#   envs/mj       MuJoCo 3.17 + Menagerie models + PyTorch (the Go1 trainer, the whole sorting cell, the benchmarks),
#                 Gymnasium, SB3, mujoco_playground / MJX / Brax on JAX-CPU (benchmarks only), ONNX
#   envs/genesis  Genesis 1.4.5 + rsl_rl (the Go2 trainer), CoACD (YCB processing), pytest (rl factory-test)
#   envs/mjbatch  only with --with-mjbatch: the archived mjbatch package with its pinned MuJoCo 3.14
#                 (rl train go1-mjbatch, rl play)
# Two main venvs because Genesis pins its own stack and should not fight MuJoCo + JAX. Python 3.12: the widest wheel
# coverage across mujoco, jax, torch and genesis as of 2026-11. Then: scripts/fetch_vendor.sh and bin/rl cell build.
set -u
cd "$(dirname "$0")"
mkdir -p envs logs
PY=3.12
log() { echo "$(date +%H:%M:%S) $*"; }
mk() { # mk <name> <pkgs...>
  local name=$1; shift
  if [ ! -x envs/$name/bin/python ]; then log "create envs/$name"; uv venv -q -p $PY envs/$name || return 1; fi
  log "install into envs/$name"
  uv pip install -q -p envs/$name/bin/python "$@" || { log "FAILED envs/$name"; return 1; }
  log "ok envs/$name"
}
mk mj mujoco==3.17.0 mujoco-menagerie==2026.10.6 torch==2.15.0 numpy==2.5.4 scipy==1.18.1 trimesh==5.2.0 rtree==1.4.1 \
   "gymnasium[mujoco]==1.4.0" stable-baselines3==2.9.2 tensorboard==2.21.0 imageio==2.38.1 "imageio[ffmpeg]" av==19.0.2 \
   mediapy==1.2.7 matplotlib==3.11.2 pillow tqdm pyyaml playground==0.2.0 brax==0.14.2 jax==0.12.0.1 flax==0.12.12 \
   optax==0.2.8 onnx==1.23.2 onnxruntime==1.31.1 2>&1 | tee logs/setup-mj.log
mk genesis genesis-world==1.4.5 torch==2.15.0 rsl-rl-lib==5.5.2 tensordict==0.15.4 numpy==2.5.4 scipy==1.18.1 \
   mujoco==3.17.0 trimesh==5.2.0 coacd==1.0.14 rtree==1.4.1 tensorboard==2.21.0 imageio==2.38.1 "imageio[ffmpeg]" \
   av==19.0.2 matplotlib==3.11.2 pytest==9.2.0 onnx==1.23.2 onnxruntime==1.23.2 2>&1 | tee logs/setup-genesis.log
if [ "${1:-}" = --with-mjbatch ]; then
  mk mjbatch mjbatch==0.1.4 torch==2.15.0 numpy==2.5.4 "imageio[ffmpeg]" av mujoco-menagerie==2026.10.6 glfw==2.11.0 2>&1 | tee logs/setup-mjbatch.log
fi
log "versions:"
for e in envs/*(/); do echo "--- $e"; uv pip list -p $e/bin/python 2>/dev/null | grep -iE "^(mujoco|mujoco-menagerie|mjbatch|gymnasium|stable-baselines3|torch|jax|brax|playground|genesis-world|rsl-rl-lib|numpy) "; done 2>&1 | tee logs/setup-versions.log
log "setup finished"
