#!/bin/zsh
# fetch_vendor.sh — clone the upstream projects the code reads by path into vendor/ (gitignored), at the commits the
# results in this repo were produced with. Nothing in vendor/ is ever edited: patched copies live in lib/ and say so.
#   mujoco_menagerie  robot models: UR5e, Robotiq 2F-85, RealSense D435i, SO-101 (factory/cell, factory/desk), Go2 scene (scripts/bench_sim.py)
#   mjbatch           examples/assets/theme.xml (the Go1 trainer's scene) and examples/go1_joystick.py (rl play, rl train go1-mjbatch)
#   mujoco_mpc        optional: the MJPC predictive-control demo, built separately (see README)
# The Go2 trainer needs no clone: lib/genesis_go2/ is a patched copy of Genesis's examples/locomotion, and Genesis
# itself is installed from PyPI by setup.sh. Usage: scripts/fetch_vendor.sh [--with-mpc]
set -eu -o pipefail
R=${0:A:h:h}; mkdir -p $R/vendor; cd $R/vendor
get() { # get <dir> <url> <commit>
  if [ ! -d $1/.git ]; then git clone -q $2 $1; fi
  git -C $1 fetch -q origin $3 2>/dev/null || true
  git -C $1 checkout -q $3 && echo "vendor/$1 at $(git -C $1 rev-parse --short HEAD)"
}
get mujoco_menagerie https://github.com/google-deepmind/mujoco_menagerie.git b0655057a9df2abe39ea7fc989914ba6f248301c
get mjbatch https://github.com/kevinzakka/mjbatch.git 4cba7799b0e5265cdb96f6d520a491ace4dad9ae
if [ "${1:-}" = --with-mpc ]; then get mujoco_mpc https://github.com/google-deepmind/mujoco_mpc.git ff572a21e7c2bf9fda62e1862a758da7e9a8719b; fi
