#!/bin/zsh
# fetch_ycb.sh — download the 11 YCB objects of the sorting cell (docs/sorting-cell-spec.md §2) as the "google_16k"
# archives (16k-face textured meshes, metres, z-up) into factory/cell/assets/ycb/_dl/ and unpack them (2026-11-02).
# Source and licence: factory/cell/ASSETS.md. Idempotent: an object already unpacked is skipped.
set -eu -o pipefail
R=${0:A:h:h}; D=$R/factory/cell/assets/ycb/_dl; mkdir -p $D; cd $D
ITEMS=(002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can 006_mustard_bottle 007_tuna_fish_can
       008_pudding_box 009_gelatin_box 010_potted_meat_can 021_bleach_cleanser 022_windex_bottle)
for o in $ITEMS; do
  if [ -f $o/google_16k/textured.obj ]; then echo "have $o"; continue; fi
  u="https://ycb-benchmarks.s3.amazonaws.com/data/google/${o}_google_16k.tgz"
  curl -s -f -L --max-time 300 -o ${o}_google_16k.tgz "$u" && tar -xzf ${o}_google_16k.tgz && echo "fetched $o" || { echo "FAILED $o ($u)"; exit 1; }
done
ls -d */ | wc -l | xargs echo "objects unpacked:"
