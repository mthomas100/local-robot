# Sorting-cell assets: sources, licences, processing (2026-11-02)

## Robot, gripper, camera (vendor/mujoco_menagerie, pristine clone, used by path)

| model | Menagerie directory | licence (from the model's README and LICENSE) |
|---|---|---|
| Universal Robots UR5e | `universal_robots_ur5e` | BSD-3-Clause (ROS-Industrial) |
| Robotiq 2F-85 | `robotiq_2f85` | BSD-2-Clause (ROS-Industrial) |
| Intel RealSense D435i | `realsense_d435i` | Apache-2.0 |

`factory/cell/build_mjcf.py` attaches the 2F-85 to the UR5e's `attachment_site` with MjSpec and rewrites mesh paths to
absolute ones; nothing in `vendor/` is edited. Each model's licence is listed in the top-level README.

## Objects: the YCB Object and Model Set

- **Source**: the per-object "google_16k" archives (16k-face textured scans, metres, z-up) from
  `https://ycb-benchmarks.s3.amazonaws.com/data/google/<id>_google_16k.tgz`, downloaded 2026-11-02 by
  `scripts/fetch_ycb.sh` into `factory/cell/assets/ycb/_dl/` (gitignored; about 270 MB unpacked).
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0), stated in the License field of the AWS Open
  Data Registry entry "Yale-CMU-Berkeley (YCB) Object and Model Set" (managed by Yale University and Berkeley), which
  fronts the S3 bucket the archives come from: https://registry.opendata.aws/ycb-benchmarks (checked 2026-11-02 in a
  research pass). The archives and the bucket carry no licence file; ycbbenchmarks.com returned HTTP 500 that day, so
  its own wording is unchecked. Attribution request: "Yale-CMU-Berkeley (YCB) Object and Model Set was accessed on DATE
  from https://registry.opendata.aws/ycb-benchmarks." Also cite the papers below in anything published.
- **Items** (the set in `docs/sorting-cell-spec.md` §1; the material class decides the chute): plastic `006_mustard_bottle`,
  `021_bleach_cleanser`, `022_windex_bottle`; metal `002_master_chef_can`, `005_tomato_soup_can`, `007_tuna_fish_can`,
  `010_potted_meat_can`; cardboard `003_cracker_box`, `004_sugar_box`, `008_pudding_box`, `009_gelatin_box`.
- **Masses** (`ITEMS` in `factory/cell/ycb.py`, grams): 414, 411, 514, 349, 603, 171, 187, 97, 370, 1131, 1022 for
  002, 003, 004, 005, 006, 007, 008, 009, 010, 021, 022: identical to RAM 2015 Table 2 (p. 42) and ICAR 2015 Table I,
  which agree (checked 2026-11-02). The arXiv preprint 1502.03143 has older ids and different masses: do not use it.
- **Citations**: B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, A. M. Dollar, "The YCB Object and Model Set:
  Towards Common Benchmarks for Manipulation Research," ICAR 2015, pp. 510-517, doi:10.1109/ICAR.2015.7251504.
  B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel, A. M. Dollar, "Benchmarking in Manipulation Research: Using
  the Yale-CMU-Berkeley Object and Model Set," IEEE Robotics & Automation Magazine 22(3):36-52, 2015,
  doi:10.1109/MRA.2015.2448951.

## Processing (`factory/cell/ycb.py`, idempotent; every number lands in `assets/ycb/index.json`, which is in git)

1. **Convex decomposition**: CoACD (threshold 0.04, at most 12 pieces, seed 0) on the scan as downloaded; cached in
   `_dl/<id>/coacd/`. 2-12 pieces per item.
2. **Canonical frame**: the scans' origins are off the objects (the soup can's by 8 cm) and the Windex scan is tilted
   8.2°. Each item is rotated onto the most upright stable resting pose of its convex hull (trimesh
   `compute_stable_poses` with the real centre of mass), by the smallest rotation (no spin), and translated so the
   origin is the bottom centre of its bounding box. The 4x4 transform is `scan_to_item` in the index.
3. **Base pad**: a 3 mm convex pad shaped like the 2D outline of the bottom 1 cm (`hull_base.obj`). Turntable scans
   capture an object's underside worst; real containers stand on their rim. The Windex uses its bottom 4.5 cm: its scan
   is broken (transparent bottle: the trigger head floats above the neck, the bottom is cut off diagonally;
   `factory/cell/f0_media.py` renders it as `docs/media/F0-windex-scan.png`, not shipped), and its lower body is prismatic.
4. **Mass and inertia**: uniform density over the pieces, scaled to the real mass; full inertia about the centre of mass.
   Friction (sliding): plastic 0.6, metal 0.5, cardboard 0.7 (plausible, untuned).

## Open
- Confirm the licence wording on ycbbenchmarks.com itself when the site is reachable again.
