# F6 acceptance evaluation, attempt 3 (system identification only): protocol fixed before the run (2026-11-03)

> This repository's public history is a fresh export. In the original development history this protocol was
> committed before the run it governs; commit references to that history have been removed.

Committed before the run. Attempt 2 (`f6-eval-protocol-attempt2.md`; results in `f6-eval.json`)
passed calibration (worst 1.46 mm, 0.27 deg), picks (classical 200/200, ONNX 189/200) and rate (0 late of 6,300 ticks),
and failed system identification on 2 of its 3 fresh seeds (seed 12 bus delay +45.9%, seed 13 gripper damping
+13.8%). Each part has its own rule, and by attempt 2's own rule the failed part is re-run under a new protocol with
fresh seeds. The plant, the servo model, the camera, the calibration and the policies are unchanged since attempt 2
(the commits since touch only `sysid.py`, its tests, docs, and the F7 line), so the passed parts stand.

- **What changed in system identification** (the diagnosis in sysid.py's docstring and
  the F6 write-up in the README): the record and fit models agree at the truth (0.30 counts rms); the failure was a fast
  planted motor (no-load speed 6.2-6.4 rad/s) whose swings outlived the 0.4-0.6 s segments, which made the cost a
  needle. Now: longer segments (settle 1.5 s, step 1.3 s, pre-coast 1.3 s, hold 1.0 s), only informative reads
  fitted (creeps, each step until it first reaches its goal, each coast, the chirps), and each experiment kind
  weighted equally (the creeps' many correlated reads had outvoted the coasts that measure damping).
- **Development only on spent seeds** (< 20): 2, 12, 13 (the hard ones: worst 7.8%, 2.9%, 3.9%), 3, 7, 9, 16
  (2.4%, 2.5%, 3.7%, 4.8%; 9 and 16 also plant fast motors), all passing (runs/f6-sysid-dev/v3-balance*.log), and the
  attempt-1 campaign's 1-3. The live test (`test_live_reduced_identification`, seed 7, one joint) passes in 43 s.
- **The run**: `sysid.py --seeds 21,22,23 --workers 10` (fresh planting seeds; 21-30 were reserved for this
  protocol and never run), writing `factory/desk/results/f6-sysid.json` and `docs/media/F6-sysid.png`.
  The planting distribution is unchanged (every armature, damping and frictionloss of all six joints, the bus delay
  and the no-load speed, each 20-50% off nominal in a seeded random direction).
- **Pass**: every planted parameter of every seed recovered within 10% (`all_within_tolerance`). Nothing else is
  gated; fit times and residuals are reported.
- **If it fails**: reported as failed with its numbers; a re-run needs a new protocol with fresh seeds (24-30).
