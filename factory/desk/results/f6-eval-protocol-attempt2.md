# F6 acceptance evaluation, attempt 2: protocol fixed before the run (2026-11-03)

> This repository's public history is a fresh export. In the original development history this protocol was
> committed before the run it governs; commit references to that history have been removed.

Committed before the run. Attempt 1 (`f6-eval-protocol-attempt1.md`, results `f6-eval-attempt1.json`) passed the
picks (classical 200/200, ONNX 186/200) and failed the calibration (2.21 and 2.04 mm on 2 of 5 planted errors, bar
2 mm). By its own rule a re-run needs a new protocol with fresh seeds. Since attempt 1 (development, on spent seeds
only):

- **the plant changed**: the horn play's end stops are stiff (`so101.PLAY_SOLREF`); MuJoCo's default
  soft limit had let the +/- 0.5 deg play open to -2.6 / +1.5 deg under load. So every part is re-run, picks too;
- **the calibration changed**: poses raised off the belt rail (96 poses at tool heights 8-17 cm), a robust refit
  and a bundle adjustment with one pan-play angle per pose, and the forward kinematics now reads each
  pitch joint's gravity torque where the links really hang. That last fix removed a 1.5 mm
  systematic error: the wrist's play sat 0.47 deg off where the old model assumed it. Development re-runs on the
  spent seeds 4242-4246: 0.32, 0.32, 0.92, 0.17, 0.42 mm, rotation <= 0.2 deg.

Everything else is attempt 1's protocol, unchanged:

- **Question** (docs/sorting-cell-spec.md §4, row F6): with the real-device kit on, does the classical pick succeed on at
  least 90% and the ONNX policy on at least 80% of 200 seeded L0 trials of the admitted items; does system
  identification recover every planted parameter within 10%; does the calibration routine recover the planted
  hand-eye error within 2 mm and 1 degree; does the loop hold its rate?
- **Admitted items** (`admitted.json`): tomato soup can, potted meat can (YCB, metal), the PET-bottle stand-in
  (plastic), the tea-box stand-in (cardboard). Trial `seed` uses item `admitted[seed % 4]`: 50 trials per item.
- **The real-device kit, on in every part**: servos `DeviceKit.real_kit()` (as in attempt 1, with the stiff play
  stops), plant `REAL_PLANT`, camera `CameraKit.real_kit(seed)` (depth z^2 noise, 2% holes plus edge holes, RGB
  noise, 45 +/- 10 ms latency, a planted hand-eye error of 8 mm and 2.5 deg in seeded random directions). The pick
  sees only the camera, after the cell calibrated it.
- **Calibration**: `eval_f6.py calib`, five planted errors, **camera seeds 4300-4304 (fresh)**. Pass: every one
  recovered within 2 mm and 1 deg (`hand_eye_error()` of the calibrated belief against the true camera pose).
- **Picks**: `eval_f6.py picks`, **seeds 21000-21199 (fresh), camera seed 4300**, each seed run by the classical
  pipeline (`desk_pick.trial`) and the ONNX policy (`bc.policy_trial`, 10.5 s at 30 Hz) on the same scene. Success
  = the item ends inside the bin of its material. Pass: classical >= 180/200 and ONNX >= 160/200; Wilson 95%
  intervals reported, not gated. Sample size: attempt 1 measured 100% and 93%; at those rates 200 trials put the
  Wilson lower bound (98.1%, 88.6%) above both bars, so 200 is enough to tell a pass from a fail.
- **Which ONNX model**: `factory/desk/models/pick.onnx`, the same file as attempt 1 (trained on the soft-stop
  kit; not retrained), **sha256 7610b151222e74e1c5d4b0d1110b12a78e1de62dcca56679fe6272716f64fabc**. Checked on the
  stiff-stop kit on the check seeds 90000-90039 (camera seed 77) before this protocol: 40/40, calibration 0.71 mm.
- **Rate**: `eval_f6.py rate`, 20 trials, **seeds 21200-21219 (fresh)**, the loop paced against the wall clock,
  nothing else heavy running (no other evaluation, no live view). Pass: mean tick period within 1% of 33.33 ms and no
  tick later than 1.5 periods (50 ms); inference time (p99) reported.
- **System identification**: `sysid.py --seeds 11,12,13` (**fresh planting seeds**; attempt 1's campaign used 1-3
  and is kept as `f6-sysid-attempt1.json`), re-run because its runs may predate the stiff play stops. Its rule as
  in its docstring: every planted parameter recovered within 10%.
- **Commands**: `bin/rl desk eval calib`, `bin/rl desk eval picks --workers 10`, `bin/rl desk eval rate`,
  `bin/rl desk sysid --seeds 11,12,13` -> `factory/desk/results/f6-eval.json`, `f6-sysid.json`.
- **If a part fails**: it is reported as failed with its numbers; a re-run needs a new protocol with fresh seeds.
