# F6 acceptance evaluation: protocol fixed before the run (2026-11-03)

> This repository's public history is a fresh export. In the original development history this protocol was
> committed before the run it governs; commit references to that history have been removed.

Committed before the acceptance run. Before it, only development runs on other seeds had been made (scratch trials
on seeds 0-11, 500-619, 1000-1005 (payload table), the media seeds 7 and 11, the behaviour-cloning training scenes
100000+ and its checks 90000+).

- **Question** (docs/sorting-cell-spec.md §4, row F6): with the real-device kit on, does the classical pick succeed on at
  least 90% and the ONNX policy on at least 80% of 200 seeded L0 trials of the admitted items; does system
  identification recover every planted parameter within 10%; does the calibration routine recover the planted
  hand-eye error within 2 mm and 1 degree; does the loop hold its rate?
- **Admitted items** (`factory/desk/results/admitted.json`, from the payload-and-jaw table): tomato soup can, potted
  meat can (YCB, metal), the PET-bottle stand-in (plastic), the tea-box stand-in (cardboard). Trial `seed` uses item
  `admitted[seed % 4]`: 50 trials per item.
- **The real-device kit, on in every part**: servos `DeviceKit.real_kit()` (PD loop inside the servo, kp 998.22 /
  kv 2.731 on a 10 ms encoder-difference velocity, 12-bit encoder on the servo shaft, deadband 1 count, 4 ms bus
  delay on writes and reads, back-EMF torque-speed line, +/- 0.5 deg horn play beyond the encoder, gripper torque
  limit 50%); plant `REAL_PLANT`; camera `CameraKit.real_kit(seed)` (depth noise z^2 * 0.08 / (f * 50 mm), 2% holes
  plus edge holes, RGB noise 4/255 and 3% gain, 45 ms +/- 10 ms latency, a planted hand-eye error of 8 mm and
  2.5 deg in seeded random directions). The pick sees only the camera: the item's pose and its class (hence its bin)
  come from the detector after the cell calibrated its camera.
- **Calibration**: `eval_f6.py calib`, five planted errors (camera seeds 4242-4246, fresh). Pass: every one recovered
  within 2 mm and 1 deg (`hand_eye_error()` of the calibrated belief against the true camera pose).
- **Picks**: `eval_f6.py picks`, seeds 20000-20199 (fresh), camera seed 4242, each seed run by the classical pipeline
  (`desk_pick.trial`) and the ONNX policy (`bc.policy_trial`, 10.5 s at 30 Hz) on the same scene. Success = the item
  ends inside the bin of its material (outcome `ok`). Pass: classical >= 180/200 and ONNX >= 160/200 (the plan's 90%
  and 80% as point thresholds on these 200 trials); Wilson 95% intervals are reported, not gated.
- **Which ONNX model**: `factory/desk/models/pick.onnx` as trained by `bc.py train` on demonstrations from training
  scenes only (2,661 successful demonstrations of 3,000 training scenes, seeds 100000-102999, 12 calibrated sessions with
  camera seeds 200-211), observation = the camera's estimate, class and time (9 inputs), 40 epochs:
  **sha256 7610b151222e74e1c5d4b0d1110b12a78e1de62dcca56679fe6272716f64fabc**. Model choices were made only on the check seeds 90000-90039: the first policy (which also read the
  servo positions) scored 6/40 there, this one 40/40.
- **Rate**: `eval_f6.py rate`, 20 trials (seeds 20200-20219) with the loop paced against the wall clock, nothing else
  heavy running. Pass: mean tick period within 1% of 33.33 ms and no tick later than 1.5 periods (50 ms); the
  inference time (p99) is reported.
- **System identification**: `sysid.py`, its planting seeds and its rule (every planted parameter within 10%) as
  written in its module docstring; the results file `factory/desk/results/f6-sysid.json`.
- **Commands**: `bin/rl desk eval calib`, `bin/rl desk eval picks --workers 10`, `bin/rl desk eval rate`,
  `bin/rl desk sysid` -> `factory/desk/results/f6-eval.json`, `f6-sysid.json`.
- **If a part fails**: it is reported as failed with its numbers; a re-run needs a new protocol with fresh seeds.
