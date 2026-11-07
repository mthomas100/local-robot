# F4 acceptance evaluation: protocol fixed before the run (2026-11-02)

> This repository's public history is a fresh export. In the original development history this protocol was
> committed before the run it governs; commit references to that history have been removed.

Committed before any F4 line evaluation ran (a 25-second, 1-seed smoke run of the harness only).

- **Question** (docs/sorting-cell-spec.md §4 F4): is L4 (vision only, no privileged poses) within 10 points of L2 (privileged
  poses) on success?
- **Cells**: L2 = privileged detections (`tracking.Detector`) + the F3 carry policy (`factory/learned/policy.json`, the
  teacher, reads the held item's true pose); L4 = `vision.VisionDetector` (overhead D435i at 424x240, ~23 ms per frame,
  80 ms latency) + the student carry (`factory/learned/student.json`, `CarryCtl(privileged=False)`). Reported but not
  gating: the same two perceptions with F2's classical carry (what perception alone costs).
- **Which student**: chosen by mean return on the F3 training start states only (no line data, no held-out items),
  among the zero-shot student, the DAgger rounds and an ARS fine-tune of the student, before this evaluation runs.
- **Stream**: L2 mixed, all 11 items, 16 arrivals/min, lookahead scheduler, 180 s per run, belt 0.15 and 0.3 m/s.
- **Seeds 300-315** (16; fresh: F3 used 0-9, 100-123, 200-239).
- **Success** = sort rate (items sorted / items that arrived and left the line).
- **Rule**: at each belt speed, the 95% t-interval of the paired difference L2 - L4 (same seed) must have its upper end
  below 0.10 (non-inferiority: "within 10 points" beyond the noise, not only on average). Accepted when both speeds pass.
- **Command**: `bin/memguard 56 envs/mj/bin/python -I
  factory/vision/eval_f4.py --seeds 16 --seconds 180 --workers 8 --name f4-eval` -> `factory/vision/results/f4-eval.json`.
