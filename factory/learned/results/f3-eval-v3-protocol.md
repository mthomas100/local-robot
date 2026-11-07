# F3 evaluation, attempt 3: protocol fixed before the run (2026-11-02)

> This repository's public history is a fresh export. In the original development history this protocol was
> committed before the run it governs; commit references to that history have been removed.

Committed before any attempt-3 episode ran. Policy, line, shield and verdict rule are unchanged from attempt 2
(`factory/learned/policy.json` = ARS run 2, iteration 60; `eval_f3.py` `verdict`).

- **Why another attempt.** Attempt 2 (10 seeds x 180 s, seeds 0-9) was underpowered, not negative: every point estimate
  of learned - F2 (held-out items sorted per minute) was positive (+0.60, +1.00, +0.80, +0.30), two intervals cleared
  zero and two did not. With attempt 2's paired spreads, the probability that the L2 0.3 m/s condition clears zero at
  10 x 180 s is about 0.3 even if the +0.30 gain is real. Attempt 2's own data is not reused.
- **Sample size, from a power calculation on attempt 2** (sd of the paired difference scaled by sqrt(180/T)):
  40 seeds x 360 s gives a 95% half-width of 0.16-0.24 per condition and power >= 0.93 in each condition at
  attempt 2's point estimates (0.93 is the weakest, L2 0.3 m/s). Those estimates are likely optimistic (winner's
  curse), so a failure is a real possibility and will be recorded as such.
- **Seeds 200-239** (fresh: attempts 1-2 used 0-9, the training start states came from 100-123).
- **Conditions**: streams held-out (8/min) and mixed L2 (16/min) x belt 0.15 and 0.3 m/s; carries f2, prior, learned.
- **Rule** (unchanged): F3 is accepted only if, in all four conditions, the 95% t-interval of the mean paired gain
  (learned - F2, same seed, held-out items sorted per minute) lies above zero. No further attempt on new seeds will be
  run with this policy if attempt 3 fails; a later attempt needs a changed policy.
- **Command**: `bin/memguard 48 envs/mj/bin/python -I factory/learned/eval_f3.py --seed0 200 --seeds 40 --seconds 360
  --workers 8 --name f3-eval-v3` -> `factory/learned/results/f3-eval-v3.json`, `runs/f3-eval-v3/episodes.jsonl`.
  If accepted it is copied to `f3-eval.json` (the gate the test reads); attempt 2 is kept as `f3-eval-attempt2.json`.
- Every attempt (1: 6 seeds, before two bug fixes; 2: 10 seeds; 3: this) is reported (README, F3 chapter).
