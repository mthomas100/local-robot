# F7 acceptance: protocol fixed before the run (2026-11-03)

> This repository's public history is a fresh export. In the original development history this protocol was
> committed before the run it governs; commit references to that history have been removed.

Committed before the run. Development used only seeds 7000-7001 (the v1 envelope), 7100-7105 (ablations rounds 1 and
2), 7200-7205 (tuning 'early') and 1-3 (privileged probes).

- **Question** (docs/sorting-cell-spec.md §4, row F7): at 0.5 m/s does the v2 cell beat the v1 cell's sorted/min beyond
  the noise floor, with 0 wrong chutes and no new protective-stop class, and is it not worse at 0.3 m/s?
- **Cells**: v1 = the F5 cell (`LearnedLine`, student carry, vision only, lookahead scheduler, `v2=()`); v2 = the
  same with `v2=("fast", "early", "deadline")` as committed in `factory/line/line.py` at this commit.
- **Stream**: the F5 shift's (12 arrivals/min, 25% as touching pairs, all 11 items), 180 s per run.
- **Seeds**: **8000-8011 (12, fresh)**, each run by both cells at both speeds (48 runs), paired by seed.
- **Rule** (`factory/ops/f7.py eval`): at 0.5 m/s the 95% CI (normal, on the 12 paired per-seed differences
  v2 - v1 in sorted/min) lies above 0; at 0.3 m/s its lower bound lies above **-0.5 sorted/min** (the
  non-inferiority margin, about 8% of v1's 6.0); summed over all v2 runs **0 wrong chutes**; and **no new
  protective-stop class** in v2 (a class is the stop's kind: a named fixture contact or a tracking-error stop).
- **Sample size**: the round-2 ablation's paired differences had sd 0.86 (0.3 m/s) and 0.89 (0.5 m/s) sorted/min.
  For the 0.3 m/s non-inferiority test with a true difference of half the observed +1.17, 80% power needs
  n = ((1.96 + 0.84) x 0.86 / (0.58 + 0.5))^2, about 5; at 0.5 m/s the observed +3.78 is over 4 sd. 12 seeds
  is more than twice that, which also gives the wrong-chute check 24 runs of exposure.
- **Command**: `bin/rl shift f7 eval --seeds 12 --seed0 8000 --workers 7` -> `factory/ops/results/f7-eval.json`,
  `runs/f7-eval/episodes.jsonl`.
- **Also recorded, not gated**: the v2 envelope (`rl shift sweep --v2 fast,early,deadline --name f7-sweep-v2 --seed0
  7000`, the v1 sweep's seeds, so the two envelopes are paired), the report and figure (`rl shift f7 report`).
- **If it fails**: reported as failed with its numbers; a re-run needs a new protocol with fresh seeds.
