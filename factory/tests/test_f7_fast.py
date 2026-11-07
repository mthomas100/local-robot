"""F7 as regression tests (docs/sorting-cell-spec.md §4 row F7; 2026-11-03): the fast belt. The v1 cell is
unchanged by the v2 flags' defaults; on a short 0.5 m/s run v1 makes no attempt (its approach is too slow) and v2
sorts; the recorded envelope, ablation and pre-registered acceptance; the F7 report and lab page."""
import json, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("learned", "line", "cell", "ops"): sys.path.insert(0, os.path.join(ROOT, "factory", p))
RES = os.path.join(ROOT, "factory", "ops", "results")
V2 = ("fast", "early", "deadline")


def _line(v2, seed=1, speed=0.5):
    from line_learned import LearnedLine
    return LearnedLine(policy=os.path.join(ROOT, "factory", "learned", "student.json"), privileged_carry=False, speed=speed, rate_per_min=12,
                       scheduler="lookahead", seed=seed, perception="privileged", clutter=0.25, v2=v2)


def test_v1_defaults_unchanged():
    """The flags are off by default and v1's planner constants are the ones F2-F5 were measured with."""
    import line as LN
    ln = _line(()); lp = ln.lp
    assert ln.v2 == () and ln.min_confirm == 10 and abs(ln._last_x() - (LN.WINDOW[1] - 0.15)) < 1e-12
    assert np.allclose(lp.vmax, LN.VMAX) and np.allclose(lp.amax, LN.AMAX)
    assert (lp.t_lo_min, lp.scan_step, lp.n_scan, lp.t_close, lp.t_brake, lp.decision_s, lp.v_desc, lp.t_lift) == (0.6, 0.35, 6, 0.40, 0.3, 0.1, 0.3, 0.35)
    assert np.allclose(ln.planner.arm.fk(ln.q_ready)[0], [-0.15, 0.42, 1.12], atol=1e-3)
    v = _line(V2)
    assert np.all(v.lp.vmax <= np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28]))   # inside the UR5e's rated joint speeds
    with pytest.raises(AssertionError): _line(("warp",))


def test_fast_belt_v1_never_tries_v2_sorts():
    """Disciplines 8, 9: 40 s at 0.5 m/s (privileged poses, F5 stream): v1 makes no attempt at all (found 2026-11-03:
    every pre-grasp is out of reach by the time its approach could get there); v2 sorts at least 2, 0 wrong chutes."""
    a = _line((), seed=1).run(seconds=40)["summary"]; b = _line(V2, seed=1).run(seconds=40)["summary"]
    assert sum(a["attempts"].values()) == 0 and a["outcomes"].get("sorted", 0) == 0, a
    assert b["outcomes"].get("sorted", 0) >= 2 and b["outcomes"].get("wrong_chute", 0) == 0, b


def test_recorded_envelope():
    S = json.load(open(os.path.join(RES, "f7-sweep-v1.json"))); rows = S["rows"]
    assert {r["speed"] for r in rows} == {0.15, 0.3, 0.5} and len(rows) == 18
    assert all(r["sorted_per_min"] == 0 for r in rows if r["speed"] == 0.5)
    assert os.path.exists(os.path.join(RES, "f7-sweep-v2.json")), "the v2 envelope (rl shift sweep --v2 fast,early,deadline --name f7-sweep-v2)"
    V = json.load(open(os.path.join(RES, "f7-sweep-v2.json")))
    assert all(r["sorted_per_min"] > 0 for r in V["rows"] if r["speed"] == 0.5)


def test_recorded_ablation_and_acceptance():
    A = json.load(open(os.path.join(RES, "f7-ablation.json")))
    names = {"+".join(r["v2"]) or "v1" for r in A["table"].values()}
    assert {"v1", "fast", "early", "deadline", "+".join(V2)} <= names   # each change measured on its own
    E = json.load(open(os.path.join(RES, "f7-eval.json"))); v = E["verdict"]
    assert v["accepted"] and v["fast_beats_v1"] and v["not_worse_at_0_3"] and v["wrong_chutes_v2"] == 0 and not v["new_stop_classes"], v
    assert E["paired"]["0.50"]["lo"] > 0 and E["paired"]["0.30"]["lo"] > -E["margin"]
    assert os.path.exists(os.path.join(RES, "f7-eval-protocol.md"))
    rep = open(os.path.join(ROOT, "docs", "ops", "F7-operations-report.md")).read()
    assert "v1 envelope" in rep and "v2 envelope" in rep and "Acceptance" in rep


def test_lab_page():
    page = json.load(open(os.path.join(ROOT, "lab", "phases", "F7.json")))
    assert page["id"] == "F7" and page["blurb"] and page["jobs"] and page["live"]["job"] == "live-fast"
