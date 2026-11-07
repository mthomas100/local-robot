"""F3 acceptance as regression tests (docs/sorting-cell-spec.md §4 F3, discipline 10 and 13; 2026-11-02): the learned carry
sorts a fresh F2 grasp, the safety shield holds whatever the policy asks, the line runs with the learned carry and
still catches a planted drop, the policy was chosen without the held-out items, and the recorded evaluation beats F2
on held-out items beyond the noise floor at both belt speeds."""
import json, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("learned", "line", "classical"): sys.path.insert(0, os.path.join(ROOT, "factory", p))
RES = os.path.join(ROOT, "factory", "learned", "results")
POLICY = os.path.join(ROOT, "factory", "learned", "policy.json")


@pytest.fixture(scope="module")
def snaps(mj):
    import snapshots
    S = snapshots.collect(dict(speed=0.3, rate=12.0, seed=100, seconds=40))
    assert len(S) >= 2, "the F2 line made fewer than 2 grasps in 40 s"
    return S


@pytest.fixture(scope="module")
def env(mj):
    import carry
    return carry.Env(cell=mj)


def test_policy_file():
    import carry
    P = json.load(open(POLICY)); th = carry.Policy(P["theta"]).theta
    assert th.shape == (carry.NP,) and np.all(np.isfinite(th)) and "held-out" not in P["chosen_by"].replace("no held-out", "")


def test_learned_carry_sorts_fresh_grasps(env, snaps):
    """Start states from a line seed the policy never saw; every carry ends in the right chute, faster than F2's."""
    import carry
    pol = carry.Policy(json.load(open(POLICY))["theta"])
    for s in snaps[:3]:
        r = env.episode(s, pol)
        assert r["ok"], r
        if s["f2_free_s"]: assert r["t_free"] < s["f2_free_s"], (r, s["f2_free_s"])


def test_shield_never_releases_over_the_belt(env, snaps):
    """A policy that always wants to let go (release logit pinned high) still cannot drop the item onto the belt or rail:
    the classical shield blocks release until the item hangs clear of the rail."""
    import carry
    th = np.zeros(carry.NP); th[3 * carry.NF + carry.NF - 1] = 50.0; pol = carry.Policy(th); seen = {}   # row 3 (release), last column (bias)

    def grab(c, ctl, t): seen["ctl"] = ctl
    r = env.episode(snaps[0], pol, on_frame=grab, frame_every=5); ctl = seen["ctl"]
    rel = [row for row in ctl.log if row[-1] == 1]
    assert rel, "the policy never released"
    y_at_release = rel[0][2]
    assert y_at_release + ctl.r_item <= ctl.rail_y - 0.012 + 1e-6, (y_at_release, ctl.r_item, ctl.rail_y)
    assert r["outcome"] in ("sorted", "off_target", "wrong_chute", "timeout")


def test_learned_line_sorts(mj):
    from line_learned import LearnedLine
    ln = LearnedLine(speed=0.15, rate_per_min=8, scheduler="lookahead", seed=2, cell=mj)
    S = ln.run(seconds=90)["summary"]
    assert S["outcomes"].get("sorted", 0) >= 4 and S["collisions"] == 0 and S["attempt_success"] >= 0.75, S
    assert S["cycle_s"] < 4.8, S   # F2's cycle is 5.3-5.8 s


def test_planted_drop_still_caught_with_the_learned_carry(mj):
    from line_learned import LearnedLine
    ln = LearnedLine(speed=0.15, rate_per_min=4, scheduler="lookahead", seed=1, cell=mj, inject={1: "drop"})
    out = ln.run(seconds=75); first = out["attempts"][0]
    assert first["outcome"] == "drop", first
    assert any(e["outcome"] == "ok" for e in out["attempts"][1:]), out["attempts"]


def test_recorded_f3_evaluation():
    """The acceptance run on record: the learned carry beats F2 on held-out items at both belt speeds, the 95% interval
    of the paired gain above zero; the training record exists."""
    E = json.load(open(os.path.join(RES, "f3-eval.json"))); T = json.load(open(os.path.join(RES, "f3-train.json")))
    assert E["accepted"] and E["seeds"] >= 5, E["verdict"]
    speeds = {k.split(" · ")[1] for k in E["verdict"] if k.startswith("held-out")}
    assert {"0.15 m/s", "0.30 m/s"} <= speeds
    assert T["episodes"] > 1000 and T["history"]


def test_f3_lab_page():
    sys.path.insert(0, os.path.join(ROOT, "lab")); import server
    page = json.load(open(os.path.join(ROOT, "lab", "phases", "F3.json")))
    assert page["id"] == "F3" and {"blurb", "media", "jobs", "numbers", "live"} <= set(page)
    for j in page["jobs"] + [{"id": page["live"]["job"]}]: assert j["id"] in server.JOBS
