"""F6 system identification as regression tests (docs/sorting-cell-spec.md §4 F6, discipline 14; 2026-11-03): "system
identification recovers every planted plant parameter within 10%". The recorded campaign (factory/desk/sysid.py, three
planting seeds, all six joints) is checked from its JSON; a reduced identification (one joint, one fresh seed, the
short experiment) runs live (~30 s)."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "factory", "desk"))
RESULT = os.path.join(ROOT, "factory", "desk", "results", "f6-sysid.json")


def test_recorded_identification_within_10pct():
    R = json.load(open(RESULT))
    assert R["tolerance"] == 0.10 and R["all_within_tolerance"] is True
    seeds = R["seeds"]; assert len({s["seed"] for s in seeds}) >= 3
    for s in seeds:
        rows = s["rows"]
        assert {(r["param"], r["joint"]) for r in rows} >= {(k, j) for k in ("armature", "damping", "frictionloss")
                                                            for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")}
        assert {"bus_delay_s", "no_load_speed"} <= {r["param"] for r in rows}
        for r in rows:
            assert abs(r["planted"] / r["nominal"] - 1) >= 0.19, r          # really planted off nominal
            assert abs(r["recovered"] - r["planted"]) <= 0.10 * abs(r["planted"]), (s["seed"], r)
    assert os.path.exists(os.path.join(ROOT, "docs", "media", "F6-sysid.png"))


def test_live_reduced_identification():
    """One joint (shoulder_lift, under gravity), seed 7 (not one of the recorded seeds), the short experiment, the other
    five joints given at their planted values (they hold, but not rigidly: sysid.py's docstring): the joint's three
    parameters, the bus delay and the no-load speed, each within 10%, starting from nominal. ~30 s on 16 cores."""
    import sysid as S
    j, seed = 1, 7
    P, exps = S.record(seed, joints=[j], short=True)
    with ProcessPoolExecutor(min(10, os.cpu_count() or 4)) as pool:
        R, steps = S.identify(exps, pool, log=lambda *a: None, base=P)
    rows = S.compare(P, R, [j])
    bad = [r for r in rows if abs(r["rel_err"]) > S.TOL]
    assert not bad, (bad, steps)
