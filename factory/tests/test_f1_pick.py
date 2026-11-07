"""F1 acceptance as regression tests (docs/sorting-cell-spec.md §3-§4, disciplines 3-6 and 11; 2026-11-02): inverse
kinematics on 1000 zone poses, a grasp that holds every item under gravity and a shake, zero collisions over 1000
planned picks, servo step responses without oscillation, the pick pipeline's success rate on a live sample and in the
recorded 200-trial evaluation, the failure taxonomy, bit-exact replay, and the F1 lab page."""
import json, math, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "factory", "classical"))
RESULTS = os.path.join(ROOT, "factory", "classical", "results")


@pytest.fixture(scope="module")
def planner(mj):
    import pick
    return pick.Planner(mj)


def test_ik_1000_zone_poses():
    """Discipline 3: every solution is within 1 mm / 1 deg and inside the joint limits (checked by forward kinematics);
    every pose reported infeasible is also infeasible for the exact top-down solver of the reach map."""
    from kinematics import Arm, grasp_rot, angle_between
    import reach_map
    from layout import DEFAULT as L
    arm = Arm(); b = L.belt; rng = np.random.default_rng(42); exact = reach_map.TopDownIK(L); solved = 0
    for k in range(1000):
        p = np.array([rng.uniform(*b.zone), rng.uniform(b.y - b.width / 2, b.y + b.width / 2), b.surface_z + rng.uniform(0.03, 0.30)])
        R = grasp_rot(rng.uniform(-math.pi, math.pi))
        ok, q, info = arm.ik(p, R, restarts=8, rng=np.random.default_rng(k))
        if ok:
            pc, Rc = arm.fk(q); solved += 1
            assert np.linalg.norm(pc - p) <= 1e-3 and math.degrees(angle_between(Rc, R)) <= 1.0, (k, p)
            assert np.all(q >= arm.lo - 1e-9) and np.all(q <= arm.hi + 1e-9), (k, q)
        else:
            assert not exact.solve(tuple(p))[0], f"pose {k} {p} reported infeasible but the exact solver reaches it"
    assert solved >= 880, solved   # 910 on 2026-11-02; the rest lie beyond the far belt edge at grasp height


@pytest.mark.parametrize("item", ["master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "tuna_fish_can",
                                  "pudding_box", "gelatin_box", "potted_meat_can", "bleach_cleanser", "windex_bottle"])
def test_grasp_holds_under_gravity_and_shake(mj, planner, item):
    """Discipline 4: pick the item at the zone centre, hold it 0.5 s, then shake the arm (pan and lift +-0.05 rad at
    1 Hz, about 1.2 m/s^2 at the tool); the held point may not slip more than 1 cm."""
    import pick
    mj.reset(); mj.belt(0.0, ramp=0); mj.spawn(item, x=0.0, y=mj.lay.belt.y, yaw=0.3, with_belt_velocity=False); mj.step(150)
    s = mj.item_state(item); rec = mj.items[item]["rec"]
    plan, why = planner.plan(item, s["pos"], s["quat"], mj.lay.chute(rec["material"])); assert plan is not None, why
    k = [sg.name for sg in plan.segments].index("lift"); plan.segments = plan.segments[:k + 1]
    ex = pick.Executor(mj); out = ex.run(plan, item); assert out["outcome"] == "done", out
    d = mj.data; q0 = d.qpos[mj.robot_qpos].copy(); dt = mj.model.opt.timestep; worst = 0.0
    for j in range(int(2.5 / dt)):
        t = j * dt; amp = 0.05 if t > 0.5 else 0.0; w = 2 * math.pi
        q = q0 + amp * np.array([math.sin(w * t), math.sin(w * t + 1.0), 0, 0, 0, 0]); qd = amp * w * np.array([math.cos(w * t), math.cos(w * t + 1.0), 0, 0, 0, 0])
        d.ctrl[mj.robot_act] = q + ex.kv_over_kp * qd; d.qfrc_applied[mj.robot_dof] = d.qfrc_bias[mj.robot_dof]; mj.step()
        if j % 20 == 0: worst = max(worst, ex._slip_swing(item, ex.held)[0])
    assert worst < 0.01, (item, plan.tool, worst)


def test_zero_collisions_over_1000_planned_picks():
    """Discipline 5: plan 1000 picks over a zone wider than the eval's and sweep every path kinematically (the carried
    item rides with the tool); no robot-fixture contact, no carried-item penetration; unplannable poses are reported."""
    import eval_f1
    S = eval_f1.plan_sweep(1000, workers=4)
    assert S["robot_collisions"] == 0 and S["item_hits"] == 0, S["examples"][:3]
    assert S["planned"] >= 980 and set(S["unplanned"]) <= {"ik_fail", "no_grasp"}, S["unplanned"]


def test_servos_settle_without_oscillation(mj):
    """Discipline 6: a 0.1 rad step on each joint settles within 1 s with < 2% overshoot and no ringing."""
    import control
    for j in range(6):
        r = control.step_response(mj, j)
        assert r["overshoot_pct"] < 2.0 and r["zero_crossings"] <= 1 and r["settling_s"] is not None and r["settling_s"] < 1.0, r


def test_pick_success_live_sample():
    """The pipeline end to end on 44 fresh trials (4 per item, seed 3): >= 95% success and no collision."""
    import eval_f1, multiprocessing as mp
    specs = eval_f1.trial_specs(44, seed=3)
    with mp.get_context("spawn").Pool(4, initializer=eval_f1._worker_init) as pool: rows = pool.map(eval_f1._run, specs)
    ok = sum(r["outcome"] == "success" for r in rows)
    assert ok >= 42 and not any(r["outcome"] == "collision" for r in rows), [(r["item"], r["outcome"], r.get("detail")) for r in rows if r["outcome"] != "success"]


def test_recorded_f1_evaluation_passes():
    """The acceptance run on record: >= 200 L0 trials over all 11 items, >= 95% success, zero collisions, tracking
    error under 0.01 rad, every outcome a class of the failure taxonomy."""
    S = json.load(open(os.path.join(RESULTS, "f1-eval.json")))
    assert S["trials"] >= 200 and S["success_rate"] >= 0.95 and S["collisions"] == 0 and len(S["per_item"]) == 11, S["failures"]
    assert S["max_track_err_rad"] < 0.01
    import pick
    taxonomy = {line.split()[0] for line in pick.__doc__.split("Failure taxonomy")[1].splitlines()[1:] if line.startswith("  ") and line.split() and not line.startswith("    ")}
    assert {"no_grasp", "ik_fail", "collision", "knocked", "miss", "drop", "wrong_chute", "off_target", "timeout"} <= taxonomy
    assert set(S["failures"]) <= taxonomy, S["failures"]


def test_trial_replays_bit_for_bit(mj, planner):
    """Discipline 11: the same trial twice gives the identical final state."""
    import pick
    finals = []
    for _ in range(2):
        r = pick.trial(mj, planner, "sugar_box", 0.12, 0.57, 0.8)
        finals.append((r["outcome"], mj.data.qpos.copy()))
    assert finals[0][0] == "success" and np.array_equal(finals[0][1], finals[1][1])


def test_f1_lab_page():
    sys.path.insert(0, os.path.join(ROOT, "lab")); import server
    page = json.load(open(os.path.join(ROOT, "lab", "phases", "F1.json")))
    assert {"blurb", "media", "jobs", "numbers"} <= set(page) and page["id"] == "F1"
    for j in page["jobs"]: assert j["id"] in server.JOBS
    for m in page["media"]: assert m["src"].endswith(".png") is False or os.path.exists(os.path.join(ROOT, m["src"])), m["src"]
