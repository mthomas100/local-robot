"""F6 acceptance as regression tests (docs/sorting-cell-spec.md §4 row F6; 2026-11-03): the SO-101 desk cell. Top-down IK
on random zone poses, the payload-and-jaw table and what it admits, the real-device servo holding still and tracking a
step, the calibration routine on a planted hand-eye error, a live sample of camera-guided classical and ONNX picks
with the whole real-device kit on, the ONNX loop's inference time, and the recorded acceptance results (picks, rate,
calibration); the F6 lab page. System identification has its own file (test_f6_sysid.py)."""
import json, math, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "factory", "desk"))
RESULTS = os.path.join(ROOT, "factory", "desk", "results")


def test_topdown_ik_zone_poses():
    """Discipline 3: 300 random top-down poses over the L0 zone at grasp heights: every solution within 1 mm and
    1 deg and inside the joint limits; the few failures lie above the arm's top-down ceiling (reach.json)."""
    import desk_ik, so101
    from desk_layout import DEFAULT as L
    s = so101.arm_spec(); s.body("base").pos = L.base; arm = desk_ik.Arm(model=s.compile()); b = L.belt
    rng = np.random.default_rng(6); ok = 0; ceiling = json.load(open(os.path.join(RESULTS, "reach.json")))["max_topdown_tool_z"]
    for k in range(300):
        p = np.array([rng.uniform(b.x - 0.04, b.x + 0.04), rng.uniform(*b.zone), rng.uniform(b.surface_z + 0.02, 0.18)])
        good, q, info = arm.ik(p, rng.uniform(-math.pi, math.pi), restarts=6, rng=np.random.default_rng(k))
        if good:
            ok += 1; pc, R = arm.fk(q); assert np.linalg.norm(pc - p) <= 1e-3 and R[2, 0] < -math.cos(math.radians(1.0)), (k, p)
            assert np.all(q >= arm.lo - 1e-9) and np.all(q <= arm.hi + 1e-9)
        else:
            assert p[2] > ceiling - 0.02, f"pose {k} {p} below the top-down ceiling {ceiling} reported infeasible"
    assert ok >= 270, ok


def test_payload_table_admits_three_materials():
    t = json.load(open(os.path.join(RESULTS, "payload.json"))); adm = t["admitted"]
    import desk_items as I
    assert {I.info(n)["material"] for n in adm} == {"metal", "plastic", "cardboard"}
    rows = {r["item"]: r for r in t["rows"]}
    for n in adm: assert rows[n]["checks_ok"] and rows[n]["trials_ok"] == t["trials_per_item"], n
    for n in adm:
        if rows[n]["standin"]: assert n.startswith("standin_") and n in t["standins_because"]
    # the table predicts the physics: items the jaw check rejects but the planner can reach also fail when tried
    for r in t["rows"]:
        if r["reach_ok"] and not r["jaw_ok"]: assert r["trials_ok"] <= t["trials_per_item"] // 2, r["item"]


def test_real_servo_holds_and_tracks():
    """Discipline 6/14: the real-device servo model reaches a step goal (lift +0.3, elbow -0.2 rad) within 0.25 s and
    then holds without a limit cycle (servo-shaft jitter < 0.1 deg rms)."""
    from desk_sim import DeskSim
    from servo import DeviceKit
    sim = DeskSim(items=["standin_tea_box"], kit=DeviceKit.real_kit()); q0 = np.array(sim.lay.home_q); g = q0.copy(); g[1] += 0.3; g[2] -= 0.2
    sim.bus.write(g, sim.t); qs = []
    for k in range(600):
        sim.step(1)
        if k % 17 == 0: sim.bus.write(g, sim.t)
        qs.append(sim.ix.servo_q(sim.data))
    qs = np.array(qs); assert np.all(np.abs(qs[125] - g)[:3] < math.radians(1.0)), np.degrees(qs[125] - g)
    assert np.degrees(qs[300:, :4].std(0)).max() < 0.1


def test_marker_fk_models_the_play():
    """Discipline 14: the calibration's robot side. With the real kit's horn play, the play the calibration assumes
    (`camera.gravity_play`: each pitch joint's link pushed to the end of its play by gravity) matches where the links
    really sit on 12 calibration poses: rms error per pitch joint <= 0.3 deg (the play is +/- 0.5 deg), so forward
    kinematics of the encoders puts the marker within 2.5 mm of its true place (the unloaded pan's play: up to
    1.7 mm). The first version read the wrist's near-zero torque at the shaft angles and left its play centred:
    0.47 deg rms off, a 0.8 mm systematic error that kept the calibration above its 2 mm bar (2026-11-03)."""
    import math, mujoco, camera as C, desk_ik
    from desk_sim import DeskSim
    from servo import DeviceKit
    sk = DeviceKit.real_kit(); sim = DeskSim(items=["standin_tea_box"], kit=sk); arm = desk_ik.Arm(model=sim.model)
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "marker_center"); steps = int(round(1 / 30 / sim.model.opt.timestep))
    perr, errs = [], []
    for p, yaw in C.calibration_poses(sim.lay)[::8]:
        ok, q5, _ = arm.ik_pose(p, desk_ik.topdown_rot(yaw), restarts=4); assert ok
        goal = np.r_[q5, 0.0]; q0 = sim.ix.servo_q(sim.data).copy()
        for i in range(1, 31): sim.bus.write(q0 + (goal - q0) * i / 30, sim.t); sim.step(steps)
        for _ in range(15): sim.bus.write(goal, sim.t); sim.step(steps)
        q = sim.bus.read(sim.t)
        perr.append(C.gravity_play(sim, q, sk.backlash_rad)[1:4] - sim.data.qpos[sim.ix.bl_qadr][1:4])
        errs.append(np.linalg.norm(C.marker_world_fk(sim, q, sk.backlash_rad) - sim.data.site_xpos[sid]) * 1000)
    rms = np.degrees(np.sqrt(np.mean(np.array(perr) ** 2, axis=0)))
    assert np.all(rms <= 0.3), rms.round(3)
    assert max(errs) <= 2.5, np.round(errs, 2)


@pytest.fixture(scope="module")
def session():
    """One calibrated full-kit session (camera seed 31337, not an evaluation seed) shared by the live checks."""
    import camera as C
    from desk_sim import DeskSim
    from servo import DeviceKit
    sk = DeviceKit.real_kit(); sim = DeskSim(kit=sk); cam = C.Camera(sim, C.CameraKit.real_kit(seed=31337), rng=np.random.default_rng(5))
    rep = C.calibrate(sim, cam, sk.backlash_rad); yield sim, cam, rep; cam.close()


def test_calibration_removes_planted_error(session):
    sim, cam, rep = session
    assert rep["before_mm"] > 7 and rep["before_deg"] > 2
    assert rep["after_mm"] <= 2.0 and rep["after_deg"] <= 1.0, rep


def test_live_classical_picks_with_kit(session):
    import desk_pick as P, desk_items as I
    sim, cam, _ = session; items = I.admitted()
    outs = [P.trial(sim, items[s % 4], 40_000 + s, cam=cam)["outcome"] for s in range(8)]
    assert outs.count("ok") >= 7, outs


def test_live_onnx_picks_with_kit(session):
    import bc, desk_items as I
    if not os.path.exists(bc.ONNX): pytest.fail(f"no ONNX policy at {bc.ONNX}: run bin/rl desk bc collect, then bin/rl desk bc train")
    sim, cam, _ = session; pol = bc.OnnxPolicy(); items = I.admitted(); outs = []; p99 = []
    for s in range(6):
        r = bc.policy_trial(sim, pol, items[s % 4], 41_000 + s, cam); outs.append(r["outcome"]); p99.append(r["timing"]["infer_ms_p99"])
    assert outs.count("ok") >= 4, outs
    assert max(p99) < 5.0, p99   # 33 ms per tick at 30 Hz: inference is a small part of it


def test_recorded_acceptance():
    r = json.load(open(os.path.join(RESULTS, "f6-eval.json")))
    assert r["calib"]["passed"] and len(r["calib"]["rows"]) == 5
    p = r["picks"]; assert p["classical"]["n"] == 200 and p["onnx"]["n"] == 200
    assert p["classical"]["rate"] >= 0.90 and p["onnx"]["rate"] >= 0.80 and p["passed"]
    assert r["rate"]["summary"]["passed"]
    assert os.path.exists(os.path.join(RESULTS, "f6-eval-protocol.md"))


def test_lab_page():
    page = json.load(open(os.path.join(ROOT, "lab", "phases", "F6.json")))
    assert page["id"] == "F6" and page["blurb"] and page.get("jobs")
