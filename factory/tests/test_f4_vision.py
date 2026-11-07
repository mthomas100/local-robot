"""F4 acceptance as regression tests (docs/sorting-cell-spec.md §4 F4, disciplines 7, 8, 10; 2026-11-02): the detector
names and places items from the camera alone, never mistakes the robot for an item (with a planted control showing
the check can fail), splits touching items, the student carry reads nothing of the held item's true state, the
vision-only line sorts, and the recorded L2-vs-L4 evaluation is within 10 points beyond the noise at both speeds."""
import json, math, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("vision", "learned", "line", "classical"): sys.path.insert(0, os.path.join(ROOT, "factory", p))
RES = os.path.join(ROOT, "factory", "vision", "results")


@pytest.fixture(scope="module")
def cam(mj):
    import vision
    return vision.Cam(mj)


def test_detector_names_and_places_items(mj):
    import vision
    S = vision.measure(n_scenes=6, seed=21, belt_speed=0.3)
    assert S["reports"] > 100 and S["identity_accuracy"] >= 0.97, S
    assert S["position_err_mm"]["mean"] < 6.0, S["position_err_mm"]


def test_robot_is_never_an_item(mj, cam):
    """The arm and gripper low over the belt with nothing on it: no item blob. Control: without the robot
    self-filter (no segmentation image) the same frame does show blobs, so the check can fail."""
    import vision
    from line import Line
    ln = Line(speed=0.0, cell=mj); ln.reset(); arm = ln.planner.arm
    ok, q, _ = arm.ik(np.array([0.0, mj.lay.belt.y, mj.lay.belt.surface_z + 0.12]), ln.lp.p.arm.fk(ln.q_ready)[1], q0=ln.q_ready)
    assert ok
    mj.data.qpos[mj.robot_qpos] = q; import mujoco; mujoco.mj_forward(mj.model, mj.data)
    rgb, depth, seg = cam.grab()
    assert [b for b in vision.blobs(cam, rgb, depth, seg) if not b["partial"]] == []
    assert len(vision.blobs(cam, rgb, depth, None)) >= 1   # the planted control


def test_touching_items_are_split(mj, cam):
    import vision
    idm = vision.IdentityModel(); mj.reset(); mj.belt(0.0, ramp=0); y = mj.lay.belt.y
    mj.spawn("tomato_soup_can", x=0.0, y=y, yaw=0.0, with_belt_velocity=False); mj.spawn("cracker_box", x=0.0345 + 0.037 + 0.003, y=y, yaw=0.0, with_belt_velocity=False)
    mj.step(300); rgb, depth, seg = cam.grab()
    assert sorted(str(idm.predict(vision.feat(b))) for b in vision.blobs(cam, rgb, depth, seg, idm)) == ["cracker_box", "tomato_soup_can"]
    R = json.load(open(os.path.join(RES, "f5-clutter-perception.json")))
    assert R["split"]["both_named"] >= 0.9 * R["scenes"] and R["split"]["wrong_names"] == 0, R["split"]


def test_student_carry_reads_no_item_state(mj):
    """While the student carry decides (CarryCtl.tick), nothing reads the simulator's item poses."""
    import carry
    from line_learned import LearnedLine
    ln = LearnedLine(policy=os.path.join(ROOT, "factory", "learned", "student.json"), privileged_carry=False, speed=0.15, rate_per_min=8, seed=2, cell=mj)
    inside = {"on": False, "ticks": 0}; orig_state = mj.item_state; orig_tick = carry.CarryCtl.tick

    def guarded(name):
        assert not inside["on"], "the student carry read the item's true state"
        return orig_state(name)

    def tick(self):
        inside["on"] = True; inside["ticks"] += 1
        try: return orig_tick(self)
        finally: inside["on"] = False
    mj.item_state = guarded; carry.CarryCtl.tick = tick
    try: S = ln.run(seconds=45)["summary"]
    finally: mj.item_state = orig_state; carry.CarryCtl.tick = orig_tick
    assert inside["ticks"] > 50 and S["outcomes"].get("sorted", 0) >= 2, (inside, S)


def test_vision_line_sorts(mj):
    from line_learned import LearnedLine
    ln = LearnedLine(policy=os.path.join(ROOT, "factory", "learned", "student.json"), privileged_carry=False, speed=0.15, rate_per_min=8, seed=3, cell=mj, perception="vision")
    S = ln.run(seconds=60)["summary"]
    assert S["outcomes"].get("sorted", 0) >= 3 and S["outcomes"].get("wrong_chute", 0) == 0 and S["collisions"] == 0, S


def test_recorded_f4_evaluation():
    E = json.load(open(os.path.join(RES, "f4-eval.json")))
    assert E["accepted"] and E["seeds"] >= 10, E["verdict"]
    assert set(E["verdict"]) == {"0.15 m/s", "0.30 m/s"} and all(v["ci95"][1] < 0.10 for v in E["verdict"].values()), E["verdict"]


def test_f4_lab_page():
    sys.path.insert(0, os.path.join(ROOT, "lab")); import server
    page = json.load(open(os.path.join(ROOT, "lab", "phases", "F4.json")))
    assert page["id"] == "F4" and {"blurb", "media", "jobs", "numbers", "live"} <= set(page)
    for j in page["jobs"] + [{"id": page["live"]["job"]}]: assert j["id"] in server.JOBS
    assert "## F4 · Seeing" in open(os.path.join(ROOT, "README.md")).read()   # the write-up
