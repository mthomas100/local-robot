"""F0 acceptance as regression tests (docs/sorting-cell-spec.md §4 F0, 2026-11-02): the layout is consistent, the assets are
sane, the cell builds in MuJoCo, items ride the belt in MuJoCo and Genesis, the chutes catch items in both, the reach
zone and the chutes lie inside the UR5e envelope, and the lab serves its pages."""
import json, math, os, sys, threading, urllib.request, urllib.error
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_layout_is_consistent():
    from layout import DEFAULT as L
    b = L.belt
    assert b.x_min < b.zone[0] < b.zone[1] < b.x_max and b.zone[1] - b.zone[0] == pytest.approx(1.0)
    for c in L.chutes:   # chutes between pedestal and belt, not overlapping either, rims below the belt surface
        half = c.inner / 2 + c.wall
        assert c.y - half > L.pedestal[1] / 2 or abs(c.x) - half > L.pedestal[0] / 2
        assert c.y < b.y - b.width / 2                                  # the opening is centred in front of the belt's near edge
        assert c.rim_z < b.surface_z - b.plate_thickness - 0.1          # the rim passes under the belt plate with room to spare
    assert L.camera.pos[2] - b.surface_z == pytest.approx(1.0)
    assert L.cycle_budget()[0.3]["zone_seconds"] == pytest.approx(3.33, abs=0.01)


def test_assets(index):
    assert len(index) == 11 and {r["material"] for r in index.values()} == {"plastic", "metal", "cardboard"}
    import ycb
    for r in index.values():
        d = os.path.join(ycb.ASSETS, r["id"])
        assert 0.05 < r["mass_kg"] < 1.5 and r["bounds"][0][2] == pytest.approx(0.0, abs=1e-6)   # canonical frame: bottom at z = 0
        assert abs(r["bounds"][0][0] + r["bounds"][1][0]) < 1e-3 and abs(r["bounds"][0][1] + r["bounds"][1][1]) < 1e-3   # centred
        assert "hull_base.obj" in r["hulls"] and all(os.path.exists(os.path.join(d, h)) for h in r["hulls"])
        assert os.path.exists(os.path.join(d, "visual.obj")) and os.path.exists(os.path.join(d, "texture_map.png"))


def test_cell_builds(built):
    m, paths = built
    names = {m.actuator(i).name for i in range(m.nu)}
    assert {"shoulder_pan", "elbow", "2f85/fingers_actuator", "belt_drive"} <= names and m.ncam == 5
    import mujoco
    for k in ("robot.xml", "fixtures.xml", "belt.xml"): mujoco.MjModel.from_xml_path(paths[k])


def test_home_pose_clears_the_belt(mj):
    """The F0 lesson: the gripper at home must be off the belt and above the tallest item."""
    mj.reset(); p = mj.pinch_pos(); b = mj.lay.belt
    tallest = max(r["extents"][2] for r in mj.index.values())
    assert p[1] < b.y - b.width / 2 or p[2] - 0.1 > b.surface_z + tallest


@pytest.mark.parametrize("speed", [0.15, 0.5])
def test_items_ride_the_belt_mujoco(mj, speed):
    from belt_test import ride
    rows = [ride(mj, n, speed, 0.7) for n in mj.items]
    bad = [r for r in rows if not r["ok"]]
    assert not bad, bad


def test_items_ride_the_belt_genesis(gs_cell):
    from belt_test import ride
    rows = [ride(gs_cell, n, 0.5, 0.7) for n in gs_cell.items]
    bad = [r for r in rows if not r["ok"]]
    assert not bad, bad


def test_engines_agree(mj, gs_cell):
    """Same scene, same push: positions after 3 s on the belt agree within 1 cm."""
    out = {}
    for name, cell in (("mj", mj), ("gs", gs_cell)):
        cell.reset(); cell.belt(0.3); cell.step(500); cell.spawn("cracker_box", x=-0.9, yaw=0.5); cell.step(1500)
        out[name] = cell.item_state("cracker_box")["pos"]
    assert np.linalg.norm(out["mj"] - out["gs"]) < 0.01, out


def test_chutes_catch_items(mj):
    for ch in mj.lay.chutes:
        mj.reset(); it = mj.items["tomato_soup_can"]; q = it["qpos"]
        mj.data.qpos[q:q + 3] = (ch.x, ch.y, ch.rim_z + 0.05); mj.data.qpos[q + 3:q + 7] = (1, 0, 0, 0); mj.step(800)
        z = mj.item_state("tomato_soup_can")["pos"][2]
        assert z == pytest.approx(ch.rim_z - ch.depth, abs=0.005), (ch.material, z)


def test_reach_zone_inside_envelope():
    """Exact top-down IK on a 2 cm grid: the zone at a low grasp height and every chute opening are reachable with the
    gripper pointing straight down (the sampled coverage in the plot depends on the sample count; this does not)."""
    import reach_map
    from layout import DEFAULT as L
    b = L.belt; ik = reach_map.TopDownIK(L)
    zone, _, _, _ = ik.grid_coverage((b.zone[0], b.zone[1], b.y - b.width / 2, b.y + b.width / 2), b.surface_z + 0.10)
    assert zone > 0.95, zone
    for c in L.chutes:
        h = c.inner / 2; cov, _, _, _ = ik.grid_coverage((c.x - h, c.x + h, c.y - h, c.y + h), c.rim_z + 0.25)
        assert cov == 1.0, (c.material, cov)
    ok, q, err = ik.solve((0.0, b.y, b.surface_z + 0.10)); assert ok and err < 1e-3   # the zone centre, exactly


def test_lab_serves_pages():
    sys.path.insert(0, os.path.join(ROOT, "lab")); import server
    srv = server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler); port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        get = lambda p: urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=5)
        assert json.load(get("/api/phases"))["phases"]["F0"]["title"].startswith("F0")
        assert get("/phase/F0").status == 200 and get("/").status == 200
        page = json.load(get("/api/phase/F0")); assert {"blurb", "media", "jobs", "numbers", "live"} <= set(page)
        with pytest.raises(urllib.error.HTTPError) as e: get("/api/file?path=../../etc/passwd")
        assert e.value.code == 403
        for j in page["jobs"]: assert j["id"] in server.JOBS
    finally:
        srv.shutdown()
