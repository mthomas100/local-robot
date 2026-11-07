"""F5 acceptance as regression tests (docs/sorting-cell-spec.md §4 F5, disciplines 9, 12, 15; 2026-11-02): clutter
arrives touching, a short shift writes continuous KPIs the dashboard API serves, and the recorded shift is a
10-minute continuous vision-only run with its KPIs, its dashboard, its operations report."""
import json, os, subprocess, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("line", "ops"): sys.path.insert(0, os.path.join(ROOT, "factory", p))
MJ = os.path.join(ROOT, "envs", "mj", "bin", "python")


def test_clutter_arrives_touching(mj):
    from line import Line
    ln = Line(speed=0.3, rate_per_min=30, seed=4, cell=mj, clutter=1.0); ln.reset()
    assert ln._spawn(0.0)
    pair = [n for n, it in ln.items.items() if it.get("clutter")]
    assert len(pair) == 2, pair
    xa, xb = (mj.item_state(n)["pos"][0] for n in pair); ext = [mj.items[n]["rec"]["extents"] for n in pair]
    assert abs(xa - xb) <= 0.5 * (max(ext[0][:2]) + max(ext[1][:2])) * 1.45 + 0.01   # side by side, not spread out


def test_short_shift_and_dashboard_api():
    name = "f5-test"; r = subprocess.run([MJ, "-I", os.path.join(ROOT, "factory", "ops", "shift.py"), "--minutes", "0.3", "--pace", "0", "--port", "0", "--no-video", "--name", name, "--perception", "privileged"],
                                         capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", name, "kpis.jsonl"))]
    assert len(rows) >= 17 and all(b["t"] - a["t"] == pytest.approx(1.0, abs=0.05) for a, b in zip(rows, rows[1:]))
    for k in rows[-1:]:
        assert 0 <= k["oee"] <= 1 and k["oee"] == pytest.approx(k["availability"] * k["performance"] * k["quality"], abs=0.002)
    sys.path.insert(0, os.path.join(ROOT, "lab")); import server
    d = server.shift_state(name); assert d["name"] == name and d["live"]["running"] is False and len(d["series"]) == len(rows)


def test_recorded_shift():
    S = json.load(open(os.path.join(ROOT, "factory", "ops", "results", "f5-shift.json")))
    assert S["config"]["minutes"] >= 10 and S["sim_s"] >= 600 and S["config"]["perception"] == "vision", S["config"]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", "f5-shift", "kpis.jsonl"))] if os.path.exists(os.path.join(ROOT, "runs", "f5-shift", "kpis.jsonl")) else None
    assert S["kpi_rows"] >= 599
    if rows: assert len(rows) >= 599 and rows[-1]["t"] >= 599
    f = S["final"]
    for k in ("sorted_per_min", "sort_rate", "availability", "performance", "quality", "oee", "losses", "protective_stops"): assert k in f
    assert f["sorted"] >= 30, f
    rep = open(os.path.join(ROOT, "docs", "ops", "F5-operations-report.md")).read()
    assert "OEE" in rep and "Trade-offs" in rep
    assert os.path.exists(os.path.join(ROOT, "lab", "static", "dashboard.html"))


def test_f5_lab_page():
    sys.path.insert(0, os.path.join(ROOT, "lab")); import server
    page = json.load(open(os.path.join(ROOT, "lab", "phases", "F5.json")))
    assert page["id"] == "F5" and {"blurb", "media", "jobs", "numbers"} <= set(page)
    for j in page["jobs"]: assert j["id"] in server.JOBS
    assert "## F5 · The factory" in open(os.path.join(ROOT, "README.md")).read()   # the write-up
