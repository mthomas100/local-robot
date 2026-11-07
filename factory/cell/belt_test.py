"""belt_test.py — F0 acceptance, discipline 2: do items ride the belt stably at every speed, without penetration, in
both engines? (2026-11-02, docs/sorting-cell-spec.md §4 F0.)

For each belt speed and each item, with the belt already running, the item is placed upright at the upstream end
with a yaw and rides until it leaves the reach zone. Measured per ride: speed error vs the belt over the zone (slip),
peak tilt (a tip-over shows as a large tilt), sideways drift, deepest penetration, whether it stayed on the belt.

    envs/mj/bin/python -I factory/cell/belt_test.py --engine mujoco [--yaws 4] [--out factory/cell/results/belt-test-mujoco.json]
    envs/genesis/bin/python -I factory/cell/belt_test.py --engine genesis [--backend cpu]
Pass criteria (written before the run): every ride stays on the belt, peak tilt < 15 deg, mean |v_item - v_belt|
in the zone < 2% of the belt speed, sideways drift < 2 cm, deepest penetration < 3 mm.
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CRIT = dict(max_tilt_deg=15.0, slip_frac=0.02, drift_m=0.02, penetration_m=0.003)


def ride(cell, name, speed, yaw, settle=1.0):
    lay = cell.lay; b = lay.belt; dt = lay.timestep
    cell.reset(); cell.belt(speed); cell.step(int(max(settle, speed / b.ramp + 0.3) / dt))
    x0 = b.zone[0] - 0.35
    cell.spawn(name, x=x0, yaw=yaw)
    y0 = float(cell.item_state(name)["pos"][1])
    tilt, pen, slips, drift, on_belt, t = 0.0, 0.0, [], 0.0, True, 0.0
    t_max = (b.zone[1] - x0 + 0.1) / speed + 1.0
    while t < t_max:
        cell.step(5); t += 5 * dt
        s = cell.item_state(name); vb = cell.belt_velocity()
        tilt = max(tilt, s["tilt_deg"]); pen = max(pen, cell.penetration(name))
        drift = max(drift, abs(float(s["pos"][1]) - y0))
        if s["pos"][2] < b.surface_z - 0.05: on_belt = False; break
        if b.zone[0] <= s["pos"][0] <= b.zone[1]: slips.append(abs(float(s["linvel"][0]) - vb))
        if s["pos"][0] > b.zone[1]: break
    slip = float(np.mean(slips)) if slips else float("nan")
    ok = on_belt and tilt < CRIT["max_tilt_deg"] and slips and slip < CRIT["slip_frac"] * speed and drift < CRIT["drift_m"] and pen < CRIT["penetration_m"]
    return dict(item=name, speed=speed, yaw=round(yaw, 3), on_belt=on_belt, peak_tilt_deg=round(tilt, 2), mean_slip_mps=round(slip, 5),
                slip_frac=round(slip / speed, 4) if slips else None, drift_m=round(drift, 4), max_penetration_mm=round(pen * 1000, 3), samples_in_zone=len(slips), ok=bool(ok))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--engine", choices=("mujoco", "genesis"), default="mujoco"); p.add_argument("--backend", default="cpu")
    p.add_argument("--yaws", type=int, default=4); p.add_argument("--items", default="all"); p.add_argument("--speeds", default=None); p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.engine == "mujoco":
        from cell_mujoco import CellMJ; cell = CellMJ()
    else:
        from cell_genesis import CellGS; cell = CellGS(backend=a.backend, cameras=False)
    lay = cell.lay; speeds = [float(s) for s in a.speeds.split(",")] if a.speeds else list(lay.belt.speeds)
    names = list(cell.items) if a.items == "all" else a.items.split(",")
    yaws = [k * math.pi / a.yaws for k in range(a.yaws)]   # 0..180 deg: the items are near-symmetric under a half turn
    rows = []; t0 = time.perf_counter()
    for v in speeds:
        for n in names:
            for y in yaws:
                r = ride(cell, n, v, y); rows.append(r)
                if not r["ok"]: print("FAIL", json.dumps(r), flush=True)
        sub = [r for r in rows if r["speed"] == v]
        print(f"{a.engine} {v:.2f} m/s: {sum(r['ok'] for r in sub)}/{len(sub)} rides pass; worst tilt {max(r['peak_tilt_deg'] for r in sub):.1f} deg, "
              f"worst slip {max((r['slip_frac'] or 0) for r in sub) * 100:.2f}%, worst drift {max(r['drift_m'] for r in sub) * 100:.2f} cm, "
              f"worst penetration {max(r['max_penetration_mm'] for r in sub):.2f} mm", flush=True)
    summary = dict(engine=a.engine, backend=a.backend if a.engine == "genesis" else "cpu", criteria=CRIT, speeds=speeds, items=names, yaws_per_item=a.yaws,
                   rides=len(rows), passed=sum(r["ok"] for r in rows), all_pass=all(r["ok"] for r in rows), wall_seconds=round(time.perf_counter() - t0, 1),
                   per_speed={str(v): dict(passed=sum(r["ok"] for r in rows if r["speed"] == v), rides=sum(1 for r in rows if r["speed"] == v),
                                           worst_tilt_deg=max(r["peak_tilt_deg"] for r in rows if r["speed"] == v),
                                           worst_slip_frac=max((r["slip_frac"] or 0) for r in rows if r["speed"] == v),
                                           worst_drift_m=max(r["drift_m"] for r in rows if r["speed"] == v),
                                           worst_penetration_mm=max(r["max_penetration_mm"] for r in rows if r["speed"] == v)) for v in speeds},
                   date="2026-11-02", rows=rows)
    out = a.out or os.path.join(HERE, "results", f"belt-test-{a.engine}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"})); print(out)


if __name__ == "__main__":
    main()
