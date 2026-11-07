"""payload.py — which items can the SO-101 handle? The payload-and-jaw table (2026-11-03, F6; disciplines 1, 4, 6).

A desk arm is not a small industrial arm: its six hobby servos stall at 2.94 N*m and its jaw is a hinged finger
against a fixed one. Before any pick is planned, each candidate item is checked against four limits, every one
computed from the model, then confirmed with physical picks in simulation (real-device kit on):

  jaw      the grasp width (the narrow footprint axis; a can's diameter) plus the 15 mm pre-grasp clearance must fit
           the jaw where it closes on a *face*: probe boxes between the jaws (`JAW_ANGLE` in desk_pick.py) show the
           moving jaw meets the face up to ~95 mm at 20-30 mm depth and only the box's end beyond that, so W <= 80 mm
  reach    the planner (desk_pick.Planner) must find the pick at the zone's centre and both ends: above the item, the
           grasp, the lift to carry height (the item's bottom 2.5 cm above the belt rails and bin rims) and the drop.
           Tall items fail here: the top-down tool frame cannot rise more than ~9 cm above the arm's base
  torque   the static holding torque on every arm joint at every waypoint of that plan, item included (hung at the
           tool, its centre of mass half its hang below the grasp), must stay under `TORQUE_BUDGET` = 50% of stall.
           The rule is ours, an assumption stated here: a DC servo's copper loss goes with torque squared, so holding
           at half stall heats at a quarter of the stall rate and leaves half the torque for acceleration
  grip     friction must hold the item with a safety factor of 2: 2 * mu * F >= 2 * m * g, where F is the squeeze the
           gripper servo makes at its torque limit (50%, as LeRobot caps the gripper) through ~7 cm of jaw

Items that pass all four get `TRIALS` seeded L0 picks with the real-device kit; an item is admitted if all of them
succeed. If a material has no admitted YCB item, its stand-in (desk_items.STANDINS) completes the set, and is named
as a stand-in everywhere. Writes results/payload.json (the whole table) and results/admitted.json.

    envs/mj/bin/python -I factory/desk/payload.py
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import desk_items as I  # noqa: E402
import desk_pick as P  # noqa: E402
from desk_sim import DeskSim  # noqa: E402
from servo import DeviceKit  # noqa: E402
import so101  # noqa: E402

TORQUE_BUDGET = 0.5 * so101.STALL_TORQUE
JAW_MAX_WIDTH = 0.080
GRIP_LEVER = 0.07
SAFETY = 2.0
TRIALS = 6


def static_torques(sim, q5, mass, hang):
    """Gravity torque on each arm joint at q5 with an item of `mass` hung `hang` below the tool frame (its centre of
    mass at hang/2)."""
    m, d = sim.model, sim.data; ix = sim.ix
    q = np.zeros(6); q[:5] = q5; ix.set_q(d, q); d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    tau = d.qfrc_bias[ix.dadr[:5]].copy()
    if mass > 0:
        p = d.site_xpos[ix.site] + np.array([0, 0, -hang / 2])
        jp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, None, p, m.site_bodyid[ix.site])
        tau += jp[:, ix.dadr[:5]].T @ np.array([0, 0, mass * 9.81])
    return tau


def table():
    sim = DeskSim(items=I.ALL, kit=DeviceKit.real_kit())
    planner = P.Planner(sim); b = sim.lay.belt; rows = []
    # the bare arm's own holding torque at home, for reference
    for name in I.ALL:
        inf = I.info(name); row = dict(item=name, material=inf["material"], standin=inf["standin"], mass_kg=inf["mass_kg"],
                                       extents_mm=[round(1000 * e) for e in inf["extents"]], height_mm=round(1000 * inf["height"]))
        ex, ey = inf["extents"][:2]; W = min(ex, ey); row["grasp_width_mm"] = round(1000 * W)
        row["jaw_ok"] = bool(W <= JAW_MAX_WIDTH)
        plans = []
        for y in (b.zone[0], 0.0, b.zone[1]):
            pl = planner.plan(name, (b.x, y, 0.0)); plans.append(pl)
        row["reach_ok"] = all(p is not None for p in plans)
        if row["reach_ok"]:
            hang = inf["height"] - plans[0].grasp["depth"]; worst = np.zeros(5)
            for pl in plans:
                for seg in pl.segments:
                    if seg.name in ("lift", "transfer", "release"):
                        for q5 in seg.path: worst = np.maximum(worst, np.abs(static_torques(sim, q5, inf["mass_kg"], hang)))
            row["max_hold_torque_Nm"] = [round(float(v), 3) for v in worst]
            row["torque_ratio"] = round(float(worst.max() / TORQUE_BUDGET), 3)
            row["torque_ok"] = bool(worst.max() <= TORQUE_BUDGET)
            row["z_lift_m"] = round(plans[0].grasp["z_lift"], 3)
        else:
            row["torque_ok"] = None
        F = 0.5 * so101.STALL_TORQUE / GRIP_LEVER
        row["grip_force_N"] = round(F, 1); row["grip_ratio"] = round(2 * inf["friction"] * F / (SAFETY * inf["mass_kg"] * 9.81), 2)
        row["grip_ok"] = bool(row["grip_ratio"] >= 1.0)
        row["checks_ok"] = bool(row["jaw_ok"] and row["reach_ok"] and row["torque_ok"] and row["grip_ok"])
        rows.append(row)
    # physical confirmation: seeded L0 picks with the real-device kit, for every item the planner can plan (so the
    # table's predictions are tested on the items it rejects too)
    for row in rows:
        if not row["reach_ok"]: row["trials"] = None; continue
        sim_i = DeskSim(items=[row["item"]], kit=DeviceKit.real_kit())
        outs = [P.trial(sim_i, row["item"], 1000 + s)["outcome"] for s in range(TRIALS)]
        row["trials"] = {o: outs.count(o) for o in sorted(set(outs))}; row["trials_ok"] = outs.count("ok")
        print(f"{row['item']:22s} checks {row['checks_ok']!s:5} trials {row['trials']}", flush=True)
    return rows


def table_view(rows):
    return {r["item"]: dict(material=r["material"], mass_kg=r["mass_kg"], width_mm=r["grasp_width_mm"], height_mm=r["height_mm"], jaw=r["jaw_ok"], reach=r["reach_ok"],
                            torque_vs_budget=r.get("torque_ratio"), grip_margin=r["grip_ratio"], picks_ok=f"{r.get('trials_ok', '-')}/{TRIALS}" if r.get("trials") else "-") for r in rows}


def admit(rows):
    adm = [r["item"] for r in rows if r["checks_ok"] and not r["standin"] and r.get("trials_ok") == TRIALS]
    mats = {I.info(n)["material"] for n in adm}; why = {}
    for r in rows:
        if r["standin"] and r["material"] not in mats and r["checks_ok"] and r.get("trials_ok") == TRIALS:
            adm.append(r["item"]); mats.add(r["material"]); why[r["item"]] = f"no YCB {r['material']} item passes the table"
    return adm, why


if __name__ == "__main__":
    t0 = time.time(); rows = table(); adm, why = admit(rows)
    os.makedirs(I.RESULTS, exist_ok=True)
    out = dict(date="2026-11-03", torque_budget_Nm=TORQUE_BUDGET, jaw_max_width_mm=1000 * JAW_MAX_WIDTH, grip_lever_m=GRIP_LEVER, safety=SAFETY,
               trials_per_item=TRIALS, kit="real", rows=rows, table=table_view(rows), admitted=adm, standins_because=why, seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(I.RESULTS, "payload.json"), "w"), indent=1)
    json.dump(dict(date="2026-11-03", admitted=adm, standins_because=why, source="factory/desk/payload.py"), open(os.path.join(I.RESULTS, "admitted.json"), "w"), indent=1)
    print("admitted:", adm, why)
    for r in rows:
        print(f"{r['item']:22s} {r['material']:9s} {r['mass_kg']:.3f} kg W {r['grasp_width_mm']:3d} mm h {r['height_mm']:3d} | jaw {r['jaw_ok']!s:5} reach {r['reach_ok']!s:5} "
              f"torque {r.get('torque_ratio', '-')!s:5} grip {r['grip_ratio']:5.2f} | trials {r.get('trials')}")
