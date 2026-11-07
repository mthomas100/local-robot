"""control.py — discipline 6 (2026-11-02, F1): measure the arm's joint servos the pick pipeline runs on. Each UR5e joint
is a position servo (MuJoCo `general` actuator: force = kp*(ctrl - q) - kv*qdot, Menagerie gains) with gravity
compensation (qfrc_applied = qfrc_bias, as the UR controller does). This steps each joint by a small angle from the
home pose and records rise time, overshoot, settling time (2% band) and steady-state error, so "no oscillation at
the chosen gains" is a number, not a claim.

    envs/mj/bin/python -I factory/classical/control.py      -> factory/classical/results/f1-control.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))


def step_response(cell, joint, step=0.1, seconds=1.5):
    m, d = cell.model, cell.data; cell.reset(); q0 = np.array(cell.lay.home_qpos)
    for _ in range(200):   # settle at home under gravity compensation
        d.ctrl[cell.robot_act] = q0; d.qfrc_applied[cell.robot_dof] = d.qfrc_bias[cell.robot_dof]; cell.step()
    target = q0.copy(); target[joint] += step; start = float(d.qpos[cell.robot_qpos[joint]]); ys = []
    for _ in range(int(seconds / m.opt.timestep)):
        d.ctrl[cell.robot_act] = target; d.qfrc_applied[cell.robot_dof] = d.qfrc_bias[cell.robot_dof]; cell.step()
        ys.append(float(d.qpos[cell.robot_qpos[joint]]))
    y = (np.array(ys) - start) / step; t = np.arange(1, len(y) + 1) * m.opt.timestep
    rise = float(t[np.argmax(y >= 0.9)] - t[np.argmax(y >= 0.1)]) if (y >= 0.9).any() else None
    outside = np.flatnonzero(np.abs(y - 1.0) > 0.02); settle = float(t[outside[-1]]) if len(outside) and outside[-1] < len(t) - 1 else (0.0 if not len(outside) else None)
    # oscillation: sign changes of the error after the first crossing
    e = y - 1.0; crossings = int(np.sum(np.diff(np.sign(e[np.argmax(y >= 1.0):])) != 0)) if (y >= 1.0).any() else 0
    return dict(joint=cell.robot_joints[joint], step_rad=step, rise_s=rise, overshoot_pct=round(float(max(0.0, y.max() - 1.0) * 100), 2),
                settling_s=settle, steady_state_err_rad=round(float(abs(e[-1]) * step), 5), zero_crossings=crossings)


def main():
    import warnings; warnings.filterwarnings("ignore")
    from cell_mujoco import CellMJ
    cell = CellMJ(); rows = [step_response(cell, j) for j in range(6)]
    kp = [float(cell.model.actuator_gainprm[a][0]) for a in cell.robot_act]; kv = [float(-cell.model.actuator_biasprm[a][2]) for a in cell.robot_act]
    out = dict(date=__import__("time").strftime("%Y-%m-%d"), kp=kp, kv=kv, gravity_compensation=True, steps=rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "f1-control.json"), "w") as f: json.dump(out, f, indent=1)
    for r in rows: print(r)
    print("kp", kp, "kv", kv)


if __name__ == "__main__":
    main()
