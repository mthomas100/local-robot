"""pick.py — the classical pick pipeline (2026-11-02, F1; disciplines 3-6 and 11): perceive (privileged pose in F1),
choose a grasp (library: fingers, or the suction cup for items wider than the 2F-85's stroke), plan waypoints with
inverse kinematics (Cartesian straight lines solved densely), time them (minimum-jerk under joint-speed, joint-
acceleration and tool-speed limits), track them with the arm's position servos plus gravity compensation, and
classify the outcome.

Module boundaries are ROS-2-shaped (discipline 11): `plan_pick(state) -> Plan` takes poses and returns joint
trajectories plus gripper commands; `Executor.run(plan)` turns a plan into actuator commands at the 500 Hz control
rate and watches for failures; `classify` reads the final state. A trial is reproducible from its seed.

Failure taxonomy (F1; FMEA-style, every class detected by a rule below and counted):
  no_grasp      no grasp in the library is reachable for this pose (planning failure)
  ik_fail       a waypoint or a Cartesian segment has no IK solution inside the joint limits (planning failure)
  collision     the robot (arm, gripper, cup) touched a fixture (pedestal, rails, belt plate, chutes, floor)
  knocked       the item was pushed off its spot (> 3 cm) or tipped (> 20 deg) before the grasp closed
  miss          the gripper closed but the item did not come up with the tool, or the cup found no face to seal on
  drop          the item left the tool while carried: the point the tool held slid > 3 cm from the pinch (a swing about
                the held point is recorded as max_swing_deg but is not a drop), or the vacuum seal broke (force limit)
  wrong_chute   the item ended in another material's chute
  off_target    released, but the item is in no chute (floor, belt, chute rim)
  timeout       the episode ran past its time budget
"""
from __future__ import annotations
import json, math, os, sys, time
from dataclasses import dataclass, field
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
from kinematics import Arm, grasp_rot, angle_between  # noqa: E402
import grasps as G  # noqa: E402
import build_mjcf  # noqa: E402

LIMITS = dict(vmax=np.array([1.6, 1.6, 2.0, 2.6, 2.6, 2.6]), amax=np.array([5.0, 5.0, 6.0, 8.0, 8.0, 8.0]), vtool=0.6, vtool_contact=0.12,
              atool=4.0, carry_moment=0.25)
# carry_moment (N*m/(m/s^2)... i.e. kg*m): a finger-held item whose mass times the depth of its centre of mass below the
# pinch is m*h swings about the closing axis like a pendulum when the tool accelerates sideways; the carry segments
# limit the tool's peak acceleration to carry_moment / (m*h) (the 1.1 kg bleach bottle held by its neck: ~1.5 m/s^2).
# Found 2026-11-02: short transfers next to the plastic chute were limited only by joint acceleration and reached
# ~6 m/s^2 at the tool; the bleach bottle swung out of the pads (2 of 18 bleach trials).
CLEAR = 0.05          # m: item bottom above the belt rails while carried
GRIP_CLOSED = 255.0
SUCTION_STROKE = build_mjcf.SUCTION["extended_tip"] - build_mjcf.SUCTION["retracted_tip"]
CUP_BEYOND_PINCH = build_mjcf.SUCTION["extended_tip"] - 0.145


@dataclass
class Segment:
    name: str
    path: np.ndarray            # (K, 6) joint path
    T: float                    # duration, s
    grip: float = None          # 2F-85 command during/after the segment (0 open .. 255 closed); None = keep
    suction: float = None       # adhesion 0..1; None = keep
    extend: float = None        # cup slide target (m); None = keep
    dwell: float = 0.0          # hold the end pose this long after the segment (gripper settling)


@dataclass
class Plan:
    tool: str
    grasp: dict
    segments: list = field(default_factory=list)
    plan_seconds: float = 0.0


def rot_y(a):
    c, s_ = math.cos(a), math.sin(a); return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def rot_z(a):
    c, s_ = math.cos(a), math.sin(a); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def minjerk(tau):
    tau = np.clip(tau, 0.0, 1.0); return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def duration(path, cart_len=0.0, vtool=LIMITS["vtool"], atool=LIMITS["atool"], vmax=None, amax=None):
    """Minimum-jerk duration: peak joint speed 1.875*D/T <= vmax, peak acceleration 5.77*D/T^2 <= amax, peak tool
    speed 1.875*L/T <= vtool, peak tool acceleration 5.77*L/T^2 <= atool (D = per-joint path length, L = Cartesian
    length)."""
    D = np.abs(np.diff(path, axis=0)).sum(0); vmax = LIMITS["vmax"] if vmax is None else vmax; amax = LIMITS["amax"] if amax is None else amax
    T = max(np.max(1.875 * D / vmax), np.max(np.sqrt(5.77 * D / amax)), 1.875 * cart_len / vtool, math.sqrt(5.77 * cart_len / atool), 0.15)
    return float(T)


def sample(seg: Segment, t):
    """Joint position on a segment at time t (path parameterised by cumulative joint-space chord length)."""
    P = seg.path
    if len(P) == 1: return P[0]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9: return P[-1]
    u = minjerk(t / seg.T) * d[-1]; k = int(np.clip(np.searchsorted(d, u) - 1, 0, len(P) - 2))
    w = (u - d[k]) / max(d[k + 1] - d[k], 1e-12); return P[k] * (1 - w) + P[k + 1] * w


class Planner:
    def __init__(self, cell, lib=None):
        self.cell = cell; self.lay = cell.lay; self.lib = lib or G.load()
        self.arm = Arm(self.lay, model=cell.model)   # the cell's own model: same kinematics as the simulation

    def line(self, q0, p0, p1, R, n=None, tol=5e-4, R1=None):
        """Dense IK along a straight Cartesian line, the orientation slerped from R to R1 (default: constant); None if
        any point fails."""
        n = n or max(3, int(np.linalg.norm(np.asarray(p1) - p0) / 0.01) + 1)
        if R1 is not None:
            q_a = np.zeros(4); q_b = np.zeros(4); mujoco.mju_mat2Quat(q_a, R.reshape(-1)); mujoco.mju_mat2Quat(q_b, R1.reshape(-1))
            if q_a @ q_b < 0: q_b = -q_b
        qs = [q0]; q = q0
        for k in range(1, n + 1):
            f = k / n; p = np.asarray(p0) + (np.asarray(p1) - p0) * f
            Rk = R
            if R1 is not None:
                om = math.acos(min(1.0, abs(float(q_a @ q_b))))
                qk = q_a if om < 1e-6 else (math.sin((1 - f) * om) * q_a + math.sin(f * om) * q_b) / math.sin(om)
                Rk = np.zeros(9); mujoco.mju_quat2Mat(Rk, qk / np.linalg.norm(qk)); Rk = Rk.reshape(3, 3)
            ok, q, info = self.arm.ik(p, Rk, q0=q, restarts=0, iters=60, tol_pos=tol)
            if not ok: return None
            qs.append(q)
        return np.array(qs)

    def reach(self, q0, p, R):
        ok, q, info = self.arm.ik(p, R, q0=q0, restarts=6)
        if ok and np.max(np.abs(q - q0)) > 2.5:   # prefer a solution near the current configuration
            ok2, q2, _ = self.arm.ik(p, R, q0=q0, restarts=0, iters=300)
            if ok2: q = q2
        return q if ok else None

    def candidates(self, name, item_pos, item_quat):
        """(tool, grasp) candidates in the world, best first: finger grasps from the library, else suction on the top."""
        rec = self.cell.items[name]["rec"]; out = []
        for g in G.world_grasps(self.lib, name, item_pos, item_quat)[:12]:
            out.append(("fingers", g))
        if not out:
            top = np.asarray(item_pos) + np.array([0, 0, rec["bounds"][1][2]])
            com_xy = np.asarray(item_pos)[:2]
            for yaw in (0.0, math.pi / 2, math.pi / 4, -math.pi / 4):
                out.append(("suction", dict(pos=np.array([com_xy[0], com_xy[1], top[2]]), yaw=yaw, width=0.0, depth=0.0, score=1.0)))
        return out

    # tilts of the tool about the fingers' closing axis (rad): the approach leans while both pads stay on the same two
    # faces. Tall items on the belt's far side are out of top-down reach (the wrist would sit ~0.96 m from the
    # shoulder; 4 of 200 trials, 2026-11-02); leaning the approach toward the robot brings the wrist back in.
    TILTS = (0.0, 0.35, -0.35, 0.6, -0.6)

    def plan(self, name, item_pos, item_quat, chute):
        t0 = time.perf_counter(); rec = self.cell.items[name]["rec"]; b = self.lay.belt; q_home = np.array(self.lay.home_qpos)
        q_now = self.cell.data.qpos[self.cell.robot_qpos].copy()
        mass = rec["mass_kg"]; com_z = item_pos[2] + rec["com"][2]; failure = "no_grasp"
        for tool, g in self.candidates(name, item_pos, item_quat):
            for tilt in (self.TILTS if tool == "fingers" else (0.0,)):
                for yaw in (g["yaw"], g["yaw"] + math.pi):   # the closing axis is symmetric; try both wrist turns
                    R = grasp_rot(yaw) @ rot_y(tilt); approach = R[:, 2]
                    if tool == "fingers":
                        p_grasp = np.asarray(g["pos"], float)
                        a_carry = min(LIMITS["atool"], LIMITS["carry_moment"] / max(mass * (p_grasp[2] - com_z), 1e-3))
                    else:
                        p_grasp = np.asarray(g["pos"], float) + np.array([0, 0, CUP_BEYOND_PINCH - 0.004]); a_carry = LIMITS["atool"]
                    hang = p_grasp[2] - item_pos[2]                                  # pinch height above the item's bottom (kept: the item stays upright)
                    p_pre = p_grasp - approach * (0.07 if tool == "fingers" else 0.06)
                    carry_z = b.surface_z + b.rail_height + CLEAR + hang            # pinch height that carries the item over the rails
                    p_lift = np.array([p_grasp[0], p_grasp[1], max(carry_z, p_pre[2])])
                    drop_y = chute.y - 0.03                                          # a little toward the robot, away from the belt rail
                    p_over = np.array([chute.x, drop_y, p_lift[2]])
                    p_drop = np.array([chute.x, drop_y, chute.rim_z + 0.04 + hang])
                    # over the chute the fingers close along the belt (x), so neither finger reaches toward the rail; the item
                    # turns with the wrist about the vertical on the way (the half turn nearer the grasp yaw) and stays upright
                    yd = 0.0 if math.cos(yaw) >= 0 else math.pi; R_drop = rot_z(yd - yaw) @ R
                    q_pre = self.reach(q_now, p_pre, R)
                    if q_pre is None: failure = "ik_fail"; continue
                    down = self.line(q_pre, p_pre, p_grasp, R)
                    if down is None: failure = "ik_fail"; continue
                    up = self.line(down[-1], p_grasp, p_lift, R)
                    if up is None: failure = "ik_fail"; continue
                    across = self.line(up[-1], p_lift, p_over, R, n=max(10, int(np.linalg.norm(p_over - p_lift) / 0.02)), R1=R_drop)
                    if across is None: failure = "ik_fail"; continue
                    lower = self.line(across[-1], p_over, p_drop, R_drop)
                    if lower is None: failure = "ik_fail"; continue
                    # retreat straight up before the joint-space move home: from a low drop pose (flat boxes) that move
                    # swung the open fingers through the belt's near rail (182 of 1000 planned picks, 2026-11-02)
                    retreat = self.line(lower[-1], p_drop, p_over, R_drop)
                    if retreat is None: failure = "ik_fail"; continue
                    q_after = retreat[-1]
                    segs = []
                    to_pre = np.array([q_now, q_pre]); d_pre = float(np.linalg.norm(p_pre - p_grasp))
                    if tool == "fingers":
                        segs.append(Segment("approach", to_pre, duration(to_pre), grip=0.0, extend=0.0, suction=0.0))
                        segs.append(Segment("descend", down, duration(down, d_pre, LIMITS["vtool_contact"] * 2)))
                        segs.append(Segment("close", down[-1:], 0.05, grip=GRIP_CLOSED, dwell=0.45))
                    else:
                        segs.append(Segment("approach", to_pre, duration(to_pre), grip=0.0, extend=SUCTION_STROKE, suction=0.0, dwell=0.15))
                        segs.append(Segment("descend", down, duration(down, d_pre, LIMITS["vtool_contact"])))
                        segs.append(Segment("suck", down[-1:], 0.05, suction=1.0, dwell=0.25))
                    segs.append(Segment("lift", up, duration(up, np.linalg.norm(p_lift - p_grasp), LIMITS["vtool_contact"] * 3, a_carry)))
                    segs.append(Segment("transfer", across, duration(across, np.linalg.norm(p_over - p_lift), LIMITS["vtool"], a_carry)))
                    segs.append(Segment("lower", lower, duration(lower, np.linalg.norm(p_over - p_drop), LIMITS["vtool"], a_carry)))
                    if tool == "fingers": segs.append(Segment("release", lower[-1:], 0.05, grip=0.0, dwell=0.35))
                    else: segs.append(Segment("release", lower[-1:], 0.05, suction=0.0, dwell=0.35))
                    segs.append(Segment("retreat", retreat, duration(retreat, np.linalg.norm(p_over - p_drop))))
                    back = np.array([q_after, q_home])
                    segs.append(Segment("home", back, duration(back), extend=0.0))
                    return Plan(tool, dict(g, yaw=float(yaw), tilt=float(tilt), a_carry=round(float(a_carry), 2), p_grasp=p_grasp.tolist()), segs, time.perf_counter() - t0), None
        return None, failure


class Executor:
    """Runs a plan on the cell at the control rate: position targets for the arm with gravity compensation, the
    gripper and the suction cup; watches for failures while it runs."""

    def __init__(self, cell, on_frame=None, frame_every=None):
        self.cell = cell; m = cell.model
        self.on_frame = on_frame; self.frame_every = frame_every
        self.act_grip = cell.grip_act; self.act_ext = m.actuator("suction_extend").id
        robot_bodies = {i for i in range(m.nbody) if (m.body(i).name in ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "suction")
                                                         or m.body(i).name.startswith("2f85/"))}
        self.robot_geom = np.array([m.geom_bodyid[g] in robot_bodies for g in range(m.ngeom)])
        item_bodies = {it["body"] for it in cell.items.values()}
        self.fixture_geom = np.array([(m.geom_bodyid[g] not in robot_bodies) and (m.geom_bodyid[g] not in item_bodies) and m.body(m.geom_bodyid[g]).name != "base" for g in range(m.ngeom)])
        self.pad_bodies = {m.body("2f85/right_pad").id, m.body("2f85/left_pad").id, m.body("suction").id}
        self.track_err = 0.0
        # the arm's servos are PD loops: force = kp*(ctrl - q) - kv*qdot. Feeding ctrl = q_ref + (kv/kp)*qdot_ref makes
        # it kp*(q_ref - q) + kv*(qdot_ref - qdot): it tracks the reference velocity instead of braking toward zero
        self.kv_over_kp = np.array([-m.actuator_biasprm[a][2] / m.actuator_gainprm[a][0] for a in cell.robot_act])

    def robot_fixture_contact(self):
        d = self.cell.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < -0.001 and ((self.robot_geom[c.geom1] and self.fixture_geom[c.geom2]) or (self.robot_geom[c.geom2] and self.fixture_geom[c.geom1])):
                return self.cell.model.geom(c.geom1).name or self.cell.model.body(self.cell.model.geom_bodyid[c.geom1]).name, \
                       self.cell.model.geom(c.geom2).name or self.cell.model.body(self.cell.model.geom_bodyid[c.geom2]).name
        return None

    def _slip_swing(self, name, held):
        """How far the point the tool holds has slid from the pinch (m), and how far the item has turned in the hand
        (deg), relative to the moment the grasp closed."""
        c = self.cell; s = c.item_state(name); R = quat_mat(s["quat"]); Rt = c.data.site_xmat[c.pinch_site].reshape(3, 3)
        slip = float(np.linalg.norm(s["pos"] + R @ held[0] - c.pinch_pos()))
        swing = math.degrees(angle_between(Rt.T @ R, held[1]))
        return slip, swing

    def run(self, plan: Plan, name, budget=25.0):
        c = self.cell; m, d = c.model, c.data; dt = m.opt.timestep
        self.held = None; start = c.item_state(name); k = 0; t_run = 0.0; self.max_slip = 0.0; self.max_swing = 0.0
        for seg in plan.segments:
            if seg.grip is not None: d.ctrl[self.act_grip] = seg.grip
            if seg.suction is not None: c.vacuum(seg.suction > 0.5)
            if seg.extend is not None: d.ctrl[self.act_ext] = seg.extend
            steps = int(math.ceil((seg.T + seg.dwell) / dt)); carrying = self.held is not None and seg.name in ("lift", "transfer", "lower")
            for j in range(steps):
                tj = min(j * dt, seg.T); q_ref = sample(seg, tj)
                qd_ref = (sample(seg, min(tj + dt, seg.T)) - q_ref) / dt if tj < seg.T else 0.0 * q_ref
                d.ctrl[c.robot_act] = q_ref + self.kv_over_kp * qd_ref
                d.qfrc_applied[c.robot_dof] = d.qfrc_bias[c.robot_dof]   # gravity (and Coriolis) compensation, as the UR controller does
                c.step(); k += 1; t_run += dt
                if seg.name in ("transfer", "lift", "descend"): self.track_err = max(self.track_err, float(np.max(np.abs(d.qpos[c.robot_qpos] - q_ref))))
                if self.on_frame and k % self.frame_every == 0: self.on_frame(c)
                if k % 10 == 0:
                    hit = self.robot_fixture_contact()
                    if hit: return dict(outcome="collision", detail=f"{hit[0]} x {hit[1]}", segment=seg.name, t=t_run)
                    if carrying:
                        slip, swing = self._slip_swing(name, self.held); self.max_slip = max(self.max_slip, slip); self.max_swing = max(self.max_swing, swing)
                        if slip > 0.03: return dict(outcome="drop", segment=seg.name, t=t_run, slip=round(slip, 3), swing=round(swing, 1))
                if t_run > budget: return dict(outcome="timeout", segment=seg.name, t=t_run)
            s = c.item_state(name)
            if seg.name in ("close", "suck"):
                moved = float(np.linalg.norm(s["pos"][:2] - start["pos"][:2]))
                if moved > 0.03 or s["tilt_deg"] > 20: return dict(outcome="knocked", segment=seg.name, t=t_run, moved=round(moved, 3), tilt=round(s["tilt_deg"], 1))
                if seg.name == "suck" and c.vac_item != name: return dict(outcome="miss", segment=seg.name, t=t_run, why="no seal")
                R = quat_mat(s["quat"]); Rt = d.site_xmat[c.pinch_site].reshape(3, 3)
                self.held = (R.T @ (c.pinch_pos() - s["pos"]), Rt.T @ R)    # the held point in the item frame; the item's attitude in the hand
            if seg.name == "lift" and s["pos"][2] < start["pos"][2] + 0.03:
                return dict(outcome="miss", segment=seg.name, t=t_run, rise=round(float(s["pos"][2] - start["pos"][2]), 3))
            if seg.name == "release": self.held = None
        return dict(outcome="done", t=t_run)


def quat_mat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def classify(cell, name, target):
    """Where did the item end up? success / wrong_chute / off_target, by its centre of mass."""
    s = cell.item_state(name); rec = cell.items[name]["rec"]
    w, x, y, z = s["quat"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    com = R @ np.array(rec["com"]) + s["pos"]
    for ch in cell.lay.chutes:
        if abs(com[0] - ch.x) < ch.inner / 2 and abs(com[1] - ch.y) < ch.inner / 2 and com[2] < ch.rim_z:
            return ("success" if ch.material == target else "wrong_chute"), ch.material, com
    return "off_target", None, com


def trial(cell, planner, name, x, y, yaw, seed=0, on_frame=None, frame_every=17):
    """One L0 trial: the belt stopped, one item standing at (x, y) with yaw; pick it and drop it in its chute."""
    cell.reset(); cell.belt(0.0, ramp=0); cell.spawn(name, x=x, y=y, yaw=yaw, with_belt_velocity=False); cell.step(150)
    s = cell.item_state(name); rec = cell.items[name]["rec"]; chute = cell.lay.chute(rec["material"])
    plan, why = planner.plan(name, s["pos"], s["quat"], chute)
    res = dict(item=name, material=rec["material"], x=round(x, 3), y=round(y, 3), yaw=round(yaw, 3), seed=seed)
    if plan is None: return dict(res, outcome=why, failure=why, tool=None, cycle_s=None)
    ex = Executor(cell, on_frame, frame_every)
    out = ex.run(plan, name)
    res.update(tool=plan.tool, grasp_yaw=round(plan.grasp["yaw"], 3), grasp_width=plan.grasp.get("width"), plan_ms=round(plan.plan_seconds * 1000, 1),
               cycle_s=round(out["t"], 2), max_track_err_rad=round(ex.track_err, 4), segments=[s.name for s in plan.segments],
               tilt_deg=round(math.degrees(plan.grasp.get("tilt", 0.0)), 1), a_carry=plan.grasp.get("a_carry"), max_slip_m=round(ex.max_slip, 4), max_swing_deg=round(ex.max_swing, 1))
    if out["outcome"] != "done": return dict(res, outcome=out["outcome"], failure=out["outcome"], detail={k: v for k, v in out.items() if k not in ("outcome", "t")})
    cell.step(400)   # let it settle in the chute
    where, chute_hit, com = classify(cell, name, rec["material"])
    return dict(res, outcome=where, failure=None if where == "success" else where, landed_in=chute_hit, com=[round(float(v), 3) for v in com])
