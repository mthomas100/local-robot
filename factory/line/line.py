"""line.py — F2: the classical pipeline on a moving belt (2026-11-02; disciplines 8, 9, 12; docs/sorting-cell-spec.md §4).

The belt never stops. Items arrive upstream as a random stream (L1: one at a time; L2: random spacing, mixed
materials), ride into the camera's view, are tracked (tracking.py: noisy, late detections -> a Kalman filter per item),
and a scheduler picks which one the robot goes for next. The robot plans an *intercept*: a joint-space approach timed
to meet the item, then descends, closes and lifts while moving with the belt (every 20 ms of that phase solved by IK
on the predicted item position), rises clear of the other items, carries the item to its chute, drops it, and goes
straight for the next one. Every item ends in exactly one class (`OUTCOMES`); robot faults are counted and recovered.

    from line import Line
    ln = Line(speed=0.3, rate_per_min=8, scheduler="lookahead", seed=0); S = ln.run(seconds=180); print(S["summary"])

Failure taxonomy (per item; FMEA-style, each detected by a rule in `_lifecycle` or `_check`):
  sorted        landed in the chute for its material
  missed        left the reach window and fell off the belt end without the robot ever trying (the scheduler let it go)
  late          the robot tried (a plan was started or a pick failed) but the item still left the window unsorted
  wrong_chute   landed in another material's chute
  dropped       slipped out of the tool while carried and ended off the belt (floor, chute rim)
  knocked_off   pushed off the belt by the robot without being held
Robot events (per attempt): ok, knocked (item moved/tipped at the grasp), miss (did not come up / no seal), drop (slip
while carried), collision (protective stop: freeze 0.5 s, open, retreat), late (no feasible plan any more). Recovery:
open the gripper and vacuum, rise 12 cm, return to the ready pose, re-plan; an item that fell back on the belt can be
picked again (up to 2 attempts).
"""
from __future__ import annotations
import json, math, os, sys, time
from dataclasses import dataclass, field
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE)
for p in (HERE, os.path.join(FACTORY, "classical"), os.path.join(FACTORY, "cell")): sys.path.insert(0, p)
import pick as PK  # noqa: E402
from pick import Planner, Segment, duration, sample, minjerk, grasp_rot, rot_y, rot_z, LIMITS, GRIP_CLOSED, SUCTION_STROKE, CUP_BEYOND_PINCH, quat_mat, classify  # noqa: E402
from tracking import Detector, DetectorCfg, Tracker  # noqa: E402

DT_S = 0.008          # trajectory sample period, s
OUTCOMES = ("sorted", "missed", "late", "wrong_chute", "dropped", "knocked_off")
WINDOW = (-0.55, 0.55)
# the line's joint limits (rad/s, rad/s^2): faster than F1's careful profile, below the UR5e's rated 3.14 rad/s (base,
# shoulder, elbow) and 6.28 rad/s (wrists). From the downstream chute to an upstream item the base pans ~1.6 rad.
TRACK_STOP = 0.15   # rad: joint tracking error that triggers a protective stop (normal motion stays under ~0.05)
VMAX = np.array([2.4, 2.4, 3.0, 3.0, 3.0, 3.0]); AMAX = np.array([8.0, 8.0, 10.0, 12.0, 12.0, 12.0])  # x range where the robot may hold an item on the belt (reach map: >= 98% of the zone at grasp height)


# --- timed joint trajectories -------------------------------------------------------------------------------------
@dataclass
class Traj:
    name: str
    t: np.ndarray
    q: np.ndarray
    grip: float = None
    suction: float = None
    extend: float = None

    @property
    def T(self): return float(self.t[-1])

    def at(self, tau):
        t, q = self.t, self.q
        if tau >= t[-1] or len(t) == 1: return q[-1], np.zeros(q.shape[1])
        i = int(min(np.searchsorted(t, tau, "right") - 1, len(t) - 2)); h = t[i + 1] - t[i]; w = (tau - t[i]) / h
        return q[i] * (1 - w) + q[i + 1] * w, (q[i + 1] - q[i]) / h


def from_segment(seg: Segment, **cmd):
    ts = list(np.arange(0.0, seg.T, DT_S)) + [seg.T]; qs = [sample(seg, x) for x in ts]
    if seg.dwell: ts.append(seg.T + seg.dwell); qs.append(qs[-1])
    return Traj(seg.name, np.array(ts), np.array(qs), **cmd)


def hold(name, q, T, **cmd): return Traj(name, np.array([0.0, T]), np.array([q, q]), **cmd)


def quintic(name, q0, v0, q1, v1, T, **cmd):
    """Joint quintic from (q0, v0, a=0) to (q1, v1, a=0) in T seconds, sampled every DT_S."""
    q0, v0, q1, v1 = (np.asarray(a, float) for a in (q0, v0, q1, v1))
    a3 = (20 * (q1 - q0) - (8 * v1 + 12 * v0) * T) / (2 * T ** 3); a4 = (30 * (q0 - q1) + (14 * v1 + 16 * v0) * T) / (2 * T ** 4)
    a5 = (12 * (q1 - q0) - (6 * v1 + 6 * v0) * T) / (2 * T ** 5)
    ts = np.append(np.arange(0.0, T, DT_S), T)[:, None]
    qs = q0 + v0 * ts + a3 * ts ** 3 + a4 * ts ** 4 + a5 * ts ** 5
    return Traj(name, ts[:, 0], qs, **cmd)


def quintic_time(q0, v0, q1, v1, T_min=0.3, vmax=VMAX, amax=AMAX):
    T = max(T_min, duration(np.array([q0, q1]), vmax=vmax, amax=amax))
    for _ in range(25):
        tr = quintic("x", q0, v0, q1, v1, T); qd = np.diff(tr.q, axis=0) / DT_S; qdd = np.diff(qd, axis=0) / DT_S
        if np.all(np.abs(qd).max(0) <= vmax * 1.02) and np.all(np.abs(qdd).max(0) <= amax * 1.05): return T
        T *= 1.12
    return T


# --- F7: the fast belt (Factory v2, 2026-11-03; docs/sorting-cell-spec.md §4 row F7) ----------------------
# At 0.5 m/s an item crosses the 1.1 m reach window in 2.2 s, and the v1 intercept needs ~0.6 s of approach plus ~1.4 s
# of tracking (descend, close, lift, brake) during which the item must stay in the window. Each change below is a flag
# of `Line(v2=...)`, measured on its own (factory/ops/f7.py); v1 is `v2=()`.
#   fast      the approach and the retreat use joint limits nearer the UR5e's rated speeds (base/shoulder/elbow
#             3.14 rad/s, wrists 6.28 rad/s; v1 used 2.4 / 3.0) and a faster descent and lift
#   early     the intercept is searched from 0.4 s after the decision (v1 0.6 s) in 0.15 s steps (v1 0.35 s), the
#             tracking phase is shorter (close 0.30 s, brake 0.2 s; v1 0.40, 0.3), a track is a candidate after 5
#             detections (v1 10: at 0.5 m/s ten frames are 16 cm of belt, and the overhead camera first sees an item
#             only 11 cm inside the window), and the robot waits over x = -0.30 m (v1 -0.15), so items are met further
#             upstream (measured 2026-11-03: with vision, tracks confirmed at x = -0.29 m and the deadline at -0.17 m
#             left a 12 cm slot)
#   deadline  the scheduler's horizons follow the belt speed: an item is a candidate only while a pick can still finish
#             in the window, and lookahead counts the items still catchable after this pick (time to the next grasp
#             from the measured approach, not a fixed 2 s)
VMAX_FAST = np.array([3.0, 3.0, 3.1, 4.5, 4.5, 4.5]); AMAX_FAST = np.array([12.0, 12.0, 15.0, 18.0, 18.0, 18.0])
V2_FLAGS = ("fast", "early", "deadline")


# --- the intercept planner ----------------------------------------------------------------------------------------
@dataclass
class LinePlan:
    name: str
    tool: str
    trajs: list
    t_start: float
    t_grasp: float           # when the pads close / the cup seals (absolute sim time)
    t_done: float            # when the retreat ends (the robot is free)
    x_grasp: float
    p_pred: np.ndarray       # predicted item position at t_grasp
    chute: object
    meta: dict = field(default_factory=dict)


class LinePlanner:
    """Plans a pick of a moving item from the robot's current (resting) configuration."""
    TILTS = (0.0, 0.35, -0.35)

    def __init__(self, cell, planner: Planner, h_pre=0.06, v_desc=0.3, t_close=0.40, t_seal=0.20, h_clear=0.06, t_lift=0.35, t_brake=0.3, decision_s=0.1, max_horizon=2.5, v_carry=1.0,
                 vmax=VMAX, amax=AMAX, t_lo_min=0.6, scan_step=0.35, n_scan=6):
        self.c = cell; self.p = planner; self.arm = planner.arm; self.lay = cell.lay; self.t_brake = t_brake; self.max_horizon = max_horizon
        self.vmax, self.amax, self.t_lo_min, self.scan_step, self.n_scan = vmax, amax, t_lo_min, scan_step, n_scan
        self.v_carry = v_carry   # tool speed limit while carrying (F1: 0.6 m/s); the UR5e's tool goes to 1 m/s and more
        self.h_pre, self.v_desc, self.t_close, self.t_seal, self.h_clear, self.t_lift, self.decision_s = h_pre, v_desc, t_close, t_seal, h_clear, t_lift, decision_s

    def _reach(self, q0, p, R):
        """A cheaper IK than Planner.reach (planning runs every 50 ms while the robot waits): the seed, then two restarts."""
        ok, q, _ = self.arm.ik(p, R, q0=q0, restarts=2, iters=120, rng=np.random.default_rng(0))
        return q if ok else None

    def item_quat(self, yaw): return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])

    def carry_height(self, x0, x1, tracks, skip):
        """Bottom of a carried item must clear the rails by CLEAR and every other tracked item between x0 and x1."""
        b = self.lay.belt; z = b.surface_z + b.rail_height + PK.CLEAR; lo, hi = min(x0, x1) - 0.15, max(x0, x1) + 0.15
        for n, tr in tracks.items():
            if n == skip: continue
            xn = tr.predict(tr.t)[0][0]
            if lo <= xn <= hi: z = max(z, b.surface_z + self.c.items[n]["rec"]["extents"][2] + 0.03)
        return z

    def plan(self, name, track, t_now, q_now, tracks):
        rec = self.c.items[name]["rec"]; chute = self.lay.chute(rec["material"]); b = self.lay.belt
        t0 = t_now + self.decision_s
        p_ref, v = track.predict(t0); v3 = np.array([v[0], v[1], 0.0]); p_ref = np.array([p_ref[0], p_ref[1], b.surface_z])
        quat = self.item_quat(track.yaw); mass = rec["mass_kg"]; com_z = p_ref[2] + rec["com"][2]
        if v[0] > 1e-3 and p_ref[0] + v[0] * self.max_horizon < WINDOW[0]: return None, "early"   # not in reach within the horizon: no need to plan yet
        why = "no_grasp"
        R_now = self.arm.fk(q_now)[1]; yaw_now = math.atan2(R_now[1, 0], R_now[0, 0]) - math.pi / 2   # grasp_rot(yaw) has tool x at yaw + 90 deg
        wrap = lambda a: abs((a + math.pi) % (2 * math.pi) - math.pi)
        cands = self.p.candidates(name, p_ref, quat); cands = [c for c in cands if c[0] != "fingers"] + [c for c in cands if c[0] == "fingers"][:6]
        for tool, g in cands:
            fingers = tool == "fingers"
            for tilt in (self.TILTS if fingers else (0.0,)):
                for yaw in sorted((g["yaw"], g["yaw"] + math.pi), key=lambda y: wrap(y - yaw_now)):   # the wrist turn nearer the current one first
                    R = grasp_rot(yaw) @ rot_y(tilt); appr = R[:, 2]
                    d_g = np.asarray(g["pos"], float) - p_ref
                    if not fingers: d_g = d_g + np.array([0, 0, CUP_BEYOND_PINCH - 0.004])
                    hang = d_g[2]   # pinch above the item's bottom
                    a_carry = min(LIMITS["atool"], LIMITS["carry_moment"] / max(mass * (p_ref[2] + d_g[2] - com_z), 1e-3)) if fingers else LIMITS["atool"]
                    T_d = max(0.25, 1.875 * self.h_pre / (self.v_desc if fingers else self.v_desc * 0.7)); T_c = self.t_close if fingers else self.t_seal; T_l = self.t_lift
                    T_b = self.t_brake; T_trk = T_d + T_c + T_l + T_b
                    # the item must be inside the window for the whole tracking phase (descend, close, lift, brake)
                    item_x = lambda t: p_ref[0] + v[0] * (t - t0)
                    t_enter = t0 + (WINDOW[0] - p_ref[0]) / v[0] if v[0] > 1e-3 and p_ref[0] < WINDOW[0] else t0
                    # the intercept: scan descent start times from the earliest possible; for each, find whether the
                    # approach gets there in time (it is then stretched to arrive exactly at t_g). A later intercept puts
                    # the item nearer the robot: tall items entering the window are out of reach at its upstream edge.
                    feasible = False; q_pre = None; t_lo = max(t0 + self.t_lo_min, t_enter)
                    for k_try in range(self.n_scan):
                        t_try = t_lo + self.scan_step * k_try; T = t_try - t0; q_seed = q_now
                        if item_x(t_try + T_trk) > WINDOW[1] or T > self.max_horizon: break
                        for _ in range(6):
                            t_g = max(t0 + T, t_try)
                            p_pre = p_ref + v3 * (t_g - t0) + d_g - appr * self.h_pre
                            q_pre = self._reach(q_seed, p_pre, R)
                            if q_pre is None: break
                            q_seed = q_pre
                            J = self.arm.jacobian(q_pre); qd_pre = np.linalg.lstsq(J, np.r_[v3, 0, 0, 0], rcond=None)[0]
                            T_req = quintic_time(q_now, np.zeros(6), q_pre, qd_pre, vmax=self.vmax, amax=self.amax)
                            if T_req <= t_g - t0 + 1e-3: feasible = True; break
                            T = T_req * 1.03
                            if T > self.max_horizon: break
                        if feasible: break
                    if not feasible: why = "ik_fail:pre" if q_pre is None else "late"; continue
                    T_app = t_g - t0
                    if item_x(t_g + T_trk) > WINDOW[1]: why = "late"; continue
                    if T_app > self.max_horizon: why = "early"; continue   # too far ahead to predict well: decide again later
                    # the tracking phase, sampled every 20 ms and solved by IK on the predicted item position: the tool
                    # moves with the belt while it descends, closes and lifts the item clear, then brakes to a stop
                    taus = np.append(np.arange(0.0, T_trk, 0.02), T_trk); qs = [q_pre]; ok = True; q = q_pre
                    for tau in taus[1:]:
                        off = np.zeros(3); tb = tau   # tb: how far (in belt-seconds) the grasp point has been carried
                        if tau < T_d: off = -appr * self.h_pre * (1 - minjerk(tau / T_d))
                        elif tau > T_d + T_c + T_l:
                            s = (tau - T_d - T_c - T_l) / T_b; off = np.array([0, 0, self.h_clear + 0.02 * minjerk(s)]); tb = T_d + T_c + T_l + T_b * (s - s * s / 2)
                        elif tau > T_d + T_c: off = np.array([0, 0, self.h_clear * minjerk((tau - T_d - T_c) / T_l)])
                        p = p_ref + v3 * (t_g - t0 + tb) + d_g + off
                        okk, q, _ = self.arm.ik(p, R, q0=q, restarts=0, iters=60, tol_pos=5e-4)
                        if not okk: ok = False; break
                        qs.append(q)
                    if not ok: why = "ik_fail:track"; continue
                    qs = np.array(qs); i_d = int(np.searchsorted(taus, T_d)); i_c = int(np.searchsorted(taus, T_d + T_c))
                    p_lift = self.arm.fk(qs[-1])[0]
                    # carry: rise clear of the other items, cross to the chute turning the wrist (the item stays upright), lower, drop, retreat
                    yd = 0.0 if math.cos(yaw) >= 0 else math.pi; R_drop = rot_z(yd - yaw) @ R
                    carry_z = max(self.carry_height(p_lift[0], chute.x, tracks, name) + hang, b.surface_z + b.rail_height + PK.CLEAR + hang)
                    p_rise = np.array([p_lift[0], p_lift[1], max(carry_z, p_lift[2])]); drop_y = chute.y - 0.03
                    p_over = np.array([chute.x, drop_y, p_rise[2]]); p_drop = np.array([chute.x, drop_y, chute.rim_z + 0.04 + hang])
                    rise = self.p.line(qs[-1], p_lift, p_rise, R) if p_rise[2] - p_lift[2] > 0.005 else qs[-1:]
                    if rise is None: why = "ik_fail:rise"; continue
                    across = self.p.line(rise[-1], p_rise, p_over, R, n=max(10, int(np.linalg.norm(p_over - p_rise) / 0.02)), R1=R_drop)
                    if across is None: why = "ik_fail:across"; continue
                    lower = self.p.line(across[-1], p_over, p_drop, R_drop)
                    if lower is None: why = "ik_fail:lower"; continue
                    retreat = self.p.line(lower[-1], p_drop, p_over, R_drop)
                    if retreat is None: why = "ik_fail:retreat"; continue
                    T = lambda path, L, vt=self.v_carry, at=a_carry: duration(path, L, vt, at, VMAX, AMAX)
                    cmd_open = dict(grip=0.0, suction=0.0, extend=0.0 if fingers else SUCTION_STROKE)
                    trajs = [hold("decide", q_now, self.decision_s, **cmd_open), quintic("approach", q_now, np.zeros(6), q_pre, qd_pre, T_app)]
                    trajs.append(Traj("descend", taus[:i_d + 1], qs[:i_d + 1]))
                    trajs.append(Traj("close", taus[i_d:i_c + 1] - taus[i_d], qs[i_d:i_c + 1], **(dict(grip=GRIP_CLOSED) if fingers else dict(suction=1.0))))
                    trajs.append(Traj("lift", taus[i_c:] - taus[i_c], qs[i_c:]))
                    if len(rise) > 1: trajs.append(from_segment(Segment("rise", rise, T(rise, np.linalg.norm(p_rise - p_lift)))))
                    trajs.append(from_segment(Segment("transfer", across, T(across, np.linalg.norm(p_over - p_rise)))))
                    trajs.append(from_segment(Segment("lower", lower, T(lower, np.linalg.norm(p_over - p_drop)))))
                    trajs.append(hold("release", lower[-1], 0.25, **(dict(grip=0.0) if fingers else dict(suction=0.0))))
                    trajs.append(from_segment(Segment("retreat", retreat, duration(retreat, np.linalg.norm(p_over - p_drop), self.v_carry, LIMITS["atool"], self.vmax, self.amax))))
                    total = sum(tr.T for tr in trajs)
                    return LinePlan(name, tool, trajs, t_now, t_g + T_d, t_now + total, float(item_x(t_g + T_d)), p_ref + v3 * (t_g + T_d - t0), chute,
                                    dict(tilt=round(math.degrees(tilt), 1), a_carry=round(a_carry, 2), carry_z=round(float(carry_z), 3), T_app=round(T_app, 3), v=v3,
                                         T_trk=round(T_trk, 3))), None
        return None, why


# --- the line -----------------------------------------------------------------------------------------------------
class Line:
    def __init__(self, speed=0.3, rate_per_min=8.0, scheduler="lookahead", seed=0, min_gap=0.25, y_jitter=0.06, items=None,
                 detector: DetectorCfg = None, cell=None, max_attempts=2, log=None, inject=None, perception="privileged", vision_cfg=None, clutter=0.0, v2=()):
        from cell_mujoco import CellMJ
        self.c = cell or CellMJ(); self.lay = self.c.lay; self.speed = speed; self.rate = rate_per_min; self.sched = scheduler
        self.rng = np.random.default_rng(seed); self.min_gap = min_gap; self.y_jitter = y_jitter; self.max_attempts = max_attempts
        self.pool = list(items or self.c.items); self.detcfg = detector or DetectorCfg(); self.log = log
        self.clutter = clutter   # L3 (F5): the fraction of arrivals that come as two items touching side by side
        self.perception = perception; self.vision_cfg = vision_cfg; self.vision = None   # "vision": F4's overhead camera replaces the privileged Detector (L4)
        self.inject = dict(inject or {})   # planted faults for the safety tests: {attempt number: "drop" | "shove" | "dive"}
        self.v2 = tuple(v2 or ()); assert set(self.v2) <= set(V2_FLAGS), self.v2
        kw = {}
        if "fast" in self.v2: kw.update(vmax=VMAX_FAST, amax=AMAX_FAST, v_desc=0.45, t_lift=0.28)
        if "early" in self.v2: kw.update(t_lo_min=0.3, decision_s=0.05, scan_step=0.15, n_scan=14, t_close=0.30, t_brake=0.2)
        self.min_confirm = 5 if "early" in self.v2 else 10; ready_x = -0.30 if "early" in self.v2 else -0.15
        self.planner = Planner(self.c); self.lp = LinePlanner(self.c, self.planner, **kw)
        self.ex = PK.Executor(self.c)   # for its contact sets, servo constants and slip measure
        q_ready_ok, q_ready, _ = self.planner.arm.ik(np.array([ready_x, 0.42, 1.12]), grasp_rot(0.0), q0=np.array(self.lay.home_qpos))   # over the upstream half of the zone
        assert q_ready_ok or ready_x == -0.15, "the v2 ready pose is out of reach"
        self.q_ready = q_ready if q_ready_ok else np.array(self.lay.home_qpos)

    # bookkeeping
    def _rec(self, name): return self.items[name]

    def reset(self):
        c = self.c; c.reset(); c.belt(self.speed); self.det = Detector(c, self.detcfg, rng=np.random.default_rng(self.rng.integers(1 << 30)))
        if self.perception == "vision":
            sys.path.insert(0, os.path.join(FACTORY, "vision")); from vision import VisionDetector
            if self.vision is None: self.vision = VisionDetector(c, self.vision_cfg, belt_speed=self.speed)   # one renderer per process
            self.vision.__init__(c, self.vision_cfg, belt_speed=self.speed, cam=self.vision.cam, model=self.vision.id); self.det = self.vision
        self.trk = Tracker(v_prior=(self.speed, 0.0), v_sigma=0.02)   # the belt encoder's speed as the prior
        self.items = {n: dict(state="parked") for n in c.items}; self.log_items = []; self.events = []
        c.data.qpos[c.robot_qpos] = self.q_ready; c.data.ctrl[c.robot_act] = self.q_ready; mujoco.mj_forward(c.model, c.data)
        self.n_attempt = 0; self.max_track_err = 0.0; self.q_ref = self.q_ready.copy(); self.queue = []; self.cur = None; self.k_traj = 0; self.tau = 0.0; self.target = None; self.next_arrival = 1.0 / max(self.speed, 0.05) * 0.5
        self.q_hold = self.q_ready.copy(); self.recovering = False

    def _spawn(self, t):
        free = [n for n in self.pool if self.items[n]["state"] == "parked"]
        upstream = [self.c.item_state(n)["pos"][0] for n in self.pool if self.items[n]["state"] in ("belt",)]
        if not free or (upstream and min(upstream) < self.lay.spawn_x + self.min_gap): return False
        n = free[int(self.rng.integers(len(free)))]
        y0 = self.lay.belt.y + float(self.rng.uniform(-self.y_jitter, self.y_jitter)); yaw0 = float(self.rng.uniform(-math.pi, math.pi))
        self.c.spawn(n, y=y0, yaw=yaw0, rng=self.rng)
        self.items[n] = dict(state="belt", spawn_t=round(t, 3), attempts=0, held=False, name=n, material=self.c.items[n]["rec"]["material"])
        if self.clutter > 0 and self.rng.random() < self.clutter and len(free) > 1:   # a second item touching the first, just downstream of it
            m = [f for f in free if f != n][int(self.rng.integers(len(free) - 1))]; yaw1 = float(self.rng.uniform(-math.pi, math.pi))
            hx = lambda name, yaw: 0.5 * (abs(self.c.items[name]["rec"]["extents"][0] * math.cos(yaw)) + abs(self.c.items[name]["rec"]["extents"][1] * math.sin(yaw)))
            self.c.spawn(m, x=self.lay.spawn_x + hx(n, yaw0) + hx(m, yaw1) + 0.002, y=y0 + float(self.rng.uniform(-0.02, 0.02)), yaw=yaw1, rng=self.rng)
            self.items[m] = dict(state="belt", spawn_t=round(t, 3), attempts=0, held=False, name=m, material=self.c.items[m]["rec"]["material"], clutter=True); self.items[n]["clutter"] = True
        return True

    def _finish(self, name, outcome, t, extra=None):
        it = self.items[name]; it.update(state="parked", outcome=outcome, end_t=round(t, 3), **(extra or {})); self.log_items.append(dict(it))
        self.c.park(name, list(self.c.items).index(name)); self.trk.drop(name)
        if self.log: self.log(f"{t:7.2f}  {name:16s} -> {outcome}")

    def _lifecycle(self, t):
        b = self.lay.belt
        for n, it in self.items.items():
            if it["state"] not in ("belt", "held", "settling"): continue
            s = self.c.item_state(n); p = s["pos"]
            if it["state"] == "settling":
                if t - it["settle_t"] > 0.6:
                    where, ch, com = classify(self.c, n, it["material"])
                    if where == "success": self._finish(n, "sorted", t)
                    elif where == "wrong_chute": self._finish(n, "wrong_chute", t, dict(landed_in=ch))
                    else: self._finish(n, "dropped" if it["held"] else "knocked_off", t, dict(where="chute rim"))
                continue
            if it["state"] == "held": continue
            for ch in self.lay.chutes:
                if abs(p[0] - ch.x) < ch.inner / 2 + 0.01 and abs(p[1] - ch.y) < ch.inner / 2 + 0.01 and p[2] < ch.rim_z:
                    it["state"] = "settling"; it["settle_t"] = t; break
            if it["state"] == "settling": continue
            if p[0] > b.x_max + 0.05 or p[2] < b.surface_z - 0.25:
                if p[0] > b.x_max: self._finish(n, "missed" if it["attempts"] == 0 else "late", t)
                else: self._finish(n, "dropped" if it["held"] else "knocked_off", t)

    def _candidates(self, t):
        conf = self.trk.confirmed(min_n=self.min_confirm)
        return {n: tr for n, tr in conf.items() if self.items.get(n, {}).get("state") == "belt" and self.items[n]["attempts"] < self.max_attempts and t >= self.items[n].get("retry_at", 0.0)
                and tr.predict(t)[0][0] < self._last_x() and self._tilt(n, tr) < 20}

    def _last_x(self):
        """The most downstream position at which an item is still worth planning for. v1: 0.15 m short of the window's
        end. deadline: where a pick decided now can still finish its tracking phase inside the window (the shortest
        approach the planner allows plus the shortest tracking phase, at the belt speed)."""
        if "deadline" not in self.v2: return WINDOW[1] - 0.15
        lp = self.lp; t_min = lp.decision_s + lp.t_lo_min + max(0.25, 1.875 * lp.h_pre / lp.v_desc) + min(lp.t_close, lp.t_seal) + lp.t_lift + lp.t_brake
        return WINDOW[1] - self.speed * t_min

    def _tilt(self, n, tr):
        """Is the item standing? Privileged: its true tilt. Vision: the detector's estimate (top height vs the class's)."""
        return getattr(tr, "tilt", 0.0) if self.perception == "vision" else self.c.item_state(n)["tilt_deg"]

    def _choose(self, t, q_now):
        cands = self._candidates(t)
        if not cands: return None
        cands = dict(sorted(cands.items(), key=lambda kv: -kv[1].predict(t)[0][0])[:4])   # plan for the 4 most downstream at most
        plans = []
        for n, tr in cands.items():
            pl, why = self.lp.plan(n, tr, t, q_now, self.trk.tracks)
            if pl is not None: plans.append(pl)
            else: self.items[n]["last_why"] = why; self.items[n]["retry_at"] = t + (0.25 if why != "early" else 0.1)
        if not plans: return None
        if self.sched == "greedy":       # the nearest job first: whatever finishes soonest
            return min(plans, key=lambda p: p.t_done)
        if self.sched == "edf":          # earliest deadline first among the feasible: the most downstream item
            return max(plans, key=lambda p: p.p_pred[0])
        # lookahead: prefer the pick after which the most other candidates can still be caught (each needs ~2 s from
        # the moment the robot is free); ties go to the earliest finish
        def keeps(pl):
            if "deadline" in self.v2:   # still catchable after this pick: its grasp can start before the item's last moment
                need = 0.6 + pl.meta.get("T_trk", 1.4)
                return sum(1 for n, tr in cands.items() if n != pl.name and tr.predict(pl.t_done + need)[0][0] <= WINDOW[1])
            return sum(1 for n, tr in cands.items() if n != pl.name and tr.predict(pl.t_done + 2.0)[0][0] <= WINDOW[1])
        return max(plans, key=lambda p: (keeps(p), -p.t_done))

    def _start(self, plan, t):
        self.cur = plan; self.k_traj = 0; self.tau = 0.0; self.items[plan.name]["attempts"] += 1; self.held = None; self.n_attempt += 1
        self.delta = np.zeros(2); self.delta_frozen = None
        self.attempt = dict(item=plan.name, tool=plan.tool, t_start=round(t, 3), t_grasp_plan=round(plan.t_grasp, 3), x_grasp=round(plan.x_grasp, 3), **{k: v for k, v in plan.meta.items() if k != "v"})
        self._apply(plan.trajs[0])

    def _apply(self, tr):
        c = self.c; d = c.data
        if tr.grip is not None: d.ctrl[c.grip_act] = tr.grip
        if tr.suction is not None: c.vacuum(tr.suction > 0.5)
        if tr.extend is not None: d.ctrl[self.ex.act_ext] = tr.extend
        fault = None if self.recovering else self.inject.get(self.n_attempt)
        if fault == "drop" and tr.name == "transfer": d.ctrl[c.grip_act] = 0.0; c.vacuum(False)            # the gripper lets go mid-carry
        if fault == "shove" and tr.name == "close":                                                         # something bumps the item as the pads arrive
            q = c.items[self.cur.name]["qpos"]; d.qpos[q + 1] += 0.06; mujoco.mj_forward(c.model, d)

    def _abort(self, t, why, detail=None):
        """Recovery: open, rise, back to ready. Collisions first freeze (a protective stop)."""
        c = self.c; d = c.data; q = d.qpos[c.robot_qpos].copy(); name = self.cur.name if self.cur else None
        self.attempt.update(outcome=why, t_end=round(t, 3), detail=detail); self.events.append(self.attempt)
        if self.log: self.log(f"{t:7.2f}  {name:16s} attempt failed: {why} {detail or ''}")
        if name and self.items[name]["state"] == "held": self.items[name]["state"] = "belt"
        trajs = []
        if why == "collision": trajs.append(hold("protective_stop", q, 0.5, grip=0.0, suction=0.0))
        else: trajs.append(hold("open", q, 0.15, grip=0.0, suction=0.0))
        p, R = self.planner.arm.fk(q); up = self.planner.line(q, p, p + np.array([0, 0, 0.12]), R)
        q1 = up[-1] if up is not None else q
        if up is not None: trajs.append(from_segment(Segment("rise", up, duration(up, 0.12, vmax=VMAX, amax=AMAX))))
        trajs.append(quintic("to_ready", q1, np.zeros(6), self.q_ready, np.zeros(6), quintic_time(q1, np.zeros(6), self.q_ready, np.zeros(6), 0.6), extend=0.0))
        self.cur = LinePlan(name, "none", trajs, t, t, t + sum(x.T for x in trajs), 0.0, np.zeros(3), None, dict(recovery=why))
        self.k_traj = 0; self.tau = 0.0; self.recovering = True; self._apply(trajs[0])

    def _check(self, tr, t):
        """At the end of a trajectory: did the grasp happen, did the item come up."""
        c = self.c; n = self.cur.name; s = c.item_state(n)
        if tr.name == "descend":   # the intercept: where the item is when the pads arrive, against where the plan aimed (+ the correction)
            self.attempt["pred_err_mm"] = round(1000 * float(np.linalg.norm(s["pos"][:2] - self.cur.p_pred[:2])), 1)
            self.attempt["grasp_err_mm"] = round(1000 * float(np.linalg.norm(s["pos"][:2] - self.cur.p_pred[:2] - self.delta)), 1)
        if tr.name == "close":   # knocked: the item is not where the tool went (plan + correction, moved with the belt) or it tipped
            pred = self.cur.p_pred[:2] + self.cur.meta["v"][:2] * (t - self.cur.t_grasp) + self.delta
            off = float(np.linalg.norm(s["pos"][:2] - pred))
            if s["tilt_deg"] > 20 or off > 0.03: return "knocked", dict(tilt=round(s["tilt_deg"], 1), off=round(off, 3))
            if self.cur.tool == "suction" and c.vac_item != n: return "miss", dict(why="no seal")
            R = quat_mat(s["quat"]); Rt = c.data.site_xmat[c.pinch_site].reshape(3, 3)
            self.held = (R.T @ (c.pinch_pos() - s["pos"]), Rt.T @ R); self.items[n]["state"] = "held"; self.items[n]["held"] = True
        if tr.name == "lift" and s["pos"][2] < self.lay.belt.surface_z + 0.02: return "miss", dict(rise=round(float(s["pos"][2] - self.lay.belt.surface_z), 3))
        if tr.name == "release":
            self.items[n]["state"] = "belt"; self.held = None   # in flight: the lifecycle sees it land in a chute (or not)
        return None, None

    def step_robot(self, t):
        c = self.c; d = c.data
        if self.cur is None:
            if t >= getattr(self, "next_decision", 0.0):
                self.next_decision = t + 0.05
                plan = self._choose(t, d.qpos[c.robot_qpos].copy())
                if plan is not None: self._start(plan, t)
            if self.cur is None and np.max(np.abs(self.q_hold - self.q_ready)) > 0.05:   # nothing to do: wait over the zone, not over a chute
                q = self.q_hold; T = quintic_time(q, np.zeros(6), self.q_ready, np.zeros(6), 0.5)
                self.cur = LinePlan(None, "none", [quintic("to_ready", q, np.zeros(6), self.q_ready, np.zeros(6), T, extend=0.0)], t, t, t + T, 0.0, np.zeros(3), None, dict(park=True))
                self.k_traj = 0; self.tau = 0.0; self.recovering = True; self._apply(self.cur.trajs[0])
            if self.cur is None:
                d.ctrl[c.robot_act] = self.q_hold; d.qfrc_applied[c.robot_dof] = d.qfrc_bias[c.robot_dof]; return
        tr = self.cur.trajs[self.k_traj]; q_ref, qd_ref = tr.at(self.tau)
        if not self.recovering and self.cur.name in self.trk.tracks and tr.name in ("approach", "descend", "close", "lift"):
            q_ref = q_ref + self._conveyor_correction(tr, t, q_ref)
        if not self.recovering and self.inject.get(self.n_attempt) == "dive" and tr.name in ("descend", "close"):
            q_ref = q_ref + np.array([0, 0.6 * min(1.0, self.tau / 0.2), 0, 0, 0, 0])   # a bad reference drives the tool down into the belt
        d.ctrl[c.robot_act] = q_ref + self.ex.kv_over_kp * qd_ref; self.q_ref = q_ref; d.qfrc_applied[c.robot_dof] = d.qfrc_bias[c.robot_dof]
        self.tau += c.model.opt.timestep

    def _conveyor_correction(self, tr, t, q_ref):
        """Conveyor tracking closes the loop: the plan aimed at a prediction made when the pick was decided; while the
        tool approaches and descends, the newest track estimate shifts the reference by (estimate - plan), ramped in over
        the approach's last 0.5 s, held through the grasp and ramped out during the lift (the item is in hand then)."""
        if tr.name == "lift":
            if self.delta_frozen is None: self.delta_frozen = self.delta.copy()
            self.delta = self.delta_frozen * max(0.0, 1.0 - self.tau / max(tr.T * 0.5, 1e-3))
        else:
            est = self.trk.tracks[self.cur.name].predict(t)[0][:2]; v = self.cur.meta["v"][:2]
            plan_pos = self.cur.p_pred[:2] + v * (t - self.cur.t_grasp)
            w = max(0.0, 1.0 - (tr.T - self.tau) / 0.5) if tr.name == "approach" else 1.0
            if tr.name == "close" and self.delta_frozen is None and self.tau > 0.1: self.delta_frozen = self.delta.copy()   # the pads are on: stop chasing
            self.delta = self.delta_frozen if self.delta_frozen is not None else w * np.clip(est - plan_pos, -0.05, 0.05)
        if not np.any(self.delta): return 0.0
        J = self.planner.arm.jacobian(q_ref); return np.linalg.lstsq(J, np.r_[self.delta, 0, 0, 0, 0], rcond=None)[0]

    def after_step(self, t, k):
        c = self.c
        if self.cur is None: return
        tr = self.cur.trajs[self.k_traj]
        if not self.recovering and k % 5 == 0:
            hit = self.ex.robot_fixture_contact()
            if hit: return self._abort(t, "collision", f"{hit[0]} x {hit[1]}")
            # the second collision detector, as on the UR controller: a joint far behind its reference means something is
            # pushing back (the UR's protective stop watches joint torque against its model; tracking error is our proxy)
            err = float(np.max(np.abs(c.data.qpos[c.robot_qpos] - self.q_ref))); self.max_track_err = max(self.max_track_err, err)
            if err > TRACK_STOP: return self._abort(t, "collision", f"tracking error {err:.3f} rad (contact force)")
            if self.held is not None and tr.name in ("lift", "rise", "transfer", "lower"):
                slip, swing = self.ex._slip_swing(self.cur.name, self.held)
                if slip > 0.03 or (self.cur.tool == "suction" and c.vac_item != self.cur.name):
                    self.items[self.cur.name]["state"] = "belt"; return self._abort(t, "drop", dict(slip=round(slip, 3)))
        if self.tau >= tr.T:
            if not self.recovering:
                why, detail = self._check(tr, t)
                if why: return self._abort(t, why, detail)
            self.k_traj += 1; self.tau = 0.0
            if self.k_traj >= len(self.cur.trajs):
                if not self.recovering: self.attempt.update(outcome="ok", t_end=round(t, 3), cycle_s=round(t - self.attempt["t_start"], 2)); self.events.append(self.attempt)
                self.q_hold = self.cur.trajs[-1].q[-1].copy(); self.cur = None; self.recovering = False
            else: self._apply(self.cur.trajs[self.k_traj])

    def run(self, seconds=180.0, on_frame=None, frame_every=17, warmup=0.0):
        self.reset(); c = self.c; dt = c.model.opt.timestep; k = 0; t0 = time.perf_counter()
        while c.data.time < seconds:
            t = c.data.time
            if t >= self.next_arrival:
                if self._spawn(t): self.next_arrival = t + max(self.min_gap / max(self.speed, 0.02), float(self.rng.exponential(60.0 / self.rate)))
                else: self.next_arrival = t + 0.1
            names = [n for n, it in self.items.items() if it["state"] in ("belt", "held")]
            self.trk.feed(self.det.poll(t, names), t)
            self.step_robot(t); c.step(); k += 1
            self.after_step(c.data.time, k)
            if k % 10 == 0: self._lifecycle(c.data.time)
            if on_frame and k % frame_every == 0: on_frame(self)
        return self.summary(seconds, time.perf_counter() - t0)

    def summary(self, seconds, wall):
        from collections import Counter
        done = self.log_items; cnt = Counter(i["outcome"] for i in done); n = len(done)
        att = Counter(e["outcome"] for e in self.events); ok = [e for e in self.events if e["outcome"] == "ok"]
        S = dict(speed=self.speed, rate_per_min=self.rate, scheduler=self.sched, seconds=seconds, wall_s=round(wall, 1),
                 items_done=n, outcomes=dict(cnt), sorted_per_min=round(cnt["sorted"] / (seconds / 60.0), 2), arrivals_per_min=round(n / (seconds / 60.0), 2),
                 sort_rate=round(cnt["sorted"] / max(n, 1), 3), miss_rate=round(1 - cnt["sorted"] / max(n, 1), 3),
                 attempts=dict(att), attempt_success=round(len(ok) / max(len(self.events), 1), 3),
                 cycle_s=round(float(np.mean([e["cycle_s"] for e in ok])), 2) if ok else None,
                 grasp_err_mm=dict(mean=round(float(np.mean([e["grasp_err_mm"] for e in self.events if "grasp_err_mm" in e])), 1),
                                   max=round(float(np.max([e["grasp_err_mm"] for e in self.events if "grasp_err_mm" in e])), 1)) if any("grasp_err_mm" in e for e in self.events) else None,
                 collisions=att.get("collision", 0), max_track_err_rad=round(self.max_track_err, 4))
        return dict(summary=S, items=done, attempts=self.events)
