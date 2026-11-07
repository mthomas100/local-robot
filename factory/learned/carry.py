"""carry.py — F3: the learned carry (2026-11-02; discipline 10; docs/sorting-cell-spec.md §4 F3).

F2 spends half of each 5.4 s cycle carrying the held item to its chute: rise, cross, lower to the rim, open, wait,
retreat. It is planned open-loop and cautiously, because a bottle held by its neck swings like a pendulum (the
load-dependent acceleration limit of F1). F3 replaces that part with a closed-loop *carry policy*:

    every 20 ms it sees  the vector to the drop point, the tool's velocity, how far the item has swung in the hand
                         and how fast, where the item would land if let go now, its mass, how far it hangs, the tool
    and commands         the tool's acceleration (3) and whether to let go (1)

so it can accelerate harder when the item is not swinging, damp the swing when it is, and *let go in flight* over the
chute instead of stopping, lowering and waiting. The policy is a residual on a hand-written prior (a PD move to the
drop point plus a ballistic release rule: `Policy.zero()`): the prior is the warm start, learning adjusts it (residual
policy learning: Johannink et al., "Residual Reinforcement Learning for Robot Control", ICRA 2019; Silver et al., "Residual Policy Learning", 2018). A classical **safety shield** wraps the policy: the item
must clear the belt rail before it may move sideways or be released, the tool stays within 8 cm of the box spanned by
its start and the drop point, tool speed is capped at 1.5 m/s and every joint at the line's limits, and the orientation turns the item upright for the chute as in F2.

The same `CarryCtl` runs in training (`episode`, from snapshots of real F2 grasps) and on the line (`line_learned.py`).
"""
from __future__ import annotations
import math, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE)
for p in (HERE, os.path.join(FACTORY, "line"), os.path.join(FACTORY, "classical"), os.path.join(FACTORY, "cell")):
    if p not in sys.path: sys.path.insert(0, p)
from pick import minjerk, quat_mat, classify  # noqa: E402
import line as LN  # noqa: E402

TICK = 0.02                 # policy rate 50 Hz
G = 9.81
V_TOOL_MAX = 1.5            # m/s, shield
A_MAX_CAP = 12.0            # m/s^2, shield (the line's joint limits bind first on long moves)
HELD_OUT = ("windex_bottle", "sugar_box", "tuna_fish_can")   # one per material, one on the suction cup; never trained on
NF = 17                     # features (16 until run v4 added the item's height; older 66-number policies load with a zero weight on it)
NA = 4                      # actions: tool acceleration xyz, release logit
NP = NA * NF + 2            # parameters: the residual matrix and two scalars (gain, acceleration-limit scale)
RES = 0.5                   # the residual's authority: at most half the acceleration limit (run 1 with 1.0 overshot held-out bottles)
BOX = 0.08                  # shield: the tool stays within this of the box spanned by the start and the goal (m)
TALL_LOWER = bool(int(os.environ.get("CARRY_TALL_LOWER", "0")))   # experiment (F3): lower items taller than 18 cm to the rim before release
SUCCESS_R = 25.0            # return for a sorted item; minus the seconds until the robot is free
FEATURES = ("to goal x", "to goal y", "to goal z", "vel x", "vel y", "vel z", "swing x", "swing y", "swing rate x", "swing rate y",
            "landing margin", "speed", "mass", "hang", "suction", "height", "bias")


class Policy:
    """theta = [W (4 x 16) row-major, log gain scale, log acceleration-limit scale]; zero = the hand-written prior."""

    def __init__(self, theta=None):
        th = np.zeros(NP) if theta is None else np.asarray(theta, float).copy()
        if len(th) == NA * 16 + 2:   # a policy from before the height feature: insert a zero column before the bias
            W = th[:NA * 16].reshape(NA, 16); th = np.concatenate([np.insert(W, 15, 0.0, axis=1).reshape(-1), th[-2:]])
        assert len(th) == NP, len(th)
        self.theta = th

    @classmethod
    def zero(cls): return cls()

    @property
    def W(self): return self.theta[:NA * NF].reshape(NA, NF)

    @property
    def k(self): return 6.0 * math.exp(float(np.clip(self.theta[-2], -1.0, 1.0)))

    def a_max(self, a_carry): return float(np.clip(a_carry * math.exp(float(np.clip(self.theta[-1], -1.0, 1.5))), 0.5, A_MAX_CAP))

    def to_json(self): return dict(theta=[round(float(x), 6) for x in self.theta], features=FEATURES)


def slerp(Ra, Rb, f):
    qa = np.zeros(4); qb = np.zeros(4); mujoco.mju_mat2Quat(qa, Ra.reshape(-1)); mujoco.mju_mat2Quat(qb, Rb.reshape(-1))
    if qa @ qb < 0: qb = -qb
    om = math.acos(min(1.0, abs(float(qa @ qb))))
    q = qa if om < 1e-6 else (math.sin((1 - f) * om) * qa + math.sin(f * om) * qb) / math.sin(om)
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q / np.linalg.norm(q)); return R.reshape(3, 3)


class CarryCtl:
    """The carry, from the end of the lift (item held, tool at rest above the belt) until the tool has stopped after
    letting go. `begin(q)` then `at(t)` once per physics step (returns the joint reference and its velocity)."""

    def __init__(self, cell, arm, policy: Policy, item, tool, carry_z, a_carry, R_drop, held, privileged=True, hang=None):
        """privileged=False (F4's student): nothing is read from the item's true state. The item is assumed rigid in the
        hand where the grasp put it (`hang`: pinch height above the item's bottom, from the robot's pose when it closed),
        the swing features read 0, the landing point is the rigid estimate's, and the item counts as clear of the
        fingers when it has had time to fall clear (free fall over its known height above the pinch)."""
        self.privileged = privileged; self.hang_given = hang; self.probe = None; self.teacher = None
        self.c = cell; self.arm = arm; self.pol = policy; self.item = item; self.tool = tool; self.carry_z = float(carry_z)
        self.a_carry = float(a_carry); self.R1 = R_drop; self.held = held
        rec = cell.items[item]["rec"]; self.rec = rec; self.chute = cell.lay.chute(rec["material"]); b = cell.lay.belt
        self.mass = float(rec["mass_kg"]); self.r_item = 0.5 * float(max(rec["extents"][0], rec["extents"][1]))
        self.rail_y = b.y - b.width / 2 - b.rail_thickness           # outer face of the near rail
        self.goal = np.array([self.chute.x, self.chute.y - 0.03, 0.0])
        self.released = False; self.t_release = None; self.t_stop = None; self.fail = None; self.log = []

    # -- state ----------------------------------------------------------------------------------------------
    def begin(self, q0):
        self.q0 = np.asarray(q0, float).copy(); p0, R0 = self.arm.fk(self.q0); self.p = p0.copy(); self.v = np.zeros(3); self.R0 = R0
        self.q_prev = self.q0.copy(); self.q_next = self.q0.copy(); self.t_tick = 0.0; self.k_tick = 0
        if self.privileged:
            s = self.c.item_state(self.item); com = s["pos"] + quat_mat(s["quat"]) @ np.asarray(self.rec["com"])
            self.hang = float(p0[2] - (s["pos"][2] + self.rec["bounds"][0][2]))  # pinch above the item's bottom
        else:   # rigid estimate: the item hangs straight below the pinch, its bottom `hang` below it
            self.hang = float(self.hang_given); com = p0 - np.array([0.0, 0.0, self.hang - (self.rec["com"][2] - self.rec["bounds"][0][2])])
        self.com_in_tool = R0.T @ (com - p0)                                  # where the item's centre sits in the hand at the start
        self.goal[1] = min(self.goal[1], self.rail_y - 0.025 - self.r_item)   # a wide item must hang clear of the rail there
        self.goal[2] = max(self.carry_z, p0[2]); self.p_start = p0.copy(); self.e0 = float(np.linalg.norm((self.goal - p0)[:2]))
        self.z_clear = self.carry_z                                           # pinch height while the item is over the belt
        self.z_chute = self.c.lay.chute(self.rec["material"]).rim_z + self.hang + 0.03
        self.tall = TALL_LOWER and float(self.rec["extents"][2]) > 0.18   # bottles and the cracker box
        self.last_com = com; self.started = True
        if self.probe is not None:   # distillation: what the teacher knows (true hang and where the item sits in the hand)
            st = self.c.item_state(self.item); com_t = st["pos"] + quat_mat(st["quat"]) @ np.asarray(self.rec["com"])
            self.hang_t = float(p0[2] - (st["pos"][2] + self.rec["bounds"][0][2])); self.com_in_tool_t = R0.T @ (com_t - p0)

    def _item(self, priv=False):
        if not (self.privileged or priv):   # rigid in the hand: the item's centre moves with the tool
            Rt = self.c.data.site_xmat[self.c.pinch_site].reshape(3, 3); return self.c.pinch_pos() + Rt @ self.com_in_tool, self.v.copy()
        s = self.c.item_state(self.item); R = quat_mat(s["quat"]); com = s["pos"] + R @ np.asarray(self.rec["com"])
        return com, s["linvel"]

    def features(self, priv=None):
        """The 17 features. priv=True on a student computes what the privileged teacher would see in the same state
        (for distillation: `probe`)."""
        priv = self.privileged if priv is None else priv
        if priv and not self.privileged: cit, hang = self.com_in_tool_t, self.hang_t; com, vel = self._item(True)
        else: cit, hang = self.com_in_tool, self.hang; com, vel = self._item()
        p, v = self.p, self.v; Rt = self.c.data.site_xmat[self.c.pinch_site].reshape(3, 3)
        rigid = self.c.pinch_pos() + Rt @ cit; d = (com - rigid)[:2]; dv = (vel - v)[:2]
        land, margin = self.landing(com, vel)
        e = self.goal - p
        f = np.array([e[0] / 0.5, e[1] / 0.5, e[2] / 0.5, v[0], v[1], v[2], d[0] / 0.05, d[1] / 0.05, dv[0] / 0.5, dv[1] / 0.5,
                      margin / 0.05, float(np.linalg.norm(v)), self.mass, hang / 0.15, 1.0 if self.tool == "suction" else 0.0,
                      float(self.rec["extents"][2]) / 0.3, 1.0])
        return f, margin, com

    def landing(self, com, vel):
        """Where the item's centre would cross the chute rim if let go now (ballistic), and the margin to the opening's
        edge (m, positive = inside with room for the item's half width)."""
        ch = self.chute; h = max(com[2] - ch.rim_z, 0.0); vz = float(vel[2])
        t_fall = (vz + math.sqrt(vz * vz + 2 * G * h)) / G
        land = com[:2] + vel[:2] * t_fall
        margin = min(ch.inner / 2 - abs(land[0] - ch.x), ch.inner / 2 - abs(land[1] - ch.y)) - 0.6 * self.r_item
        return land, float(margin)

    def over_belt(self, y_tool):
        """Is the held item's footprint still over the belt or its rail (the tool's y, the item hangs below it)?"""
        return y_tool + self.r_item > self.rail_y - 0.012

    # -- one policy step -------------------------------------------------------------------------------------
    def tick(self):
        pol = self.pol; f, margin, com = self.features(); a_max = pol.a_max(self.a_carry); k = pol.k
        if self.probe is not None and not self.released: self.probe.append((f, self.teacher.W @ self.features(True)[0]))
        u = pol.W @ f
        if not self.released:
            e = self.goal - self.p
            if self.tall and not self.over_belt(self.p[1]): e[2] = self.z_chute - self.p[2]   # prior: a tall item goes down to the rim before it is let go
            a = k * k * e - 2 * k * self.v + RES * a_max * np.tanh(u[:3])
            # shield: rise first, then cross; no lateral move while the item could still clip the rail
            if self.over_belt(self.p[1]) and self.p[2] < self.z_clear - 0.005:
                a[:2] = -self.v[:2] / TICK; a[2] = k * k * (self.z_clear - self.p[2]) - 2 * k * self.v[2]
            n = np.linalg.norm(a)
            if n > a_max: a *= a_max / n
            # release rule: the prior lets go when the predicted landing is well inside the opening and the tool slow
            logit = margin / 0.02 - float(np.linalg.norm(self.v)) / 0.5 - 1.0 + 3.0 * math.tanh(u[3])
            low_enough = (not self.tall) or self.p[2] < self.z_chute + 0.04
            if logit > 0 and low_enough and not self.over_belt(self.p[1]) and self.k_tick >= 3:
                self.released = True; self.t_release = self.t_tick
        if self.released:   # brake to a stop
            sp = float(np.linalg.norm(self.v)); a = -self.v / max(sp, 1e-9) * min(a_max, sp / TICK) if sp > 1e-6 else np.zeros(3)
        v1 = self.v + a * TICK; sp = float(np.linalg.norm(v1))
        if sp > V_TOOL_MAX: v1 *= V_TOOL_MAX / sp
        p1 = self.p + 0.5 * (self.v + v1) * TICK
        lo = np.minimum(self.p_start, self.goal)[:2] - BOX; hi = np.maximum(self.p_start, self.goal)[:2] + BOX   # shield: stay in the corridor
        clipped = np.clip(p1[:2], lo, hi); v1[:2] = np.where(clipped != p1[:2], 0.0, v1[:2]); p1[:2] = clipped
        zmin = self.z_clear if self.over_belt(p1[1]) else self.z_chute
        if p1[2] < zmin - 0.002 and not self.released: p1[2] = max(p1[2], min(zmin, self.p[2])); v1[2] = max(v1[2], 0.0)
        frac = minjerk(float(np.clip(1.0 - np.linalg.norm((self.goal - p1)[:2]) / max(self.e0, 1e-3), 0.0, 1.0)))
        R = slerp(self.R0, self.R1, frac)
        ok, q, info = self.arm.ik(p1, R, q0=self.q_next, restarts=0, iters=60, tol_pos=5e-4)
        if not ok:   # a hard step (a fast wrist turn): try longer before giving up
            ok2, q2, info2 = self.arm.ik(p1, R, q0=self.q_next, restarts=0, iters=400, tol_pos=5e-4)
            if ok2 or info2["pos_err"] + info2["rot_err"] < info["pos_err"] + info["rot_err"]: ok, q, info = ok2, q2, info2
        if not ok and info["pos_err"] > 0.005:
            # shield: an unreachable step holds the last reachable pose (an abort opens the gripper mid-air); only half a
            # second of them is a failure (found 2026-11-02: the learned carry overshooting the plastic chute with a bottle)
            self.n_ik_fail = getattr(self, "n_ik_fail", 0) + 1
            if self.n_ik_fail > 25: self.fail = self.fail or "ik"
            q = self.q_next.copy(); p1 = self.p.copy(); v1 = np.zeros(3)
        else: self.n_ik_fail = 0
        dq = np.clip(q - self.q_next, -LN.VMAX * TICK, LN.VMAX * TICK); q = self.q_next + dq
        if np.any(np.abs(dq) >= LN.VMAX * TICK * 0.999):   # a joint is at its speed limit: the tool goes where the joints put it
            p1 = self.arm.fk(q)[0]; v1 = (p1 - self.p) / TICK
        self.q_prev, self.q_next = self.q_next, q; self.p, self.v = p1, v1
        self.log.append((round(self.t_tick, 3), *np.round(p1, 4), *np.round(v1, 3), round(margin, 4), int(self.released)))
        self.t_tick += TICK; self.k_tick += 1
        if self.released and self.t_stop is None and float(np.linalg.norm(self.v)) < 0.02 and self.cleared(): self.t_stop = self.t_tick

    def cleared(self):
        """Has the let-go item fallen clear of the open fingers (its top 2 cm below the pads)? Until then the robot is not
        free: on the line, moving off at once yanked the fingers away while a tall bottle's neck was still between them
        and knocked it out of the chute (found 2026-11-02, F3 line evaluation)."""
        if not self.privileged:   # free fall from rest over (item height above the pinch + 2 cm)
            drop = float(self.rec["extents"][2]) - self.hang + 0.02; return self.t_tick - self.t_release > math.sqrt(2 * max(drop, 0.0) / G) + 0.03
        s = self.c.item_state(self.item); top = s["pos"][2] + self.rec["bounds"][1][2]
        return top < self.c.pinch_pos()[2] - 0.02

    def at(self, t):
        while t >= self.t_tick - 1e-9: self.tick()
        w = 1.0 - (self.t_tick - t) / TICK
        return self.q_prev + (self.q_next - self.q_prev) * w, (self.q_next - self.q_prev) / TICK


from procs import die_with_parent  # noqa: E402,F401  (factory/cell/procs.py; pools: initializer=die_with_parent)


# --- training episodes from snapshots -------------------------------------------------------------------------
class Env:
    """One cell per process. `episode(snap, policy, dr)` restores a snapshot taken at the end of an F2 lift and runs the
    carry until the tool has stopped and the item has landed (or failed)."""

    def __init__(self, cell=None):
        from cell_mujoco import CellMJ
        from pick import Planner, Executor
        self.c = cell or CellMJ(); self.arm = Planner(self.c).arm; self.ex = Executor(self.c)
        m = self.c.model; self.mass0 = m.body_mass.copy(); self.inertia0 = m.body_inertia.copy(); self.fric0 = m.geom_friction.copy()

    def restore(self, snap):
        c = self.c; m, d = c.model, c.data
        mujoco.mj_setState(m, d, snap["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        for n, row in snap["eq"].items(): m.eq_data[c.vac_eq[n]] = row
        c.vac_cmd = snap["vac_cmd"]; c.vac_item = snap["vac_item"]; c.belt_target = snap["belt_target"]; c.belt_ramp = snap["belt_ramp"]
        mujoco.mj_forward(m, d)

    def randomize(self, item, dr, rng):
        m = self.c.model; b = self.c.items[item]["body"]
        m.body_mass[:] = self.mass0; m.body_inertia[:] = self.inertia0; m.geom_friction[:] = self.fric0
        if not dr: return {}
        ms = float(rng.uniform(0.85, 1.2)); fs = float(rng.uniform(0.7, 1.1))
        m.body_mass[b] *= ms; m.body_inertia[b] *= ms
        gi = np.flatnonzero(m.geom_bodyid == b); m.geom_friction[gi, 0] *= fs
        return dict(mass_scale=round(ms, 3), friction_scale=round(fs, 3))

    def episode(self, snap, policy: Policy, dr=False, seed=0, on_frame=None, frame_every=17, budget=4.0, privileged=True, teacher=None):
        """privileged=False runs the student carry (F4); with a `teacher` Policy it also records (student features,
        teacher residual output) at every tick in the result's "probe" (DAgger data)."""
        c = self.c; d = c.data; dt = c.model.opt.timestep; rng = np.random.default_rng(seed)
        self.restore(snap); dri = self.randomize(snap["item"], dr, rng); mujoco.mj_forward(c.model, d)
        hang = None
        if not privileged:   # the hang the line knows from the robot's pose when the fingers closed: here the snapshot's true one (within mm of it)
            p0 = self.arm.fk(np.asarray(snap["q_ref"], float))[0]; st = c.item_state(snap["item"]); hang = float(p0[2] - (st["pos"][2] + c.items[snap["item"]]["rec"]["bounds"][0][2]))
        ctl = CarryCtl(c, self.arm, policy, snap["item"], snap["tool"], snap["carry_z"], snap["a_carry"], snap["R_drop"], snap["held"], privileged=privileged, hang=hang)
        if teacher is not None: ctl.probe = []; ctl.teacher = teacher
        ctl.begin(snap["q_ref"]); kvkp = self.ex.kv_over_kp; t = 0.0; k = 0; out = None; landed_t = None; max_slip = 0.0; max_err = 0.0
        name = snap["item"]; mat = c.items[name]["rec"]["material"]
        while out is None:
            q_ref, qd_ref = ctl.at(t)
            if ctl.fail: out = ("ik", None); break
            d.ctrl[c.robot_act] = q_ref + kvkp * qd_ref; d.qfrc_applied[c.robot_dof] = d.qfrc_bias[c.robot_dof]
            if ctl.released and not getattr(ctl, "_cmd_sent", False):
                d.ctrl[c.grip_act] = 0.0; c.vacuum(False); ctl._cmd_sent = True
            c.step(); t += dt; k += 1
            if on_frame and k % frame_every == 0: on_frame(c, ctl, t)
            if k % 5 == 0:
                if self.ex.robot_fixture_contact(): out = ("collision", None); break
                err = float(np.max(np.abs(d.qpos[c.robot_qpos] - q_ref))); max_err = max(max_err, err)
                if err > LN.TRACK_STOP: out = ("collision", "tracking"); break
                if not ctl.released:
                    slip, _ = self.ex._slip_swing(name, snap["held"]); max_slip = max(max_slip, slip)
                    if slip > 0.03 or (snap["tool"] == "suction" and c.vac_item != name): out = ("drop", None); break
                else:
                    s = c.item_state(name); ch = ctl.chute
                    if landed_t is None and s["pos"][2] < ch.rim_z: landed_t = t
                    if landed_t is not None and t - landed_t > 0.5 and ctl.t_stop is not None:
                        where, _, _ = classify(c, name, mat); out = ({"success": "sorted"}.get(where, where), None); break
                    if s["pos"][2] < c.lay.belt.surface_z - 0.25 and landed_t is None: out = ("off_target", None); break
            if t > budget: out = ("timeout", None); break
        outcome = out[0]; ok = outcome == "sorted"
        t_free = ctl.t_stop if (ok and ctl.t_stop is not None) else budget
        return dict(item=name, outcome=outcome, detail=out[1], ok=ok, t_free=round(float(t_free), 3), t_release=ctl.t_release,
                    max_slip=round(max_slip, 4), max_track_err=round(max_err, 4), dr=dri, reward=round((SUCCESS_R if ok else 0.0) - float(t_free), 4),
                    **({"probe": ctl.probe} if ctl.probe is not None else {}))
