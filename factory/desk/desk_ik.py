"""kinematics.py — forward and inverse kinematics of the SO-101 for top-down picks (2026-11-03, F6; discipline 3).

The SO-101 has five arm joints: pan (about the vertical), then three parallel pitch joints (shoulder lift, elbow,
wrist flex) and a wrist roll about the tool axis. Six numbers fix a gripper pose, so a 5-DoF arm cannot reach every
pose; it can reach every *top-down* pose in its workspace: the tool pointing straight down (two constraints that the
three pitch joints satisfy in the arm's vertical plane, where the tool axis always lies) at a position (three) with a
jaw yaw about the vertical (the roll joint). What it cannot do is tilt the tool sideways, out of its plane; poses that
need that are reported infeasible, never approximated silently.

`Arm.ik(p, yaw)` solves position + yaw with the tool pointing down by damped least squares on the 6-D pose error
(exactly reachable when the pose is reachable at all), from several starts, and checks the answer by forward
kinematics: <= 1 mm and <= 1 deg, inside the joint limits, or it says infeasible. `Arm.ik_pose(p, R)` takes any
rotation (used by the calibration routine and by tilted approaches).
"""
from __future__ import annotations
import math
import numpy as np
import mujoco

import so101


def topdown_rot(yaw):
    """gripperframe orientation for a top-down grasp: approach (site x) = world -z, the moving jaw opens along
    (cos yaw, sin yaw, 0) (site z); site y completes the frame."""
    x = np.array([0.0, 0.0, -1.0]); z = np.array([math.cos(yaw), math.sin(yaw), 0.0]); y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def angle_between(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2; return math.acos(max(-1.0, min(1.0, c)))


def rot_err(R, Rt):
    """Rotation-vector error that takes R to Rt (world frame)."""
    E = Rt @ R.T; q = np.zeros(4); mujoco.mju_mat2Quat(q, E.reshape(-1)); v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0); return v


class Arm:
    """Kinematics on a private copy of the bare arm (or of `model`, if given: then the base pose is the cell's)."""

    def __init__(self, model=None, base_pos=(0, 0, 0)):
        if model is None:
            s = so101.arm_spec(); s.body("base").pos = base_pos; model = s.compile()
        self.m = model; self.d = mujoco.MjData(model); self.ix = so101.Indices(model)
        self.lo = self.ix.lo[:5]; self.hi = self.ix.hi[:5]
        self.nv = model.nv; self.cols = self.ix.dadr[:5]

    def fk(self, q5, grip=0.0):
        q = np.zeros(6); q[:5] = q5; q[5] = grip
        self.ix.set_q(self.d, q); mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        s = self.ix.site; return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def jac(self):
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv)); mujoco.mj_jacSite(self.m, self.d, jp, jr, self.ix.site)
        return np.vstack([jp[:, self.cols], jr[:, self.cols]])

    def ik_pose(self, p, R, q0=None, restarts=6, iters=100, tol_pos=1e-3, tol_rot_deg=1.0, rng=None, w_rot=0.1):
        rng = rng or np.random.default_rng(0); p = np.asarray(p, float)
        starts = [np.zeros(5) if q0 is None else np.asarray(q0, float)]
        # a good generic start for top-down picks: pan toward the target, elbow bent, wrist down
        pan = math.atan2(p[1], p[0]) if q0 is None else starts[0][0]
        starts.append(np.array([pan, 0.0, 0.5, 1.2, 0.0]))
        for _ in range(restarts): starts.append(rng.uniform(self.lo, self.hi))
        best = (False, None, dict(pos_err=1e9, rot_err_deg=1e9))
        for q in starts:
            q = np.clip(q.copy(), self.lo, self.hi); lam = 1e-2
            for it in range(iters):
                pc, Rc = self.fk(q)
                e = np.r_[p - pc, w_rot * rot_err(Rc, R)]
                if np.linalg.norm(e[:3]) < tol_pos * 0.2 and np.linalg.norm(e[3:]) / w_rot < math.radians(tol_rot_deg) * 0.2: break
                J = self.jac(); J[3:] *= w_rot
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
                q = np.clip(q + dq, self.lo, self.hi)
            pc, Rc = self.fk(q); pe = float(np.linalg.norm(p - pc)); re = math.degrees(angle_between(Rc, R))
            if pe <= tol_pos and re <= tol_rot_deg: return True, q, dict(pos_err=pe, rot_err_deg=re)
            if pe + re * 1e-3 < best[2]["pos_err"] + best[2]["rot_err_deg"] * 1e-3: best = (False, q, dict(pos_err=pe, rot_err_deg=re))
        return best

    def ik(self, p, yaw, **kw):
        """Top-down IK; the jaw yaw is taken modulo pi (the grasp is symmetric) and both choices are tried."""
        ok, q, info = self.ik_pose(p, topdown_rot(yaw), **kw)
        if ok: return ok, q, info
        return self.ik_pose(p, topdown_rot(yaw + math.pi), **kw)
