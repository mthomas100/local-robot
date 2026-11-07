"""kinematics.py — discipline 3 (2026-11-02, F1): forward kinematics, numerical inverse kinematics with joint limits,
Jacobians and a singularity measure for the UR5e + 2F-85, on MuJoCo's kinematics of the same robot the cell simulates.

The tool frame is the 2F-85 "pinch" site: the point between the fingertips, its z axis along the approach direction
(out of the gripper), x across the fingers' opening direction... (see `tool_axes`). A pose is (position, 3x3 rotation)
of that frame in the world.

    from kinematics import Arm
    arm = Arm()                                   # its own MjModel of the robot on the cell's pedestal
    p, R = arm.fk(q)                              # pinch pose for joint angles q (6,)
    ok, q, info = arm.ik(p_target, R_target, q0)  # damped least squares, joint limits, several restarts
    J = arm.jacobian(q); w = arm.manipulability(q)
"""
from __future__ import annotations
import math, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
from layout import Layout, DEFAULT  # noqa: E402
import build_mjcf  # noqa: E402

JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def rot_err(R, Rt):
    """Rotation error as an axis-angle vector (world frame) taking R to Rt."""
    E = Rt @ R.T; q = np.zeros(4); mujoco.mju_mat2Quat(q, E.reshape(-1)); v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0); return v


def angle_between(R, Rt):
    return float(np.linalg.norm(rot_err(R, Rt)))


def topdown(yaw=0.0):
    """Rotation of a tool pointing straight down (tool z = world -z), rotated by yaw about the vertical: the fingers
    close along world (cos yaw, sin yaw, 0)."""
    c, s = math.cos(yaw), math.sin(yaw)
    x = np.array([c, s, 0.0]); z = np.array([0.0, 0.0, -1.0]); y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def grasp_rot(yaw):
    """Tool rotation for a top-down 2F-85 grasp whose fingers close along world (cos yaw, sin yaw, 0): the 2F-85
    closes along its base frame's y axis and approaches along z, so tool y = the closing axis, tool z = world -z."""
    return topdown(yaw + math.pi / 2)


class Arm:
    def __init__(self, lay: Layout = DEFAULT, model=None, data=None, site="2f85/pinch"):
        if model is None:
            s = build_mjcf.robot_spec(); s.body("base").pos = lay.robot_base; model = s.compile()
        self.m = model; self.d = data if data is not None else mujoco.MjData(model)
        self.jid = np.array([self.m.joint(j).id for j in JOINTS]); self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.jid])
        self.lo = self.m.jnt_range[self.jid, 0].copy(); self.hi = self.m.jnt_range[self.jid, 1].copy()
        self.site = self.m.site(site).id; self.lay = lay
        self._scratch = mujoco.MjData(self.m)   # FK/IK never disturb the simulation's data

    def fk(self, q):
        d = self._scratch; d.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def jacobian(self, q):
        d = self._scratch; d.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv)); mujoco.mj_jacSite(self.m, d, jp, jr, self.site)
        return np.vstack([jp[:, self.dadr], jr[:, self.dadr]])

    def manipulability(self, q):
        """Yoshikawa's measure sqrt(det(J J^T)): zero at a singularity."""
        J = self.jacobian(q); return float(math.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def ik(self, p, R=None, q0=None, tol_pos=1e-3, tol_rot=math.radians(1.0), iters=200, restarts=8, damping=1e-2, rng=None, pos_only=False):
        """Damped least squares with joint-limit clamping. Tries q0 first, then the home pose, then random restarts.
        Returns (ok, q, info) where info has the position/rotation error, iterations and the manipulability."""
        rng = rng or np.random.default_rng(0); p = np.asarray(p, float)
        seeds = [np.asarray(q0, float)] if q0 is not None else []
        seeds += [np.array(self.lay.home_qpos)] + [rng.uniform(np.maximum(self.lo, -math.pi), np.minimum(self.hi, math.pi)) for _ in range(restarts)]
        best = None; total = 0
        for q in seeds:
            q = np.clip(q.copy(), self.lo, self.hi)
            for k in range(iters):
                pc, Rc = self.fk(q); ep = p - pc; er = np.zeros(3) if (R is None or pos_only) else rot_err(Rc, R)
                epn, ern = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
                if epn < tol_pos and ern < tol_rot:
                    total += k; return True, q, dict(pos_err=epn, rot_err=ern, iters=total, manipulability=self.manipulability(q))
                J = self.jacobian(q)
                if R is None or pos_only: J, e = J[:3], ep
                else: e = np.concatenate([ep, er])
                lam = damping * (1.0 + 10.0 * min(1.0, epn))   # more damping far from the goal, less near it
                dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(J.shape[0]), e)
                n = np.linalg.norm(dq)
                if n > 0.5: dq *= 0.5 / n   # trust region
                q = np.clip(q + dq, self.lo, self.hi)
            total += iters
            if best is None or epn + ern < best[2]["pos_err"] + best[2]["rot_err"]:
                best = (False, q, dict(pos_err=epn, rot_err=ern, iters=total, manipulability=self.manipulability(q)))
        return best

    @staticmethod
    def wrap_near(q, ref):
        """Shift each joint by multiples of 2 pi to lie nearest the reference (UR joints span +-2 pi)."""
        return ref + (np.asarray(q) - ref + math.pi) % (2 * math.pi) - math.pi
