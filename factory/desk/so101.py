"""so101.py — the SO-101 desk arm as an MjSpec, ready for the real-device servo model (2026-11-03, F6; disciplines 2, 6, 14).

The model is Menagerie's `robotstudio_so101/so101.xml` (Apache-2.0, vendor/ untouched), changed here only in memory:
  * mesh paths made absolute (the cell XML then needs no asset copies);
  * the six `position` actuators replaced by `motor` (torque) actuators with the same 2.94 N*m limit, so that the
    servo's own position loop runs in `servo.py`: the ideal servo there reproduces Menagerie's position actuator
    (kp 998.22, kv 2.731) and the real-device kit adds what a Feetech STS3215 does on a real bus (deadband, encoder
    quantization, bus delay, a torque-speed curve, a torque-limit register);
  * optionally (`backlash=True`) one extra hinge per joint, on the same axis, limited to +/- `backlash_rad`: the play
    between the servo's output shaft and the 3D-printed link (spline, horn screws, printed bores). Menagerie ships this
    as a commented-out default class ("backlash", +/- 0.5 deg, not added to any joints); here it is added. The servo's
    magnetic encoder sits on its output shaft, so it reads the joint, not the link: the play is invisible to the
    servo and shows up as link error (gravity pushes the pitch joints to one end of it). A first version put the
    encoder beyond the play; the unloaded pan joint then hunted +/- 0.7 deg at 1 rad/s rms (2026-11-03).

Frames (measured 2026-11-03 with the desk cell's IK, `factory/desk/desk_ik.py`): at q = 0 the arm is horizontal along +x; the `gripperframe` site's x axis
is the approach direction, its z axis the direction the moving jaw opens (the fixed jaw is the flat plate on -z), its
origin 3 mm short of the fixed jaw's tip. Joint 6 (`gripper`) opens the moving jaw: tip gap 16 mm at 0 rad, 86 mm at
0.92, 133 mm at 1.74.
"""
from __future__ import annotations
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SO101_DIR = os.path.join(ROOT, "vendor", "mujoco_menagerie", "robotstudio_so101")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = JOINTS[:5]
# The play's end stops: MuJoCo's default soft limit (time constant 20 ms) let the +/- 0.5 deg play open to -2.6 / +1.5
# deg under load (found 2026-11-03 by the calibration's residual); a 4 ms time constant (two physics steps, the
# stiffest stable value) and a hard impedance make it a stop.
PLAY_SOLREF = (0.004, 1.0)
PLAY_SOLIMP = (0.95, 0.99, 0.0005, 0.5, 2.0)
BACKLASH_FRICTION = 0.0   # N*m, output-bearing friction inside the gear play (see servo.py)
STALL_TORQUE = 2.94          # N*m, Menagerie's forcerange: the STS3215 (12 V) stall torque, 30 kg*cm
MENAGERIE_KP, MENAGERIE_KV = 998.22, 2.731


def _absolute_meshes(spec, model_dir):
    for m in spec.meshes:
        if not os.path.isabs(m.file): m.file = os.path.join(model_dir, spec.meshdir or "", m.file)
    for t in spec.textures:
        if t.file and not os.path.isabs(t.file): t.file = os.path.join(model_dir, spec.texturedir or spec.meshdir or "", t.file)
    spec.meshdir = ""; spec.texturedir = ""


def arm_spec(backlash: bool = False, backlash_rad: float = 0.0087267, plant: dict | None = None):
    """The SO-101 as an MjSpec with torque actuators named like the joints. `plant` overrides per-joint physical
    parameters, {"armature": [6], "damping": [6], "frictionloss": [6]} (system identification plants these)."""
    s = mujoco.MjSpec.from_file(os.path.join(SO101_DIR, "so101.xml")); _absolute_meshes(s, SO101_DIR)
    for a in list(s.actuators): s.delete(a)
    for k, n in enumerate(JOINTS):
        j = s.joint(n)
        if plant:
            if "armature" in plant: j.armature = float(plant["armature"][k])
            if "frictionloss" in plant: j.frictionloss = float(plant["frictionloss"][k])
            if "damping" in plant: j.damping = [float(plant["damping"][k]), 0.0, 0.0]   # MuJoCo 3.17: damping is a 3-vector (linear term first)
        if backlash:
            body = j.parent if hasattr(j, "parent") else None
            body = body or _joint_body(s, n)
            b = body.add_joint(name=f"{n}_backlash", type=mujoco.mjtJoint.mjJNT_HINGE, axis=j.axis, pos=j.pos,
                               range=(-backlash_rad, backlash_rad), damping=0.01, armature=0.0005, frictionloss=BACKLASH_FRICTION)
            b.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            b.solref_limit = PLAY_SOLREF; b.solimp_limit = PLAY_SOLIMP
        s.add_actuator(name=n, target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                       biastype=mujoco.mjtBias.mjBIAS_NONE, gainprm=(1.0,) + (0.0,) * 9,
                       ctrlrange=(-STALL_TORQUE, STALL_TORQUE), ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE)
    s.modelname = "so101_servo"
    return s


def _joint_body(spec, name):
    for b in spec.bodies:
        for j in b.joints:
            if j.name == name: return b
    raise KeyError(name)


class Indices:
    """qpos / dof addresses of the arm in any model that contains it (the desk cell or the bare arm)."""
    def __init__(self, m):
        self.jid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS])
        self.qadr = m.jnt_qposadr[self.jid]; self.dadr = m.jnt_dofadr[self.jid]
        bl = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_backlash") for n in JOINTS]
        self.has_backlash = all(b >= 0 for b in bl)
        self.bl_qadr = m.jnt_qposadr[bl] if self.has_backlash else None
        self.bl_dadr = m.jnt_dofadr[bl] if self.has_backlash else None
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
        self.lo = m.jnt_range[self.jid, 0].copy(); self.hi = m.jnt_range[self.jid, 1].copy()
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    def servo_q(self, d):
        """Servo output-shaft angles (what the servo's encoder sees)."""
        return d.qpos[self.qadr].copy()

    def servo_qd(self, d):
        return d.qvel[self.dadr].copy()

    def link_q(self, d):
        """Link angles: servo shaft + the play beyond it (what forward kinematics of the real arm needs)."""
        q = d.qpos[self.qadr].copy()
        if self.has_backlash: q += d.qpos[self.bl_qadr]
        return q

    def link_qd(self, d):
        v = d.qvel[self.dadr].copy()
        if self.has_backlash: v += d.qvel[self.bl_dadr]
        return v

    def set_q(self, d, q):
        d.qpos[self.qadr] = q
        if self.has_backlash: d.qpos[self.bl_qadr] = 0.0
