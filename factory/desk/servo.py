"""servo.py — the SO-101's six STS3215 servos on their serial bus, ideal or real-device (2026-11-03, F6; disciplines 6, 14).

What a hobby serial-bus servo is: a DC motor, a gearbox, a magnetic encoder on the output shaft and a small
microcontroller that runs the position loop *inside the servo*. The host (LeRobot, 30 Hz on the SO-101) only writes a
goal position to each servo over a half-duplex bus and reads the present position back. So the controller the host
designs is not a torque controller: it is a setpoint stream, and everything between setpoint and motion belongs to
the servo. That is the part this file models.

Ideal servo (`DeviceKit.ideal()`): torque = kp * (goal - q) - kv * qdot, clipped to the stall torque, every physics
step, with exact state: Menagerie's position actuator, reproduced.

Real-device servo (`DeviceKit.real()`), per joint:
  encoder     12 bits per turn (4096 counts, 0.088 deg) on the servo's output shaft; the loop and the host both see
              the quantized shaft angle, not the link beyond the play
  deadband    no drive while |goal - encoder| <= deadband counts (the STS3215's CW/CCW dead-zone registers, 1 count)
  loop        duty = clip((kp * error - kv * v_est) / stall, +/- torque_limit) at the servo's loop rate (1 kHz on the
              real chip, here every physics step, 2 ms): the chip's PD loop (LeRobot sets P 16, D 32, I 0), with
              Menagerie's kp and kv, the error from the quantized encoder and v_est from encoder differences over
              `vel_window_s` (10 ms; one count per 2 ms is 0.77 rad/s, so the chip must average). Without the D term
              the loop chatters in a +/- 2 deg limit cycle (found 2026-11-03)
  motor       torque = stall * duty - (stall / no_load_speed) * qdot: the DC motor's torque-speed line (back-EMF), so
              the servo cannot exceed ~4.7 rad/s and is weaker while moving
  backlash    play +/- 0.5 deg between the shaft and the link (so101.arm_spec(backlash=True)): the link moves freely
              inside it and the encoder does not see it
  bus         goal writes reach the servo `bus_delay_s` after the host sends them, position reads are that old
              when the host gets them (a sync-write + sync-read round at 1 Mbaud: a few ms)
  host loop   the host runs at `host_hz` (30 Hz); between writes the servo holds the last goal

The plant's physical parameters (rotor inertia = armature, viscous damping, Coulomb friction = frictionloss) live in
the MJCF (so101.arm_spec(plant=...)); system identification (sysid.py) recovers planted values from recorded data.
Menagerie's joint damping (0.60) is the motor's back-EMF folded into the joint; the real kit models back-EMF here
explicitly, so its plant uses `REAL_PLANT` (gearbox viscous damping 0.10, same armature and friction).
"""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field, asdict
import numpy as np
import mujoco

from so101 import Indices, STALL_TORQUE, MENAGERIE_KP, MENAGERIE_KV

REAL_PLANT = dict(armature=[0.028] * 6, damping=[0.10] * 6, frictionloss=[0.052] * 6)
COUNTS = 4096
RAD_PER_COUNT = 2 * math.pi / COUNTS


@dataclass
class DeviceKit:
    """The servo-side half of the real-device kit (the camera half is in camera.py)."""
    real: bool = False
    kp: float = MENAGERIE_KP            # N*m/rad of position error (the ideal servo); the real one uses it per count
    kv: float = MENAGERIE_KV            # N*m*s/rad, ideal servo only (the real servo's damping is its back-EMF)
    stall: float = STALL_TORQUE         # N*m
    no_load_speed: float = 4.71         # rad/s (0.222 s / 60 deg at 12 V)
    torque_limit: float = 1.0           # fraction of stall (the servo's torque-limit register)
    deadband_counts: int = 1
    quantize: bool = True
    bus_delay_s: float = 0.004
    backlash_rad: float = 0.0087267     # +/- 0.5 deg; used by so101.arm_spec when the kit is real
    gripper_torque_limit: float = 0.5   # LeRobot caps the gripper servo's torque (it grips without burning out)
    vel_window_s: float = 0.010

    @staticmethod
    def ideal(): return DeviceKit(real=False)

    @staticmethod
    def real_kit(**kw): return DeviceKit(real=True, **kw)

    def to_dict(self): return asdict(self)


class ServoBus:
    """Drives the arm's torque actuators from goal positions. Call `write(goal, t)` from the host loop, `step(t)` before
    every mj_step, `read(t)` to get the present positions the host would see."""

    def __init__(self, model, data, kit: DeviceKit, rng=None):
        self.m, self.d, self.kit = model, data, kit
        self.ix = Indices(model)
        self.rng = rng or np.random.default_rng(0)
        self.goal = None                      # what the servos are holding
        self.pending = deque()                # (arrival time, goal) in flight on the bus
        self.history = deque(maxlen=256)      # (t, encoder angles): delayed reads and the velocity estimate
        self.limit = np.full(6, kit.torque_limit); self.limit[5] = min(kit.torque_limit, kit.gripper_torque_limit)

    def encoder(self):
        q = self.ix.servo_q(self.d) if self.kit.real else self.ix.link_q(self.d)
        if self.kit.real and self.kit.quantize: q = np.round(q / RAD_PER_COUNT) * RAD_PER_COUNT
        return q

    def reset(self, q, t=0.0):
        self.ix.set_q(self.d, q); self.d.qvel[self.ix.dadr] = 0
        if self.ix.has_backlash: self.d.qvel[self.ix.bl_dadr] = 0
        mujoco.mj_forward(self.m, self.d)
        self.goal = np.array(q, float); self.pending.clear(); self.history.clear(); self.history.append((t, self.encoder()))

    def write(self, goal, t):
        goal = np.clip(np.asarray(goal, float), self.ix.lo, self.ix.hi)
        if self.kit.real and self.kit.bus_delay_s > 0: self.pending.append((t + self.kit.bus_delay_s, goal))
        else: self.goal = goal

    def set_torque_limit(self, joint: int, frac: float):
        self.limit[joint] = frac

    def read(self, t):
        """Present positions as the host sees them at time t (bus_delay_s old on the real kit)."""
        if not (self.kit.real and self.kit.bus_delay_s > 0): return self.encoder()
        target = t - self.kit.bus_delay_s; best = self.history[0][1]
        for (th, q) in reversed(self.history):
            if th <= target + 1e-9: best = q; break
        return best.copy()

    def step(self, t):
        k = self.kit
        while self.pending and self.pending[0][0] <= t + 1e-9: self.goal = self.pending.popleft()[1]
        qd = self.ix.servo_qd(self.d)
        if not k.real:
            q = self.ix.link_q(self.d)
            tau = np.clip(k.kp * (self.goal - q) - k.kv * qd, -k.stall, k.stall)
        else:
            enc = self.encoder(); self.history.append((t, enc))
            th, qold = self.history[0]
            for (th, qold) in reversed(self.history):
                if th <= t - k.vel_window_s + 1e-9: break
            v_est = (enc - qold) / max(t - th, 1e-6) if t > th else np.zeros(6)
            err = (self.goal - enc) / RAD_PER_COUNT          # counts
            err = np.where(np.abs(err) <= k.deadband_counts + 1e-9, 0.0, err)
            duty = np.clip((k.kp * RAD_PER_COUNT * err - k.kv * v_est) / k.stall, -self.limit, self.limit)
            tau = k.stall * duty - (k.stall / k.no_load_speed) * qd
            tau = np.clip(tau, -k.stall, k.stall)
        self.d.ctrl[self.ix.act] = tau
        return tau
