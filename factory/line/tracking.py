"""tracking.py — discipline 8 (2026-11-02, F2): state estimation for items riding the belt.

The overhead D435i (layout.Camera) sees the belt between x = +-0.69 m (its 69-degree axis at 1 m). `Detector` stands in
for perception until F4: at the camera rate it reports each visible item's position and yaw with Gaussian noise, each
report `latency` seconds old when it arrives (exposure + transfer + inference), and with a miss probability. Identity is
privileged in F2 (each report carries the item's name); F4 replaces this with real detection and association.

`Track` is a constant-velocity Kalman filter on (x, y, vx, vy) per item, fed the time-stamped reports in order. Its
`predict(t)` extrapolates to any time, so a planner can aim at where the item *will be* when the gripper arrives
(the intercept), compensating the latency because every report keeps the time it was taken, not when it arrived.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectorCfg:
    rate_hz: float = 30.0
    latency_s: float = 0.06
    pos_sigma: float = 0.003      # m, per axis (x, y); z is reported with the same noise
    yaw_sigma_deg: float = 2.0
    miss_prob: float = 0.05
    x_view: tuple = (-0.69, 0.69)
    compensate_latency: bool = True   # False: the filter pretends a report is fresh (for the latency study)


class Detector:
    """Samples the simulation's item poses at the camera rate and delivers them `latency` later."""

    def __init__(self, cell, cfg: DetectorCfg = DetectorCfg(), rng=None):
        self.cell = cell; self.cfg = cfg; self.rng = rng or np.random.default_rng(0)
        self.next_t = 0.0; self.queue = []   # (deliver_at, report)

    def poll(self, t, names):
        """Call every control step; returns the reports that arrive at time t."""
        c = self.cfg
        if t >= self.next_t:
            self.next_t = t + 1.0 / c.rate_hz
            for n in names:
                s = self.cell.item_state(n)
                if not (c.x_view[0] <= s["pos"][0] <= c.x_view[1]) or self.rng.random() < c.miss_prob: continue
                w, x, y, z = s["quat"]; yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                rep = dict(name=n, t=t, pos=s["pos"] + self.rng.normal(0, c.pos_sigma, 3), yaw=yaw + math.radians(self.rng.normal(0, c.yaw_sigma_deg)), tilt=s["tilt_deg"])
                self.queue.append((t + c.latency_s, rep))
        out = [r for (ta, r) in self.queue if ta <= t]; self.queue = [(ta, r) for (ta, r) in self.queue if ta > t]
        if not c.compensate_latency:
            out = [dict(r, t=t) for r in out]
        return out


@dataclass
class Track:
    name: str
    t: float
    x: np.ndarray                     # (x, y, vx, vy)
    P: np.ndarray
    z: float = 0.0
    yaw: float = 0.0
    n: int = 1
    last_seen: float = 0.0
    q_acc: float = 1e-4               # process noise, (m/s^2)^2/Hz: items on a belt move at nearly constant velocity (0.05 let vx wander 15%)
    r: float = 0.003 ** 2

    @classmethod
    def start(cls, rep, v_prior=(0.0, 0.0), v_sigma=0.5, r=0.003 ** 2):
        x = np.array([rep["pos"][0], rep["pos"][1], v_prior[0], v_prior[1]])
        P = np.diag([r, r, v_sigma ** 2, v_sigma ** 2])
        return cls(rep["name"], rep["t"], x, P, z=float(rep["pos"][2]), yaw=float(rep["yaw"]), last_seen=rep["t"], r=r)

    def _F(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = self.q_acc; Q = np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2], [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]]) * q
        return F, Q

    def update(self, rep):
        dt = rep["t"] - self.t
        if dt < 0: return   # out of order: drop
        F, Q = self._F(dt); x = F @ self.x; P = F @ self.P @ F.T + Q
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.]]); S = H @ P @ H.T + self.r * np.eye(2); K = P @ H.T @ np.linalg.inv(S)
        self.x = x + K @ (rep["pos"][:2] - H @ x); self.P = (np.eye(4) - K @ H) @ P; self.t = rep["t"]
        self.z += 0.2 * (float(rep["pos"][2]) - self.z); self.n += 1; self.last_seen = rep["t"]
        dy = (rep["yaw"] - self.yaw + math.pi) % (2 * math.pi) - math.pi; self.yaw += 0.2 * dy

    def predict(self, t):
        """(x, y, z), (vx, vy) at time t, extrapolated with constant velocity."""
        dt = t - self.t
        return np.array([self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt, self.z]), self.x[2:].copy()

    def sigma(self, t):
        F, Q = self._F(max(0.0, t - self.t)); P = F @ self.P @ F.T + Q; return float(math.sqrt(P[0, 0])), float(math.sqrt(P[1, 1]))


class Tracker:
    """One track per item name; tracks expire when unseen for `expire_s`."""

    def __init__(self, v_prior=(0.0, 0.0), expire_s=0.5, r=0.003 ** 2, v_sigma=0.5):
        """v_prior/v_sigma: what is known about an item's velocity before it is seen. A conveyor drive's encoder gives
        the belt speed, so the line starts each track at the belt velocity +-0.02 m/s; the tracking study starts at 0
        +-0.5 m/s (vision alone)."""
        self.tracks = {}; self.v_prior = v_prior; self.expire_s = expire_s; self.r = r; self.v_sigma = v_sigma

    def feed(self, reports, t):
        for rep in reports:
            tr = self.tracks.get(rep["name"])
            if tr is None or rep["t"] - tr.last_seen > self.expire_s: self.tracks[rep["name"]] = Track.start(rep, self.v_prior, self.v_sigma, r=self.r)
            else: tr.update(rep)
            self.tracks[rep["name"]].tilt = float(rep.get("tilt", 0.0))
        for n in [n for n, tr in self.tracks.items() if t - tr.last_seen > self.expire_s + 0.2]: del self.tracks[n]

    def confirmed(self, min_n=10):
        return {n: tr for n, tr in self.tracks.items() if tr.n >= min_n}

    def drop(self, name): self.tracks.pop(name, None)
