"""desk_pick.py — the classical L0 pick on the desk cell (2026-11-03, F6; disciplines 3, 4, 6): one item at rest on the
stopped belt section, picked top-down and dropped in the bin for its material. No learning.

Grasp (discipline 4), for the SO-101's jaw: one jaw is fixed (a flat plate on the gripper body) and one swings on a
hinge, so the grasp is not centred. The fixed jaw's pad goes 3 mm from the item's face across its narrow horizontal
axis, the moving jaw opens just wider than the item (the contact angle measured in `jaw_table`), the tool descends to
`depth` below the item's top, the moving jaw closes against the servo's torque limit and pushes the item onto the fixed
pad. Both sides of the narrow axis are tried; the first one IK reaches wins.

Plan (disciplines 3, 5): waypoints solved with top-down IK (desk_ik.py): above the grasp, the grasp (a vertical line,
solved every 5 mm), the lift, above the bin, then home. Each segment is timed as a minimum-jerk profile under joint
speed and acceleration limits and a tool-speed limit, as in the v1 cell (factory/classical/pick.py).

Execution (disciplines 6, 14): a host loop at `HOST_HZ` (30 Hz, as LeRobot drives the SO-101) samples the plan and
writes goal positions to the servo bus (servo.py); the servos run their own position loops at the physics rate.

Outcomes (the v1 taxonomy, renamed for bins): ok, unseen (the camera found no item), ik_fail, collision (a robot link touched the desk, riser, belt
frame or a bin), knocked (the item moved > 2 cm or tipped > 20 deg before the jaw closed), miss (the item did not come
up), drop (it left the jaw on the way), wrong_bin, off_target (released but in no bin), timeout.
"""
from __future__ import annotations
import math, os, sys, time
from dataclasses import dataclass, field
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from desk_ik import Arm, topdown_rot  # noqa: E402
import desk_items as I  # noqa: E402

HOST_HZ = 30.0
FIXED_PAD = -0.020        # site-z of the fixed jaw's pad face (measured: pads at -0.021 .. -0.026, 1 mm thick)
PAD_GAP = 0.003           # pre-grasp gap between the item's face and the fixed pad
GRIP_OPEN_MAX = 1.74
GRIP_CLOSED = -0.17
LIMITS = dict(vmax=np.array([2.0, 2.0, 2.0, 2.5, 3.0]), amax=np.array([8.0, 8.0, 8.0, 10.0, 12.0]), vtool=0.30, vtool_slow=0.08, atool=2.0)
ABOVE = 0.025            # pre-grasp: tool frame this far above the item's top
CLEAR = 0.025             # item bottom above the belt rails / bin rims while carried

# jaw angle (rad) at which the moving jaw first touches the face of an item of width W (m), at 20 and 30 mm grasp
# depth (static probe boxes, measured 2026-11-03 with the bare arm; see jaw_table())
JAW_W = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
JAW_ANGLE = {0.02: np.array([0.24, 0.44, 0.60, 0.74, 0.88, 0.98, 1.06]), 0.03: np.array([0.31, 0.54, 0.73, 0.88, 0.98, 1.10, 1.19])}


def jaw_open_for(width, depth):
    """Jaw command that clears an item of this width by ~15 mm at this depth."""
    tab = JAW_ANGLE[0.02] if depth < 0.025 else JAW_ANGLE[0.03]
    return float(min(GRIP_OPEN_MAX, np.interp(width + 0.015, JAW_W, tab, right=GRIP_OPEN_MAX) + 0.08))


def minjerk(tau):
    tau = np.clip(tau, 0.0, 1.0); return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


@dataclass
class Segment:
    name: str
    path: np.ndarray          # (K, 5) arm joint path
    T: float
    grip: float               # jaw goal during the segment
    dwell: float = 0.0


@dataclass
class Plan:
    item: str
    material: str
    grasp: dict
    segments: list = field(default_factory=list)
    plan_seconds: float = 0.0

    @property
    def T(self): return sum(s.T + s.dwell for s in self.segments)


def duration(path, cart_len=0.0, vtool=LIMITS["vtool"]):
    D = np.abs(np.diff(path, axis=0)).sum(0) if len(path) > 1 else np.zeros(5)
    return float(max(np.max(1.875 * D / LIMITS["vmax"]), np.max(np.sqrt(5.77 * D / LIMITS["amax"])), 1.875 * cart_len / vtool,
                     math.sqrt(5.77 * cart_len / LIMITS["atool"]), 0.2))


def sample(seg: Segment, t):
    P = seg.path
    if len(P) == 1: return P[0]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9: return P[-1]
    u = minjerk(t / seg.T) * d[-1]; k = int(np.clip(np.searchsorted(d, u) - 1, 0, len(P) - 2))
    w = (u - d[k]) / max(d[k + 1] - d[k], 1e-12); return P[k] * (1 - w) + P[k + 1] * w


class Planner:
    def __init__(self, sim):
        self.sim = sim; self.lay = sim.lay
        self.arm = Arm(model=sim.model)   # kinematics on the cell's own model

    def line(self, q0, p0, p1, R, step=0.005):
        n = max(2, int(np.linalg.norm(np.asarray(p1) - p0) / step) + 1); qs = [q0]; q = q0
        for k in range(1, n + 1):
            p = np.asarray(p0) + (np.asarray(p1) - p0) * k / n
            ok, q, info = self.arm.ik_pose(p, R, q0=q, restarts=0, iters=60)
            if not ok: return None
            qs.append(q)
        return np.array(qs)

    def grasps(self, x, y, yaw, inf):
        """Candidate grasps: across the narrow footprint axis (any axis for a round item), both jaw sides."""
        ex, ey = inf["extents"][0], inf["extents"][1]
        round_ = abs(ex - ey) < 0.004 and inf["name"] in ("tomato_soup_can", "master_chef_can", "tuna_fish_can", "standin_pet_bottle", "standin_small_tin")
        if round_:
            radial = math.atan2(y, x); axes = [radial + math.pi / 2, radial - math.pi / 2, radial + math.pi / 4, radial - math.pi / 4]; W = ex
        else:   # the jaw side is a rule, not "whichever IK reaches first", so the choice is a function of what the camera
            # sees (the box's axis, which has no head or tail): the moving jaw opens toward +y (a in [0, pi)), the
            # other side only if that fails. Behaviour cloning (bc.py) needs a consistent expert.
            a = (yaw if ex <= ey else yaw + math.pi / 2) % math.pi; axes = [a, a + math.pi]; W = min(ex, ey)
        h = inf["height"]; depth = float(np.clip(0.45 * h, 0.012, 0.03))
        top = self.lay.belt.surface_z + h
        out = []
        for a in axes:
            u = np.array([math.cos(a), math.sin(a), 0.0])
            p = np.array([x, y, top - depth]) - u * (FIXED_PAD + PAD_GAP + W / 2)
            out.append(dict(axis=a, u=u, width=W, depth=depth, pos=p, top=top, open=jaw_open_for(W, depth)))
        return out

    def plan(self, name, est, q_start=None):
        """est = (x, y, yaw) of the item's footprint centre (privileged or from the camera)."""
        t0 = time.perf_counter(); inf = I.info(name); x, y, yaw = est
        q_home = np.array(self.lay.home_q[:5]); q_start = q_home if q_start is None else q_start
        b = self.lay.bin(inf["material"]); bx, by = b.xy
        belt_top = self.lay.belt.surface_z + self.lay.belt.rail_height
        for g in self.grasps(x, y, yaw, inf):
            R = topdown_rot(g["axis"]); p = g["pos"]; hang = inf["height"] - g["depth"]     # item below the tool frame
            z_lift = max(belt_top, b.rim_z) + CLEAR + hang
            above = np.array([p[0], p[1], g["top"] + ABOVE])   # the jaw tips 3 mm below the tool frame clear the top by ~2 cm
            ok, q_above, _ = self.arm.ik_pose(above, R, q0=None, restarts=4)
            if not ok: continue
            down = self.line(q_above, above, p, R)
            if down is None: continue
            lift_p = np.array([p[0], p[1], z_lift])
            up = self.line(down[-1], p, lift_p, R)
            if up is None: continue
            # over the bin: keep the jaw axis tangential there, at the carry height, top-down
            ba = math.atan2(by, bx) + math.pi / 2
            ok, q_bin, _ = self.arm.ik_pose(np.array([bx, by, z_lift]), topdown_rot(ba), q0=up[-1], restarts=4)
            if not ok: ok, q_bin, _ = self.arm.ik_pose(np.array([bx, by, z_lift]), topdown_rot(ba + math.pi), q0=up[-1], restarts=4)
            if not ok: continue
            segs = []
            P = np.array([q_start, q_above]); segs.append(Segment("approach", P, duration(P), grip=g["open"]))
            segs.append(Segment("descend", down, duration(down, above[2] - p[2], LIMITS["vtool_slow"]), grip=g["open"], dwell=0.15))
            segs.append(Segment("close", down[-1:], 0.45, grip=GRIP_CLOSED))
            segs.append(Segment("lift", up, duration(up, z_lift - p[2], 0.15), grip=GRIP_CLOSED))
            P = np.array([up[-1], q_bin]); segs.append(Segment("transfer", P, duration(P) * 1.3, grip=GRIP_CLOSED, dwell=0.2))
            segs.append(Segment("release", P[-1:], 0.5, grip=g["open"]))
            P = np.array([q_bin, q_home]); segs.append(Segment("return", P, duration(P), grip=0.0))
            g = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in g.items()}; g["z_lift"] = float(z_lift)
            return Plan(name, inf["material"], g, segs, time.perf_counter() - t0)
        return None


class Executor:
    """Runs a plan on the sim: the host writes servo goals at HOST_HZ; the item's state is watched for failures."""

    def __init__(self, sim, record=None):
        self.sim = sim; self.record = record   # record(sim) is called at every host tick (videos, BC datasets)

    def run(self, plan: Plan, name, on_tick=None):
        sim = self.sim; dt_host = 1.0 / HOST_HZ; steps = max(1, int(round(dt_host / sim.model.opt.timestep)))
        p0 = sim.item_pose(name)[0]; ev = dict(collision=None, knocked=False, lifted=False, max_tilt=0.0, drop=False)
        for seg in plan.segments:
            n = int(math.ceil((seg.T + seg.dwell) / dt_host))
            for k in range(n):
                q5 = sample(seg, min(k * dt_host, seg.T)); goal = np.r_[q5, seg.grip]
                t = sim.t; sim.bus.write(goal, t)
                if on_tick: on_tick(sim, seg, goal)
                for _ in range(steps):
                    sim.step(1)
                    c = sim.robot_fixture_contact()
                    if c and ev["collision"] is None: ev["collision"] = dict(t=round(sim.t, 3), seg=seg.name, pair=c)
                p, R = sim.item_pose(name)
                if seg.name in ("approach", "descend") and (np.linalg.norm(p[:2] - p0[:2]) > 0.02 or sim.tilt_deg(name) > 20): ev["knocked"] = True
                if seg.name == "transfer" and p[2] > p0[2] + 0.02: ev["lifted"] = True
                if seg.name in ("lift", "transfer"):
                    tool = sim.tool()[0]
                    if seg.name == "transfer" and np.linalg.norm(tool - p) > 0.25: ev["drop"] = True
            if seg.name == "lift":
                p = sim.item_pose(name)[0]; ev["lifted_after_lift"] = bool(p[2] > p0[2] + 0.01)
        sim.step(int(0.8 / sim.model.opt.timestep))   # let the item settle in the bin
        return ev


def classify(sim, plan, name, ev):
    inf = I.info(name); p = sim.item_pose(name)[0]
    where = None
    for b in sim.lay.bins:
        bx, by = b.xy; a = math.radians(b.angle_deg); d = np.array([p[0] - bx, p[1] - by])
        lx, ly = d @ [math.cos(a), math.sin(a)], d @ [-math.sin(a), math.cos(a)]
        if abs(lx) < b.inner / 2 and abs(ly) < b.inner / 2 and p[2] < b.rim_z + 0.06: where = b.material
    if ev["collision"]: outcome = "collision"
    elif where == inf["material"]: outcome = "ok"
    elif where is not None: outcome = "wrong_bin"
    elif ev["knocked"]: outcome = "knocked"
    elif not ev.get("lifted_after_lift", False): outcome = "miss"
    elif ev["drop"]: outcome = "drop"
    else: outcome = "off_target"
    return outcome, where


def scene(sim, name, seed):
    """Place the item of an L0 trial: a random yaw, a random y in the zone, and a random x across the belt such that the
    turned footprint stays 5 mm inside the rails (a first version spawned the 103 mm can across a rail, which threw it)."""
    rng = np.random.default_rng(seed); b = sim.lay.belt; ex, ey = I.info(name)["extents"][:2]
    yaw = rng.uniform(-math.pi, math.pi); y = rng.uniform(*b.zone)
    half_x = 0.5 * (abs(ex * math.cos(yaw)) + abs(ey * math.sin(yaw))); room = max(0.0, b.width / 2 - 0.005 - half_x)
    x = b.x + rng.uniform(-room, room)
    sim.reset(); sim.spawn(name, x, y, yaw); sim.step(int(0.3 / sim.model.opt.timestep))
    return rng, (x, y, yaw)


def trial(sim, name, seed, cam=None, record=None):
    """One seeded L0 trial: the item placed by `scene`, seen (the camera's estimate and class if `cam` is given,
    else the privileged pose and the true class), planned, picked, classified."""
    rng, (x, y, yaw) = scene(sim, name, seed)
    if cam is not None:
        cam.flush(); e = cam.estimate()
        if e is None: return dict(item=name, seed=seed, outcome="unseen", x=x, y=y, yaw=yaw)
        seen, est = e["item"], (e["x"], e["y"], e["yaw"])
    else:
        seen, est = name, sim.item_xy_yaw(name)
    plan = Planner(sim).plan(seen, est)
    if plan is None: return dict(item=name, seen=seen, seed=seed, outcome="ik_fail", x=x, y=y, yaw=yaw)
    ev = Executor(sim).run(plan, name, on_tick=record)
    outcome, where = classify(sim, plan, name, ev)
    return dict(item=name, seen=seen, seed=seed, outcome=outcome, bin=where, x=round(x, 4), y=round(y, 4), yaw=round(yaw, 3), T=round(plan.T, 2),
                collision=ev["collision"], est=[round(float(v), 4) for v in est])
