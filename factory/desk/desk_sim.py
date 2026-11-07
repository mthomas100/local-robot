"""desk_sim.py — the desk cell at run time (2026-11-03, F6): the compiled model, the servo bus (ideal or real-device),
item placement, stepping, contacts and rendering. One `DeskSim` per process; every trial starts from `reset`.
"""
from __future__ import annotations
import math, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_desk  # noqa: E402
import desk_items as I  # noqa: E402
import so101  # noqa: E402
from desk_layout import DEFAULT  # noqa: E402
from servo import ServoBus, DeviceKit, REAL_PLANT  # noqa: E402


def yaw_quat(yaw): return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


class DeskSim:
    def __init__(self, items=None, kit: DeviceKit | None = None, lay=DEFAULT, plant=None, rng=None, render=False):
        self.lay = lay; self.kit = kit or DeviceKit.ideal()
        self.items = list(I.admitted() if items is None else items)
        if plant is None and self.kit.real: plant = REAL_PLANT
        self.plant = plant
        spec = build_desk.desk_spec(lay, self.items, backlash=self.kit.real and self.kit.backlash_rad > 0, backlash_rad=self.kit.backlash_rad, plant=plant)
        self.model = spec.compile(); self.data = mujoco.MjData(self.model)
        self.ix = so101.Indices(self.model)
        self.bus = ServoBus(self.model, self.data, self.kit, rng=rng)
        m = self.model
        self.item_body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.items}
        self.item_qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")] for n in self.items}
        self.item_dadr = {n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")] for n in self.items}
        self.belt_dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "belt_slide")]
        self.belt_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "belt_slide")]
        self.belt_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "belt_drive")
        robot_bodies = set()
        for b in range(m.nbody):
            r = b
            while r > 0 and m.body(r).name != "base": r = m.body_parentid[r]
            if r > 0: robot_bodies.add(b)
        self.robot_bodies = robot_bodies
        self.jaw_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gripper", "moving_jaw_so101_v1")}
        self.fixture_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 or m.body(m.geom_bodyid[g]).name.startswith("bin_")}
        self._renderers = {}
        self.reset()

    # ---- state
    def reset(self, q=None):
        mujoco.mj_resetData(self.model, self.data)
        for k, n in enumerate(self.items): self.park(n, k)
        self.data.ctrl[self.belt_act] = 0.0
        self.bus.reset(np.array(self.lay.home_q if q is None else q, float), t=0.0)
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self):
        """Everything needed to put the sim back exactly (physics state and the servo bus)."""
        import copy
        d = self.data
        return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(), time=d.time,
                    qacc_warmstart=d.qacc_warmstart.copy(), bus=copy.deepcopy((self.bus.goal, list(self.bus.pending), list(self.bus.history))))

    def restore(self, snap):
        d = self.data
        d.qpos[:] = snap["qpos"]; d.qvel[:] = snap["qvel"]; d.act[:] = snap["act"]; d.ctrl[:] = snap["ctrl"]; d.time = snap["time"]
        d.qacc_warmstart[:] = snap["qacc_warmstart"]
        g, pend, hist = snap["bus"]; self.bus.goal = g; self.bus.pending.clear(); self.bus.pending.extend(pend); self.bus.history.clear(); self.bus.history.extend(hist)
        mujoco.mj_forward(self.model, d)

    def park(self, name, k=None):
        k = self.items.index(name) if k is None else k; a = self.item_qadr[name]
        self.data.qpos[a:a + 3] = (self.lay.park[0] + 0.15 * k, self.lay.park[1], self.lay.park[2]); self.data.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        self.data.qvel[self.item_dadr[name]:self.item_dadr[name] + 6] = 0

    def spawn(self, name, x, y, yaw):
        """Place an item at rest on the belt: its footprint centre at (x, y), turned by yaw."""
        inf = I.info(name); a = self.item_qadr[name]
        cx, cy = inf["center_xy"]; c, s = math.cos(yaw), math.sin(yaw)
        self.data.qpos[a:a + 3] = (x - (c * cx - s * cy), y - (s * cx + c * cy), self.lay.belt.surface_z - inf["bottom"] + 0.001)
        self.data.qpos[a + 3:a + 7] = yaw_quat(yaw); self.data.qvel[self.item_dadr[name]:self.item_dadr[name] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def belt(self, speed): self.data.ctrl[self.belt_act] = speed

    @property
    def t(self): return self.data.time

    def step(self, n=1):
        for _ in range(n):
            self.bus.step(self.data.time); mujoco.mj_step(self.model, self.data)

    def item_pose(self, name):
        b = self.item_body[name]; return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def item_xy_yaw(self, name):
        """Footprint centre and yaw, as the planner and the camera report them."""
        p, R = self.item_pose(name); inf = I.info(name); cx, cy = inf["center_xy"]
        c = p + R @ np.array([cx, cy, 0.0]); return c[0], c[1], math.atan2(R[1, 0], R[0, 0])

    def tilt_deg(self, name):
        _, R = self.item_pose(name); return math.degrees(math.acos(max(-1, min(1, R[2, 2]))))

    def tool(self):
        s = self.ix.site; return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    # ---- contacts
    def contact_pairs(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0: continue
            yield c.geom1, c.geom2, m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]

    def robot_fixture_contact(self):
        """A robot geom touching the desk, riser (except the base, which sits on it), belt or a bin."""
        m = self.model; base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"); belt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "belt_plate")
        for g1, g2, b1, b2 in self.contact_pairs():
            for ga, ba, gb, bb in ((g1, b1, g2, b2), (g2, b2, g1, b1)):
                if ba in self.robot_bodies and ba != base and (gb in self.fixture_geoms or bb == belt): return m.geom(ga).name or m.body(ba).name, m.geom(gb).name
        return None

    def jaw_contacts(self, name):
        """Which jaw bodies touch the item: {'fixed': bool, 'moving': bool}."""
        b = self.item_body[name]; m = self.model
        fixed = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper"); moving = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        out = dict(fixed=False, moving=False)
        for g1, g2, b1, b2 in self.contact_pairs():
            if b in (b1, b2):
                o = b2 if b1 == b else b1
                if o == fixed: out["fixed"] = True
                if o == moving: out["moving"] = True
        return out

    # ---- pictures
    def render(self, camera="desk", width=640, height=360, depth=False, segmentation=False):
        key = (width, height)
        if key not in self._renderers: self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = self._renderers[key]
        if depth: r.enable_depth_rendering()
        elif segmentation: r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=camera); img = r.render()
        if depth: r.disable_depth_rendering()
        if segmentation: r.disable_segmentation_rendering()
        return img

    def still(self, path, camera="desk", width=1280, height=720):
        import imageio.v2 as iio
        iio.imwrite(path, self.render(camera, width, height)); return path
