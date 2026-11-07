"""cell_mujoco.py — the sorting cell running in MuJoCo (2026-11-02, discipline 2; the cross-check engine).

    from cell_mujoco import CellMJ
    cell = CellMJ()                       # builds factory/cell/build/cell.xml if missing, loads it, parks the items
    cell.spawn("mustard_bottle", x=-1.0)  # put an item on the belt (upright, random yaw)
    cell.belt(0.3)                        # belt speed in m/s (the treadmill plate teleports back by itself)
    cell.step(500)                        # 1 s at the layout timestep
    cell.render("cell")                   # an RGB frame from a named camera
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from layout import Layout, DEFAULT  # noqa: E402
import ycb  # noqa: E402

PARK = (2.6, -2.2, 0.0)   # parked items wait here, out of every camera's view


class CellMJ:
    def __init__(self, lay: Layout = DEFAULT, xml=None, rebuild=False):
        xml = xml or os.path.join(HERE, "build", "cell.xml")
        if rebuild or not os.path.exists(xml):
            import build_mjcf; build_mjcf.write(lay)
        self.lay = lay; self.index = ycb.load_index()
        self.model = mujoco.MjModel.from_xml_path(xml); self.data = mujoco.MjData(self.model)
        m = self.model
        self.robot_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.robot_qpos = np.array([m.joint(j).qposadr[0] for j in self.robot_joints])
        self.robot_dof = np.array([m.joint(j).dofadr[0] for j in self.robot_joints])
        self.robot_act = np.array([m.actuator(a).id for a in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")])
        self.grip_act = m.actuator("2f85/fingers_actuator").id
        self.belt_act = m.actuator("belt_drive").id
        self.belt_qpos = m.joint("belt_slide").qposadr[0]; self.belt_dof = m.joint("belt_slide").dofadr[0]
        self.belt_body = m.body("belt_plate").id
        self.items = {}
        for rec in self.index.values():
            n = rec["name"]
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) < 0: continue
            self.items[n] = dict(body=m.body(n).id, qpos=m.joint(f"{n}_free").qposadr[0], dof=m.joint(f"{n}_free").dofadr[0], rec=rec)
        self.pinch_site = m.site("2f85/pinch").id
        # the vacuum (build_mjcf.add_suction / add_vacuum_welds): one inactive weld per item, sealed by vacuum(True)
        self.vac_eq = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"vac_{n}") for n in self.items}
        self.vac_eq = {n: e for n, e in self.vac_eq.items() if e >= 0}
        self.suction_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "suction")
        self.cup_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "suction_cup")
        self.item_of_body = {it["body"]: n for n, it in self.items.items()}
        import build_mjcf
        self.vac_cfg = dict(build_mjcf.VACUUM)
        self._renderers = {}
        self.reset()

    # --- state -------------------------------------------------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.robot_qpos] = self.lay.home_qpos; self.data.ctrl[self.robot_act] = self.lay.home_qpos
        self.data.ctrl[self.grip_act] = 0.0
        for k, n in enumerate(self.items): self.park(n, k)
        self.belt_target = 0.0; self.belt_ramp = self.lay.belt.ramp; self.data.ctrl[self.belt_act] = 0.0
        self.vac_cmd = False; self.vac_item = None; self.vac_events = []
        for e in self.vac_eq.values(): self.data.eq_active[e] = 0   # only the vacuum welds: the 2F-85's linkage is made of equality constraints too
        mujoco.mj_forward(self.model, self.data)

    def park(self, name, k=0):
        it = self.items[name]; q = it["qpos"]
        self.data.qpos[q:q + 3] = (PARK[0] + 0.3 * (k % 6), PARK[1] - 0.3 * (k // 6), PARK[2] - it["rec"]["bounds"][0][2] + 0.002)
        self.data.qpos[q + 3:q + 7] = (1, 0, 0, 0); self.data.qvel[it["dof"]:it["dof"] + 6] = 0

    def spawn(self, name, x=None, y=None, yaw=None, lift=0.003, rng=None, with_belt_velocity=True):
        """Place an item upright on the belt at x (default the layout's spawn_x), centred on the belt, with a yaw
        (default: random). Its mesh frame's lowest point ends `lift` above the belt surface."""
        rng = rng or np.random.default_rng(); it = self.items[name]; q = it["qpos"]; b = self.lay.belt
        x = self.lay.spawn_x if x is None else x; y = b.y if y is None else y; yaw = rng.uniform(-math.pi, math.pi) if yaw is None else yaw
        self.data.qpos[q:q + 3] = (x, y, b.surface_z - it["rec"]["bounds"][0][2] + lift)
        self.data.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qvel[it["dof"]:it["dof"] + 6] = 0
        if with_belt_velocity: self.data.qvel[it["dof"]:it["dof"] + 3] = np.asarray(b.direction) * self.belt_velocity()   # it arrives from upstream, already moving
        mujoco.mj_forward(self.model, self.data)

    def belt(self, speed, ramp=None):
        """Command the belt speed (m/s). The command ramps at `ramp` m/s^2 (default the layout's belt.ramp) like a real
        conveyor drive, so a standing item is not jolted over."""
        self.belt_target = float(speed); self.belt_ramp = self.lay.belt.ramp if ramp is None else ramp
        if ramp == 0: self.data.ctrl[self.belt_act] = speed

    def step(self, n=1):
        b = self.lay.belt; m, d = self.model, self.data; dt = m.opt.timestep
        for _ in range(n):
            c = d.ctrl[self.belt_act]; dv = self.belt_target - c
            if dv: d.ctrl[self.belt_act] = c + np.clip(dv, -self.belt_ramp * dt, self.belt_ramp * dt)
            mujoco.mj_step(m, d)
            if d.qpos[self.belt_qpos] >= b.reset_travel: d.qpos[self.belt_qpos] -= b.reset_travel     # the treadmill: slide back, keep velocity
            elif d.qpos[self.belt_qpos] <= -b.reset_travel: d.qpos[self.belt_qpos] += b.reset_travel
            if self.vac_cmd: self._vacuum_update()

    # --- the vacuum cup -------------------------------------------------------------------------------------------
    def vacuum(self, on):
        """Command the vacuum. On: the cup seals on the first item face it touches squarely (contact normal within
        seal_angle_deg of the cup axis) and holds it with a weld; off: the weld releases at once."""
        self.vac_cmd = bool(on)
        if not on and self.vac_item is not None:
            self.data.eq_active[self.vac_eq[self.vac_item]] = 0; self.vac_events.append(("release", self.vac_item, round(self.data.time, 3))); self.vac_item = None

    def _vacuum_update(self):
        m, d = self.model, self.data
        if self.vac_item is None:
            axis = d.xmat[self.suction_body].reshape(3, 3)[:, 2]; cos_max = math.cos(math.radians(self.vac_cfg["seal_angle_deg"]))
            for i in range(d.ncon):
                c = d.contact[i]
                if self.cup_geom not in (c.geom1, c.geom2) or c.dist > 0.0025: continue   # a bellows lip seals across a few mm (the cup has a 2 mm margin)
                n = self.item_of_body.get(m.geom_bodyid[c.geom2 if c.geom1 == self.cup_geom else c.geom1])
                if n is None or abs(float(c.frame[:3] @ axis)) < cos_max: continue
                self._seal(n); return
        else:
            f = self.vacuum_force()
            if f > self.vac_cfg["max_force"]:   # the seal breaks: the item falls
                self.vac_events.append(("seal_broke", self.vac_item, round(d.time, 3), round(f, 1)))
                d.eq_active[self.vac_eq[self.vac_item]] = 0; self.vac_item = None

    def _seal(self, name):
        """Weld the item to the cup in its current relative pose (anchor at the item origin; relpose = item in cup frame)."""
        m, d = self.model, self.data; e = self.vac_eq[name]; b1, b2 = self.suction_body, self.items[name]["body"]
        R1 = d.xmat[b1].reshape(3, 3); relpos = R1.T @ (d.xpos[b2] - d.xpos[b1])
        q1i = np.zeros(4); mujoco.mju_negQuat(q1i, d.xquat[b1]); relq = np.zeros(4); mujoco.mju_mulQuat(relq, q1i, d.xquat[b2])
        m.eq_data[e, 0:3] = 0.0; m.eq_data[e, 3:6] = relpos; m.eq_data[e, 6:10] = relq; m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1; self.vac_item = name; self.vac_events.append(("seal", name, round(d.time, 3)))

    def vacuum_force(self):
        """Force (N) the active vacuum weld transmits (the item's weight plus its inertial load), 0 when nothing is held."""
        if self.vac_item is None: return 0.0
        d = self.data; e = self.vac_eq[self.vac_item]; n = d.nefc
        rows = np.flatnonzero((d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:n] == e))
        return float(np.linalg.norm(d.efc_force[rows[:3]])) if len(rows) else 0.0

    @property
    def t(self): return self.data.time

    def item_state(self, name):
        it = self.items[name]; q, v = it["qpos"], it["dof"]; d = self.data
        R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[q + 3:q + 7]); R = R.reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))   # angle between the item's z axis and world up
        return dict(pos=d.qpos[q:q + 3].copy(), quat=d.qpos[q + 3:q + 7].copy(), linvel=d.qvel[v:v + 3].copy(), angvel=d.qvel[v + 3:v + 6].copy(), tilt_deg=tilt)

    def belt_velocity(self): return float(self.data.qvel[self.belt_dof])

    def penetration(self, name=None):
        """Deepest contact penetration (m, positive = overlap) over the active contacts of one item, or over all
        contacts when name is None."""
        n = self.data.ncon
        if not n: return 0.0
        if name is None: return float(max(0.0, -self.data.contact.dist[:n].min()))
        return float(max([0.0] + [-d for *_, d in self.contacts_with(name)]))

    def contacts_with(self, name):
        it = self.items[name]; m, d = self.model, self.data; out = []
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if it["body"] in (b1, b2): out.append((m.body(b1).name, m.body(b2).name, float(c.dist)))
        return out

    def pinch_pos(self): return self.data.site_xpos[self.pinch_site].copy()

    # --- rendering ---------------------------------------------------------------------------------------------
    def render(self, camera="cell", width=None, height=None):
        if camera == self.lay.camera.name: width, height = width or self.lay.camera.width, height or self.lay.camera.height
        width, height = width or 1280, height or 720
        key = (width, height)
        if key not in self._renderers: self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._renderers[key]; r.update_scene(self.data, camera=camera); return r.render()

    def record(self, path, seconds, camera="cell", fps=30, width=1280, height=720, on_frame=None):
        import imageio.v3 as iio
        every = max(1, round(1.0 / (self.model.opt.timestep * fps))); frames = []
        for k in range(int(seconds / self.model.opt.timestep)):
            self.step()
            if k % every == 0:
                frames.append(self.render(camera, width, height).copy())
                if on_frame: on_frame(self, frames[-1])
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", plugin="pyav")
        return len(frames)

    def still(self, path, camera="cell", width=1280, height=720):
        import imageio.v3 as iio
        iio.imwrite(path, self.render(camera, width, height)); return path


if __name__ == "__main__":
    cell = CellMJ()
    print("model: nbody", cell.model.nbody, "ngeom", cell.model.ngeom, "items", list(cell.items))
    cell.belt(0.3); cell.step(500)   # the belt is already running when the items arrive, as in a real line
    cell.spawn("mustard_bottle", yaw=0.3); cell.spawn("tomato_soup_can", x=cell.lay.spawn_x + 0.3); cell.spawn("cracker_box", x=cell.lay.spawn_x + 0.6)
    t0 = time.perf_counter(); cell.step(1500)
    print(f"3 s simulated in {time.perf_counter() - t0:.2f} s; belt v {cell.belt_velocity():.3f}; penetration {cell.penetration() * 1000:.2f} mm; pinch {cell.pinch_pos().round(3)}")
    for n in ("mustard_bottle", "tomato_soup_can", "cracker_box"):
        s = cell.item_state(n); print(f"  {n:18s} pos {s['pos'].round(3)} v {s['linvel'].round(3)} tilt {s['tilt_deg']:.1f} deg")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for cam in ("cell", "overhead", "belt"): print(cell.still(os.path.join(HERE, "results", f"smoke-{cam}.png"), cam))
