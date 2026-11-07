"""cell_genesis.py — the sorting cell running in Genesis (2026-11-02, discipline 2; the batched engine for RL later).

Loads the same build/ files as MuJoCo (robot.xml, fixtures.xml, belt.xml, items/<id>.xml, all written by
build_mjcf.py from layout.py), so the two engines simulate one scene definition.

    from cell_genesis import CellGS
    cell = CellGS(backend="cpu")          # "metal" for the Apple GPU; CPU renders, Metal does not
    cell.spawn("mustard_bottle"); cell.belt(0.3); cell.step(500); cell.render("cell")
"""
from __future__ import annotations
import os, sys, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from layout import Layout, DEFAULT  # noqa: E402
import ycb  # noqa: E402

PARK = (2.6, -2.2, 0.0)
ROBOT_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
_inited = False


def init(backend="cpu"):
    global _inited
    import genesis as gs
    if not _inited:
        gs.init(backend=gs.cpu if backend == "cpu" else gs.metal, precision="32", logging_level="warning"); _inited = True
    return gs


class CellGS:
    def __init__(self, lay: Layout = DEFAULT, backend="cpu", items=None, cameras=True, n_envs=1, show_viewer=False, rebuild=False):
        gs = init(backend); self.gs = gs; self.lay = lay; self.index = ycb.load_index(); build = os.path.join(HERE, "build")
        if rebuild or not os.path.exists(os.path.join(build, "cell.xml")):
            import build_mjcf; build_mjcf.write(lay)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=lay.timestep, substeps=1),
            rigid_options=gs.options.RigidOptions(dt=lay.timestep, enable_collision=True, enable_joint_limit=True, enable_self_collision=False,
                                                friction_cone=gs.friction_cone.elliptic, enable_torsional_friction=True),
            vis_options=gs.options.VisOptions(show_world_frame=False, background_color=(0.72, 0.78, 0.88), ambient_light=(0.35, 0.35, 0.35)),
            viewer_options=gs.options.ViewerOptions(camera_pos=lay.views["cell"][0], camera_lookat=lay.views["cell"][1], camera_fov=45),
            show_viewer=show_viewer, show_FPS=False)
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.fixtures = self.scene.add_entity(gs.morphs.MJCF(file=os.path.join(build, "fixtures.xml")))
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=os.path.join(build, "robot.xml"), pos=lay.robot_base))
        self.belt_ent = self.scene.add_entity(gs.morphs.MJCF(file=os.path.join(build, "belt.xml")))
        self.items = {}
        ids = list(self.index) if items is None else [i for i in self.index if i in items or self.index[i]["name"] in items]
        for k, iid in enumerate(ids):
            rec = self.index[iid]
            ent = self.scene.add_entity(gs.morphs.MJCF(file=os.path.join(build, "items", f"{iid}.xml"), pos=self._park_pos(rec, k)))
            self.items[rec["name"]] = dict(ent=ent, rec=rec, k=k)
        self.cams = {}
        if cameras:
            c = lay.camera
            self.cams[c.name] = self.scene.add_camera(res=(c.width, c.height), pos=c.pos, lookat=c.lookat, up=c.up, fov=c.fovy, GUI=False)
            for name, (pos, target) in lay.views.items():
                self.cams[name] = self.scene.add_camera(res=(1280, 720), pos=pos, lookat=target, fov=45, GUI=False)
        self.scene.build(n_envs=n_envs)
        self.n_envs = n_envs
        self.belt_dof = [self.belt_ent.get_joint("belt_slide").dofs_idx_local[0]]
        self.belt_ent.set_dofs_kv([5000.0], self.belt_dof)
        self.robot_dofs = [self.robot.get_joint(j).dofs_idx_local[0] for j in ROBOT_JOINTS]
        self.robot.set_dofs_kp([2000, 2000, 2000, 500, 500, 500], self.robot_dofs); self.robot.set_dofs_kv([400, 400, 400, 100, 100, 100], self.robot_dofs)
        self.reset()

    def _park_pos(self, rec, k): return (PARK[0] + 0.3 * (k % 6), PARK[1] - 0.3 * (k // 6), PARK[2] - rec["bounds"][0][2] + 0.002)

    # --- state -------------------------------------------------------------------------------------------------
    def reset(self):
        self.scene.reset()
        self.robot.set_dofs_position(np.array(self.lay.home_qpos), self.robot_dofs); self.robot.control_dofs_position(np.array(self.lay.home_qpos), self.robot_dofs)
        for n, it in self.items.items(): self.park(n)
        self.belt_target = 0.0; self.belt_ramp = self.lay.belt.ramp; self.belt_cmd = 0.0; self.belt_ent.control_dofs_velocity([0.0], self.belt_dof)
        self.scene.step()

    def park(self, name):
        it = self.items[name]; it["ent"].set_pos(np.array(self._park_pos(it["rec"], it["k"])), zero_velocity=True); it["ent"].set_quat(np.array([1, 0, 0, 0]), zero_velocity=True)

    def spawn(self, name, x=None, y=None, yaw=None, lift=0.003, rng=None, with_belt_velocity=True):
        rng = rng or np.random.default_rng(); it = self.items[name]; b = self.lay.belt; ent = it["ent"]
        x = self.lay.spawn_x if x is None else x; y = b.y if y is None else y; yaw = rng.uniform(-math.pi, math.pi) if yaw is None else yaw
        ent.set_pos(np.array([x, y, b.surface_z - it["rec"]["bounds"][0][2] + lift]), zero_velocity=True)
        ent.set_quat(np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), zero_velocity=True)
        if with_belt_velocity:
            v = self.belt_velocity(); ent.set_dofs_velocity(np.array([b.direction[0] * v, b.direction[1] * v, b.direction[2] * v, 0, 0, 0]))

    def belt(self, speed, ramp=None):
        self.belt_target = float(speed); self.belt_ramp = self.lay.belt.ramp if ramp is None else ramp
        if ramp == 0: self.belt_cmd = speed

    def step(self, n=1):
        b = self.lay.belt; dt = self.lay.timestep
        for _ in range(n):
            dv = self.belt_target - self.belt_cmd
            if dv: self.belt_cmd += float(np.clip(dv, -self.belt_ramp * dt, self.belt_ramp * dt))
            self.belt_ent.control_dofs_velocity([self.belt_cmd], self.belt_dof)
            self.scene.step()
            q = float(self._np(self.belt_ent.get_dofs_position(self.belt_dof)).reshape(-1)[0])
            if q >= b.reset_travel: self.belt_ent.set_dofs_position([q - b.reset_travel], self.belt_dof)
            elif q <= -b.reset_travel: self.belt_ent.set_dofs_position([q + b.reset_travel], self.belt_dof)

    @staticmethod
    def _np(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    @property
    def t(self): return self.scene.t if hasattr(self.scene, "t") else None

    def belt_velocity(self): return float(self._np(self.belt_ent.get_dofs_velocity(self.belt_dof)).reshape(-1)[0])

    def item_state(self, name):
        ent = self.items[name]["ent"]
        pos = self._np(ent.get_pos()).reshape(-1)[:3]; quat = self._np(ent.get_quat()).reshape(-1)[:4]
        linvel = self._np(ent.get_vel()).reshape(-1)[:3]; angvel = self._np(ent.get_ang()).reshape(-1)[:3]
        w, x, y, z = quat; r33 = 1 - 2 * (x * x + y * y)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, r33))))
        return dict(pos=pos, quat=quat, linvel=linvel, angvel=angvel, tilt_deg=tilt)

    def penetration(self, name=None):
        """Deepest penetration (m) in the contacts of one item (or all items) after the last step; 0 if the engine
        reports none."""
        worst = 0.0
        for n, it in self.items.items():
            if name and n != name: continue
            try:
                c = it["ent"].get_contacts()
                pen = self._np(c["penetration"]).reshape(-1)
                if pen.size: worst = max(worst, float(pen.max()))
            except Exception:
                pass
        return worst

    def pinch_pos(self): return self._np(self.robot.get_link("2f85/pinch" if False else "2f85/base").get_pos()).reshape(-1)[:3]

    # --- rendering ---------------------------------------------------------------------------------------------
    def render(self, camera="cell"):
        out = self.cams[camera].render(rgb=True, depth=False, segmentation=False, normal=False)
        rgb = out[0] if isinstance(out, (tuple, list)) else out
        return np.asarray(rgb)[..., :3]

    def record(self, path, seconds, camera="cell", fps=30, on_frame=None):
        import imageio.v3 as iio
        every = max(1, round(1.0 / (self.lay.timestep * fps))); frames = []
        for k in range(int(seconds / self.lay.timestep)):
            self.step()
            if k % every == 0:
                frames.append(self.render(camera).copy())
                if on_frame: on_frame(self, frames[-1])
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", plugin="pyav")
        return len(frames)

    def still(self, path, camera="cell"):
        import imageio.v3 as iio
        iio.imwrite(path, self.render(camera)); return path


if __name__ == "__main__":
    t0 = time.perf_counter()
    cell = CellGS(backend=sys.argv[1] if len(sys.argv) > 1 else "cpu", items=["mustard_bottle", "tomato_soup_can", "cracker_box"])
    print(f"built in {time.perf_counter() - t0:.1f} s; items {list(cell.items)}")
    cell.belt(0.3); cell.step(500)
    cell.spawn("mustard_bottle", yaw=0.3); cell.spawn("tomato_soup_can", x=cell.lay.spawn_x + 0.3); cell.spawn("cracker_box", x=cell.lay.spawn_x + 0.6)
    t0 = time.perf_counter(); cell.step(1500)
    print(f"3 s simulated in {time.perf_counter() - t0:.2f} s; belt v {cell.belt_velocity():.3f}; penetration {cell.penetration() * 1000:.2f} mm")
    for n in cell.items:
        s = cell.item_state(n); print(f"  {n:18s} pos {s['pos'].round(3)} v {s['linvel'].round(3)} tilt {s['tilt_deg']:.1f} deg")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for cam in ("cell", "overhead", "belt"): print(cell.still(os.path.join(HERE, "results", f"smoke-gs-{cam}.png"), cam))
