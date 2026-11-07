"""build_desk.py — the desk cell as MJCF (2026-11-03, F6; discipline 2): a desk, the SO-101 on its riser, a short
belt section (a sliding plate with a velocity drive, as in the v1 cell), three bins, an overhead RGB-D camera on a
boom, lights, video cameras and the items (YCB scans through the v1 cell's `add_item`, plus desk-scale stand-ins built
from primitives, see `items.py`). Unused items wait off the desk at `park`.

    envs/mj/bin/python -I factory/desk/build_desk.py [--items admitted|all|<name>,...] [--backlash]
Writes factory/desk/build/desk.xml (absolute mesh paths, like factory/cell/build).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "factory", "cell"))
import so101  # noqa: E402
from desk_layout import DEFAULT, DeskLayout  # noqa: E402
import desk_items as I  # noqa: E402
import build_mjcf as V1  # noqa: E402  (the v1 cell: add_item, lookat_quat, the item contact settings)

BUILD = os.path.join(HERE, "build")
GEOM = mujoco.mjtGeom
FIX = dict(contype=1, conaffinity=1, group=0)
VIS = dict(contype=0, conaffinity=0, group=2)
RGBA = dict(desk=(0.55, 0.42, 0.30, 1), riser=(0.30, 0.31, 0.33, 1), rail=(0.25, 0.27, 0.3, 1), plate=(0.12, 0.12, 0.13, 1),
            plastic=(0.2, 0.5, 0.9, 1), metal=(0.72, 0.72, 0.76, 1), cardboard=(0.75, 0.55, 0.3, 1), boom=(0.2, 0.2, 0.22, 1))


def _options(s, lay):
    s.option.timestep = lay.timestep
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; s.option.impratio = 10; s.option.noslip_iterations = 3
    s.option.iterations = 30; s.option.ls_iterations = 30
    s.visual.global_.offwidth = 1280; s.visual.global_.offheight = 720; s.visual.map.znear = 0.005
    s.visual.headlight.diffuse[:] = (0.45, 0.45, 0.45); s.visual.headlight.ambient[:] = (0.3, 0.3, 0.3)


def add_desk(s, lay: DeskLayout):
    s.add_texture(name="desk_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=(0.62, 0.48, 0.34), rgb2=(0.5, 0.38, 0.26), width=64, height=64)
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=(0.55, 0.65, 0.8), rgb2=(0.9, 0.92, 0.95), width=256, height=1536)
    s.worldbody.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=(3, 3, 0.1), pos=(0, 0, -0.75), rgba=(0.8, 0.8, 0.82, 1), **FIX)
    s.worldbody.add_geom(name="desk", type=GEOM.mjGEOM_BOX, size=(0.45, 0.6, 0.02), pos=(0.15, 0, -0.02), rgba=RGBA["desk"], **FIX)
    rx, ry, rz = lay.riser
    s.worldbody.add_geom(name="riser", type=GEOM.mjGEOM_BOX, size=(rx / 2, ry / 2, rz / 2), pos=(0.0, 0, rz / 2), rgba=RGBA["riser"], **FIX)
    s.worldbody.add_light(name="sun", pos=(0.4, -0.4, 1.6), dir=(-0.2, 0.25, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.6, 0.6, 0.6), specular=(0.15, 0.15, 0.15), castshadow=True)
    s.worldbody.add_light(name="fill", pos=(-0.8, 0.8, 1.2), dir=(0.5, -0.5, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.3, 0.3, 0.3), specular=(0, 0, 0), castshadow=False)


def add_belt(s, lay: DeskLayout):
    b = lay.belt; L = (b.y_max - b.y_min) / 2; cy = (b.y_max + b.y_min) / 2
    for side, sgn in (("near", -1), ("far", 1)):
        x = b.x + sgn * (b.width / 2 + b.rail_gap + b.rail_thickness / 2)
        s.worldbody.add_geom(name=f"rail_{side}", type=GEOM.mjGEOM_BOX, size=(b.rail_thickness / 2, L, (b.rail_height + b.plate_thickness) / 2),
                             pos=(x, cy, b.surface_z - b.plate_thickness + (b.rail_height + b.plate_thickness) / 2), rgba=RGBA["rail"], **FIX)
    s.worldbody.add_geom(name="belt_frame", type=GEOM.mjGEOM_BOX, size=(b.width / 2 + b.rail_thickness, L, (b.surface_z - b.plate_thickness) / 2),
                         pos=(b.x, cy, (b.surface_z - b.plate_thickness) / 2), rgba=RGBA["rail"], **FIX)
    plate_len = 2 * L + b.reset_travel
    body = s.worldbody.add_body(name="belt_plate", pos=(b.x, cy + b.reset_travel / 2, b.surface_z - b.plate_thickness / 2))
    body.add_joint(name="belt_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=b.direction, damping=2.0)
    body.mass = 3.0; body.explicitinertial = True; body.ipos = (0, 0, 0); body.inertia = (0.1, 0.02, 0.1)
    body.add_geom(name="belt_plate", type=GEOM.mjGEOM_BOX, size=(b.width / 2, plate_len / 2, b.plate_thickness / 2), rgba=RGBA["plate"],
                  friction=(0.9, 0.005, 0.0001), priority=1, condim=4, solref=(0.004, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2), contype=1, conaffinity=3, group=0)
    s.add_actuator(name="belt_drive", target="belt_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                   biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=(800.0,) + (0.0,) * 9, biasprm=(0.0, 0.0, -800.0) + (0.0,) * 7, ctrlrange=(-0.5, 0.5), forcerange=(-300, 300))
    for name, y in (("zone_a", b.zone[0]), ("zone_b", b.zone[1])):
        s.worldbody.add_geom(name=name, type=GEOM.mjGEOM_BOX, size=(b.width / 2 + b.rail_thickness, 0.002, 0.0005), pos=(b.x, y, b.surface_z + b.rail_height + 0.0012), rgba=(1, 0.85, 0.1, 1), **VIS)


def add_bins(s, lay: DeskLayout):
    for c in lay.bins:
        x, y = c.xy; half = c.inner / 2; hz = c.rim_z / 2
        yaw = np.radians(c.angle_deg); q = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        body = s.worldbody.add_body(name=f"bin_{c.material}", pos=(x, y, 0), quat=q)
        body.add_geom(name=f"bin_{c.material}_floor", type=GEOM.mjGEOM_BOX, size=(half, half, c.floor / 2), pos=(0, 0, c.floor / 2), rgba=RGBA[c.material], **FIX)
        for side, (dx, dy, sx, sy) in dict(w=(-1, 0, c.wall / 2, half + c.wall), e=(1, 0, c.wall / 2, half + c.wall), s=(0, -1, half, c.wall / 2), n=(0, 1, half, c.wall / 2)).items():
            body.add_geom(name=f"bin_{c.material}_{side}", type=GEOM.mjGEOM_BOX, size=(sx, sy, hz), pos=(dx * (half + c.wall / 2), dy * (half + c.wall / 2), hz), rgba=RGBA[c.material], **FIX)


def add_camera(s, lay: DeskLayout):
    c = lay.camera
    s.worldbody.add_camera(name=c.name, pos=c.pos, quat=V1.lookat_quat(c.pos, c.lookat, c.up), fovy=c.fovy)
    # a stalk up to the ceiling (out of frame) and the camera body (visual only)
    s.worldbody.add_geom(name="boom", type=GEOM.mjGEOM_CYLINDER, size=(0.006, 0.25), pos=(c.pos[0], c.pos[1], c.pos[2] + 0.27), rgba=RGBA["boom"], **VIS)
    s.worldbody.add_geom(name="camera_body", type=GEOM.mjGEOM_BOX, size=(0.0125, 0.045, 0.0125), pos=(c.pos[0], c.pos[1], c.pos[2] + 0.012), rgba=(0.15, 0.15, 0.16, 1), **VIS)
    for name, (pos, target) in lay.views.items():
        s.worldbody.add_camera(name=name, pos=pos, quat=V1.lookat_quat(pos, target), fovy=45)


MARKER_RADIUS = 0.007
MARKER_BODY_POS = (0.012, -0.0352, -0.058)   # gripper-body frame: 4 cm behind the tool frame, 3.5 cm to the side (site -y)


def add_marker(s):
    """The hand-eye calibration marker (camera.py): a 14 mm magenta ball on a short stalk, rigid to the gripper body,
    off to the side of the jaws so the overhead camera sees it from above at any wrist roll. It is a calibration
    fixture, so it is visual only (it never touches items or bins)."""
    g = s.body("gripper")
    g.add_geom(name="marker", type=GEOM.mjGEOM_SPHERE, size=(MARKER_RADIUS, 0, 0), pos=MARKER_BODY_POS, rgba=(1.0, 0.1, 0.6, 1), contype=0, conaffinity=0, group=1)
    g.add_geom(name="marker_stalk", type=GEOM.mjGEOM_CAPSULE, size=(0.002, 0.012, 0), pos=(0.012, -0.0232, -0.058), euler=(1.5708, 0, 0), rgba=(0.2, 0.2, 0.22, 1), contype=0, conaffinity=0, group=1)
    g.add_site(name="marker_center", pos=MARKER_BODY_POS, size=(0.002, 0, 0), group=5)


def desk_spec(lay: DeskLayout = DEFAULT, items=None, backlash=False, plant=None, backlash_rad=0.0087267):
    """The whole desk cell. `items`: item names to include (default: the admitted set)."""
    s = so101.arm_spec(backlash=backlash, backlash_rad=backlash_rad, plant=plant); s.modelname = "desk_cell"; _options(s, lay)
    s.body("base").pos = lay.base
    for lt in list(s.lights):
        try: s.delete(lt)
        except Exception: pass
    add_desk(s, lay); add_belt(s, lay); add_bins(s, lay); add_camera(s, lay); add_marker(s)
    names = I.admitted() if items is None else items
    for k, n in enumerate(names):
        I.add(s, n, pos=(lay.park[0] + 0.15 * k, lay.park[1], lay.park[2]))
    return s


def write(lay=DEFAULT, items=None, backlash=False, out=BUILD):
    os.makedirs(out, exist_ok=True)
    s = desk_spec(lay, items, backlash); m = s.compile(); p = os.path.join(out, "desk.xml")
    with open(p, "w") as f: f.write(s.to_xml())
    with open(os.path.join(out, "layout.json"), "w") as f: f.write(lay.to_json())
    return m, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--items", default="admitted"); ap.add_argument("--backlash", action="store_true")
    a = ap.parse_args()
    items = None if a.items == "admitted" else (I.ALL if a.items == "all" else a.items.split(","))
    m, p = write(DEFAULT, items, a.backlash)
    print(f"{p}: nbody {m.nbody} ngeom {m.ngeom} nq {m.nq} nv {m.nv} nu {m.nu}")
