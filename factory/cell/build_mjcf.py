"""build_mjcf.py — assemble the sorting cell as MJCF from layout.py and the YCB index, with MuJoCo's MjSpec
(2026-11-02, docs/sorting-cell-spec.md §1-§2, discipline 2).

Writes factory/cell/build/:
  cell.xml        the whole cell for MuJoCo: floor, pedestal, UR5e + Robotiq 2F-85 (Menagerie, attached at the UR5e's
                  attachment_site), belt (a sliding plate with a velocity actuator), chutes, the D435i, lights, cameras,
                  and every item from the index as a free body (visual mesh + CoACD hulls + explicit inertia)
  robot.xml       UR5e + 2F-85 alone (Genesis loads it as one entity)
  fixtures.xml    pedestal, belt rails, chutes, camera housing (static)
  belt.xml        the sliding plate alone
  items/<id>.xml  one free body per item
  layout.json     the layout the files were built from
Mesh and texture paths are absolute (this Mac), so the files need no asset copies; vendor/ stays untouched.

    envs/mj/bin/python -I factory/cell/build_mjcf.py [--items all|<id>,...]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layout import Layout, DEFAULT  # noqa: E402
import ycb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
MENAGERIE = os.path.join(ROOT, "vendor", "mujoco_menagerie")
BUILD = os.path.join(HERE, "build")
GEOM = mujoco.mjtGeom; JNT = mujoco.mjtJoint

COL = dict(contype=1, conaffinity=1, group=3)   # collision-only geoms (item hulls): hidden in renders
FIX = dict(contype=1, conaffinity=1, group=0)   # fixtures that collide and are seen (floor, pedestal, rails, plate, chutes)
VIS = dict(contype=0, conaffinity=0, group=2)
RGBA = dict(pedestal=(0.35, 0.36, 0.38, 1), rail=(0.25, 0.27, 0.3, 1), plate=(0.12, 0.12, 0.13, 1), floor=(0.8, 0.8, 0.8, 1),
            plastic=(0.2, 0.5, 0.9, 1), metal=(0.75, 0.75, 0.78, 1), cardboard=(0.75, 0.55, 0.3, 1), hull=(0.9, 0.3, 0.3, 0.3))


def lookat_quat(pos, target, up=(0, 0, 1)):
    """MuJoCo camera frame (looks down -z, +y up) as a wxyz quaternion."""
    pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
    z = pos - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1); q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q


def _absolute_meshes(spec, model_dir):
    """Rewrite every mesh/texture file of a Menagerie spec to an absolute path so it survives attach + to_xml."""
    for m in spec.meshes:
        if not os.path.isabs(m.file): m.file = os.path.join(model_dir, spec.meshdir or "", m.file)
    for t in spec.textures:
        if t.file and not os.path.isabs(t.file): t.file = os.path.join(model_dir, spec.texturedir or spec.meshdir or "", t.file)
    spec.meshdir = ""; spec.texturedir = ""


def robot_spec():
    """UR5e with the 2F-85 on its flange (both Menagerie, BSD-3 / BSD-2), as one MjSpec."""
    arm_dir = os.path.join(MENAGERIE, "universal_robots_ur5e"); grip_dir = os.path.join(MENAGERIE, "robotiq_2f85")
    arm = mujoco.MjSpec.from_file(os.path.join(arm_dir, "ur5e.xml")); _absolute_meshes(arm, arm_dir)
    grip = mujoco.MjSpec.from_file(os.path.join(grip_dir, "2f85.xml")); _absolute_meshes(grip, grip_dir)
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; arm.option.impratio = 10   # the gripper's contact settings, so attach has no conflict
    arm.site("attachment_site").attach_body(grip.worldbody.first_body(), "2f85/", "")
    arm.modelname = "ur5e_2f85"
    add_suction(arm)
    return arm


SUCTION = dict(retracted_tip=0.085, extended_tip=0.175, cup_radius=0.02, cup_half=0.004, rod_radius=0.008, force=80.0)


def add_suction(spec):
    """The hybrid gripper (2026-11-02, F1): a retractable suction cup on the 2F-85's centre axis, as on the hybrid
    finger+suction grippers of bin-picking cells. Four of the 11 YCB items are wider than the 2F-85's 85 mm stroke
    (coffee can 104 mm, tuna can 86, gelatin and pudding boxes 90+ lying flat), the spec's 'vacuum cup for boxes the
    fingers cannot span'. In the gripper base frame (z = approach, the pinch point at z 0.145) the cup tip sits at z
    0.085 retracted (above an item held 4 cm deep between the pads) and 0.175 extended (3 cm past the fingertips).
    The vacuum itself is a weld (`add_vacuum_welds`, one inactive weld per item) that the runtime switches on when the
    cup touches a face squarely (the seal check) and off when the holding force exceeds the cup's limit (CellMJ.vacuum).
    The first version used MuJoCo's adhesion actuator; at 80 N it yanked the 97 g gelatin box into the cup, which
    tipped it against an open finger and flipped it (3 of 18 gelatin trials, 2026-11-02). The cup collides only with
    items and the belt (contype/conaffinity bit 2), never with the gripper it is part of."""
    S = SUCTION; base = spec.body("2f85/base")
    body = base.add_body(name="suction", pos=(0, 0, S["retracted_tip"] - S["cup_half"]))
    body.add_joint(name="suction_slide", type=JNT.mjJNT_SLIDE, axis=(0, 0, 1), range=(0, S["extended_tip"] - S["retracted_tip"]), damping=20.0, armature=0.01)
    body.mass = 0.08; body.explicitinertial = True; body.inertia = (2e-5, 2e-5, 1e-5)
    body.add_geom(name="suction_rod", type=GEOM.mjGEOM_CYLINDER, size=(S["rod_radius"], 0.04, 0), pos=(0, 0, -0.04), rgba=(0.6, 0.6, 0.62, 1), contype=0, conaffinity=0, group=2)
    body.add_geom(name="suction_cup", type=GEOM.mjGEOM_CYLINDER, size=(S["cup_radius"], S["cup_half"], 0), rgba=(0.15, 0.55, 0.85, 1),
                  contype=2, conaffinity=2, group=0, condim=4, friction=(1.0, 0.01, 0.001), margin=0.002, gap=0.002, solref=(0.004, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2))
    body.add_site(name="suction_tip", pos=(0, 0, S["cup_half"]), size=(0.004, 0, 0), rgba=(0.1, 0.6, 1, 1), group=5)
    spec.add_actuator(name="suction_extend", target="suction_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=(400.0,) + (0.0,) * 9, biasprm=(0.0, -400.0, -40.0) + (0.0,) * 7,
                      ctrlrange=(0.0, S["extended_tip"] - S["retracted_tip"]), forcerange=(-60, 60))


VACUUM = dict(seal_angle_deg=25.0, max_force=60.0)   # 40 mm cup at about -0.5 bar holds ~60 N; seal only on a square face


def add_vacuum_welds(spec, names):
    """One inactive weld per item between the suction body and the item; the runtime activates the one it seals on."""
    for n in names:
        e = spec.add_equality(); e.name = f"vac_{n}"; e.type = mujoco.mjtEq.mjEQ_WELD; e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = "suction"; e.name2 = n; e.active = False; e.solref = (0.005, 1.0); e.solimp = (0.95, 0.99, 0.001, 0.5, 2)


def _options(spec, lay: Layout):
    spec.option.timestep = lay.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; spec.option.impratio = 10
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1280; spec.visual.global_.offheight = 720
    spec.visual.map.znear = 0.01
    spec.visual.headlight.diffuse[:] = (0.5, 0.5, 0.5); spec.visual.headlight.ambient[:] = (0.25, 0.25, 0.25)
    spec.meshdir = ""; spec.texturedir = ""


def add_floor_and_lights(spec, lay: Layout):
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=(0.78, 0.78, 0.8), rgb2=(0.62, 0.62, 0.65), markrgb=(0.9, 0.9, 0.9), mark=mujoco.mjtMark.mjMARK_EDGE, width=512, height=512)
    spec.add_material(name="floor_mat", texrepeat=(8, 8), texuniform=True, reflectance=0.05).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=(0.55, 0.65, 0.8), rgb2=(0.9, 0.92, 0.95), width=512, height=3072)
    spec.worldbody.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=(4, 4, 0.1), material="floor_mat", **FIX)
    spec.worldbody.add_light(name="sun", pos=(0.5, -0.5, 3.0), dir=(-0.2, 0.3, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.7, 0.7, 0.7), specular=(0.2, 0.2, 0.2), castshadow=False)
    spec.worldbody.add_light(name="fill", pos=(-1.5, 2.0, 2.5), dir=(0.5, -0.6, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(0.35, 0.35, 0.35), specular=(0, 0, 0), castshadow=False)


def add_fixtures(spec, lay: Layout, camera_mesh=True):
    b = lay.belt; px, py, pz = lay.pedestal
    spec.worldbody.add_geom(name="pedestal", type=GEOM.mjGEOM_BOX, size=(px / 2, py / 2, pz / 2), pos=(0, 0, pz / 2), rgba=RGBA["pedestal"], **FIX)
    # belt frame: two side rails along the visible span, plus the underside beam
    L = (b.x_max - b.x_min) / 2; cx = (b.x_max + b.x_min) / 2
    for side, s in (("near", -1), ("far", 1)):
        y = b.y + s * (b.width / 2 + b.rail_gap + b.rail_thickness / 2)
        spec.worldbody.add_geom(name=f"rail_{side}", type=GEOM.mjGEOM_BOX, size=(L, b.rail_thickness / 2, (b.rail_height + b.plate_thickness) / 2),
                                pos=(cx, y, b.surface_z - b.plate_thickness + (b.rail_height + b.plate_thickness) / 2), rgba=RGBA["rail"], **FIX)
    spec.worldbody.add_geom(name="belt_beam", type=GEOM.mjGEOM_BOX, size=(L, b.width / 2 + b.rail_thickness, 0.04), pos=(cx, b.y, b.surface_z - b.plate_thickness - 0.045), rgba=RGBA["rail"], **VIS)
    for k in (-0.9, 0.0, 0.9):
        spec.worldbody.add_geom(name=f"belt_leg_{k:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_"), type=GEOM.mjGEOM_BOX, size=(0.03, b.width / 2 + b.rail_thickness, (b.surface_z - 0.08) / 2),
                                pos=(k, b.y, (b.surface_z - 0.08) / 2), rgba=RGBA["rail"], **VIS)
    # zone markers (visual only): two thin lines across the belt at the reach-zone ends
    for name, x in (("zone_in", b.zone[0]), ("zone_out", b.zone[1])):
        spec.worldbody.add_geom(name=name, type=GEOM.mjGEOM_BOX, size=(0.004, b.width / 2 + b.rail_thickness, 0.001), pos=(x, b.y, b.surface_z + b.rail_height + 0.002), rgba=(1, 0.85, 0.1, 1), **VIS)
    # chutes: open-top bins; the bottom plate sits at rim_z - depth
    for c in lay.chutes:   # floor top at rim_z - depth; the opening is inner x inner; walls and floor grow outward/downward
        half = c.inner / 2 + c.wall; zf = c.rim_z - c.depth; zb = zf - c.floor; hz = (c.rim_z - zb) / 2
        spec.worldbody.add_geom(name=f"chute_{c.material}_floor", type=GEOM.mjGEOM_BOX, size=(c.inner / 2, c.inner / 2, c.floor / 2), pos=(c.x, c.y, zf - c.floor / 2), rgba=RGBA[c.material], **FIX)
        for side, (dx, dy, sx, sy) in dict(w=(-1, 0, c.wall / 2, half), e=(1, 0, c.wall / 2, half), s=(0, -1, c.inner / 2, c.wall / 2), n=(0, 1, c.inner / 2, c.wall / 2)).items():
            spec.worldbody.add_geom(name=f"chute_{c.material}_{side}", type=GEOM.mjGEOM_BOX, size=(sx, sy, hz),
                                    pos=(c.x + dx * (c.inner / 2 + c.wall / 2), c.y + dy * (c.inner / 2 + c.wall / 2), zb + hz), rgba=RGBA[c.material], **FIX)
    # the overhead camera housing (Menagerie D435i, Apache-2.0): its lenses face +z of its mesh frame, so flip it
    if camera_mesh:
        cam = lay.camera
        mount = spec.worldbody.add_body(name="camera_mount", pos=cam.pos)
        mount.add_geom(name="camera_boom", type=GEOM.mjGEOM_CYLINDER, size=(0.012, 0.3), pos=(0, 0, 0.32), rgba=RGBA["rail"], **VIS)
        d_dir = os.path.join(MENAGERIE, "realsense_d435i"); d = mujoco.MjSpec.from_file(os.path.join(d_dir, "d435i.xml")); _absolute_meshes(d, d_dir)
        d.compiler.fitaabb = True
        for g in d.worldbody.first_body().geoms: g.contype = 0; g.conaffinity = 0   # the housing never collides with anything
        mount.add_frame(name="d435i_frame", pos=(0, 0, 0.015), quat=(0, 0, 1, 0)).attach_body(d.worldbody.first_body(), "d435i/", "")


def add_belt(spec, lay: Layout):
    """The treadmill: one long plate on a slide joint along the belt direction, driven at constant velocity by a
    velocity actuator; the runtime teleports it back by reset_travel whenever it has slid that far. The plate
    extends reset_travel behind x_min so the visible span is always covered."""
    b = lay.belt
    plate_len = (b.x_max - b.x_min) + b.reset_travel
    body = spec.worldbody.add_body(name="belt_plate", pos=((b.x_max + b.x_min) / 2 - b.reset_travel / 2, b.y, b.surface_z - b.plate_thickness / 2))
    body.add_joint(name="belt_slide", type=JNT.mjJNT_SLIDE, axis=b.direction, damping=5.0)
    body.mass = 40.0; body.explicitinertial = True; body.ipos = (0, 0, 0); body.inertia = (5.0, 20.0, 25.0)
    body.add_geom(name="belt_plate", type=GEOM.mjGEOM_BOX, size=(plate_len / 2, b.width / 2, b.plate_thickness / 2), rgba=RGBA["plate"],
                  friction=(0.9, 0.005, 0.0001), priority=1, condim=4,
                  solref=(0.004, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2), contype=1, conaffinity=3, group=0)   # conaffinity bit 2: the suction cup; priority 1: these win in every item-belt pair; the default 20 ms time constant let items sink 1.4 mm on landing
    spec.add_actuator(name="belt_drive", target="belt_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=(5000.0,) + (0.0,) * 9, biasprm=(0.0, 0.0, -5000.0) + (0.0,) * 7, ctrlrange=(-1.0, 1.0), forcerange=(-3000, 3000))


def add_item(spec, rec, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    """Add one YCB item to `spec` as a free body: textured visual mesh (group 2), CoACD hulls for collision (group 3),
    explicit inertia from the hulls scaled to the real mass. Names: <name>, joint <name>_free, geoms <name>_visual,
    <name>_hull_NN. (Built in place rather than attached: MjSpec.attach with an empty prefix cannot be saved.)"""
    n = rec["name"]; d = os.path.join(ycb.ASSETS, rec["id"])
    spec.add_mesh(name=f"{n}_visual", file=os.path.join(d, rec["visual"]))
    spec.add_texture(name=f"{n}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=os.path.join(d, rec["texture"]))
    spec.add_material(name=f"{n}_mat", specular=0.2, shininess=0.3).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{n}_tex"
    for h in rec["hulls"]:
        spec.add_mesh(name=f"{n}_{h[:-4]}", file=os.path.join(d, h))
    body = spec.worldbody.add_body(name=n, pos=pos, quat=quat)
    body.add_freejoint(name=f"{n}_free")
    body.explicitinertial = True; body.mass = rec["mass_kg"]; body.ipos = rec["com"]; body.fullinertia = rec["inertia_full"]
    body.add_geom(name=f"{n}_visual", type=GEOM.mjGEOM_MESH, meshname=f"{n}_visual", material=f"{n}_mat", **VIS)
    f = rec["friction"]
    for h in rec["hulls"]:
        body.add_geom(name=f"{n}_{h[:-4]}", type=GEOM.mjGEOM_MESH, meshname=f"{n}_{h[:-4]}", rgba=RGBA["hull"],
                      friction=(f, 0.005, 0.0001), condim=4, solref=(0.004, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2), contype=1, conaffinity=3, group=3)   # conaffinity bit 2: the suction cup
    return body


def item_spec(rec, lay: Layout):
    """One item alone (Genesis loads each as its own entity)."""
    s = mujoco.MjSpec(); s.modelname = rec["name"]; _options(s, lay); add_item(s, rec); return s


def add_cameras(spec, lay: Layout):
    cam = lay.camera
    spec.worldbody.add_camera(name=cam.name, pos=cam.pos, quat=lookat_quat(cam.pos, cam.lookat, cam.up), fovy=cam.fovy)
    for name, (pos, target) in lay.views.items():
        spec.worldbody.add_camera(name=name, pos=pos, quat=lookat_quat(pos, target), fovy=45)


def cell_spec(lay: Layout, items=None, index=None):
    """The whole cell. `items`: list of index ids to include (default: all)."""
    index = index or ycb.load_index(); items = list(index) if items is None else items
    # the cell spec *is* the robot spec (attaching the robot with an empty prefix breaks to_xml; a prefix would rename
    # every joint); its base body moves onto the pedestal and everything else is added around it
    s = robot_spec(); s.modelname = "sorting_cell"; _options(s, lay)
    s.body("base").pos = lay.robot_base
    for lt in list(s.lights):   # the Menagerie spotlight; the cell has its own lights
        try: s.delete(lt)
        except Exception: lt.active = 0
    add_floor_and_lights(s, lay); add_fixtures(s, lay); add_belt(s, lay); add_cameras(s, lay)
    b = lay.belt
    for k, iid in enumerate(items):
        rec = index[iid]
        z = b.surface_z - rec["bounds"][0][2] + 0.003
        add_item(s, rec, pos=(lay.spawn_x - 0.35 * k, b.y, z))
    add_vacuum_welds(s, [index[i]["name"] for i in items])
    return s


def write(lay: Layout = DEFAULT, items=None, out=BUILD):
    os.makedirs(os.path.join(out, "items"), exist_ok=True)
    index = ycb.load_index()
    paths = {}
    s = cell_spec(lay, items, index); m = s.compile(); paths["cell.xml"] = os.path.join(out, "cell.xml")
    with open(paths["cell.xml"], "w") as f: f.write(s.to_xml())
    r = robot_spec(); r.compile(); paths["robot.xml"] = os.path.join(out, "robot.xml")
    with open(paths["robot.xml"], "w") as f: f.write(r.to_xml())
    fx = mujoco.MjSpec(); fx.modelname = "fixtures"; _options(fx, lay); add_fixtures(fx, lay); fx.compile(); paths["fixtures.xml"] = os.path.join(out, "fixtures.xml")
    with open(paths["fixtures.xml"], "w") as f: f.write(fx.to_xml())
    bl = mujoco.MjSpec(); bl.modelname = "belt"; _options(bl, lay); add_belt(bl, lay); bl.compile(); paths["belt.xml"] = os.path.join(out, "belt.xml")
    with open(paths["belt.xml"], "w") as f: f.write(bl.to_xml())
    for iid, rec in index.items():
        it = item_spec(rec, lay); it.compile(); p = os.path.join(out, "items", f"{iid}.xml"); paths[f"items/{iid}.xml"] = p
        with open(p, "w") as f: f.write(it.to_xml())
    with open(os.path.join(out, "layout.json"), "w") as f: f.write(lay.to_json())
    return m, paths


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0]); p.add_argument("--items", default="all"); a = p.parse_args()
    items = None if a.items == "all" else a.items.split(",")
    m, paths = write(DEFAULT, items)
    print(f"cell.xml: nbody {m.nbody} ngeom {m.ngeom} nq {m.nq} nv {m.nv} nu {m.nu} nmesh {m.nmesh} ncam {m.ncam}")
    for k, v in paths.items(): print(f"  {k:28s} {os.path.getsize(v):8d} B")
