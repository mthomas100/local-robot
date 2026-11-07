"""items.py — what the desk cell can put on its belt (2026-11-03, F6): the 11 YCB scans of the v1 cell (same meshes,
masses, hulls and friction, `factory/cell/assets/ycb/index.json`) and desk-scale stand-ins of the same three
materials, built from primitives and always named `standin_*`. Which of them the SO-101 may handle is decided by
the payload-and-jaw table (`payload.py`, -> results/admitted.json); `admitted()` reads it.
"""
from __future__ import annotations
import json, os, sys
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "factory", "cell"))
import ycb  # noqa: E402
import build_mjcf as V1  # noqa: E402

RESULTS = os.path.join(HERE, "results")
INDEX = ycb.load_index()
YCB = {r["name"]: r for r in INDEX.values()}

# Desk-scale stand-ins: everyday objects of the same materials that a hobby arm's jaw spans. Masses are typical
# (an empty 330 ml PET bottle with its cap ~25 g, a 150 g tin of fish, a 100 g tea box).
STANDINS = {
    "standin_pet_bottle": dict(material="plastic", shape="cylinder", radius=0.028, height=0.11, mass_kg=0.045, friction=0.6, rgba=(0.55, 0.8, 0.95, 1)),
    "standin_small_tin": dict(material="metal", shape="cylinder", radius=0.03, height=0.06, mass_kg=0.15, friction=0.5, rgba=(0.78, 0.78, 0.82, 1)),
    "standin_tea_box": dict(material="cardboard", shape="box", size=(0.06, 0.045, 0.085), mass_kg=0.10, friction=0.7, rgba=(0.85, 0.35, 0.25, 1)),
}
ALL = list(YCB) + list(STANDINS)


def info(name):
    """material, mass, friction, footprint extents (x, y), height and z of the bottom in the body frame."""
    if name in YCB:
        r = YCB[name]; e = r["extents"]
        return dict(name=name, material=r["material"], mass_kg=r["mass_kg"], friction=r["friction"], extents=e, height=e[2],
                    bottom=r["bounds"][0][2], center_xy=[(r["bounds"][0][0] + r["bounds"][1][0]) / 2, (r["bounds"][0][1] + r["bounds"][1][1]) / 2], standin=False)
    st = STANDINS[name]
    if st["shape"] == "cylinder": e = [2 * st["radius"], 2 * st["radius"], st["height"]]
    else: e = list(st["size"])
    return dict(name=name, material=st["material"], mass_kg=st["mass_kg"], friction=st["friction"], extents=e, height=e[2], bottom=0.0,
                center_xy=[0.0, 0.0], standin=True)


def add(spec, name, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    if name in YCB: return V1.add_item(spec, YCB[name], pos=pos, quat=quat)
    st = STANDINS[name]; body = spec.worldbody.add_body(name=name, pos=pos, quat=quat); body.add_freejoint(name=f"{name}_free")
    kw = dict(friction=(st["friction"], 0.005, 0.0001), condim=4, solref=(0.004, 1), solimp=(0.95, 0.99, 0.001, 0.5, 2), contype=1, conaffinity=3, group=0,
              rgba=st["rgba"], mass=st["mass_kg"])
    if st["shape"] == "cylinder":
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(st["radius"], st["height"] / 2, 0), pos=(0, 0, st["height"] / 2), **kw)
    else:
        sx, sy, sz = st["size"]; body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, sz / 2), pos=(0, 0, sz / 2), **kw)
    return body


def admitted():
    p = os.path.join(RESULTS, "admitted.json")
    if os.path.exists(p): return json.load(open(p))["admitted"]
    return ["tomato_soup_can", "potted_meat_can", "standin_pet_bottle", "standin_tea_box"]   # until payload.py has run
