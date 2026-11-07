"""layout.py — the one place the sorting cell is defined (2026-11-02, docs/sorting-cell-spec.md §1, discipline 1).

Everything is in metres, world frame z-up, floor at z = 0. The UR5e stands on a pedestal; the belt runs along +x in
front of it; the three chutes sit between the pedestal and the belt; the RealSense D435i hangs over the reach zone.
`build_mjcf.py` turns this into MJCF for MuJoCo, `cell_genesis.py` into Genesis entities; the two must agree, which
is why neither of them hard-codes a number.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, math


@dataclass
class Belt:
    direction: tuple = (1.0, 0.0, 0.0)   # items travel +x
    y: float = 0.55                      # centreline, from the robot base
    surface_z: float = 0.78              # top of the belt (0.18 m above the robot base plane)
    width: float = 0.40
    x_min: float = -1.2                  # visible span: items spawn near x_min, fall off past x_max (a miss)
    x_max: float = 1.2
    zone: tuple = (-0.5, 0.5)            # the reach zone where picks happen (1.0 m, the spec's "~1 m reach zone")
    plate_thickness: float = 0.02
    reset_travel: float = 0.4            # treadmill: the plate slides this far, then teleports back (objects stay)
    speeds: tuple = (0.15, 0.30, 0.50)   # m/s, the spec's Dorner-class range
    ramp: float = 1.0                    # m/s^2: how fast the drive changes speed (a step jolt tips tall boxes)
    rail_height: float = 0.03            # side rails (visual + a wall, keep items on)
    rail_thickness: float = 0.015
    rail_gap: float = 0.005              # clearance between the sliding plate and the rails (they must never touch)


@dataclass
class Chute:
    material: str
    x: float
    y: float = 0.27
    inner: float = 0.24                  # square opening
    depth: float = 0.30                  # deep enough that the tallest item (0.27 m) ends below the rim when it tips (0.16 let it lean out)
    wall: float = 0.02                   # walls and floor are thick on purpose: a bottle falling 0.34 m sank 10 mm into a 1 cm floor and stuck
    floor: float = 0.04                  # floor slab thickness, below the floor top at rim_z - depth
    rim_z: float = 0.60                  # top of the walls; the drop point is above this


@dataclass
class Camera:
    name: str = "overhead"               # the D435i colour stream: 69 x 42 degrees, 16:9 (Intel datasheet)
    pos: tuple = (0.0, 0.55, 1.78)       # 1.0 m above the belt surface, over the zone
    lookat: tuple = (0.0, 0.55, 0.78)
    up: tuple = (0.0, 1.0, 0.0)          # the belt runs left to right in the image (its 69-degree axis covers the zone)
    fovy: float = 42.0
    width: int = 848
    height: int = 480


@dataclass
class Layout:
    pedestal: tuple = (0.24, 0.24, 0.60)  # footprint and height; the UR5e base frame sits on top at (0, 0, height)
    belt: Belt = field(default_factory=Belt)
    chutes: list = field(default_factory=lambda: [Chute("plastic", -0.50), Chute("metal", 0.0), Chute("cardboard", 0.50)])
    camera: Camera = field(default_factory=Camera)
    views: dict = field(default_factory=lambda: {   # extra cameras for videos: name -> (pos, lookat)
        "cell": ((2.1, -1.5, 1.9), (0.0, 0.45, 0.75)),
        "belt": ((0.0, 2.3, 1.5), (0.0, 0.55, 0.78)),
        "end": ((-2.3, 0.55, 1.3), (0.3, 0.55, 0.78)),
        "pick": ((1.3, -0.62, 1.7), (-0.05, 0.40, 0.78)),   # close on the arm, the zone and the chutes (F1 pick videos)
    })
    # UR5e pan, lift, elbow, w1, w2, w3 (rad). Gripper pointing down over the chute line (pinch at about (-0.05, 0.27, 1.18)),
    # off the belt and above the tallest item: the Menagerie home put the fingertips at z 0.92 over the belt and tall
    # boxes hit them (found 2026-11-02 when the cracker box kept tipping at x = -0.2).
    home_qpos: tuple = (-1.9, -1.92, 1.15, -0.8008, -1.5708, 0.0)
    spawn_x: float = -1.05                # where items are placed on the belt (before the zone)
    timestep: float = 0.002

    @property
    def robot_base(self):
        return (0.0, 0.0, self.pedestal[2])

    @property
    def shoulder_z(self):
        return self.pedestal[2] + 0.163    # ur5e.xml: shoulder_link at z 0.163 above the base

    def chute(self, material):
        return next(c for c in self.chutes if c.material == material)

    def cycle_budget(self):
        """Seconds an item spends in the reach zone at each belt speed: the per-item budget when items arrive one
        zone-length apart, and the arrival rate the belt imposes."""
        L = self.belt.zone[1] - self.belt.zone[0]
        return {v: dict(zone_seconds=round(L / v, 2), items_per_minute_at_0_3m_spacing=round(60 * v / 0.3, 1)) for v in self.belt.speeds}

    def to_json(self):
        return json.dumps(asdict(self), indent=1)


DEFAULT = Layout()

if __name__ == "__main__":
    print(DEFAULT.to_json())
    print("cycle budget:", DEFAULT.cycle_budget())
