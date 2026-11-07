"""layout.py — the desk-scale sorting cell for the SO-101 (2026-11-03, F6; discipline 1). Metres, world z-up, desk top
at z = 0.

    +y (left)
      ^        bin metal (+64 deg)        bin plastic (+104 deg)
      |
      |   [SO-101 on a 10 cm riser]  ----->  belt section across the front (x = 0.235), items travel -y
      |
      |        bin cardboard (-64 deg)
      +----------------------------------> +x (forward)

Why a riser: with the tool pointing straight down the SO-101's grasp frame cannot rise more than ~9 cm above its own
base (the three pitch joints hit their limits; measured with `rl desk reach`). Items up to ~11 cm tall must be lifted
clear of a 4 cm belt and 6 cm bins, so the base sits 10 cm above the desk and the work happens below it. Bins sit on
both sides inside the pan range (+/- 110 deg); the belt sits where the top-down reach is widest (radius 0.15-0.30 m).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, math


@dataclass
class DeskBelt:
    x: float = 0.235                    # centreline, in front of the arm
    surface_z: float = 0.04             # a desk conveyor: 4 cm high
    width: float = 0.13
    y_min: float = -0.15                # the visible section (ends short of the bins)
    y_max: float = 0.15
    zone: tuple = (-0.08, 0.08)         # L0 items are placed here (y)
    direction: tuple = (0.0, -1.0, 0.0)
    plate_thickness: float = 0.012
    rail_height: float = 0.015
    rail_thickness: float = 0.008
    rail_gap: float = 0.003
    reset_travel: float = 0.0        # the desk belt is a short section, stopped for L0 (no treadmill reset)


@dataclass
class Bin:
    material: str
    angle_deg: float
    radius: float = 0.23
    inner: float = 0.12                 # square opening
    depth: float = 0.055
    wall: float = 0.006
    floor: float = 0.006

    @property
    def xy(self):
        a = math.radians(self.angle_deg); return (self.radius * math.cos(a), self.radius * math.sin(a))

    @property
    def rim_z(self): return self.floor + self.depth


@dataclass
class DeskCamera:
    name: str = "overhead"              # a D435i-class RGB-D camera on a boom over the belt
    pos: tuple = (0.235, 0.0, 0.50)
    lookat: tuple = (0.235, 0.0, 0.04)
    up: tuple = (-1.0, 0.0, 0.0)        # image up = toward the arm
    fovy: float = 42.0
    width: int = 424
    height: int = 240


@dataclass
class DeskLayout:
    riser: tuple = (0.12, 0.10, 0.10)   # footprint x, y and height; the SO-101 base frame sits on top
    belt: DeskBelt = field(default_factory=DeskBelt)
    bins: list = field(default_factory=lambda: [Bin("metal", 64.0), Bin("plastic", 104.0), Bin("cardboard", -64.0)])
    camera: DeskCamera = field(default_factory=DeskCamera)
    views: dict = field(default_factory=lambda: {
        "desk": ((0.62, -0.48, 0.42), (0.12, 0.02, 0.06)),
        "side": ((0.25, -0.62, 0.22), (0.20, 0.0, 0.06)),
        "front": ((0.75, 0.05, 0.30), (0.10, 0.0, 0.08)),
        "top": ((0.10, 0.0, 0.95), (0.13, 0.0, 0.0)),
    })
    home_q: tuple = (0.0, -1.2, 1.1, 1.35, 0.0, 0.0)   # pan, lift, elbow, wrist flex, roll, gripper: folded, tool over the riser front
    timestep: float = 0.002
    park: tuple = (0.0, 1.5, 0.05)      # where unused items wait (off the desk)

    @property
    def base(self): return (0.0, 0.0, self.riser[2])

    def bin(self, material): return next(b for b in self.bins if b.material == material)

    def to_json(self): return json.dumps(asdict(self), indent=1)


DEFAULT = DeskLayout()
