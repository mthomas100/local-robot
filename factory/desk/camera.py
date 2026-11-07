"""camera.py — the desk cell's eyes, and the camera half of the real-device kit (2026-11-03, F6; disciplines 7, 14).

The overhead RGB-D camera (a D435-class stereo camera, 424 x 240, 46 cm above the belt) reports where the item is.
On a real desk four things go wrong between the scene and that report, and the kit plants each of them:

  depth noise    stereo depth error grows with distance squared: sigma_z = z^2 * subpixel / (focal_px * baseline)
                 (Intel's RMS formula; subpixel 0.08, baseline 50 mm, focal 216 px at 424 wide: 1.6 mm at 0.46 m),
                 plus `hole_rate` of pixels with no depth (0) at random and along depth edges
  RGB noise      Gaussian per-pixel noise and a small per-frame exposure gain
  latency        a frame taken at t reaches the host at t + latency + jitter (`latency_s`, `jitter_s`); the host
                 always uses the newest frame that has arrived
  hand-eye error the camera is not where the cell drawing says: the true mount is off by a planted translation and
                 rotation (`he_trans_m`, `he_rot_deg`). Perception computes world positions with the *believed*
                 extrinsics, so every estimate is shifted (3 deg at 46 cm is ~2.4 cm on the belt)

`calibrate()` is the routine that removes the hand-eye error: the arm carries the magenta marker ball on its wrist to
N poses over the belt; the camera finds the ball in each frame (colour blob -> median depth -> the ball's centre one
radius behind its surface); the robot side says where the ball is by forward kinematics of the servo encoders, with
the horn play preloaded by gravity (each pitch joint's link sits `backlash_rad` past its shaft, toward gravity, which
the servo cannot see); the camera pose that maps the camera-side points onto the robot-side points best (Kabsch,
least squares) is the first estimate; poses far off the fit are dropped (a pitch joint's play at its far end), then
a bundle adjustment solves the camera pose together with one unknown pan-play angle per pose (the pan joint carries
no gravity load, so its play sits anywhere in its +/- 0.5 deg) and that is the calibrated extrinsic. The planted error is recovered when |dt| <= 2 mm and |dR| <= 1 deg.

The detector (`detect`) works on the depth image alone, as F4's did: belt-region points more than 6 mm above the belt
(the robot masked out, as a real cell renders its own mask from the URDF), the top surface (within 8 mm of the highest
point), its centroid (footprint centre), principal axis (yaw) and extents, the height above the belt, and the top
surface's mean colour; the item class is the nearest admitted item on (height, short side, long side, colour).
"""
from __future__ import annotations
import math, os, sys
from collections import deque
from dataclasses import dataclass, asdict, field
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import desk_items as I  # noqa: E402
import desk_ik  # noqa: E402
import build_desk  # noqa: E402


@dataclass
class CameraKit:
    real: bool = False
    subpixel: float = 0.08
    baseline_m: float = 0.050
    hole_rate: float = 0.02
    rgb_sigma: float = 4.0            # of 255
    gain_sigma: float = 0.03
    latency_s: float = 0.045
    jitter_s: float = 0.010
    he_trans_m: tuple = (0.0, 0.0, 0.0)   # planted hand-eye error, camera frame offset in world (m)
    he_rot_deg: tuple = (0.0, 0.0, 0.0)   # planted rotation (roll, pitch, yaw about the camera's own axes, deg)

    @staticmethod
    def real_kit(seed=0, trans_mm=8.0, rot_deg=2.5, **kw):
        """A real kit with a seeded hand-eye error of the given size (random directions)."""
        rng = np.random.default_rng(10_000 + seed)
        t = rng.normal(size=3); t = t / np.linalg.norm(t) * trans_mm / 1000
        r = rng.normal(size=3); r = r / np.linalg.norm(r) * rot_deg
        return CameraKit(real=True, he_trans_m=tuple(float(v) for v in t), he_rot_deg=tuple(float(v) for v in r), **kw)

    def to_dict(self): return asdict(self)


def euler_R(deg):
    r = np.radians(deg); q = np.zeros(4); mujoco.mju_euler2Quat(q, r, "xyz"); R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)


def rotvec_deg(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0); return math.degrees(np.linalg.norm(v))


class Camera:
    """The overhead camera on a DeskSim: frames with noise and latency, believed vs true extrinsics."""

    def __init__(self, sim, kit: CameraKit | None = None, rng=None):
        self.sim, self.kit = sim, kit or CameraKit()
        self.rng = rng or np.random.default_rng(0)
        m = sim.model; self.cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, sim.lay.camera.name)
        self.W, self.H = sim.lay.camera.width, sim.lay.camera.height
        self.fy = self.H / 2 / math.tan(math.radians(m.cam_fovy[self.cid]) / 2); self.fx = self.fy
        # the believed pose = the cell drawing (the compiled model's camera); the true pose = drawing + planted error
        mujoco.mj_forward(m, sim.data)
        self.nominal_pos = sim.data.cam_xpos[self.cid].copy(); self.nominal_R = sim.data.cam_xmat[self.cid].reshape(3, 3).copy()
        self._model_pos0 = m.cam_pos[self.cid].copy(); self._model_quat0 = m.cam_quat[self.cid].copy()
        self.believed_pos = self.nominal_pos.copy(); self.believed_R = self.nominal_R.copy()
        self.apply_true_pose()
        self.renderer = mujoco.Renderer(m, height=self.H, width=self.W)
        self.queue = deque()   # (arrival time, capture time, frame)
        self.robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in sim.robot_bodies])
        self._ref = None

    def close(self):
        self.renderer.close()

    # ---- extrinsics
    def true_pose(self):
        k = self.kit; Re = euler_R(k.he_rot_deg); return self.nominal_pos + np.asarray(k.he_trans_m), self.nominal_R @ Re

    def apply_true_pose(self):
        """Move the model's camera to the true pose (world-body camera: model pos/quat are world)."""
        m = self.sim.model; p, R = self.true_pose()
        m.cam_pos[self.cid] = p; q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); m.cam_quat[self.cid] = q
        mujoco.mj_forward(m, self.sim.data)

    def hand_eye_error(self):
        """Believed vs true camera pose: (translation error m, rotation error deg)."""
        p, R = self.true_pose(); return float(np.linalg.norm(self.believed_pos - p)), rotvec_deg(self.believed_R.T @ R)

    # ---- frames
    def capture(self):
        """Render rgb, depth (metres along the optical axis; 0 = no return) and the robot mask now."""
        r, d = self.renderer, self.sim.data
        r.update_scene(d, camera=self.cid); rgb = r.render().copy()
        r.enable_depth_rendering(); r.update_scene(d, camera=self.cid); depth = r.render().copy(); r.disable_depth_rendering()
        r.enable_segmentation_rendering(); r.update_scene(d, camera=self.cid); seg = r.render()[..., 0].copy(); r.disable_segmentation_rendering()
        robot = np.isin(seg, self.robot_geoms) if len(self.robot_geoms) else np.zeros_like(depth, bool)
        if self.kit.real:
            k = self.kit; rng = self.rng
            sig = depth ** 2 * k.subpixel / (self.fx * k.baseline_m)
            depth = depth + rng.normal(size=depth.shape) * sig
            edges = np.zeros_like(depth, bool); gy, gx = np.gradient(depth); edges |= np.hypot(gx, gy) > 0.01
            holes = (rng.random(depth.shape) < k.hole_rate) | (edges & (rng.random(depth.shape) < 0.5))
            depth = np.where(holes, 0.0, depth)
            g = 1.0 + rng.normal() * k.gain_sigma
            rgb = np.clip(rgb.astype(np.float32) * g + rng.normal(size=rgb.shape) * k.rgb_sigma, 0, 255).astype(np.uint8)
        return dict(t=self.sim.t, rgb=rgb, depth=depth.astype(np.float32), robot=robot)

    def tick(self):
        """Capture a frame now and put it in flight; returns the newest frame that has arrived (or None)."""
        f = self.capture(); lat = 0.0
        if self.kit.real: lat = max(0.0, self.kit.latency_s + self.rng.normal() * self.kit.jitter_s)
        self.queue.append((self.sim.t + lat, f["t"], f))
        return self.latest()

    def flush(self):
        """Drop frames in flight (call after the scene was reset: their capture times belong to another trial)."""
        self.queue.clear(); self._last = None

    def latest(self):
        best = None
        while self.queue and self.queue[0][0] <= self.sim.t + 1e-9: best = self.queue.popleft()[2]; self._last = best
        return best if best is not None else getattr(self, "_last", None)

    # ---- geometry
    def points_world(self, depth, mask=None, pos=None, R=None):
        """Back-project pixels (u, v) with depth to world points using the given (default: believed) extrinsics."""
        pos = self.believed_pos if pos is None else pos; R = self.believed_R if R is None else R
        v, u = np.nonzero((depth > 0) if mask is None else (mask & (depth > 0)))
        z = depth[v, u]; x = (u + 0.5 - self.W / 2) / self.fx * z; y = -(v + 0.5 - self.H / 2) / self.fy * z
        pc = np.stack([x, y, -z], axis=1)          # MuJoCo camera frame: looks down -z, +y up
        return pc @ R.T + pos, (v, u)

    def detect(self, frame):
        """The item on the belt from one frame: dict(x, y, yaw, height, short, long, rgb, n) or None."""
        lay = self.sim.lay; b = lay.belt; depth = frame["depth"]
        pw, (v, u) = self.points_world(depth, mask=~frame["robot"])
        sel = (np.abs(pw[:, 0] - b.x) < b.width / 2 - 0.002) & (pw[:, 1] > b.y_min) & (pw[:, 1] < b.y_max) & (pw[:, 2] > b.surface_z + 0.006) & (pw[:, 2] < b.surface_z + 0.20)
        if sel.sum() < 30: return None
        P = pw[sel]; zmax = np.percentile(P[:, 2], 98); top = P[P[:, 2] > zmax - 0.008]
        if len(top) < 15: return None
        c = top[:, :2].mean(0); X = top[:, :2] - c; C = X.T @ X / len(X); w, V = np.linalg.eigh(C)
        long_ax = V[:, 1]; yaw = math.atan2(long_ax[1], long_ax[0])
        proj_l = X @ long_ax; proj_s = X @ V[:, 0]
        long_ = float(np.percentile(proj_l, 99) - np.percentile(proj_l, 1)); short = float(np.percentile(proj_s, 99) - np.percentile(proj_s, 1))
        rgb = frame["rgb"][v[sel][P[:, 2] > zmax - 0.008], u[sel][P[:, 2] > zmax - 0.008]].mean(0)
        return dict(x=float(c[0]), y=float(c[1]), yaw=yaw, height=float(zmax - b.surface_z), short=short, long=long_, rgb=rgb.tolist(), n=int(len(top)), t=frame["t"])

    # ---- identity
    def references(self):
        """Per admitted item: height, short and long top-surface extents and top colour, from one clean frame each."""
        if self._ref is not None: return self._ref
        sim = self.sim; saved = (self.kit, self.believed_pos.copy(), self.believed_R.copy()); snap = sim.snapshot()
        self.kit = CameraKit(); self.apply_true_pose(); self.believed_pos, self.believed_R = self.nominal_pos.copy(), self.nominal_R.copy()
        ref = {}
        for n in sim.items:
            sim.reset(); sim.spawn(n, sim.lay.belt.x, 0.0, 0.3); sim.step(100); d = self.detect(self.capture())
            if d: ref[n] = np.array([d["height"], d["short"], d["long"], *(np.array(d["rgb"]) / 255.0)])
        self.kit, self.believed_pos, self.believed_R = saved; self.apply_true_pose(); sim.restore(snap)   # a first version reset the scene here, mid-trial
        self._ref = ref; return ref

    def classify(self, det):
        ref = self.references(); f = np.array([det["height"], det["short"], det["long"], *(np.array(det["rgb"]) / 255.0)])
        w = np.array([20.0, 20.0, 20.0, 1.0, 1.0, 1.0])   # metres matter at the cm scale, colour at the 0.1 scale
        return min(ref, key=lambda n: float(np.sum(w * (ref[n] - f) ** 2)))

    def estimate(self, settle_frames=4):
        """What the pick needs: (x, y, yaw) of the footprint centre and the item class, from the newest frames
        (the median of `settle_frames` arrived detections)."""
        dets = []; dt = 1.0 / 30.0; steps = max(1, int(round(dt / self.sim.model.opt.timestep))); t0 = self.sim.t
        for _ in range(settle_frames * 6):
            f = self.tick()
            if f is not None and f["t"] >= t0 - 1e-9:
                d = self.detect(f)
                if d is not None and (not dets or d["t"] != dets[-1]["t"]): dets.append(d)
            if len(dets) >= settle_frames: break
            self.sim.bus.write(self.sim.bus.goal, self.sim.t); self.sim.step(steps)
        if not dets: return None
        x = float(np.median([d["x"] for d in dets])); y = float(np.median([d["y"] for d in dets]))
        c2 = np.median([math.cos(2 * d["yaw"]) for d in dets]); s2 = np.median([math.sin(2 * d["yaw"]) for d in dets])
        yaw_long = 0.5 * math.atan2(s2, c2); last = dets[-1]
        name = self.classify(last)
        # the planner's yaw is the item body's yaw; for a box the long axis is the body axis with the larger extent
        inf = I.info(name); ex, ey = inf["extents"][:2]
        yaw = yaw_long if ex >= ey else yaw_long - math.pi / 2
        return dict(x=x, y=y, yaw=yaw, item=name, n_frames=len(dets), det=last)


# ---- hand-eye calibration
def gravity_play(sim, q_servo, backlash_rad, d=None):
    """The play angle each joint's link sits at beyond its shaft, as the calibration models it: each pitch joint's
    (lift, elbow, wrist) play preloaded by gravity, the link backlash_rad past the shaft in the direction gravity
    pulls it; the pan and the roll carry no gravity load and are left centred (the bundle adjustment takes the pan).
    The gravity torque is read where the links really hang: the wrist pitch carries almost no torque when the tool
    points straight down, so the sign of its few mN*m is set by the lift and elbow play, which tilts the gripper about
    1 deg. A first version read the torque at the shaft angles and left joints under 0.02 N*m centred: the wrist then
    sat 0.47 deg off on average, a 0.8 mm systematic error in the calibration (found 2026-11-03). Two
    passes reach the fixed point."""
    m = sim.model; d = d or mujoco.MjData(m); ix = sim.ix; q = np.array(q_servo, float); play = np.zeros(6)
    if backlash_rad <= 0 or not ix.has_backlash: return play
    for _ in range(2):
        ix.set_q(d, q); d.qpos[ix.bl_qadr] = play; mujoco.mj_forward(m, d)
        g = d.qfrc_bias[ix.bl_dadr]            # gravity torque at rest (+ = gravity pushes the joint negative)
        play = np.zeros(6)
        for j in (1, 2, 3): play[j] = -math.copysign(backlash_rad, g[j]) if g[j] != 0 else 0.0
    return play


def marker_world_fk(sim, q_servo, backlash_rad):
    """Where forward kinematics says the marker is, from the servo shaft angles and the play `gravity_play` models."""
    m = sim.model; d = mujoco.MjData(m); ix = sim.ix
    play = gravity_play(sim, q_servo, backlash_rad, d)
    ix.set_q(d, np.array(q_servo, float))
    if ix.has_backlash: d.qpos[ix.bl_qadr] = play
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "marker_center"); return d.site_xpos[sid].copy()


def find_marker(cam, frame, min_pixels=25):
    """The marker ball's centre in the camera frame, or None: magenta pixels with depth -> 3-D surface points -> the
    centre of the sphere of known radius that fits them best (Gauss-Newton from the cap centroid pushed 2r/3 back).
    Blobs smaller than `min_pixels` (the ball half hidden by the wrist) are rejected: a partial cap biases the fit."""
    rgb = frame["rgb"].astype(np.int32); depth = frame["depth"]
    mask = (rgb[..., 0] > 150) & (rgb[..., 1] < 110) & (rgb[..., 2] > 70) & (rgb[..., 2] < 220) & (depth > 0)
    if mask.sum() < min_pixels: return None
    v, u = np.nonzero(mask); z = depth[v, u]
    near = np.abs(z - np.median(z)) < 1.5 * build_desk.MARKER_RADIUS   # edge pixels carry the background's depth
    v, u, z = v[near], u[near], z[near]
    if len(z) < min_pixels: return None
    P = np.stack([(u + 0.5 - cam.W / 2) / cam.fx * z, -(v + 0.5 - cam.H / 2) / cam.fy * z, -z], axis=1)
    r = build_desk.MARKER_RADIUS; ray = P.mean(0) / np.linalg.norm(P.mean(0)); c = P.mean(0) + ray * (2 * r / 3)
    keep = np.ones(len(P), bool)
    for it in range(12):
        D = P[keep] - c; n = np.linalg.norm(D, axis=1); res = n - r; J = -D / n[:, None]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]; c = c + step
        if it == 5:   # one outlier pass: drop points > 3 sigma off the sphere, refit
            allres = np.abs(np.linalg.norm(P - c, axis=1) - r); keep = allres < max(3 * allres.std(), 0.0015)
            if keep.sum() < min_pixels: return None
    rms = float(np.sqrt(np.mean((np.linalg.norm(P[keep] - c, axis=1) - r) ** 2)))
    return c if rms < 0.003 else None


def kabsch(A, B):
    """Rigid (R, t) minimizing sum |R a + t - b|^2."""
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T; return R, cb - R @ ca


def calibration_poses(lay, n_xy=6, heights=(0.08, 0.11, 0.14, 0.17), rng=None):
    """Marker poses: a 4 x 6 grid of tool positions at four heights (96 poses, ~18 s) (spread in depth: it pins the camera's distance),
    on the arm's side of the belt centreline, the wrist turned so the marker (on the jaw's -y side) faces away from the
    arm, +/- 0.4 rad of variation. Measured 2026-11-03: beyond the centreline the gripper hides the marker; at a tool
    height of 6 cm the jaw rests on the near belt rail and the contact holds the links off their gravity stops."""
    rng = rng or np.random.default_rng(0); b = lay.belt; out = []
    for z in heights:
        for xi in (b.x - 0.08, b.x - 0.057, b.x - 0.033, b.x - 0.01):
            for yi in np.linspace(b.zone[0] - 0.03, b.zone[1] + 0.03, n_xy):
                out.append((np.array([xi, yi, z]), math.atan2(yi, xi) + math.pi / 2 + rng.uniform(-0.4, 0.4)))
    return out


def _bundle(pc, pw, R0, t0, axis_pt, b, prior_sigma=0.5):
    """Least squares over (camera rotation vector, camera position, pan play per pose): residual = R pc + t -
    Rz(theta_k) (pw - a) - a, plus theta_k / (prior_sigma * b). Returns R, t, theta."""
    from scipy.optimize import least_squares
    n = len(pc); a = np.array([axis_pt[0], axis_pt[1], 0.0])
    def rot(v):
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, v / (np.linalg.norm(v) + 1e-12), np.linalg.norm(v)); M = np.zeros(9); mujoco.mju_quat2Mat(M, q); return M.reshape(3, 3)
    def res(x):
        R = rot(x[:3]) @ R0; t = x[3:6]; th = x[6:]
        c, s_ = np.cos(th), np.sin(th); d = pw - a
        pr = np.stack([c * d[:, 0] - s_ * d[:, 1], s_ * d[:, 0] + c * d[:, 1], d[:, 2]], 1) + a
        return np.r_[(pc @ R.T + t - pr).ravel(), th / (prior_sigma * b) * 1e-3]
    x0 = np.r_[np.zeros(3), t0, np.zeros(n)]; lo = np.r_[np.full(3, -0.2), t0 - 0.05, np.full(n, -b)]; hi = np.r_[np.full(3, 0.2), t0 + 0.05, np.full(n, b)]
    sol = least_squares(res, x0, bounds=(lo, hi), x_scale=np.r_[np.full(3, 0.01), np.full(3, 0.005), np.full(n, b)])
    return rot(sol.x[:3]) @ R0, sol.x[3:6], sol.x[6:]


def calibrate(sim, cam, backlash_rad, frames_per_pose=3, log=None):
    """Run the routine; sets cam.believed_* to the calibrated pose and returns a report."""
    arm = desk_ik.Arm(model=sim.model); pc, pw = [], []; dt = 1.0 / 30.0; steps = int(round(dt / sim.model.opt.timestep))
    sim.reset(); cam.flush(); skipped = dict(ik=0, unseen=0)
    for k, (p, yaw) in enumerate(calibration_poses(sim.lay)):
        ok, q5, _ = arm.ik_pose(p, desk_ik.topdown_rot(yaw), restarts=4)
        if not ok: skipped["ik"] += 1; continue
        goal = np.r_[q5, 0.0]
        # move there (a straight joint-space ramp at the host rate), settle, then take frames
        q0 = sim.ix.servo_q(sim.data).copy()
        for i in range(1, 31): sim.bus.write(q0 + (goal - q0) * i / 30, sim.t); sim.step(steps)
        for _ in range(15): sim.bus.write(goal, sim.t); sim.step(steps)
        obs = []
        for _ in range(frames_per_pose * 4):
            f = cam.tick(); sim.bus.write(goal, sim.t); sim.step(steps)
            if f is not None and f["t"] >= sim.t - 0.4:
                mc = find_marker(cam, f)
                if mc is not None: obs.append(mc)
            if len(obs) >= frames_per_pose: break
        if not obs: skipped["unseen"] += 1; continue
        q_read = sim.bus.read(sim.t)
        pc.append(np.median(obs, axis=0)); pw.append(marker_world_fk(sim, q_read, backlash_rad))
        if log: log(k, p, pc[-1], pw[-1])
    pc, pw = np.array(pc), np.array(pw)
    if len(pc) < 6: return dict(ok=False, n=len(pc))
    R, t = kabsch(pc, pw)     # world = R * cam + t: R is the camera's orientation, t its position
    # robust refit: poses where a pitch joint's play sat at the far end (or the jaw touched something) are off by
    # 5-8 mm; drop points whose residual exceeds 2.5x the median and refit, twice
    keep = np.ones(len(pc), bool)
    for _ in range(2):
        res = np.linalg.norm(pc @ R.T + t - pw, axis=1); keep = res <= 2.5 * np.median(res)
        R, t = kabsch(pc[keep], pw[keep])
    # bundle adjustment: the pan joint carries no gravity load, so its play sits anywhere in +/- backlash_rad and turns
    # the marker about the pan axis by an unknown angle per pose (+/- 1.7 mm at 20 cm). Solve the camera pose (6) and
    # one pan angle per pose together, the angles bounded by the play and pulled weakly to zero (they average zero)
    if backlash_rad > 0 and keep.sum() >= 8:
        R, t, pan = _bundle(pc[keep], pw[keep], R, t, sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "shoulder")].copy(), backlash_rad)
    before = cam.hand_eye_error()
    resid_before = float(np.sqrt(np.mean(np.sum((pc @ cam.believed_R.T + cam.believed_pos - pw) ** 2, 1))))
    cam.believed_R, cam.believed_pos = R, t
    after = cam.hand_eye_error()
    resid = float(np.sqrt(np.mean(np.sum((pc[keep] @ R.T + t - pw[keep]) ** 2, 1))))   # (before the pan angles)
    sim.reset(); cam.flush()
    return dict(ok=True, n=len(pc), kept=int(keep.sum()), skipped=skipped, before_mm=round(1000 * before[0], 2), before_deg=round(before[1], 3), after_mm=round(1000 * after[0], 2),
                after_deg=round(after[1], 3), residual_before_mm=round(1000 * resid_before, 2), residual_after_mm=round(1000 * resid, 2))
