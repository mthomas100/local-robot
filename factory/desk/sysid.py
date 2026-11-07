"""sysid.py — system identification of the SO-101's servos and joints from step, creep, coast and chirp recordings
(2026-11-03, F6; discipline 14, sim-to-real).

What system identification is for. A simulator is only as useful as its parameters. The CAD gives the link masses and
inertias, but not what sits inside each servo: the rotor's inertia seen through the gearbox (1:345 on the follower
arm's STS3215; MuJoCo's `armature`, 4-25x the link inertia on every joint, so it nearly *is* the inertia), the gearbox's
viscous drag (`damping`), its Coulomb friction (`frictionloss`), the motor's back-EMF (the no-load speed) and the bus
delay. System identification records what a real arm does under chosen inputs, then searches for the parameters with
which the simulator does the same thing. Here the "real arm" is a planted plant: the same simulator with parameters
drawn 20-50% off the nominal ones (`plant`), so we know the right answer and can check that the procedure finds it.

What is planted and what is not (per seed, `plant()`):
  per joint (all six, the gripper too)   armature, damping, frictionloss         each x (1 +/- U(0.2, 0.5))
  per bus / motor type                   bus_delay_s, no_load_speed (back-EMF)    each x (1 +/- U(0.2, 0.5))
  known, not identified                  kp, kv, deadband, encoder (the servo's registers say them), link masses (CAD)
  not identifiable from servo data       backlash: the play sits *beyond* the servo's encoder (so101.py), so no reading
                                         the servo can give depends on it except through tiny load effects; it stays at
                                         its default in the plant and in the model. Measuring it needs an outside
                                         sensor on the link (a camera, a dial gauge), not the servo bus.

The recording (what a real campaign can record). One joint is excited at a time, the others hold their pose (not
rigidly: see below; all six are fitted together). The host writes a goal and reads the present position of the six
servos at `HOST_HZ` = 250 Hz, with its own clock logged (it jitters +/- 1 ms against the servos' clock, as a USB-serial
host does). Why 250 Hz: the servo's position loop with this armature has a natural frequency near
sqrt(kp / I) / 2 pi = 29 Hz, so its dynamics need well over 60 Hz of sampling; LeRobot's 30 Hz control loop would
alias them. A sync-write of six goals plus a sync-read of six positions is ~90 bytes, ~0.9 ms at 1 Mbaud, plus the USB
adapter's 1 ms latency timer, so a 4 ms cycle has margin; 500 Hz would not. The reads are the 12-bit encoder
(0.088 deg per count) and arrive one bus delay late. Per joint, 72 s of four kinds of experiment (`design`), every
full-torque move given 1.0-1.5 s to come to rest before the next one starts (see the last lesson below):
  creep    the servo's torque-limit register (STS3215 address 48) lowered to a little above what gravity at the test
           pose needs, then a far goal written: the duty saturates at the limit and the joint creeps at the speed where
           the drive balances friction plus viscous drag. Three limits, both directions. Identifies the friction.
  steps    goal steps of 0.02, 0.2, 0.6 and 1.2 rad at full torque, both directions: the inertia and the top speed
           (the back-EMF).
  coasts   a 1.2 rad step, and 0.04-0.12 s after it, with the joint near top speed, the torque switched off
           (Torque_Enable, address 40) for 0.2-0.35 s: the joint coasts on the gearbox alone. Four timings, both
           directions. Identifies the damping.
  chirps   a swept sine of the goal (0.2 rad, 0.2 -> 3 Hz; 0.04 rad, 1 -> 12 Hz): the inertia and the delay.

The fit (`identify`). Every candidate is scored by simulating the same goal stream, at the same host times, through
the same servo model, and comparing the simulated shaft angle at the read times with the recorded counts: the whole
recording is simulated, but only the reads `fit_weights` picks are compared (the creeps, each step until the joint
first reaches its goal, each coast until its torque comes back on, the chirps; not the recaptures and holds). It starts
from the nominal values, never the planted ones. All six joints' parameters, the shared no-load speed and the delay
are fitted together in log space by Gauss-Newton (scipy least_squares, trust region); the Jacobian is block-sparse (a
joint's parameters only change its own recording, so each column costs one joint simulation) and is taken from a
smooth twin of the servo (see `Fitter`). Then the delay's profile: for each delay on a grid (0.5 ms, then 0.1 ms around
the best), the rest refitted, and the delay whose refit matches best wins; then a last Gauss-Newton with it held.

What we learned building it (2026-11-03; each point is an experiment change that was needed, the tolerance never moved):
  * Back-EMF and gearbox damping are both torques proportional to speed, -(stall / no_load_speed + damping) * qdot,
    and on these servos the back-EMF term is 3-25x the damping. Steps, creeps and chirps see their sum; in this
    model they separate only while the servo brakes at full reverse duty (the motor's torque clips at stall, so the
    back-EMF drops out), and that is too short a window. Without the coast tests (`--ablate coasts`, seed 1) the fit
    put the damping 80-210% high on every joint, with the no-load speed 11% high to pay for it, and matched the
    recordings *better* than the full campaign does (0.77 against 1.03 counts rms, over less data): a close match is
    not evidence of right parameters when two of them can trade. With the torque off, the driver lets go of the
    motor, the back-EMF drives no current, and the joint slows on the gearbox's damping and friction alone. Two
    coast timings per direction still left the damping 5-11% high on several joints (seed 2, gripper +10.9%); four
    bring every damping within 4%. On a real arm the same trick works (back-drive or coast
    with the servo unpowered), and it is why motor and gearbox are often identified separately.
  * Coulomb friction (0.03-0.08 N*m here) is invisible at full torque: the loop's gain is 998 N*m/rad, so friction
    moves the holding point by 0.03 encoder counts. It shows when the drive is small, hence the creep tests with the
    torque limit lowered: at a drive of 0.1 N*m above gravity, a 10% friction error is a 5% creep-speed error, tens of
    counts over a few seconds. Without the creeps (`--ablate creeps`, seed 1) four of the six frictions miss by
    11-25%. On a real arm: identify friction with slow motion at low torque.
  * The quantized encoder and the deadband inside the servo's loop make the mismatch a noisy function of the
    parameters: finite differences of the exact model point nowhere (the first fit stalled at ~5 counts rms with the
    damping 33-96% high, even with the delay given). Slopes from the twin without quantization, scores from the exact
    model, and the same fit converges to ~1 count rms (the recording's own rounding is 0.3).
  * The bus delay looks like a staircase to the fit: a goal acts at the servo's next 2 ms tick, so delays between
    ticks only show through the spread of the host's write times against the servo's clock. The host's jitter, which
    a real campaign has anyway, is what makes a 5.2 ms delay distinguishable from 6 ms, and it must span a whole tick:
    with +/- 0.8 ms of jitter every delay from 1.8 to 2.2 ms gave the same recording to the last count (seed 3,
    planted 2.12 ms). A host whose clock drifts against the servo's does the same job. And delay trades against
    inertia (a lighter joint answers sooner): fitting everything, then scanning the delay with the rest held, left
    seed 3's 2.12 ms delay at 2.96 ms (+40%) with the armatures 1-6% low to make up for it, at 1.76 counts rms. With
    the delay held at the truth and the rest refitted the mismatch is 1.00 counts: the information is there, the
    search was the problem, hence the profile (refit at every delay) instead of the scan.
  * The holding joints are not rigid. Inside the 1-count deadband a servo does not push at all, so while one joint
    swings, the others flop within +/- 1 count on their own inertia and friction, and that feeds back. Fitting
    shoulder_lift alone with the other joints at nominal, the *true* lift parameters matched its recording at 5.6
    counts rms (seed 8; the elbow's armature alone, planted +37%, accounts for 5.2), worse than wrong ones the fit
    found, and the fit missed the lift's friction by 25%. With the other joints at their planted values the same
    one-joint fit lands within 8% on six seeds. So the campaign fits all six joints together, each joint's model
    run with every other joint's current estimate; the live test fits one joint with the others given.
  * The gripper's servo runs at half torque in the cell (LeRobot caps it); for identification its limit is raised to
    full (no object in the jaw).
  * A fast motor makes a ringing servo, and a ringing servo makes a needle of a cost (2026-11-03, acceptance attempt
    2: seeds 12 and 13 failed, delay +46% at 5.35 counts rms, gripper damping +14% at 3.04; the first campaign's
    seeds 1-3 had planted no-load speeds of 2.4-3.5 rad/s, these 6.2-6.4). Beyond 2 counts of error the servo's
    loop is saturated, so a big move is bang-bang, and after it the joint overshoots and swings about the goal until
    the energy is gone. Braking happens at the stall clip, where back-EMF does nothing, so what damps the swings is
    the back-EMF while driving, stall / no_load_speed: at 6.2 rad/s the pan joint still swung +/-20 counts 0.6 s
    after a 1.2 rad step, and the old 0.4-0.6 s segments each started mid-swing. Where a swing ends inside the
    deadband, whether a creep starts moving or sticking, when the torque goes off in a coast: each turns a small
    parameter change into a lasting offset, so the exact model's mismatch was 0.30 counts at the planted values (the
    recording's own rounding: record and fit models agree) and 5-9 counts anywhere within +/-6% of the pan's
    armature (seed 12), with no slope to follow; the delay's profile then picked 7.5 ms for 5.0. The stiff
    horn-play stops (so101.PLAY_SOLREF, new since the first campaign) are not the cause: with soft stops, or no play
    at all, the sensitivity is the same, and with the no-load speed set to 3 rad/s it drops to ~1 count. Two changes
    to the identification, none to the plant: every segment after a full-torque move is long enough for the swings
    to die out (T_SETTLE, T_STEP, T_PRECOAST, T_HOLD: 1.5, 1.3, 1.3, 1.0 s, so 72 s per joint instead of 56), and
    the fit compares only the reads before each full-torque recapture (`fit_weights`): the swings carry little the
    rest does not (the rise gives inertia and top speed, the torque-off window damping) and most of the roughness.
    With the settles the cost about the truth is a bowl again (pan armature, seed 12, every read: 1.5-1.7 counts at
    +/-1%, 4.3-4.6 at +/-6%). Settles alone passed seeds 12 and 13 at 7.0% and 8.2% worst; with the mask 3.9% and 4.1%,
    and seed 1 2.5%, but seed 2 then failed (every damping 2-15% high, gripper +14.5%) although the truth matched its
    fitted reads at 0.30 counts against 1.18 at the fit's answer: along the line from one to the other the mismatch
    stays at 1.0-1.2 counts (the deadband's floor) until the truth itself, and the fit had traded the gripper's
    coasts (726 reads, 2.95 counts at its answer) for its creeps (5716 reads, 0.41). A creep's reads are not
    independent (one offset lasts the whole creep), so counting reads let the creeps outvote the coasts, which are
    what see the damping. Hence KIND_BALANCE: the fitted reads are weighted so creeps, steps, coasts and chirps each
    carry the same total weight. (Added 2026-11-03; acceptance attempt 3 ran with it: every parameter within 3.2% on seeds 21-23, f6-sysid.json.)

Run: envs/mj/bin/python -I factory/desk/sysid.py [--seeds 1,2,3] [--out ...] [--figure ...] [--workers N]
     [--ablate coasts|creeps] [--figure-only]
(CPU only; heavy runs under bin/memguard 32).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import so101  # noqa: E402
from servo import ServoBus, DeviceKit, REAL_PLANT, RAD_PER_COUNT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "results", "f6-sysid.json")
FIGURE = os.path.join(ROOT, "docs", "media", "F6-sysid.png")
DT = 0.002                    # physics step = the servo's loop tick here (the desk cell's timestep)
HOST_HZ = 250.0               # sync-write + sync-read rate of the campaign (see the docstring)
JITTER_S = 0.0010             # +/- host clock jitter (uniform, spans a whole servo tick), logged with every read
JOINT_PARAMS = ("armature", "damping", "frictionloss")
SHARED_PARAMS = ("bus_delay_s", "no_load_speed")
NOMINAL = dict(armature=REAL_PLANT["armature"][0], damping=REAL_PLANT["damping"][0], frictionloss=REAL_PLANT["frictionloss"][0],
               bus_delay_s=DeviceKit().bus_delay_s, no_load_speed=DeviceKit().no_load_speed)
TOL = 0.10
# test pose per excited joint (the others hold it): no self-contact over the excited joint's +/- 0.8 rad sweep
POSE_MID = (0.0, -0.6, 0.6, 0.8, 0.0, 0.6)
POSE_UP = (0.0, -1.57, 1.57, 0.0, 0.0, 0.6)
POSES = {0: POSE_MID, 1: POSE_MID, 2: POSE_UP, 3: POSE_MID, 4: POSE_MID, 5: POSE_MID}
CENTER = {0: 0.0, 1: -0.6, 2: 0.84, 3: 0.8, 4: 0.0, 5: 0.68}
CREEP_NET = (0.08, 0.16, 0.32)   # N*m of drive above the worst gravity load (+ nominal friction) in the creep range
CREEP_HALF = 0.4                 # rad, creep range about the centre
STEPS = (0.02, 0.2, 0.6, 1.2)    # rad, step sizes (each both directions about the centre)
CHIRPS = ((0.2, 0.2, 3.0, 8.0), (0.04, 1.0, 12.0, 6.0))   # (amplitude rad, f0 Hz, f1 Hz, seconds), log sweeps
COASTS = ((0.06, 0.25), (0.12, 0.20), (0.04, 0.35), (0.09, 0.30))   # (torque off this long after the 1.2 rad step's write, for this long), both directions
COAST_STEP = 1.2
# segment lengths (s): long enough for the servo to come to rest before the next segment starts (see the docstring)
T_SETTLE, T_STEP, T_PRECOAST, T_HOLD = 1.5, 1.3, 1.3, 1.0


# ---------------------------------------------------------------- the plant
def plant(seed: int) -> dict:
    """The planted 'true' plant for a seed: every parameter 20-50% off nominal, up or down."""
    rng = np.random.default_rng(1000 + seed)
    off = lambda n=None: 1.0 + rng.choice([-1.0, 1.0], size=n) * rng.uniform(0.2, 0.5, size=n)
    P = {k: (NOMINAL[k] * off(6)).tolist() for k in JOINT_PARAMS}
    for k in SHARED_PARAMS: P[k] = float(NOMINAL[k] * off())
    return P


def nominal_plant() -> dict:
    P = {k: [NOMINAL[k]] * 6 for k in JOINT_PARAMS}
    for k in SHARED_PARAMS: P[k] = NOMINAL[k]
    return P


def compile_arm():
    """The bare SO-101 (base upright, fixed), backlash joints on, the desk cell's physics options."""
    s = so101.arm_spec(backlash=True, backlash_rad=DeviceKit().backlash_rad, plant=REAL_PLANT)
    s.option.timestep = DT; s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; s.option.impratio = 10; s.option.iterations = 30; s.option.ls_iterations = 30
    return s.compile()


def set_plant(m, d, ix, P):
    """Write per-joint plant parameters into a compiled model. mj_setConst recomputes what the compiler derives from
    them (dof_invweight0, which sets the friction constraint's softness): checked identical to recompiling."""
    m.dof_armature[ix.dadr] = P["armature"]; m.dof_damping[ix.dadr] = P["damping"]; m.dof_frictionloss[ix.dadr] = P["frictionloss"]
    mujoco.mj_setConst(m, d)


class CampaignBus(ServoBus):
    """The servo bus plus the one register write a campaign needs that the cell does not: Torque_Enable (STS3215
    address 40). With torque off the servo's driver lets go of the motor (no drive and no back-EMF current: the arm
    can be back-driven by hand), so the joint coasts on the gearbox's damping and friction alone. The write travels
    the bus like a goal (bus_delay_s). Everything else is ServoBus unchanged."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.enabled = np.ones(6, bool); self.pending_enable = []

    def write_torque_enable(self, joint: int, on: bool, t: float):
        dl = self.kit.bus_delay_s if self.kit.real else 0.0
        self.pending_enable.append((t + dl, joint, bool(on)))

    def step(self, t):
        while self.pending_enable and self.pending_enable[0][0] <= t + 1e-9:
            _, j, on = self.pending_enable.pop(0); self.enabled[j] = on
        tau = super().step(t)
        if not self.enabled.all():
            tau = np.where(self.enabled, tau, 0.0); self.d.ctrl[self.ix.act] = tau
        return tau


# ---------------------------------------------------------------- the experiment design
def _gravity_range(m, d, ix, j, pose, lo, hi):
    g = []
    for x in np.linspace(lo, hi, 17):
        q = np.array(pose, float); q[j] = x; mujoco.mj_resetData(m, d); ix.set_q(d, q); mujoco.mj_forward(m, d)
        g.append(d.qfrc_bias[ix.dadr[j]])
    return float(np.max(np.abs(g)))


def design(j: int, seed: int = 0, short: bool = False) -> dict:
    """The goal stream for joint j's recording: host times, the excited joint's goal and torque limit at each write,
    and the segment boundaries (for the report and the figure). `short` (the live test) keeps two creep levels, three
    step sizes and two coast timings (52 s instead of 72)."""
    m = compile_arm(); d = mujoco.MjData(m); ix = so101.Indices(m)
    pose = np.array(POSES[j], float); c = CENTER[j]; pose[j] = c
    gmax = _gravity_range(m, d, ix, j, pose, c - CREEP_HALF - 0.05, c + CREEP_HALF + 0.05)
    segs = []                                            # (name, duration, goal(t) relative function, limit)
    segs.append(("settle", T_SETTLE, lambda t: c - CREEP_HALF, 1.0))
    nets = CREEP_NET[::2] if short else CREEP_NET
    for net in nets:
        L = min(1.0, (gmax + NOMINAL["frictionloss"] + net) / so101.STALL_TORQUE)
        v = net / (so101.STALL_TORQUE / NOMINAL["no_load_speed"] + NOMINAL["damping"])
        T = float(min(5.0, 2 * CREEP_HALF / v + 0.5))
        segs.append((f"creep {net:.2f}+", T, lambda t: c + CREEP_HALF, L))
        segs.append((f"creep {net:.2f}-", T, lambda t: c - CREEP_HALF, L))
    segs.append(("recentre", T_STEP, lambda t: c, 1.0))
    for a in (STEPS[1:] if short else STEPS):
        segs.append((f"step {a:.2f}+", T_STEP, lambda t, a=a: c + a / 2, 1.0))
        segs.append((f"step {a:.2f}-", T_STEP, lambda t, a=a: c - a / 2, 1.0))
    segs.append(("hold", T_HOLD, lambda t: c, 1.0))
    for (off, T_off) in (COASTS[:2] if short else COASTS):
        for sg in (1, -1):
            segs.append(("pre-coast", T_PRECOAST, lambda t, sg=sg: c - sg * COAST_STEP / 2, 1.0))
            segs.append((f"coast {off:.2f}/{T_off:.2f} {'+' if sg > 0 else '-'}", 0.9, lambda t, sg=sg: c + sg * COAST_STEP / 2, 1.0, (off, off + T_off)))
    segs.append(("hold", T_HOLD, lambda t: c, 1.0))
    for (A, f0, f1, T) in CHIRPS:
        k = math.log(f1 / f0) / T
        segs.append((f"chirp {A:.2f} rad {f0:g}-{f1:g} Hz", T,
                     lambda t, A=A, f0=f0, k=k: c + A * math.sin(2 * math.pi * f0 * (math.exp(k * t) - 1) / k), 1.0))
        segs.append(("hold", T_HOLD, lambda t: c, 1.0))
    rng = np.random.default_rng(77 + 13 * j + 1000 * seed)
    t_host, goal, limit, enable, seg_of, bounds = [], [], [], [], [], []
    t0 = 0.0
    for si, sgm in enumerate(segs):
        name, T, fn, L = sgm[:4]; off = sgm[4] if len(sgm) > 4 else None
        n = int(round(T * HOST_HZ)); bounds.append(dict(name=name, t0=t0, t1=t0 + n / HOST_HZ, torque_limit=L, torque_off=off))
        for k in range(n):
            tn = t0 + k / HOST_HZ
            t_host.append(tn + DT / 2 + rng.uniform(-JITTER_S, JITTER_S)); goal.append(fn(k / HOST_HZ)); limit.append(L); seg_of.append(si)
            enable.append(not (off is not None and off[0] <= k / HOST_HZ < off[1]))
        t0 += n / HOST_HZ
    return dict(joint=j, pose=pose.tolist(), t_host=np.array(t_host), goal=np.clip(goal, ix.lo[j], ix.hi[j]), limit=np.array(limit),
                enable=np.array(enable), seg=np.array(seg_of), segments=bounds, gravity_max=gmax)


FIT_MASK = True           # compare only the reads fit_weights picks (False: every read)
FIT_HOLDS = False          # fit the settle / hold / pre-coast segments too (the servo's recapture ringing; see the docstring)
STEP_MARGIN_S = 0.02       # a step is fitted from its write until the recording first reaches the goal, plus this
COAST_MARGIN_S = 0.01      # a coast is fitted from its step's write until the torque comes back on, plus this
CHIRP_FIT = None           # None: every chirp in full; else a set of chirp indices to fit
KIND_BALANCE = True        # weight the fitted reads so creeps, steps, coasts and chirps count alike (see the docstring)


def fit_weights(e: dict) -> np.ndarray:
    """Which reads the fit compares (1) and which it only simulates through (0), decided once from the recording.
    Fitted: the creeps, each step until the joint first reaches the goal, each coast until its torque comes back on,
    the chirps. Not fitted: what follows a full-torque arrival, where the servo recaptures the goal (see the docstring)."""
    w = np.zeros(len(e["t_host"])); rec = e["recorded"]; ci = 0
    for si, s in enumerate(e["segments"]):
        idx = np.where(e["seg"] == si)[0]
        if len(idx) == 0: continue
        kind = s["name"].split()[0]; t_rel = e["t_host"][idx] - s["t0"]
        if kind == "creep": w[idx] = 1
        elif kind == "step":
            g = e["goal"][idx[0]]; side = np.sign(rec[idx[0]] - g) or 1.0
            hit = np.where((rec[idx] - g) * side <= 0)[0]
            end = t_rel[hit[0]] + STEP_MARGIN_S if len(hit) else np.inf
            w[idx[t_rel <= end]] = 1
        elif kind == "coast":
            w[idx[t_rel <= s["torque_off"][1] + COAST_MARGIN_S]] = 1
        elif kind == "chirp":
            if CHIRP_FIT is None or ci in CHIRP_FIT: w[idx] = 1
            ci += 1
        elif FIT_HOLDS: w[idx] = 1
    if KIND_BALANCE:                                    # each kind of experiment carries the same total weight
        kinds = np.array([s["name"].split()[0] for s in e["segments"]])[e["seg"]]; fitted = w > 0; n = fitted.sum()
        present = sorted(set(kinds[fitted]))
        for k in present:
            m = fitted & (kinds == k); w[m] = math.sqrt(n / (len(present) * m.sum()))
    return w


# ---------------------------------------------------------------- simulation (truth and candidates alike)
_CACHE = {}


def _model():
    if "m" not in _CACHE:
        m = compile_arm(); d = mujoco.MjData(m); _CACHE["m"] = (m, d, so101.Indices(m))
    return _CACHE["m"]


def simulate(exp: dict, P: dict, smooth: bool = False):
    """Run joint exp['joint']'s recording through plant P. Returns (reads the host gets: quantized encoder counts as
    radians, bus_delay_s old; the same reads unquantized: the shaft angle at the same instants). `smooth` is the
    fit's twin for slopes: the same servo without encoder quantization and deadband inside its loop."""
    m, d, ix = _model(); set_plant(m, d, ix, P)
    kit = DeviceKit.real_kit(bus_delay_s=float(P["bus_delay_s"]), no_load_speed=float(P["no_load_speed"]))
    if smooth: kit.quantize = False; kit.deadband_counts = 0
    j = exp["joint"]; th, gj, lim, en = exp["t_host"], exp["goal"], exp["limit"], exp["enable"]
    mujoco.mj_resetData(m, d); bus = CampaignBus(m, d, kit); bus.reset(np.array(exp["pose"], float)); on = True; ncon = 0
    goal = np.array(exp["pose"], float)
    n = len(th); nsteps = int(math.ceil((th[-1] + 0.02) / DT))
    qcont = np.empty(nsteps + 1)
    reads = np.empty(n); k_host = 0; dq = ix.qadr[j]; delay = kit.bus_delay_s
    for k in range(nsteps):
        tk = d.time; qcont[k] = d.qpos[dq]
        while k_host < n and th[k_host] <= tk + 1e-12:
            reads[k_host] = bus.read(th[k_host])[j]
            if lim[k_host] != bus.limit[j]: bus.set_torque_limit(j, float(lim[k_host]))
            if en[k_host] != on: on = bool(en[k_host]); bus.write_torque_enable(j, on, th[k_host])
            goal[j] = gj[k_host]; bus.write(goal, th[k_host]); k_host += 1
        bus.step(tk); mujoco.mj_step(m, d); ncon = max(ncon, d.ncon)
    qcont[nsteps] = d.qpos[dq]
    # the unquantized read: the shaft angle at the last servo tick at or before t - delay (what bus.read holds)
    idx = np.floor((th - delay) / DT + 1e-9).astype(int); idx = np.clip(idx, 0, nsteps); _CACHE["ncon"] = ncon
    return reads, qcont[idx]


def record(seed: int, joints=range(6), short: bool = False):
    P = plant(seed); out = {}
    for j in joints:
        e = design(j, seed, short=short); e["recorded"], _ = simulate(e, P)
        e["w"] = fit_weights(e) if FIT_MASK else np.ones(len(e["t_host"])); out[j] = e
    return P, out


# ---------------------------------------------------------------- the fit
NZ_J = len(JOINT_PARAMS)
H_JOINT, H_W0, H_DELAY = 0.03, 0.03, 0.15     # forward-difference steps in log space (the delay's spans ~1 servo tick)


def to_plant(z, js, base=None):
    """z = [log(p / nominal) for each joint in js and each of JOINT_PARAMS] + [log w0 / w0_nom, log delay / delay_nom].
    Joints not in js take their values from `base` (default: nominal)."""
    P = nominal_plant() if base is None else {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for i, j in enumerate(js):
        for p, k in enumerate(JOINT_PARAMS): P[k][j] = NOMINAL[k] * math.exp(z[NZ_J * i + p])
    P["no_load_speed"] = NOMINAL["no_load_speed"] * math.exp(z[-2]); P["bus_delay_s"] = NOMINAL["bus_delay_s"] * math.exp(z[-1])
    return P


def _resid_job(args):
    exp, P, smooth = args
    _, pred = simulate(exp, P, smooth)
    return exp.get("w", 1.0) * (pred - exp["recorded"]) / RAD_PER_COUNT       # counts, the unfitted reads weighted 0


class Fitter:
    """Least squares over z (see to_plant). The residual is the exact device model's; the Jacobian is the smooth
    twin's, by forward differences.

    Why a twin: the real servo's loop sees a quantized encoder and a 1-count deadband, so a 1% change of any parameter
    flips some rounding somewhere, the 1.5 N*m kick that one count of error makes sends the trajectory a little
    elsewhere, and the mismatch jumps from 0.3 counts rms (the quantization of the recording) to ~1 count. That jump
    is as large as the true slope a 2% difference would show, so finite differences of the exact model are mostly
    noise. The same servo with the quantization and the deadband taken out of its loop has the same bowl without the
    noise; its slopes steer Gauss-Newton, and the exact model decides where the bowl's bottom is."""

    def __init__(self, exps: dict, pool, base=None):
        self.exps = exps; self.js = sorted(exps); self.pool = pool; self.nsim = 0; self.base = base
        self.lens = [len(exps[j]["t_host"]) for j in self.js]; self.offs = np.cumsum([0] + self.lens)
        self.nfit = float(sum((exps[j]["w"] ** 2).sum() if "w" in exps[j] else len(exps[j]["t_host"]) for j in self.js))   # reads the fit compares

    def _run(self, jobs):
        self.nsim += len(jobs); return list(self.pool.map(_resid_job, jobs, chunksize=1)) if self.pool else [_resid_job(a) for a in jobs]

    def fun(self, z):
        P = to_plant(z, self.js, self.base); return np.concatenate(self._run([(self.exps[j], P, False) for j in self.js]))

    def jac(self, z):
        jobs, where = [], []
        P = to_plant(z, self.js, self.base)
        for i, j in enumerate(self.js): jobs.append((self.exps[j], P, True)); where.append((i, None, 0))
        for i, j in enumerate(self.js):
            for p in range(NZ_J):
                zz = z.copy(); zz[NZ_J * i + p] += H_JOINT; jobs.append((self.exps[j], to_plant(zz, self.js, self.base), True)); where.append((i, NZ_J * i + p, H_JOINT))
        for col, h in ((len(z) - 2, H_W0), (len(z) - 1, H_DELAY)):
            if col == len(z) - 1 and self.fixed_delay: continue
            zz = z.copy(); zz[col] += h; Pz = to_plant(zz, self.js, self.base)
            for i, j in enumerate(self.js): jobs.append((self.exps[j], Pz, True)); where.append((i, col, h))
        out = self._run(jobs); base = out[:len(self.js)]
        J = np.zeros((self.offs[-1], len(z)))
        for r, (i, col, h) in zip(out[len(self.js):], where[len(self.js):]):
            sl = slice(self.offs[i], self.offs[i + 1]); J[sl, col] = (r - base[i]) / h
        return J

    fixed_delay = False

    def solve(self, z0, max_nfev=40, fix_delay=False):
        """Gauss-Newton from z0; with fix_delay the last entry (the delay) is held at z0's value."""
        from scipy.optimize import least_squares
        self.fixed_delay = fix_delay; z0 = np.asarray(z0, float); n = len(z0) - (1 if fix_delay else 0)
        full = lambda y: np.concatenate([y, z0[n:]])
        lo = np.full(n, math.log(0.25)); hi = np.full(n, math.log(4.0))
        r = least_squares(lambda y: self.fun(full(y)), np.clip(z0[:n], lo + 1e-9, hi - 1e-9), jac=lambda y: self.jac(full(y))[:, :n],
                          bounds=(lo, hi), x_scale=1.0, max_nfev=max_nfev, xtol=1e-3, ftol=1e-5, method="trf")
        r.x = full(r.x); self.fixed_delay = False
        return r


def _profile_job(args):
    exps, z, dl, max_nfev, base = args
    g = Fitter(exps, None, base); zz = np.array(z, float); zz[-1] = math.log(dl / NOMINAL["bus_delay_s"])
    r = g.solve(zz, fix_delay=True, max_nfev=max_nfev)
    return float(2 * r.cost), dl, r.x, g.nsim


def profile_delay(f: Fitter, z, grid, max_nfev):
    """The delay's profile: for each delay on the grid, Gauss-Newton on everything else from z; the delay whose refit
    matches best. (A plain scan with the rest held fails: the rest has already bent to fit the wrong delay.)"""
    out = []
    if f.pool is not None and len(f.js) <= 2:          # few joints: one serial refit per worker, the grid in parallel
        out = list(f.pool.map(_profile_job, [(f.exps, z, float(dl), max_nfev, f.base) for dl in grid], chunksize=1))
        f.nsim += sum(o[3] for o in out); out = [o[:3] for o in out]
    else:
        for dl in grid:
            zz = np.array(z, float); zz[-1] = math.log(dl / NOMINAL["bus_delay_s"])
            r = f.solve(zz, fix_delay=True, max_nfev=max_nfev); out.append((float(2 * r.cost), float(dl), r.x))
    best = min(out, key=lambda o: o[0])
    return best, [(round(1e3 * dl, 3), c) for c, dl, _ in out]


def identify(exps: dict, pool=None, log=print, coarse=np.arange(0.0010, 0.0080 + 1e-9, 0.0005), fine_step=0.0001, known_delay=None, base=None):
    """From nominal: Gauss-Newton on the joints, the no-load speed and the delay together; then the delay's profile on a
    coarse grid (0.5 ms) and a fine one (0.1 ms) around the best; then Gauss-Newton with the delay held. With
    `known_delay` (s) the delay is taken as measured and only the last stage runs. Joints without a recording take
    their parameters from `base` (default nominal): the holding joints matter (see the docstring). Returns the
    recovered plant and a log of the stages."""
    js = sorted(exps); z = np.zeros(NZ_J * len(js) + 2); steps = []; f = Fitter(exps, pool, base)
    stages = ("fit", "coarse delay profile", "fine delay profile", "final fit")
    if known_delay is not None: z[-1] = math.log(known_delay / NOMINAL["bus_delay_s"]); stages = ("final fit",)
    for stage in stages:
        t0 = time.time(); n0 = f.nsim; extra = {}
        if stage == "fit":
            r = f.solve(z); z = r.x; cost = float(2 * r.cost)
        elif stage.endswith("profile"):
            if stage.startswith("coarse"): grid = coarse
            else:
                d0 = NOMINAL["bus_delay_s"] * math.exp(z[-1]); grid = d0 + fine_step * np.arange(-4, 5); grid = grid[grid > 0.0002]
            (cost, _, z), prof = profile_delay(f, z, grid, max_nfev=5 if stage.startswith("coarse") else 8); extra = dict(profile_ms_cost=prof)
        else:
            r = f.solve(z, fix_delay=True, max_nfev=40); z = r.x; cost = float(2 * r.cost)
        P = to_plant(z, js, base)
        steps.append(dict(stage=stage, cost_counts2=cost, rms_counts=math.sqrt(cost / f.nfit), sims=f.nsim - n0, s=round(time.time() - t0, 1),
                          delay_ms=round(1e3 * P["bus_delay_s"], 3), no_load_speed=round(P["no_load_speed"], 4), **extra))
        log(f"  {stage}: rms {steps[-1]['rms_counts']:.3f} counts, delay {1e3 * P['bus_delay_s']:.2f} ms, w0 {P['no_load_speed']:.3f} rad/s "
            f"({f.nsim - n0} sims, {time.time() - t0:.1f} s)")
    return to_plant(z, js, base), steps


def compare(planted, recovered, joints):
    rows = []
    for k in JOINT_PARAMS:
        for j in joints:
            a, b = planted[k][j], recovered[k][j]
            rows.append(dict(param=k, joint=so101.JOINTS[j], planted=a, recovered=b, nominal=NOMINAL[k], rel_err=(b - a) / a))
    for k in SHARED_PARAMS:
        a, b = planted[k], recovered[k]
        rows.append(dict(param=k, joint=None, planted=a, recovered=b, nominal=NOMINAL[k], rel_err=(b - a) / a))
    return rows


# ---------------------------------------------------------------- figure
SHORT = dict(shoulder_pan="pan", shoulder_lift="lift", elbow_flex="elbow", wrist_flex="w.flex", wrist_roll="w.roll", gripper="grip")


def figure(path, seed_runs, joint=1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    first = seed_runs[0]; e = first["exps"][joint]; P_fit = first["recovered"]
    _, y_fit = simulate(e, P_fit); _, y_nom = simulate(e, nominal_plant())
    fig = plt.figure(figsize=(15, 9.5), dpi=110); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.25)
    seg = lambda pre: next(s for s in e["segments"] if s["name"].startswith(pre))
    panels = ((seg("chirp 0.20"), "chirp, 0.2 rad, 0.2 to 3 Hz", 4.0, 6.5),
              (seg("step 1.20+"), "1.2 rad step, full torque", -0.05, 0.5),
              (seg("coast 0.06"), "coast: torque off 0.06 to 0.31 s", -0.05, 0.75))
    for c, (s, title, a0, a1) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c]); t0, t1 = s["t0"] + a0, s["t0"] + a1
        msk = (e["t_host"] >= t0) & (e["t_host"] <= t1); tt = e["t_host"][msk] - s["t0"]
        if s.get("torque_off"): ax.axvspan(*s["torque_off"], color="#d0442a", alpha=0.08, lw=0, label="torque off")
        ax.plot(tt, np.degrees(e["goal"][msk]), color="#999999", lw=1, ls="--", label="goal written")
        ax.plot(tt, np.degrees(y_nom[msk]), color="#d08c2a", lw=1.5, label="nominal model (the fit's start)")
        ax.plot(tt, np.degrees(e["recorded"][msk]), ".", color="#222222", ms=2.6, label="recorded: 12-bit encoder, 250 Hz")
        ax.plot(tt, np.degrees(y_fit[msk]), color="#2a6fd0", lw=1.5, label="identified model")
        ax.set_title(f"{so101.JOINTS[joint]}, seed {first['seed']}: {title}", fontsize=10.5); ax.set_xlabel("s into the segment")
        if c == 0: ax.set_ylabel("joint angle, deg")
        ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    h, l = fig.axes[1].get_legend_handles_labels()
    fig.legend(h + [Patch(color="#d0442a", alpha=0.15)], l + ["torque off (coast)"], fontsize=9, loc="center", ncol=5,
               bbox_to_anchor=(0.5, 0.505), frameon=False)
    ax = fig.add_subplot(gs[1, :]); rows0 = seed_runs[0]["rows"]
    x = np.arange(len(rows0)) + np.array([i // 6 if r["joint"] else 3 for i, r in enumerate(rows0)]) * 0.8   # gaps between groups
    colors = ("#2a6fd0", "#d0442a", "#3a9a4a", "#8a4ad0")
    ax.axhspan(-10, 10, color="#3a9a4a", alpha=0.10, lw=0)
    for si, run in enumerate(seed_runs):
        rows = run["rows"]; off = (si - (len(seed_runs) - 1) / 2) * 0.2; col = colors[si % len(colors)]
        ax.plot(x + off, [100 * (r["planted"] / r["nominal"] - 1) for r in rows], "o", mfc="none", color=col, ms=5, alpha=0.55)
        ax.plot(x + off, [100 * r["rel_err"] for r in rows], "s", color=col, ms=5.5, label=f"seed {run['seed']}")
    ax.plot([], [], "o", mfc="none", color="#555555", label="planted: % off nominal"); ax.plot([], [], "s", color="#555555", label="recovered: % error vs planted")
    for k, name in enumerate(JOINT_PARAMS):
        ax.text(x[6 * k:6 * k + 6].mean(), 56, name, ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(x[-2:].mean(), 56, "servo / bus", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r["joint"]] if r["joint"] else {"bus_delay_s": "bus\ndelay", "no_load_speed": "no-load\nspeed"}[r["param"]] for r in rows0], fontsize=8.5)
    ax.set_ylabel("%"); ax.set_ylim(-60, 60); ax.grid(axis="y", alpha=0.3)
    ax.set_title("every planted parameter, every seed: hollow = how far it was planted from nominal, filled = recovery error (green band: +/-10%)", fontsize=10.5)
    ax.legend(fontsize=8, ncol=len(seed_runs) + 2, loc="lower left")
    fig.suptitle("F6 system identification: the SO-101's planted servo and joint parameters recovered from creep, step, coast and chirp recordings", fontsize=12.5)
    os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--out", default=OUT); ap.add_argument("--figure", default=FIGURE)
    ap.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 4))
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from --out's JSON, no fitting")
    ap.add_argument("--ablate", choices=("coasts", "creeps"), help="leave one kind of experiment out (to see what it buys); "
                    "writes results/f6-sysid-no-<kind>.json and no figure unless --out/--figure are given")
    a = ap.parse_args(argv)
    global COASTS, CREEP_NET
    if a.ablate == "coasts": COASTS = ()
    if a.ablate == "creeps": CREEP_NET = ()
    if a.ablate and a.out == OUT: a.out = os.path.join(HERE, "results", f"f6-sysid-no-{a.ablate}.json")
    if a.ablate and a.figure == FIGURE: a.figure = None
    if a.figure_only:                                   # redraw from the recorded JSON (the recordings are deterministic)
        R = json.load(open(a.out)); runs = [dict(seed=sd["seed"], recovered=sd["recovered"], rows=sd["rows"], exps=record(sd["seed"])[1]) for sd in R["seeds"]]
        print("wrote", figure(a.figure, runs)); return 0
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    runs, all_ok, T0 = [], True, time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for seed in seeds:
            t0 = time.time(); P, exps = record(seed)
            print(f"seed {seed}: recorded {sum(len(e['t_host']) for e in exps.values())} reads over {sum(e['t_host'][-1] for e in exps.values()):.0f} s")
            R, steps = identify(exps, pool)
            rows = compare(P, R, range(6)); ok = all(abs(r["rel_err"]) <= TOL for r in rows); all_ok &= ok
            worst = max(rows, key=lambda r: abs(r["rel_err"]))
            print(f"seed {seed}: worst {worst['param']} {worst['joint'] or ''} {100 * worst['rel_err']:+.1f}%  -> {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.0f} s)")
            for r in rows: print(f"    {r['param']:14s} {str(r['joint'] or ''):14s} planted {r['planted']:.5g}  recovered {r['recovered']:.5g}  {100 * r['rel_err']:+6.2f}%")
            runs.append(dict(seed=seed, planted=P, recovered=R, rows=rows, ok=ok, fit_s=round(time.time() - t0, 1), steps=steps, exps=exps))
    e0 = runs[0]["exps"]
    res = dict(date="2026-11-03", phase="F6", discipline=14, tolerance=TOL, all_within_tolerance=bool(all_ok), ablated=a.ablate,
               experiment=dict(host_hz=HOST_HZ, host_jitter_s=JITTER_S, physics_dt=DT, joints=list(so101.JOINTS), poses={so101.JOINTS[j]: list(POSES[j]) for j in range(6)},
                               centres={so101.JOINTS[j]: CENTER[j] for j in range(6)}, creep_net_torque=list(CREEP_NET), creep_half_rad=CREEP_HALF,
                               coasts=[dict(torque_off_after_s=o, torque_off_for_s=T, step_rad=COAST_STEP) for o, T in COASTS],
                               steps_rad=list(STEPS), chirps=[dict(amplitude_rad=A, f0_hz=f0, f1_hz=f1, seconds=T) for A, f0, f1, T in CHIRPS],
                               segments={so101.JOINTS[j]: e0[j]["segments"] for j in e0}, seconds_per_joint={so101.JOINTS[j]: round(float(e0[j]["t_host"][-1]), 2) for j in e0},
                               reads_per_joint={so101.JOINTS[j]: int(len(e0[j]["t_host"])) for j in e0},
                               segment_lengths_s=dict(settle=T_SETTLE, step=T_STEP, pre_coast=T_PRECOAST, hold=T_HOLD),
                               fitted_reads_per_joint={so101.JOINTS[j]: int((e0[j]["w"] > 0).sum()) for j in e0},
                               fitted="creeps; each step until the recording first reaches the goal + %g s; each coast until "
                                      "the torque is back on + %g s; chirps; not the settles, holds and recaptures" % (STEP_MARGIN_S, COAST_MARGIN_S),
                               measured="present position (12-bit, bus_delay_s old) at each host cycle; goals, torque-limit and torque-enable writes logged with host time",
                               not_identified="kp, kv, deadband, encoder resolution (servo registers), link inertias (CAD), backlash (beyond the encoder: unobservable from servo data)"),
               nominal=NOMINAL,
               seeds=[dict(seed=r["seed"], ok=r["ok"], fit_s=r["fit_s"], planted=r["planted"], recovered=r["recovered"], rows=r["rows"], steps=r["steps"],
                           max_abs_rel_err=max(abs(x["rel_err"]) for x in r["rows"])) for r in runs],
               total_s=round(time.time() - T0, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, "and", figure(a.figure, runs) if a.figure else "no figure", f"({time.time() - T0:.0f} s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
