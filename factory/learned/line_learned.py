"""line_learned.py — F3 on the line (2026-11-02): F2's line with the learned carry.

Tracking, scheduling, the intercept, the grasp and the lift are F2's. Everything after the lift (rise, cross, lower,
open, wait, retreat) is replaced by the carry policy (carry.CarryCtl): two trajectory views on one controller, named
"transfer" (until it lets go; F2's slip and seal checks watch it) and "release" (the gripper opens, the tool brakes to a
stop; the item flies into the chute and the line's lifecycle sees it land). Everything else (outcomes, faults,
recovery, the live view) is unchanged.

    from line_learned import LearnedLine
    ln = LearnedLine(policy="factory/learned/policy.json", speed=0.3, rate_per_min=16, seed=0); S = ln.run(180)
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import carry  # noqa: E402
import line as LN  # noqa: E402

EST_CARRY_S = 1.2       # the scheduler's estimate of the carry (mean of the trained policy on the start states: ~1.1 s)
CARRY_BUDGET_S = 4.0
POLICY = os.path.join(HERE, "policy.json")


class CarryView:
    """A trajectory-shaped window on the carry controller, for the line's executor."""

    def __init__(self, ctl, phase, cmd=None):
        self.ctl = ctl; self.phase = phase; self.name = "transfer" if phase == 0 else "release"
        self.grip = self.suction = self.extend = None
        for k, v in (cmd or {}).items(): setattr(self, k, v)

    @property
    def T(self):
        c = self.ctl
        if self.phase == 0:
            if c.fail: return 0.0
            return c.t_release if c.t_release is not None else math.inf
        return (c.t_stop - c.t_release) if c.t_stop is not None else math.inf

    @property
    def q(self): return self.ctl.q_next[None, :]   # where it ends (the line holds the last pose when the plan is done)

    def at(self, tau): return self.ctl.at(tau if self.phase == 0 else self.ctl.t_release + tau)


def load_policy(policy):
    if policy is None or isinstance(policy, carry.Policy): return policy or carry.Policy.zero()
    if policy == "prior": return carry.Policy.zero()
    return carry.Policy(json.load(open(policy))["theta"])


class LearnedLine(LN.Line):
    def __init__(self, policy=POLICY, privileged_carry=True, **kw):
        """privileged_carry=False: F4's student carry (CarryCtl(privileged=False)): nothing read from the held item's
        true state; the hang comes from the robot's pose when the fingers closed (FK), the rest is the rigid estimate."""
        super().__init__(**kw); self.policy = load_policy(policy); arm = self.planner.arm; orig = self.lp.plan; self.privileged_carry = privileged_carry

        def plan(name, track, t_now, q_now, tracks):
            pl, why = orig(name, track, t_now, q_now, tracks)
            if pl is None: return pl, why
            k = [tr.name for tr in pl.trajs].index("lift"); lower = next(tr for tr in pl.trajs if tr.name == "lower")
            hang = None
            if not self.privileged_carry:   # pinch above the belt when the pads closed = above the item's bottom
                close = next(tr for tr in pl.trajs if tr.name == "close"); hang = float(arm.fk(close.q[-1])[0][2] - self.lay.belt.surface_z)
            ctl = carry.CarryCtl(self.c, arm, self.policy, name, pl.tool, pl.meta["carry_z"], pl.meta["a_carry"], arm.fk(lower.q[-1])[1], held=None,
                                 privileged=self.privileged_carry, hang=hang)
            rel = dict(grip=0.0) if pl.tool == "fingers" else dict(suction=0.0)
            pl.trajs = pl.trajs[:k + 1] + [CarryView(ctl, 0), CarryView(ctl, 1, rel)]
            pl.t_done = pl.t_start + sum(tr.T for tr in pl.trajs[:k + 1]) + EST_CARRY_S; pl.meta["carry"] = "learned"
            return pl, why
        self.lp.plan = plan

    def _apply(self, tr):
        if isinstance(tr, CarryView) and tr.phase == 0: tr.ctl.begin(self.cur.trajs[self.k_traj - 1].q[-1])
        return super()._apply(tr)

    def after_step(self, t, k):
        if self.cur is not None and not self.recovering:
            tr = self.cur.trajs[self.k_traj]
            if isinstance(tr, CarryView):
                if tr.ctl.fail: return self._abort(t, "carry_ik")
                if tr.phase == 0 and self.tau > CARRY_BUDGET_S: return self._abort(t, "carry_timeout")
                if tr.phase == 1 and self.tau > 1.5: tr.ctl.t_stop = tr.ctl.t_release + self.tau   # braking cannot take this long; free the robot
        return super().after_step(t, k)
