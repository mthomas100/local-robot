"""shift.py — F5: a production shift of the sorting cell with live KPIs (2026-11-02; disciplines 9, 12,
15; docs/sorting-cell-spec.md §4 F5: "a 10-minute continuous run with the KPIs on the dashboard").

The whole cell at once: items arrive at random (some in touching pairs: L3 clutter), the overhead camera is the only
source of where they are and what they are (F4 vision), the hybrid gripper (fingers, suction cup for flat boxes), the
learned carry in its vision-only student form, the safety layer (protective stops, recovery). The line never stops
for the length of the shift. Every simulated second the KPIs go to runs/<name>/kpis.jsonl and runs/<name>/live.json,
which the lab's dashboard (http://127.0.0.1:8800/dashboard) reads; the picture streams as MJPEG on --port (pick
camera with the camera's view inset), and is recorded to runs/<name>/shift.mp4.

KPIs (OEE-style accounting adapted to a sorter; definitions also in the operations report):
  throughput   sorted items per minute (cumulative and over the last 60 s)
  sort rate    sorted / items that arrived and left the line
  availability share of the shift the robot was not stopped or recovering (protective stops, drops, misses)
  performance  items the robot handled (picked at least once) / items that arrived and left: what capacity let through
  quality      sorted / items handled: of what it picked, how much ended in the right chute
  OEE          availability x performance x quality
  losses       missed, late, wrong chute, dropped, knocked off; robot events: protective stops, recoveries by kind

    envs/mj/bin/python -I factory/ops/shift.py [--minutes 10] [--speed 0.3] [--rate 12] [--clutter 0.25]
        [--perception vision] [--carry student] [--port 8804] [--pace 1.0] [--name f5-shift] [--seed 0]
    -> runs/<name>/{kpis.jsonl, live.json, items.jsonl, config.json, shift.mp4, shift-sheet.png}, factory/ops/results/<name>.json
"""
import argparse, json, os, sys, time
from collections import Counter, deque
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FACTORY = os.path.dirname(HERE); ROOT = os.path.dirname(FACTORY)
for p in (HERE, os.path.join(FACTORY, "line"), os.path.join(FACTORY, "learned"), os.path.join(FACTORY, "vision"), os.path.join(FACTORY, "cell")): sys.path.insert(0, p)

LOSSES = ("missed", "late", "wrong_chute", "dropped", "knocked_off")


class KPI:
    """Accumulates the shift's numbers from the line's own logs (items finished, attempts, the recovery flag)."""

    def __init__(self, ln): self.ln = ln; self.down_s = 0.0; self.last_t = 0.0; self.sorted_t = deque(); self.rows = []

    def tick(self, t):
        ln = self.ln; dt = t - self.last_t; self.last_t = t
        if ln.recovering and not (ln.cur is not None and ln.cur.meta.get("park")): self.down_s += dt   # the idle return to the ready pose is not downtime

    def snapshot(self, t):
        ln = self.ln; done = ln.log_items; cnt = Counter(i["outcome"] for i in done); n = len(done)
        handled = sum(1 for i in done if i.get("attempts", 0) > 0); ok = cnt["sorted"]
        ev = Counter(e["outcome"] for e in ln.events); oks = [e["cycle_s"] for e in ln.events if e["outcome"] == "ok" and "cycle_s" in e]
        recent = sum(1 for i in done if i["outcome"] == "sorted" and i.get("end_t", 0) > t - 60.0)
        A = 1.0 - self.down_s / max(t, 1e-6); P = handled / max(n, 1); Q = ok / max(handled, 1)
        mats = Counter(i["material"] for i in done if i["outcome"] == "sorted")
        on_belt = [dict(name=k, state=v["state"], x=round(float(ln.c.item_state(k)["pos"][0]), 3)) for k, v in ln.items.items() if v.get("state") in ("belt", "held", "settling")]
        robot = ln.cur.name if (ln.cur and not ln.recovering) else (("to ready" if ln.cur.meta.get("park") else "recovering") if ln.cur else "waiting")
        r = dict(t=round(t, 1), arrived=n, sorted=ok, sorted_per_min=round(ok / max(t / 60, 1e-6), 2), sorted_last_min=recent if t >= 60 else round(recent * 60 / max(t, 1e-6), 1),
                 sort_rate=round(ok / max(n, 1), 3), availability=round(A, 3), performance=round(P, 3), quality=round(Q, 3), oee=round(A * P * Q, 3),
                 losses={k: cnt.get(k, 0) for k in LOSSES}, events={k: v for k, v in ev.items()}, protective_stops=ev.get("collision", 0),
                 cycle_s=round(float(np.mean(oks)), 2) if oks else None, sorted_by_material=dict(mats), robot=robot, on_belt=on_belt,
                 vision_ms=round(float(np.mean(ln.vision.ms[-60:])), 1) if getattr(ln, "vision", None) is not None and ln.vision.ms else None)
        self.rows.append(r); return r


def main():
    import warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--minutes", type=float, default=10.0); p.add_argument("--speed", type=float, default=0.3); p.add_argument("--rate", type=float, default=12.0)
    p.add_argument("--clutter", type=float, default=0.25); p.add_argument("--perception", default="vision", choices=("vision", "privileged"))
    p.add_argument("--carry", default="student", choices=("student", "learned", "f2")); p.add_argument("--port", type=int, default=8804); p.add_argument("--pace", type=float, default=1.0, help="1 = real time, 0 = as fast as it goes")
    p.add_argument("--name", default="f5-shift"); p.add_argument("--seed", type=int, default=0); p.add_argument("--video-fps", type=float, default=12.5); p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    import mujoco, imageio.v2 as iio2
    from PIL import Image, ImageDraw
    from line import Line
    from line_learned import LearnedLine
    from vision import overlay
    run = os.path.join(ROOT, "runs", a.name); os.makedirs(run, exist_ok=True); json.dump(vars(a) | dict(date=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(run, "config.json"), "w"), indent=1)
    kw = dict(speed=a.speed, rate_per_min=a.rate, scheduler="lookahead", seed=a.seed, perception=a.perception, clutter=a.clutter)
    L = os.path.join(FACTORY, "learned")
    if a.carry == "f2": ln = Line(**kw)
    elif a.carry == "student": ln = LearnedLine(policy=os.path.join(L, "student.json"), privileged_carry=False, **kw)
    else: ln = LearnedLine(policy=os.path.join(L, "policy.json"), **kw)
    st = None
    if a.port:
        import live_stream
        live_stream.PAGE = f"<!doctype html><meta charset=utf-8><title>Sorting cell: the shift</title><body style='margin:0;background:#111'><img src='/stream.mjpg' style='width:100%;max-width:1280px;display:block;margin:auto'>"
        st = live_stream.Stream(); st.serve(a.port); print(f"shift stream at http://127.0.0.1:{a.port}/", flush=True)
    W, H = 960, 540; r = mujoco.Renderer(ln.c.model, H, W); kpi = KPI(ln); seconds = a.minutes * 60.0
    vid = None if a.no_video else iio2.get_writer(os.path.join(run, "shift.mp4"), fps=a.video_fps, codec="libx264", macro_block_size=4)
    fk = open(os.path.join(run, "kpis.jsonl"), "w"); dt = ln.c.model.opt.timestep; every = max(1, round(1.0 / (dt * 25))); vid_every = max(1, round(25 / a.video_fps))
    clock = dict(wall=time.perf_counter(), sim=0.0, next_kpi=1.0, k=0, last=None); t_start = time.perf_counter()

    def frame(Ln):
        t = Ln.c.data.time; kpi.tick(t); clock["k"] += 1
        if t >= clock["next_kpi"]:
            row = kpi.snapshot(t); clock["last"] = row; clock["next_kpi"] += 1.0
            fk.write(json.dumps({k: v for k, v in row.items() if k != "on_belt"}) + "\n"); fk.flush()
            tmp = os.path.join(run, "live.json.tmp"); json.dump(dict(row, minutes=a.minutes, name=a.name, stream=f"http://127.0.0.1:{a.port}/" if a.port else None,
                                                                    config=dict(speed=a.speed, rate=a.rate, clutter=a.clutter, perception=a.perception, carry=a.carry), running=True), open(tmp, "w")); os.replace(tmp, os.path.join(run, "live.json"))
        show = st is not None; rec = vid is not None and clock["k"] % vid_every == 0
        if show or rec:
            if Ln.vision is not None: Ln.vision.keep_view = True
            r.update_scene(Ln.c.data, camera="pick"); im = Image.fromarray(r.render()); d = ImageDraw.Draw(im)
            if Ln.vision is not None and Ln.vision.view is not None: ins = overlay(Ln.vision.view, scale=1).resize((318, 180)); im.paste(ins, (W - 326, H - 188))
            k = clock["last"] or {}
            d.rectangle((0, 0, W, 24), fill=(0, 0, 0))
            d.text((8, 5), f"shift t {t / 60:4.1f}/{a.minutes:g} min  belt {a.speed:.2f} m/s  {a.rate:g}/min  clutter {a.clutter:.0%}  {a.perception}+{a.carry}  |  sorted {k.get('sorted', 0)}/{k.get('arrived', 0)}  {k.get('sorted_per_min', 0):.1f}/min  OEE {k.get('oee', 0):.2f}", fill=(255, 255, 255))
            arr = np.asarray(im)
            if show: st.push(arr); st.state = k
            if rec: vid.append_data(arr)
        if a.pace > 0:
            lag = (t - clock["sim"]) / a.pace - (time.perf_counter() - clock["wall"])
            if lag > 0: time.sleep(lag)
            elif lag < -1.0: clock["wall"] = time.perf_counter(); clock["sim"] = t
    out = ln.run(seconds=seconds, on_frame=frame, frame_every=every)
    final = kpi.snapshot(seconds); fk.close()
    if vid is not None: vid.close(); os.system(f"{ROOT}/bin/rl-sheet {os.path.join(run, 'shift.mp4')} {os.path.join(run, 'shift-sheet.png')} >/dev/null 2>&1")
    with open(os.path.join(run, "items.jsonl"), "w") as f:
        for i in out["items"]: f.write(json.dumps(i, default=float) + "\n")
    json.dump(dict(final, minutes=a.minutes, name=a.name, running=False, config=dict(speed=a.speed, rate=a.rate, clutter=a.clutter, perception=a.perception, carry=a.carry)), open(os.path.join(run, "live.json"), "w"))
    S = dict(date=time.strftime("%Y-%m-%d %H:%M"), config=vars(a), sim_s=seconds, wall_s=round(time.perf_counter() - t_start, 1), kpi_rows=len(kpi.rows),
             final={k: v for k, v in final.items() if k != "on_belt"}, summary=out["summary"],
             clutter_items=dict(Counter(i["outcome"] for i in out["items"] if i.get("clutter"))), per_item={n: dict(Counter(i["outcome"] for i in out["items"] if i["name"] == n)) for n in sorted({i["name"] for i in out["items"]})})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(S, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
    print(json.dumps(S["final"]), flush=True)


if __name__ == "__main__":
    main()
