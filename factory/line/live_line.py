"""live_line.py — watch the line run, live in the browser (2026-11-02, F2): the moving belt, items
arriving, the tracker, the scheduler and the robot picking on the move, simulated and rendered in real time (or
faster) and streamed as MJPEG with the running score burned into the frame. Changing the belt speed, the arrival
rate, the scheduler or the carry (F2's classical carry, F3's hand-written prior or F3's learned policy) restarts the line.

    envs/mj/bin/python -I factory/line/live_line.py [--port 8803] [--speed 0.3] [--rate 8] [--scheduler lookahead] [--cam pick] [--carry f2|prior|learned]
    http://127.0.0.1:8803/   (/state JSON; /set?speed=..&rate=..&scheduler=..&carry=..&cam=..&sim=..&running=toggle)
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
from line import Line  # noqa: E402
import live_stream  # noqa: E402

CAMS = ("pick", "cell", "overhead", "belt", "end"); SCHEDS = ("lookahead", "edf", "greedy"); CARRIES = ("f2", "learned", "prior", "student"); PERCEPTIONS = ("privileged", "vision")
PAGE = """<!doctype html><meta charset=utf-8><title>Sorting cell: the line, live</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}img{width:100%;max-width:1280px;display:block;margin:auto}
.k{text-align:center;padding:8px}button,select{font:inherit;margin:2px;padding:4px 10px}</style>
<img src="/stream.mjpg"><div class=k>belt <select onchange="q('speed='+this.value)"><option>0.3</option><option>0.15</option><option>0.4</option><option>0.5</option></select> m/s
arrivals <select onchange="q('rate='+this.value)"><option>8</option><option>4</option><option>12</option><option>16</option></select>/min
scheduler <select onchange="q('scheduler='+this.value)">""" + "".join(f"<option>{s}</option>" for s in SCHEDS) + """</select>
carry <select onchange="q('carry='+this.value)">""" + "".join(f"<option>{c}</option>" for c in CARRIES) + """</select>
perception <select onchange="q('perception='+this.value)">""" + "".join(f"<option>{c}</option>" for c in PERCEPTIONS) + """</select>
camera <select onchange="q('cam='+this.value)">""" + "".join(f"<option>{c}</option>" for c in CAMS) + """</select>
speed <select onchange="q('sim='+this.value)"><option value=1>real time</option><option value=2>2x</option><option value=4>4x</option></select>
<button onclick="q('running=toggle')">pause/run</button> <span id=s></span></div>
<script>async function q(a){const st=await (await fetch('/set?'+a)).json();s.textContent=st.score||''}setInterval(()=>q(''),2000)</script>"""


class Restart(Exception): pass


def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8803); p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--rate", type=float, default=8.0); p.add_argument("--scheduler", default="lookahead", choices=SCHEDS); p.add_argument("--cam", default="pick", choices=CAMS); p.add_argument("--carry", default="f2", choices=CARRIES); p.add_argument("--perception", default="privileged", choices=PERCEPTIONS)
    p.add_argument("--v2", default="", help="F7: comma-separated line.V2_FLAGS, e.g. fast,early,deadline (empty: v1)"); p.add_argument("--sim", type=float, default=1.0); p.add_argument("--fps", type=int, default=25); p.add_argument("--width", type=int, default=960); p.add_argument("--height", type=int, default=540)
    a = p.parse_args()
    live_stream.PAGE = PAGE; st = live_stream.Stream(); st.serve(a.port)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "live-line.json"), "w") as f: json.dump(dict(url=f"http://127.0.0.1:{a.port}/", pid=os.getpid()), f)
    print(f"live line at http://127.0.0.1:{a.port}/", flush=True)
    knobs = dict(speed=a.speed, rate=a.rate, scheduler=a.scheduler, carry=a.carry, perception=a.perception, cam=a.cam, sim=a.sim, running=True); seed = 0; cell = None; renderer = None; vision_det = None

    def apply():
        restart = False
        with st.cv: cmds, st.cmds = st.cmds, []
        for q in cmds:
            for k, lo, hi in (("speed", 0.05, 0.6), ("rate", 1, 30), ("sim", 0.25, 8)):
                if k in q:
                    try:
                        v = float(np.clip(float(q[k]), lo, hi))
                        if k != "sim" and v != knobs[k]: restart = True
                        knobs[k] = v
                    except ValueError: pass
            if q.get("scheduler") in SCHEDS and q["scheduler"] != knobs["scheduler"]: knobs["scheduler"] = q["scheduler"]; restart = True
            if q.get("carry") in CARRIES and q["carry"] != knobs["carry"]: knobs["carry"] = q["carry"]; restart = True
            if q.get("perception") in PERCEPTIONS and q["perception"] != knobs["perception"]: knobs["perception"] = q["perception"]; restart = True
            if q.get("cam") in CAMS: knobs["cam"] = q["cam"]
            if q.get("running") == "toggle": knobs["running"] = not knobs["running"]
        return restart

    while True:
        kw = dict(speed=knobs["speed"], rate_per_min=knobs["rate"], scheduler=knobs["scheduler"], seed=seed, cell=cell, perception=knobs["perception"], v2=tuple(f for f in a.v2.split(",") if f))
        if knobs["carry"] == "f2": ln = Line(**kw)
        else:
            sys.path.insert(0, os.path.join(os.path.dirname(HERE), "learned")); from line_learned import LearnedLine
            L = os.path.join(os.path.dirname(HERE), "learned")
            if knobs["carry"] == "student": ln = LearnedLine(policy=os.path.join(L, "student.json"), privileged_carry=False, **kw)
            else: ln = LearnedLine(policy="prior" if knobs["carry"] == "prior" else os.path.join(L, "policy.json"), **kw)
        if vision_det is not None and knobs["perception"] == "vision": ln.vision = vision_det   # one camera renderer per process
        cell = ln.c; seed += 1
        if renderer is None: renderer = mujoco.Renderer(cell.model, a.height, a.width)
        every = max(1, round(1.0 / (cell.model.opt.timestep * a.fps))); clock = dict(wall=time.perf_counter(), sim=0.0)

        def frame(L):
            if apply(): raise Restart()
            while not knobs["running"]:
                time.sleep(0.1)
                if apply(): raise Restart()
                clock["wall"] = time.perf_counter(); clock["sim"] = L.c.data.time
            done = L.log_items; ok = sum(i["outcome"] == "sorted" for i in done); tmin = max(L.c.data.time, 1e-6) / 60
            robot = L.cur.name if (L.cur and not L.recovering) else ("recovering" if L.cur else "waiting")
            score = f"sorted {ok}/{len(done)} ({ok / tmin:.1f}/min)   robot: {robot}"
            renderer.update_scene(L.c.data, camera=knobs["cam"]); im = Image.fromarray(renderer.render()); d = ImageDraw.Draw(im)
            if L.perception == "vision" and L.vision is not None:   # inset: what the camera sees and what perception made of it
                L.vision.keep_view = True
                if L.vision.view is not None:
                    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vision")); from vision import overlay
                    ins = overlay(L.vision.view, scale=1); im.paste(ins, (a.width - ins.width - 8, a.height - ins.height - 8))
            d.rectangle((0, 0, a.width, 24), fill=(0, 0, 0))
            d.text((8, 5), f"t {L.c.data.time:6.1f} s   belt {knobs['speed']:.2f} m/s   {knobs['rate']:g}/min   {knobs['scheduler']}   carry: {knobs['carry']}   {knobs['perception']}   |   {score}", fill=(255, 255, 255))
            st.push(np.asarray(im)); st.state = dict(t=round(L.c.data.time, 1), score=score, **{k: knobs[k] for k in ("speed", "rate", "scheduler", "carry", "cam", "sim", "running")})
            lag = (L.c.data.time - clock["sim"]) / knobs["sim"] - (time.perf_counter() - clock["wall"])
            if lag > 0: time.sleep(lag)
            elif lag < -0.5: clock["wall"] = time.perf_counter(); clock["sim"] = L.c.data.time

        try: ln.run(seconds=1e9, on_frame=frame, frame_every=every)
        except Restart:
            if ln.vision is not None: vision_det = ln.vision
            continue


if __name__ == "__main__":
    main()
