"""live_pick.py — watch the classical pipeline pick, live in the browser (2026-11-02, F1). An
endless loop of L0 trials (a random item at a random pose on the stopped belt, picked and dropped in its chute),
simulated and rendered in real time (or faster), streamed as MJPEG with a running tally burned into the frame.

    envs/mj/bin/python -I factory/classical/live_pick.py [--port 8802] [--cam pick] [--rate 1.0] [--item any]
    http://127.0.0.1:8802/            the stream and its knobs; /state is JSON; /set?cam=..&rate=..&item=..&running=0|1
"""
import argparse, json, math, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cell"))
from cell_mujoco import CellMJ  # noqa: E402
from live_stream import Stream  # noqa: E402
import pick  # noqa: E402
from eval_f1 import ITEMS  # noqa: E402

CAMS = ("pick", "cell", "overhead", "belt", "end")
PAGE = """<!doctype html><meta charset=utf-8><title>Sorting cell: live picks</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}img{width:100%;max-width:1280px;display:block;margin:auto}
.k{text-align:center;padding:8px}button,select{font:inherit;margin:2px;padding:4px 10px}</style>
<img src="/stream.mjpg"><div class=k>camera <select onchange="q('cam='+this.value)">""" + "".join(f"<option>{c}</option>" for c in CAMS) + """</select>
speed <select onchange="q('rate='+this.value)"><option value=1>real time</option><option value=2>2x</option><option value=4>4x</option><option value=0.5>half</option></select>
item <select onchange="q('item='+this.value)"><option>any</option>""" + "".join(f"<option>{i}</option>" for i in ITEMS) + """</select>
<button onclick="q('running=toggle')">pause/run</button> <span id=s></span></div>
<script>async function q(a){const st=await (await fetch('/set?'+a)).json();s.textContent=st.tally||''}setInterval(()=>q(''),2000)</script>"""


def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8802); p.add_argument("--cam", default="pick", choices=CAMS)
    p.add_argument("--rate", type=float, default=1.0); p.add_argument("--item", default="any"); p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--width", type=int, default=960); p.add_argument("--height", type=int, default=540)
    a = p.parse_args()
    import live_stream; live_stream.PAGE = PAGE   # the shared server serves this page
    cell = CellMJ(); planner = pick.Planner(cell); st = Stream(); st.serve(a.port); rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "live-pick.json"), "w") as f: json.dump(dict(url=f"http://127.0.0.1:{a.port}/", pid=os.getpid()), f)
    print(f"live picks at http://127.0.0.1:{a.port}/", flush=True)
    knobs = dict(cam=a.cam, rate=a.rate, item=a.item, running=True); tally = dict(n=0, ok=0, last="")
    renderer = mujoco.Renderer(cell.model, a.height, a.width); every = max(1, round(1.0 / (cell.model.opt.timestep * a.fps)))
    clock = dict(wall=time.perf_counter(), sim=0.0); caption = [""]

    def apply_cmds():
        with st.cv: cmds, st.cmds = st.cmds, []
        for q in cmds:
            if q.get("cam") in CAMS: knobs["cam"] = q["cam"]
            if q.get("item") in ITEMS + ("any",): knobs["item"] = q["item"]
            if "rate" in q:
                try: knobs["rate"] = float(np.clip(float(q["rate"]), 0.25, 8.0))
                except ValueError: pass
            if q.get("running") == "toggle": knobs["running"] = not knobs["running"]

    def frame(c):
        apply_cmds()
        while not knobs["running"]: time.sleep(0.1); apply_cmds(); clock["wall"] = time.perf_counter(); clock["sim"] = c.data.time
        renderer.update_scene(c.data, camera=knobs["cam"]); im = Image.fromarray(renderer.render()); d = ImageDraw.Draw(im)
        d.rectangle((0, 0, a.width, 24), fill=(0, 0, 0)); d.text((8, 5), caption[0], fill=(255, 255, 255)); st.push(np.asarray(im))
        tl = f"{tally['ok']}/{tally['n']} picks sorted" + (f"  (last: {tally['last']})" if tally["last"] else "")
        st.state = dict(t=round(c.data.time, 2), cam=knobs["cam"], rate=knobs["rate"], item=knobs["item"], running=knobs["running"], tally=tl, **tally)
        lag = (c.data.time - clock["sim"]) / knobs["rate"] - (time.perf_counter() - clock["wall"])
        if lag > 0: time.sleep(lag)
        elif lag < -0.5: clock["wall"] = time.perf_counter(); clock["sim"] = c.data.time   # fell behind: resync, don't sprint

    while True:
        name = knobs["item"] if knobs["item"] != "any" else ITEMS[int(rng.integers(len(ITEMS)))]
        x, y, yaw = float(rng.uniform(-0.4, 0.4)), float(cell.lay.belt.y + rng.uniform(-0.08, 0.08)), float(rng.uniform(-math.pi, math.pi))
        caption[0] = f"pick #{tally['n'] + 1}: {name} ({cell.items[name]['rec']['material']})  x {x:+.2f} m  yaw {math.degrees(yaw):+.0f} deg   |   {tally['ok']}/{tally['n']} sorted so far"
        clock["wall"] = time.perf_counter(); clock["sim"] = 0.0
        r = pick.trial(cell, planner, name, x, y, yaw, on_frame=frame, frame_every=every)
        tally["n"] += 1; tally["ok"] += r["outcome"] == "success"; tally["last"] = f"{name}: {r['outcome']} via {r.get('tool')}, {r.get('cycle_s')} s"
        caption[0] = f"{name}: {r['outcome']}" + (f" in {r['cycle_s']} s with the {r['tool']}" if r.get("cycle_s") else "") + f"   |   {tally['ok']}/{tally['n']} sorted"
        for _ in range(int(0.8 * a.fps)): cell.step(every); frame(cell)


if __name__ == "__main__":
    main()
