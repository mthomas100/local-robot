"""desk_live.py — watch the SO-101 desk cell pick, live in the browser (2026-11-03, F6). An endless
loop of L0 trials with the real-device kit on (after the cell calibrates its camera, shown first), alternating the
classical pipeline and the behaviour-cloned ONNX policy (or one of them), rendered at the 30 Hz host rate and paced
to real time, streamed as MJPEG with the camera's own view and a running tally.

    envs/mj/bin/python -I factory/desk/desk_live.py [--port 8805] [--policy alternate|classical|onnx] [--rate 1]
    http://127.0.0.1:8805/     the stream and its knobs; /state is JSON; /set?cam=..&policy=..&rate=..&running=toggle
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "factory", "cell"))
import live_stream  # noqa: E402
from live_stream import Stream  # noqa: E402
from desk_sim import DeskSim  # noqa: E402
from servo import DeviceKit  # noqa: E402
import camera as C  # noqa: E402
import desk_pick as P  # noqa: E402
import desk_items as I  # noqa: E402

CAMS = ("desk", "front", "side", "top")
POLICIES = ("alternate", "classical", "onnx")
PAGE = """<!doctype html><meta charset=utf-8><title>Desk cell: live picks</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}img{width:100%;max-width:1272px;display:block;margin:auto}
.k{text-align:center;padding:8px}button,select{font:inherit;margin:2px;padding:4px 10px}</style>
<img src="/stream.mjpg"><div class=k>camera <select onchange="q('cam='+this.value)">""" + "".join(f"<option>{c}</option>" for c in CAMS) + """</select>
pick by <select onchange="q('policy='+this.value)">""" + "".join(f"<option>{c}</option>" for c in POLICIES) + """</select>
speed <select onchange="q('rate='+this.value)"><option value=1>real time</option><option value=2>2x</option><option value=4>4x</option><option value=0.5>half</option></select>
<button onclick="q('running=toggle')">pause/run</button> <span id=s></span></div>
<script>async function q(a){const st=await (await fetch('/set?'+a)).json();s.textContent=st.tally||''}setInterval(()=>q(''),2000)</script>"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8805); ap.add_argument("--policy", default="alternate", choices=POLICIES)
    ap.add_argument("--rate", type=float, default=1.0); ap.add_argument("--seed", type=int, default=31_000); ap.add_argument("--cam", default="desk", choices=CAMS)
    a = ap.parse_args()
    live_stream.PAGE = PAGE
    sk = DeviceKit.real_kit(); sim = DeskSim(kit=sk); cam = C.Camera(sim, C.CameraKit.real_kit(seed=a.seed), rng=np.random.default_rng(a.seed))
    pol = None
    try:
        import bc; pol = bc.OnnxPolicy()
    except Exception as e: print("no ONNX policy:", e, flush=True)
    st = Stream(); st.serve(a.port)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "live.json"), "w") as f: json.dump(dict(url=f"http://127.0.0.1:{a.port}/", pid=os.getpid()), f)
    print(f"desk live at http://127.0.0.1:{a.port}/", flush=True)
    knobs = dict(cam=a.cam, policy=a.policy, rate=a.rate, running=True); tally = {"classical": [0, 0], "onnx": [0, 0]}; caption = [""]
    clock = dict(wall=time.perf_counter(), sim=0.0)

    def apply_cmds():
        with st.cv: cmds, st.cmds = st.cmds, []
        for q in cmds:
            if q.get("cam") in CAMS: knobs["cam"] = q["cam"]
            if q.get("policy") in POLICIES: knobs["policy"] = q["policy"]
            if "rate" in q:
                try: knobs["rate"] = float(np.clip(float(q["rate"]), 0.25, 8.0))
                except ValueError: pass
            if q.get("running") == "toggle": knobs["running"] = not knobs["running"]

    def frame(sim_, seg=None, goal=None):
        apply_cmds()
        while not knobs["running"]: time.sleep(0.1); apply_cmds(); clock["wall"] = time.perf_counter(); clock["sim"] = sim.t
        big = Image.fromarray(sim.render(knobs["cam"], 848, 480)); view = cam.capture()["rgb"]
        im = Image.new("RGB", (848 + 424, 480)); im.paste(big, (0, 0)); im.paste(Image.fromarray(view), (848, 0))
        d = ImageDraw.Draw(im); d.rectangle((0, 0, 848, 22), fill=(0, 0, 0)); d.text((6, 5), caption[0] + (f"   [{seg.name}]" if seg is not None else ""), fill=(255, 255, 255))
        d.rectangle((848, 0, 1272, 22), fill=(0, 0, 0)); d.text((854, 5), "overhead camera (noise, latency; calibrated)", fill=(255, 255, 255))
        tl = "  ".join(f"{k}: {v[1]}/{v[0]} sorted" for k, v in tally.items())
        d.rectangle((848, 240, 1272, 480), fill=(20, 20, 22)); y = 250
        for line in ("SO-101 desk cell, F6", "real-device kit on:", " servos: PD loop in the servo, 12-bit encoder,", "  deadband, 4 ms bus delay, back-EMF, horn play",
                     " camera: depth z^2 noise, holes, RGB noise,", "  45 +/- 10 ms latency, planted hand-eye error", "", tl):
            d.text((860, y), line, fill=(220, 220, 220)); y += 18
        st.push(np.asarray(im)); st.state = dict(t=round(sim.t, 2), tally=tl, **knobs)
        lag = (sim.t - clock["sim"]) / knobs["rate"] - (time.perf_counter() - clock["wall"])
        if lag > 0: time.sleep(lag)
        elif lag < -0.5: clock["wall"] = time.perf_counter(); clock["sim"] = sim.t

    caption[0] = "calibrating the camera: the arm shows the magenta marker to it at 96 poses"
    clock["wall"] = time.perf_counter(); clock["sim"] = sim.t
    steps_per_tick = int(round(1 / 30 / sim.model.opt.timestep)); orig_step = sim.step; count = [0]

    def step_and_show(n=1):   # during calibration, show every host tick
        for _ in range(n):
            orig_step(1); count[0] += 1
            if count[0] % steps_per_tick == 0: frame(sim)
    sim.step = step_and_show; rep = C.calibrate(sim, cam, sk.backlash_rad); sim.step = orig_step
    caption[0] = f"calibrated: hand-eye error {rep['before_mm']} mm / {rep['before_deg']} deg -> {rep['after_mm']} mm / {rep['after_deg']} deg"
    for _ in range(60): frame(sim)
    k = 0; items = I.admitted()
    while True:
        name = items[k % len(items)]; seed = a.seed + k
        who = knobs["policy"] if knobs["policy"] != "alternate" else ("classical" if k % 2 == 0 or pol is None else "onnx")
        if who == "onnx" and pol is None: who = "classical"
        caption[0] = f"pick #{k + 1} by the {'classical pipeline' if who == 'classical' else 'ONNX policy (30 Hz)'}: {name}"
        clock["wall"] = time.perf_counter(); clock["sim"] = sim.t
        if who == "classical": r = P.trial(sim, name, seed, cam=cam, record=frame)
        else:
            import bc; r = bc.policy_trial(sim, pol, name, seed, cam, record=frame)
        tally[who][0] += 1; tally[who][1] += r["outcome"] == "ok"
        caption[0] = f"{who}: {name} -> {r['outcome']}" + (f" (bin {r.get('bin')})" if r.get("bin") else "")
        for _ in range(25): frame(sim)
        k += 1


if __name__ == "__main__":
    main()
