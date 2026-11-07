"""live_stream.py — the sorting cell running in real time, streamed to the browser as MJPEG (2026-11-02; the lab's
live viewer, lab/ pages embed it).

    envs/mj/bin/python -I factory/cell/live_stream.py [--port 8801] [--speed 0.3] [--cam cell] [--fps 20]
    http://127.0.0.1:8801/            a page with the stream and the knobs
    http://127.0.0.1:8801/stream.mjpg the stream (multipart/x-mixed-replace)
    http://127.0.0.1:8801/state       JSON: sim time, belt speed, camera, items on the belt
    http://127.0.0.1:8801/set?speed=0.4&cam=overhead&running=0|1&drop=1&reset=1&hulls=0|1
The simulation and the renderer run on the main thread (MuJoCo's GL context lives there); the HTTP server runs on a
daemon thread and only hands out the latest JPEG. Writes factory/cell/results/live-stream.json with the URL and pid.
"""
import argparse, http.server, io, json, os, sys, threading, time, urllib.parse
import numpy as np
from PIL import Image
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cell_mujoco import CellMJ  # noqa: E402
from belt_video import ORDER  # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8><title>Sorting cell live</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}img{width:100%;max-width:1280px;display:block;margin:auto}
.k{text-align:center;padding:8px}button,select{font:inherit;margin:2px;padding:4px 10px}</style>
<img src="/stream.mjpg"><div class=k>belt <button onclick="s('speed',-1)">slower</button><b id=v>?</b> m/s<button onclick="s('speed',1)">faster</button>
<button onclick="q('running=toggle')">stop/start</button><button onclick="q('drop=1')">drop item</button><button onclick="q('hulls=toggle')">collision shapes</button>
<button onclick="q('reset=1')">reset</button> camera <select onchange="q('cam='+this.value)"><option>cell</option><option>overhead</option><option>belt</option><option>end</option></select></div>
<script>let st={};async function q(a){st=await (await fetch('/set?'+a)).json();v.textContent=st.speed.toFixed(2)}
function s(k,d){q('speed='+Math.max(0,Math.min(0.8,(st.speed||0.3)+0.05*d)).toFixed(2))}q('')</script>"""


class Stream:
    def __init__(self):
        self.cv = threading.Condition(); self.seq = 0; self.jpeg = None; self.cmds = []; self.state = {}

    def push(self, rgb):
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=75)
        with self.cv: self.jpeg = buf.getvalue(); self.seq += 1; self.cv.notify_all()

    def serve(self, port):
        stream = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _json(self, obj):
                b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                if u.path == "/stream.mjpg":
                    self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers(); last = -1
                    try:
                        while True:
                            with stream.cv:
                                stream.cv.wait_for(lambda: stream.seq != last and stream.jpeg is not None, timeout=1.0)
                                if stream.seq == last: continue
                                last, data = stream.seq, stream.jpeg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data) + data + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif u.path == "/state": self._json(stream.state)
                elif u.path == "/set":
                    q = dict(urllib.parse.parse_qsl(u.query))
                    with stream.cv: stream.cmds.append(q)
                    time.sleep(0.08); self._json(stream.state)
                elif u.path in ("/", "/index.html"):
                    b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                else: self.send_error(404)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H); srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv


def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8801); p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--cam", default="cell"); p.add_argument("--fps", type=int, default=20); p.add_argument("--gap", type=float, default=1.2)
    p.add_argument("--width", type=int, default=960); p.add_argument("--height", type=int, default=540); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    c = CellMJ(); rng = np.random.default_rng(a.seed); st = Stream(); st.serve(a.port)
    url = f"http://127.0.0.1:{a.port}/"
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "live-stream.json"), "w") as f: json.dump(dict(url=url, pid=os.getpid(), port=a.port), f)
    print(f"live stream at {url}", flush=True)
    knobs = dict(speed=a.speed, running=True, cam=a.cam, hulls=False)
    opt = mujoco.MjvOption()
    renderer = mujoco.Renderer(c.model, a.height, a.width)
    live, k_item, next_spawn = [], 0, 0.5
    wall0, sim0, next_frame = time.perf_counter(), c.data.time, 0.0
    while True:
        with st.cv: cmds, st.cmds = st.cmds, []
        for q in cmds:
            if "speed" in q:
                try: knobs["speed"] = float(np.clip(float(q["speed"]), 0.0, 0.8))
                except ValueError: pass
            if q.get("running") == "toggle": knobs["running"] = not knobs["running"]
            elif q.get("running") in ("0", "1"): knobs["running"] = q["running"] == "1"
            if q.get("hulls") == "toggle": knobs["hulls"] = not knobs["hulls"]
            elif q.get("hulls") in ("0", "1"): knobs["hulls"] = q["hulls"] == "1"
            if q.get("cam") in ("cell", "overhead", "belt", "end"): knobs["cam"] = q["cam"]
            if q.get("reset") == "1": c.reset(); live.clear(); k_item = 0; next_spawn = c.data.time + 0.5
            if q.get("drop") == "1": next_spawn = c.data.time
        speed = knobs["speed"] if knobs["running"] else 0.0
        c.belt(speed)
        if c.data.time >= next_spawn and (knobs["running"] or next_spawn == c.data.time):
            name = ORDER[k_item % len(ORDER)]; k_item += 1
            if name in live: live.remove(name)
            c.spawn(name, rng=rng); live.append(name)
            next_spawn = c.data.time + a.gap * 0.3 / max(speed, 0.05)
        for n in list(live):
            if c.item_state(n)["pos"][0] > c.lay.belt.x_max + 0.05: c.park(n, ORDER.index(n)); live.remove(n)
        target = sim0 + (time.perf_counter() - wall0); n = 0
        while c.data.time < target and n < 50: c.step(); n += 1
        if c.data.time >= next_frame:
            opt.geomgroup[3] = 1 if knobs["hulls"] else 0
            renderer.update_scene(c.data, camera=knobs["cam"], scene_option=opt); st.push(renderer.render())
            next_frame = c.data.time + 1.0 / a.fps
            st.state = dict(t=round(c.data.time, 2), speed=knobs["speed"], running=knobs["running"], cam=knobs["cam"], hulls=knobs["hulls"],
                            belt_mps=round(c.belt_velocity(), 3), on_belt=list(live))
        if c.data.time - target > 0.002: time.sleep(0.002)


if __name__ == "__main__":
    main()
