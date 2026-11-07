"""lab/server.py — the sorting-cell lab: one small standard-library web server at http://127.0.0.1:8800/
(2026-11-02, docs/sorting-cell-spec.md).

Serves the static pages in lab/static/, the phase pages' content from lab/phases/<id>.json, media and result files
from whitelisted directories, and runs whitelisted jobs (the same scripts the agents run, under bin/memguard) with
their logs streamed to the page as server-sent events. No build step, no dependencies beyond the standard library.

    python3 lab/server.py [--port 8800]        (or: bin/rl lab)
"""
import argparse, http.server, json, os, re, signal, subprocess, sys, threading, time, urllib.parse

LAB = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(LAB)
STATE = os.path.join(LAB, "state", "jobs"); os.makedirs(STATE, exist_ok=True)
MEDIA_ROOTS = ("docs/media", "factory/cell/results", "factory/classical/results", "factory/line/results", "runs", "lab/static")
FILE_ROOTS = ("factory", "docs", "runs", "lab/state")
MJ = os.path.join(ROOT, "envs/mj/bin/python"); GEN = os.path.join(ROOT, "envs/genesis/bin/python"); GUARD = os.path.join(ROOT, "bin/memguard")
ITEMS = ("master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "tuna_fish_can", "pudding_box",
         "gelatin_box", "potted_meat_can", "bleach_cleanser", "windex_bottle")


# --- knob validation: every value a page sends is checked before it reaches a command line --------------------------
def _f(v, lo, hi): x = float(v); assert lo <= x <= hi, f"{x} outside [{lo}, {hi}]"; return f"{x:g}"
def _i(v, lo, hi): x = int(v); assert lo <= x <= hi, f"{x} outside [{lo}, {hi}]"; return str(x)
def _choice(v, opts): assert v in opts, f"{v!r} not one of {opts}"; return v
def _floats(v, lo, hi): return ",".join(_f(x, lo, hi) for x in str(v).split(",") if x.strip())
def _items(v):
    if v in ("", "all"): return "all"
    return ",".join(_choice(x.strip(), ITEMS) for x in v.split(","))


# job id -> (description, function(knobs) -> argv). Add a phase's jobs here when the phase lands.
JOBS = {
    "f0-belt-test": ("F0 belt test: items ride the belt at each speed (stability, slip, drift, penetration)",
                     lambda k: [GUARD, "32", MJ if k.get("engine", "mujoco") == "mujoco" else GEN, "-I", "factory/cell/belt_test.py",
                                "--engine", _choice(k.get("engine", "mujoco"), ("mujoco", "genesis")), "--speeds", _floats(k.get("speeds", "0.15,0.3,0.5"), 0.05, 0.8),
                                "--yaws", _i(k.get("yaws", 2), 1, 8), "--items", _items(k.get("items", "all")),
                                "--out", f"factory/cell/results/belt-test-{k.get('engine', 'mujoco')}-lab.json"]),
    "f0-chute-test": ("F0 chute test: drop an item into each chute",
                      lambda k: [GUARD, "16", MJ if k.get("engine", "mujoco") == "mujoco" else GEN, "-I", "factory/cell/chute_test.py",
                                 "--engine", _choice(k.get("engine", "mujoco"), ("mujoco", "genesis")), "--item", _choice(k.get("item", "tomato_soup_can"), ITEMS)]),
    "f0-reach-map": ("F0 reach map: sample the UR5e joint space and draw the reachable set over the cell",
                     lambda k: [GUARD, "16", MJ, "-I", "factory/cell/reach_map.py", "--samples", _i(k.get("samples", 100000), 10000, 2000000),
                                "--out", "factory/cell/results/reach-map-lab.png"]),
    "f0-belt-video": ("F0 belt video: record items riding the belt",
                      lambda k: [GUARD, "16", MJ, "-I", "factory/cell/belt_video.py", "--speed", _f(k.get("speed", 0.3), 0.05, 0.8),
                                 "--seconds", _f(k.get("seconds", 8), 2, 30), "--cams", _choice(k.get("cam", "cell"), ("cell", "overhead", "belt", "end")),
                                 "--seed", _i(k.get("seed", 0), 0, 10**6)]),
    "f1-eval": ("F1 evaluation: L0 picks (belt stopped, one item) over all 11 items; success rate, failures, cycle times",
                lambda k: [GUARD, "40", MJ, "-I", "factory/classical/eval_f1.py", "--trials", _i(k.get("trials", 88), 11, 1000), "--workers", _i(k.get("workers", 6), 1, 12),
                           "--seed", _i(k.get("seed", 0), 0, 10**6), "--x-half", _f(k.get("x_half", 0.4), 0.0, 0.5), "--y-half", _f(k.get("y_half", 0.08), 0.0, 0.15),
                           "--video", _i(k.get("video", 4), 0, 12), "--name", "f1-eval-lab"]),
    "f1-plan-sweep": ("F1 planning sweep: plan many picks and check every path for collisions (no physics)",
                      lambda k: [GUARD, "40", MJ, "-I", "factory/classical/eval_f1.py", "--plans", _i(k.get("plans", 1000), 50, 5000), "--workers", "6"]),
    "f1-control": ("F1 servo step responses: rise time, overshoot, settling per joint",
                   lambda k: [GUARD, "16", MJ, "-I", "factory/classical/control.py"]),
    "live-pick": ("Live view: the classical pipeline picking items one after another (port 8802)",
                  lambda k: [GUARD, "16", MJ, "-I", "factory/classical/live_pick.py", "--port", "8802", "--rate", _f(k.get("rate", 1.0), 0.25, 8.0),
                             "--item", "any" if k.get("item", "any") == "any" else _choice(k.get("item"), ITEMS)]),
    "f2-eval": ("F2 evaluation: the moving line at belt speeds x arrival rates x schedulers; throughput and miss rate",
                lambda k: [GUARD, "40", MJ, "-I", "factory/line/eval_f2.py", "--seconds", _f(k.get("seconds", 120), 30, 900), "--seeds", _i(k.get("seeds", 1), 1, 5),
                           "--speeds", _floats(k.get("speeds", "0.15,0.3"), 0.05, 0.6), "--rates", _floats(k.get("rates", "4,8,12"), 1, 30),
                           "--schedulers", ",".join(_choice(x.strip(), ("greedy", "edf", "lookahead")) for x in str(k.get("schedulers", "greedy,lookahead")).split(",")),
                           "--workers", "6", "--name", "f2-eval-lab"]),
    "f2-video": ("F2 line video: record the line running (pick camera) with the score burned in",
                 lambda k: [GUARD, "16", MJ, "-I", "factory/line/eval_f2.py", "--video", _f(k.get("seconds", 60), 10, 300), "--speed", _f(k.get("speed", 0.3), 0.05, 0.6),
                            "--rate", _f(k.get("rate", 8), 1, 30), "--scheduler", _choice(k.get("scheduler", "lookahead"), ("greedy", "edf", "lookahead")), "--name", "f2-eval-lab"]),
    "f2-tracking": ("F2 tracking study: intercept prediction error vs belt speed and camera latency",
                    lambda k: [GUARD, "16", MJ, "-I", "factory/line/tracking_study.py"]),
    "live-line": ("Live view: the line running, belt moving, robot picking on the move (port 8803)",
                  lambda k: [GUARD, "16", MJ, "-I", "factory/line/live_line.py", "--port", "8803", "--speed", _f(k.get("speed", 0.3), 0.05, 0.6), "--rate", _f(k.get("rate", 8), 1, 30),
                             "--carry", _choice(k.get("carry", "f2"), ("f2", "prior", "learned", "student")),
                             "--perception", _choice(k.get("perception", "privileged"), ("privileged", "vision"))]),
    "f3-snapshots": ("F3 start states: run the F2 line and save the simulator state at the end of every lift",
                     lambda k: [GUARD, "40", MJ, "-I", "factory/learned/snapshots.py", "--runs", _i(k.get("runs", 24), 2, 96), "--workers", "12"]),
    "f3-train": ("F3 training: Augmented Random Search on the carry policy, on 14 CPU cores (about 10 minutes for 120 iterations)",
                 lambda k: [GUARD, "40", MJ, "-I", "factory/learned/train_f3.py", "--iters", _i(k.get("iters", 120), 5, 1000), "--workers", "14",
                            "--noise", _f(k.get("noise", 0.05), 0.005, 0.5), "--step", _f(k.get("step", 0.03), 0.001, 0.5), "--seed", _i(k.get("seed", 0), 0, 10**6),
                            "--name", "f3-ars-lab"]),
    "f3-check": ("F3 check: the shipped policy (or the prior) on every start state: success and seconds until the robot is free, per item",
                 lambda k: [GUARD, "40", MJ, "-I", "factory/learned/train_f3.py", "--eval",
                            {"learned": "factory/learned/policy.json", "prior": "factory/learned/prior.json"}[_choice(k.get("policy", "learned"), ("learned", "prior"))]]),
    "f3-eval": ("F3 acceptance: F2 vs the prior vs the learned carry on the line, held-out items, paired seeds",
                lambda k: [GUARD, "40", MJ, "-I", "factory/learned/eval_f3.py", "--seeds", _i(k.get("seeds", 3), 2, 10), "--seconds", _f(k.get("seconds", 120), 30, 600),
                           "--workers", "14", "--name", "f3-eval-lab"]),
    "f5-shift": ("F5 shift: the whole cell (vision, clutter, learned carry) for N minutes with live KPIs on the dashboard (stream on port 8804)",
                 lambda k: [GUARD, "24", MJ, "-I", "factory/ops/shift.py", "--minutes", _f(k.get("minutes", 10), 1, 60), "--speed", _f(k.get("speed", 0.3), 0.05, 0.6),
                            "--rate", _f(k.get("rate", 12), 1, 30), "--clutter", _f(k.get("clutter", 0.25), 0.0, 1.0), "--perception", _choice(k.get("perception", "vision"), ("vision", "privileged")),
                            "--carry", _choice(k.get("carry", "student"), ("student", "learned", "f2")), "--seed", _i(k.get("seed", 0), 0, 10**6), "--port", "8804", "--name", "f5-shift-lab"]),
    "f4-eval": ("F4 acceptance: L2 (privileged poses) vs L4 (vision only + student carry), paired seeds, sort rate",
                lambda k: [GUARD, "56", MJ, "-I", "factory/vision/eval_f4.py", "--seeds", _i(k.get("seeds", 3), 2, 16), "--seconds", _f(k.get("seconds", 120), 30, 600),
                           "--workers", "8", "--name", "f4-eval-lab"]),
    "f4-measure": ("F4 perception check: detector reports vs the simulator's ground truth (identity, position, yaw per item)",
                   lambda k: [GUARD, "16", MJ, "-I", "factory/vision/vision.py", "measure", "--scenes", _i(k.get("scenes", 30), 5, 200), "--speed", _f(k.get("speed", 0.3), 0.0, 0.6), "--out", "f4-vision-lab.json"]),
    "live-desk": ("Live view: the SO-101 desk cell picking with the real-device kit on, classical and ONNX policy alternating (port 8805)",
                  lambda k: [GUARD, "16", MJ, "-I", "factory/desk/desk_live.py", "--port", "8805", "--policy", _choice(k.get("policy", "alternate"), ("alternate", "classical", "onnx")),
                             "--rate", _f(k.get("rate", 1.0), 0.25, 8.0), "--seed", _i(k.get("seed", 31000), 0, 10**6)]),
    "f6-reach": ("F6 reach map: where the SO-101 can point its tool straight down",
                 lambda k: [GUARD, "16", MJ, "-I", "factory/desk/desk_reach.py"]),
    "f6-payload": ("F6 payload-and-jaw table: jaw, reach, holding torque and grip for every item, then 6 real-kit picks each (about 2 minutes)",
                   lambda k: [GUARD, "32", MJ, "-I", "factory/desk/payload.py"]),
    "f6-calib": ("F6 hand-eye calibration: five planted camera errors, the marker routine, recovered within 2 mm / 1 deg? (about 1 minute)",
                 lambda k: [GUARD, "32", MJ, "-I", "factory/desk/eval_f6.py", "calib"]),
    "f6-check": ("F6 quick check: the ONNX policy on N fresh L0 trials with the whole real-device kit on",
                 lambda k: [GUARD, "32", MJ, "-I", "factory/desk/bc.py", "check", "--trials", _i(k.get("trials", 20), 4, 200)]),
    "f6-sysid": ("F6 system identification: plant servo and plant parameters, record steps and chirps, fit, compare",
                 lambda k: [GUARD, "32", MJ, "-I", "factory/desk/sysid.py", "--seeds", _i(k.get("seed", 1), 0, 10**6)]),
    "live-fast": ("Live view: the line on the fast belt, v1 or v2 (the F7 changes), vision only, student carry (port 8803)",
                  lambda k: [GUARD, "16", MJ, "-I", "factory/line/live_line.py", "--port", "8803", "--speed", _f(k.get("speed", 0.5), 0.05, 0.6), "--rate", _f(k.get("rate", 12), 1, 30),
                             "--carry", "student", "--perception", "vision", "--v2", {"v2": "fast,early,deadline", "v1": ""}[_choice(k.get("cell", "v2"), ("v2", "v1"))]]),
    "f7-ablate": ("F7 ablation: each fast-belt change alone and all together vs v1, at 0.3 and 0.5 m/s (6 seeds x 180 s: about 30 minutes)",
                  lambda k: [GUARD, "64", MJ, "-I", "factory/ops/f7.py", "ablate", "--seeds", _i(k.get("seeds", 2), 1, 12), "--seed0", _i(k.get("seed0", 9000), 0, 10**6), "--workers", "7", "--name", "f7-ablation-lab"]),
    "f7-sweep": ("F7 operating envelope: the v1 or v2 cell over belt speed x arrival rate x clutter (36 runs of 180 s, about 15 minutes)",
                 lambda k: [GUARD, "64", MJ, "-I", "factory/ops/sweep.py", "--seeds", _i(k.get("seeds", 1), 1, 4), "--seed0", _i(k.get("seed0", 9000), 0, 10**6), "--workers", "7",
                            "--name", "f7-sweep-lab", "--v2", {"v2": "fast,early,deadline", "v1": ""}[_choice(k.get("cell", "v2"), ("v2", "v1"))]]),
    "live-stream": ("Live view: the cell running in real time, streamed to this page (port 8801)",
                    lambda k: [GUARD, "16", MJ, "-I", "factory/cell/live_stream.py", "--port", "8801", "--speed", _f(k.get("speed", 0.3), 0.0, 0.8)]),
}
def shift_state(name=""):
    """F5 dashboard data: the named shift (or the newest runs/*/live.json) and its KPI rows."""
    runs = os.path.join(ROOT, "runs")
    if not re.fullmatch(r"[\w.-]*", name): return {"error": "bad name"}
    if not name:
        cands = [d for d in os.listdir(runs) if os.path.exists(os.path.join(runs, d, "live.json"))] if os.path.isdir(runs) else []
        if not cands: return {"error": "no shift has run yet (rl shift run)"}
        name = max(cands, key=lambda d: os.path.getmtime(os.path.join(runs, d, "live.json")))
    d = os.path.join(runs, name)
    try: live = json.load(open(os.path.join(d, "live.json")))
    except (OSError, ValueError): return {"error": f"no live data for {name}"}
    rows = []
    if os.path.exists(os.path.join(d, "kpis.jsonl")):
        for line in open(os.path.join(d, "kpis.jsonl")):
            try: r = json.loads(line); rows.append({k: r[k] for k in ("t", "sorted", "arrived", "sorted_last_min", "oee", "sort_rate")})
            except (ValueError, KeyError): pass
    live["stale_s"] = round(time.time() - os.path.getmtime(os.path.join(d, "live.json")), 1)
    return dict(name=name, live=live, series=rows, shifts=sorted(x for x in os.listdir(runs) if os.path.exists(os.path.join(runs, x, "live.json"))))


RUNNING = {}   # job id -> Popen
LOCK = threading.Lock()


def start_job(name, knobs):
    desc, make = JOBS[name]; argv = make(knobs)
    jid = time.strftime("%Y%m%d-%H%M%S-") + name
    log = os.path.join(STATE, jid + ".log")
    meta = dict(id=jid, job=name, desc=desc, argv=[os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in argv], knobs=knobs,
                started=time.time(), log=os.path.relpath(log, ROOT), exit=None)
    f = open(log, "w"); f.write("$ " + " ".join(meta["argv"]) + "\n"); f.flush()
    p = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    meta["pid"] = p.pid
    with LOCK: RUNNING[jid] = p
    _save(meta)

    def wait():
        rc = p.wait(); f.close(); meta["exit"] = rc; meta["ended"] = time.time(); _save(meta)
        with LOCK: RUNNING.pop(jid, None)
    threading.Thread(target=wait, daemon=True).start()
    return meta


def _save(meta):
    with open(os.path.join(STATE, meta["id"] + ".json"), "w") as f: json.dump(meta, f, indent=1)


def _jobs():
    out = []
    for fn in sorted(os.listdir(STATE), reverse=True)[:80]:
        if fn.endswith(".json"):
            try: out.append(json.load(open(os.path.join(STATE, fn))))
            except Exception: pass
    return out


def _safe(rel, roots):
    path = os.path.realpath(os.path.join(ROOT, rel))
    if not any(path.startswith(os.path.realpath(os.path.join(ROOT, r)) + os.sep) for r in roots): raise PermissionError(rel)
    return path


CT = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".json": "application/json", ".png": "image/png",
      ".jpg": "image/jpeg", ".mp4": "video/mp4", ".svg": "image/svg+xml", ".jsonl": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8"}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _file(self, path):
        if not os.path.isfile(path): return self._send(404, {"error": "not found"})
        ext = os.path.splitext(path)[1].lower(); size = os.path.getsize(path); rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):   # Safari needs ranges to play mp4
            a = int(m.group(1) or 0); b = int(m.group(2)) if m.group(2) else size - 1; b = min(b, size - 1)
            with open(path, "rb") as f: f.seek(a); data = f.read(b - a + 1)
            self.send_response(206); self.send_header("Content-Type", CT.get(ext, "application/octet-stream")); self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        with open(path, "rb") as f: data = f.read()
        self._send(200, data, CT.get(ext, "application/octet-stream"), {"Accept-Ranges": "bytes"})

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query)); p = u.path
        try:
            if p in ("/", "/index.html"): return self._file(os.path.join(LAB, "static", "index.html"))
            if p.startswith("/phase/"): return self._file(os.path.join(LAB, "static", "phase.html"))
            if p in ("/dashboard", "/dashboard/"): return self._file(os.path.join(LAB, "static", "dashboard.html"))
            if p == "/api/shift": return self._send(200, shift_state(q.get("name", "")))
            if p.startswith("/static/"): return self._file(_safe("lab/static/" + p[len("/static/"):], ("lab/static",)))
            if p.startswith("/media/"): return self._file(_safe(p[len("/media/"):], MEDIA_ROOTS))
            if p == "/api/phases":
                status = json.load(open(os.path.join(ROOT, "factory", "phases.json"))); meta = {}
                for fn in sorted(os.listdir(os.path.join(LAB, "phases"))):
                    if fn.endswith(".json"):
                        d = json.load(open(os.path.join(LAB, "phases", fn))); meta[d["id"]] = {k: d.get(k) for k in ("id", "title", "summary", "disciplines", "highlight")}
                return self._send(200, dict(status=status, phases=meta))
            if p.startswith("/api/phase/"):
                pid = p.rsplit("/", 1)[1]
                if not re.fullmatch(r"F\d+|[a-z0-9-]+", pid): return self._send(400, {"error": "bad id"})
                f = os.path.join(LAB, "phases", pid + ".json")
                return self._file(f) if os.path.exists(f) else self._send(404, {"error": f"no page for {pid} yet"})
            if p == "/api/file": return self._file(_safe(q.get("path", ""), FILE_ROOTS))
            if p == "/api/exists":
                try: return self._send(200, {"exists": os.path.exists(_safe(q.get("path", ""), MEDIA_ROOTS))})
                except PermissionError: return self._send(200, {"exists": False})
            if p == "/api/jobs": return self._send(200, dict(defined={k: v[0] for k, v in JOBS.items()}, runs=_jobs()))
            if m := re.fullmatch(r"/api/jobs/([\w.-]+)/stream", p): return self._stream(m.group(1))
            return self._send(404, {"error": "not found"})
        except PermissionError: return self._send(403, {"error": "outside the served directories"})
        except (BrokenPipeError, ConnectionResetError): return

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); n = int(self.headers.get("Content-Length") or 0)
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except Exception: return self._send(400, {"error": "bad json"})
        if m := re.fullmatch(r"/api/run/([\w-]+)", u.path):
            name = m.group(1)
            if name not in JOBS: return self._send(404, {"error": f"no job {name}"})
            if name.startswith("live-"):   # one stream of each kind at a time
                with LOCK:
                    for jid, pr in RUNNING.items():
                        if jid.endswith(name) and pr.poll() is None: return self._send(200, dict(id=jid, already=True))
            try: meta = start_job(name, body.get("knobs", {}))
            except (AssertionError, ValueError) as e: return self._send(400, {"error": f"bad knob: {e}"})
            return self._send(200, meta)
        if m := re.fullmatch(r"/api/jobs/([\w.-]+)/stop", u.path):
            with LOCK: pr = RUNNING.get(m.group(1))
            if pr and pr.poll() is None:
                try: os.killpg(pr.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                return self._send(200, {"stopped": m.group(1)})
            return self._send(200, {"stopped": None})
        return self._send(404, {"error": "not found"})

    def _stream(self, jid):
        meta_f = os.path.join(STATE, jid + ".json"); log_f = os.path.join(STATE, jid + ".log")
        if not os.path.exists(meta_f): return self._send(404, {"error": "no such job"})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close"); self.end_headers(); self.close_connection = True
        pos = 0; buf = ""
        while True:
            with open(log_f) as f: f.seek(pos); chunk = f.read(); pos = f.tell()
            buf += chunk
            *lines, buf = buf.split("\n")
            for line in lines: self.wfile.write(b"data: " + line[-2000:].encode() + b"\n\n")
            self.wfile.flush()
            meta = json.load(open(meta_f))
            if meta.get("exit") is not None and not chunk:
                if buf: self.wfile.write(b"data: " + buf.encode() + b"\n\n")
                self.wfile.write(b"event: done\ndata: " + json.dumps(meta).encode() + b"\n\n"); self.wfile.flush(); return
            time.sleep(0.4)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8800); a = p.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler); srv.daemon_threads = True
    print(f"lab at http://127.0.0.1:{a.port}/", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        with LOCK:
            for pr in RUNNING.values():
                try: os.killpg(pr.pid, signal.SIGTERM)
                except ProcessLookupError: pass


if __name__ == "__main__":
    main()
