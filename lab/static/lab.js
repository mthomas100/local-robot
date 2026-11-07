// lab.js — shared helpers and the workflow diagram for the sorting-cell lab (2026-11-02).
const $ = (s, el = document) => el.querySelector(s);
const h = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) { if (k === "html") e.innerHTML = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v); } for (const k of kids) e.append(k); return e; };
const getJSON = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); };

// The pipeline of the cell: sense (top row, left to right), think, act (bottom row, right to left), on the simulated cell.
const NODES = [
  { id: "belt", label: "Belt + items", sub: "YCB objects, 0.15-0.5 m/s", ph: ["F0"], x: 15, y: 20 },
  { id: "camera", label: "Camera", sub: "RGB-D; F6: noise, latency, calibration", ph: ["F0", "F4", "F6"], x: 212, y: 20 },
  { id: "perception", label: "Perception", sub: "detect + 6-DoF pose", ph: ["F4"], x: 410, y: 20 },
  { id: "tracking", label: "Tracking", sub: "Kalman filter, intercept", ph: ["F2"], x: 607, y: 20 },
  { id: "scheduler", label: "Scheduler", sub: "which item next", ph: ["F2", "F5"], x: 805, y: 20 },
  { id: "grasp", label: "Grasp planning", sub: "antipodal grasps", ph: ["F1"], x: 805, y: 130 },
  { id: "motion", label: "Motion planning", sub: "IK + collision-free path", ph: ["F1"], x: 607, y: 130 },
  { id: "control", label: "Control", sub: "joint PD; F6: hobby-servo model", ph: ["F1", "F6"], x: 410, y: 130 },
  { id: "robot", label: "UR5e + 2F-85", sub: "F6: the SO-101 desk arm", ph: ["F0", "F1", "F6"], x: 212, y: 130 },
  { id: "chutes", label: "Chutes", sub: "plastic · metal · cardboard", ph: ["F0"], x: 15, y: 130 },
  { id: "layout", label: "Layout + reach map", sub: "cell design, cycle budget, F6 payload table", ph: ["F0", "F6"], x: 15, y: 240, w: 375 },
  { id: "policy", label: "Learned policy", sub: "RL / BC replaces grasp → control; F6: ONNX at 30 Hz", ph: ["F3", "F4", "F6"], x: 410, y: 240, w: 375 },
  { id: "ops", label: "Safety + KPIs", sub: "recovery, dashboard, OEE", ph: ["F2", "F5"], x: 805, y: 240 },
  { id: "sim", label: "Simulated cell: MuJoCo (CPU, cross-check) + Genesis (Metal GPU, batched)", sub: "", ph: ["F0"], x: 15, y: 340, w: 960, hgt: 40 },
];
const EDGES = [["belt", "camera"], ["camera", "perception"], ["perception", "tracking"], ["tracking", "scheduler"], ["scheduler", "grasp"], ["grasp", "motion"], ["motion", "control"], ["control", "robot"], ["robot", "chutes"]];

function diagram(el, { highlight = [], status = {}, link = true } = {}) {
  const W = 170, H = 70, ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 990 395");
  svg.innerHTML = `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--todo)"/></marker></defs>`;
  const pos = Object.fromEntries(NODES.map(n => [n.id, n]));
  for (const [a, b] of EDGES) {
    const A = pos[a], B = pos[b], aw = A.w || W, bw = B.w || W; let d;
    if (A.y === B.y) { const dir = B.x > A.x ? 1 : -1; const x1 = dir > 0 ? A.x + aw : A.x, x2 = dir > 0 ? B.x : B.x + bw; d = `M${x1},${A.y + H / 2}L${x2},${B.y + H / 2}`; }
    else d = `M${A.x + aw / 2},${A.y + H}L${B.x + bw / 2},${B.y}`;
    const p = document.createElementNS(ns, "path"); p.setAttribute("d", d); p.setAttribute("class", "edge"); svg.append(p);
  }
  for (const n of NODES) {
    const g = document.createElementNS(ns, "g"); const hi = n.ph.some(p => highlight.includes(p)) || highlight.includes(n.id);
    const done = n.ph.length && n.ph.every(p => status[p] === "done");
    g.setAttribute("class", "node" + (hi ? " hi" : "") + (done ? " done" : ""));
    const w = n.w || W, hh = n.hgt || H;
    g.innerHTML = `<rect x="${n.x}" y="${n.y}" width="${w}" height="${hh}" rx="10"/>` +
      `<text x="${n.x + 12}" y="${n.y + (n.sub ? 26 : 25)}">${n.label}</text>` + (n.sub ? `<text x="${n.x + 12}" y="${n.y + 46}" class="ph">${n.sub}</text>` : "") +
      `<text x="${n.x + w - 10}" y="${n.y + 18}" class="ph" text-anchor="end">${n.ph.join(" ")}</text>`;
    if (link) g.addEventListener("click", () => { location.href = "/phase/" + n.ph[0]; });
    const t = document.createElementNS(ns, "title"); t.textContent = `${n.label}: ${n.sub} (phase ${n.ph.join(", ")})`; g.append(t);
    svg.append(g);
  }
  el.innerHTML = ""; el.append(svg);
}

function chip(st) { return h("span", { class: "chip " + (st || "todo") }, st || "todo"); }

// Stream a job's log into a <pre>; resolve with the job's final metadata.
function follow(jid, pre, onDone) {
  const es = new EventSource(`/api/jobs/${jid}/stream`);
  es.onmessage = (e) => { pre.textContent += e.data + "\n"; pre.scrollTop = pre.scrollHeight; };
  es.addEventListener("done", (e) => { es.close(); const m = JSON.parse(e.data); pre.textContent += `\n[exit ${m.exit}]\n`; onDone && onDone(m); });
  es.onerror = () => { es.close(); };
}
