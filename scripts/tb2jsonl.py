"""tb2jsonl.py <tb_dir> <out.jsonl> — flatten TensorBoard scalars (rsl_rl's logs) into one JSON line per step (2026-11-01)."""
import glob, json, sys, collections
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tb_dir, out = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(dict)
for f in glob.glob(f"{tb_dir}/**/events.out.tfevents.*", recursive=True):
    acc = EventAccumulator(f, size_guidance={"scalars": 0}); acc.Reload()
    for tag in acc.Tags().get("scalars", []):
        for ev in acc.Scalars(tag):
            rows[ev.step][tag.replace("/", "_")] = round(float(ev.value), 6); rows[ev.step].setdefault("wall", ev.wall_time)
with open(out, "w") as fh:
    for step in sorted(rows): fh.write(json.dumps({"it": step, **rows[step]}) + "\n")
print(out, len(rows), "rows")
