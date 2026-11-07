"""sheet.py <video> [out.png] [cols] [rows] — contact sheet with timestamps (see bin/rl-sheet)."""
import sys, imageio.v3 as iio, numpy as np
from PIL import Image, ImageDraw, ImageFont
video = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else video.rsplit(".", 1)[0] + "-sheet.png"
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4; rows = int(sys.argv[4]) if len(sys.argv) > 4 else 3
meta = iio.immeta(video, plugin="pyav"); fps = float(meta.get("fps", 30)); n = cols * rows
frames = list(iio.imiter(video, plugin="pyav")); total = len(frames)
idx = [min(total - 1, round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
tiles = []
try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except Exception: font = ImageFont.load_default()
for i in idx:
    im = Image.fromarray(frames[i]); w = 480; im = im.resize((w, int(im.height * w / im.width)))
    d = ImageDraw.Draw(im); t = i / fps; label = f"{t:5.2f}s  #{i}"
    d.rectangle([4, 4, 4 + 8 + len(label) * 12, 34], fill=(0, 0, 0)); d.text((8, 6), label, fill=(255, 255, 255), font=font)
    tiles.append(im)
tw, th = tiles[0].size; sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
for k, im in enumerate(tiles): sheet.paste(im, ((k % cols) * tw, (k // cols) * th))
sheet.save(out); print(out)
