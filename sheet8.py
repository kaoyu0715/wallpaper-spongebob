# 生成审计用拼图：把一组图缩到统一高度拼成一张，便于逐种子审计。
# 用法: python3 sheet8.py <out.jpg> <h_per_cell> <glob1> [glob2 ...]
import sys, glob
from PIL import Image

out, cellh = sys.argv[1], int(sys.argv[2])
paths = []
for g in sys.argv[3:]:
    paths.extend(sorted(glob.glob(g)))
cells = []
for p in paths:
    im = Image.open(p).convert('RGB')
    r = cellh / im.height
    cells.append((p, im.resize((int(im.width * r), cellh), Image.LANCZOS)))
n = len(cells)
cols = 2 if n <= 4 else 3
rows = (n + cols - 1) // cols
cw = max(c.width for _, c in cells)
W, H = cw * cols, cellh * rows
canvas = Image.new('RGB', (W, H), (240, 240, 240))
d = None
from PIL import ImageDraw
d = ImageDraw.Draw(canvas)
for i, (p, c) in enumerate(cells):
    x, y = (i % cols) * cw, (i // cols) * cellh
    canvas.paste(c, (x, y))
    d.rectangle([x, y, x + cw - 1, y + cellh - 1], outline=(120, 120, 120), width=2)
    d.text((x + 6, y + 6), p.split('/')[-1][:40], fill=(255, 0, 0))
canvas.save(out, quality=88)
print(out, n, 'cells')
