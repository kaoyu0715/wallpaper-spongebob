# 八代审计拼图：每场景一张 = 全幅缩览 + 每角色局部（头→脚，原分辨率裁切）。
# 用法: python3 audit8.py <scene-id>  → output/audit/<scene>.jpg
import sys, os, importlib.util
from PIL import Image, ImageDraw
spec = importlib.util.spec_from_file_location(
    'compose_round8', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compose-round8.py'))
m = importlib.util.module_from_spec(spec)
sys.argv_backup = sys.argv
sys.argv = ['compose-round8.py']  # 防止 argparse 干扰
spec.loader.exec_module(m)
sys.argv = sys.argv_backup
SCENES = m.SCENES

sid = sys.argv[1]
sc = SCENES[sid]
img = Image.open('output/round8/r8-%s.png' % sid).convert('RGB')
cellh = 1100
full = img.resize((int(img.width * cellh / img.height), cellh), Image.LANCZOS)
cells = [('full', full)]
for key, h, cx, gy, tilt in sc['chars']:
    x0 = max(0, int(cx - 420)); x1 = min(img.width, int(cx + 420))
    y0 = max(0, int(gy - h - 120)); y1 = min(img.height, gy + 30)
    c = img.crop((x0, y0, x1, y1))
    r = cellh / c.height
    cells.append((key, c.resize((int(c.width * r), cellh), Image.LANCZOS)))
cw = max(c.width for _, c in cells)
canvas = Image.new('RGB', (cw * len(cells), cellh), (240, 240, 240))
d = ImageDraw.Draw(canvas)
for i, (n, c) in enumerate(cells):
    x = i * cw
    canvas.paste(c, (x, 0))
    d.text((x + 6, 6), n, fill=(255, 0, 0))
os.makedirs('output/audit', exist_ok=True)
out = 'output/audit/%s.jpg' % sid
canvas.save(out, quality=86)
print(out)
