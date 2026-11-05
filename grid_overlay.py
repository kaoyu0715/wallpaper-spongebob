# 测量辅助：裁剪 + 坐标网格，便于在图上直接读原图坐标（用于样张/成品比例测量）。
# 用法: python3 grid_overlay.py <img> <x0> <y0> <x1> <y1> <out.png> [step] [target_w]
import sys
from PIL import Image, ImageDraw

img = Image.open(sys.argv[1]).convert('RGB')
x0, y0, x1, y1 = map(int, sys.argv[2:6])
out = sys.argv[6]
step = int(sys.argv[7]) if len(sys.argv) > 7 else 50
target_w = int(sys.argv[8]) if len(sys.argv) > 8 else 1200

crop = img.crop((x0, y0, x1, y1))
s = target_w / crop.width
crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
d = ImageDraw.Draw(crop)
for gx in range((x0 // step) * step, x1, step):
    px = (gx - x0) * s
    major = gx % (step * 2) == 0
    d.line([(px, 0), (px, crop.height)], fill=(255, 0, 255) if major else (255, 150, 255), width=1)
    if major:
        d.text((px + 2, 2), str(gx), fill=(255, 0, 255))
        d.text((px + 2, crop.height - 14), str(gx), fill=(255, 0, 255))
for gy in range((y0 // step) * step, y1, step):
    py = (gy - y0) * s
    major = gy % (step * 2) == 0
    d.line([(0, py), (crop.width, py)], fill=(0, 200, 255) if major else (150, 230, 255), width=1)
    if major:
        d.text((2, py + 1), str(gy), fill=(0, 160, 255))
        d.text((crop.width - 60, py + 1), str(gy), fill=(0, 160, 255))
crop.save(out)
print(out, 'scale=%.2f' % s, 'step=%d' % step)
