# -*- coding: utf-8 -*-
# 花环补画（十五代）：模型六 seed 均把花环画到圈/空中——按样张词表在珊迪领圈前弧
# 程序化补画扁平五瓣花环（draw_cast13 同思路：样张自身即扁平画法）。
import math, sys
from PIL import Image, ImageDraw


def flower(d, cx, cy, r, petal, center, outline):
    for k in range(5):
        a = math.radians(k * 72 + 18)
        px, py = cx + math.cos(a) * r * 0.72, cy + math.sin(a) * r * 0.72 * 0.62
        d.ellipse([px - r * 0.58, py - r * 0.58, px + r * 0.58, py + r * 0.58],
                  fill=petal, outline=outline, width=2)
    d.ellipse([cx - r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4], fill=center)


def draw_lei(img, cx, cy, rx, ry, r=13, n=8):
    """沿领圈前弧（模型透视下半椭圆）放 n 朵花。"""
    im = img.convert('RGB').copy()
    d = ImageDraw.Draw(im)
    outline = (90, 40, 60)
    # 藤绳
    pts = [(cx + math.cos(math.radians(a)) * rx, cy + math.sin(math.radians(a)) * ry)
           for a in range(8, 173, 6)]
    d.line(pts, fill=(74, 140, 74), width=6)
    d.line(pts, fill=(96, 168, 96), width=3)
    for i in range(n):
        a = math.radians(12 + i * (156 / (n - 1)))
        x, y = cx + math.cos(a) * rx, cy + math.sin(a) * ry
        petal = (247, 168, 196) if i % 2 else (232, 111, 160)
        flower(d, x, y, r, petal, (255, 213, 79), outline)
    return im


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    cx, cy, rx, ry = map(float, sys.argv[3:7])
    draw_lei(Image.open(src), cx, cy, rx, ry).save(dst)
    print('lei ->', dst)
