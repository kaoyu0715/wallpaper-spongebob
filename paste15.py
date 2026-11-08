# -*- coding: utf-8 -*-
# 十五代贴回掩膜（issue #31）：每分区=外圈椭圆(含漂移余量) ∪ 角色区多边形外扩，羽化。
# 邻居角色/心形/ZZ/星鱼都在掩膜外 → 原像素保留；漂移只发生在掩膜内。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
from PIL import Image, ImageDraw, ImageFilter
import mask15

# 圈外椭圆（swim-ref 坐标，实测）
RING_ELL = {
  'fish1': (342, 1102, 218, 118),
  'fish2': (720, 858, 222, 136),
  'fish3': (688, 1322, 256, 182),
}


def paste_mask(which, size=mask15.REF_SIZE, feather=18, ring_grow=26, char_grow=55):
    cx, cy, a, b = RING_ELL[which]
    big = 2
    im = Image.new('L', (size[0] * big, size[1] * big), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([2 * (cx - a - ring_grow), 2 * (cy - b - ring_grow),
               2 * (cx + a + ring_grow), 2 * (cy + b + ring_grow)], fill=255)
    poly = mask15._expand(mask15.FISHES[which], char_grow)
    d.polygon([(2 * x, 2 * y) for x, y in poly], fill=255)
    im = im.resize(size, Image.LANCZOS)
    return im.filter(ImageFilter.GaussianBlur(feather))


def paste_back(base, crop_path, which):
    """crop(矩形, 模型输出) 缩回矩形坐标后，按 paste_mask 合成到底图。"""
    from gen15crop import CROPS
    x0, y0, x1, y1 = CROPS[which]
    w, h = x1 - x0, y1 - y0
    crop = Image.open(crop_path).convert('RGB').resize((w, h), Image.LANCZOS)
    m = paste_mask(which).crop((x0, y0, x1, y1))
    out = base.convert('RGB').copy()
    out.paste(crop, (x0, y0), m)
    return out


# —— 紧凑贴回模式（fish2/3 实测教训：模型把条纹圈重画成素色面包圈 → 只贴角色+圈内孔，
#    圈管像素保留原图）——
HOLE_ELL = {
  'fish2': (720, 862, 104, 64),
  'fish3': (688, 1322, 122, 82),
}


def paste_mask_tight(which, size=mask15.REF_SIZE, feather=12, cg=38, hole_grow=14):
    cx, cy, a, b = HOLE_ELL[which]
    big = 2
    im = Image.new('L', (size[0] * big, size[1] * big), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([2 * (cx - a - hole_grow), 2 * (cy - b - hole_grow),
               2 * (cx + a + hole_grow), 2 * (cy + b + hole_grow)], fill=255)
    d.polygon([(2 * x, 2 * y) for x, y in mask15._expand(mask15.FISHES[which], cg)], fill=255)
    return im.resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(feather))


def paste_back_tight(base, crop_path, which):
    from gen15crop import CROPS
    x0, y0, x1, y1 = CROPS[which]
    w, h = x1 - x0, y1 - y0
    crop = Image.open(crop_path).convert('RGB').resize((w, h), Image.LANCZOS)
    m = paste_mask_tight(which).crop((x0, y0, x1, y1))
    out = base.convert('RGB').copy()
    out.paste(crop, (x0, y0), m)
    return out
