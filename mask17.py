# -*- coding: utf-8 -*-
# 十七代「保圈换鱼身」几何与掩膜定义（issue #35）：
#   单元替换路线（禁整区重绘）。三张圈抠图 ringcut = 角色+圈骨架与遮挡参考：
#   圈=样张原像素不许重绘；鱼身区域用「白底 LoRA 单元 + 抠图」清晰通道替换。
#
# 几何双轨：
#   OUTER_ELL 圈外缘实测椭圆（issue #35 给定 A/B；C 本脚本按同法自测：
#     抠图 alpha 下半轮廓最小二乘椭圆拟合 + 抠图原点模板匹配换算样张坐标）；
#   HOLE_ELL / OPEN_ELL / CROPS / 多边形 = 十五/十六代样张实测口径，原样沿用。
#
# 掩膜定义（全部参考帧 1080x1920 坐标）：
#   tube_band    = outer ∧ ¬hole             圈管体（样张像素保护带）
#   front_band   = tube_band ∧ y > hole_cy   前缘带（贴回样张像素 → 圈前缘压身）
#   erase        = 旧鱼足迹(OPEN ∪ poly+margin) ∧ ¬front_band ∧ ¬ZZ   内孔/上身补画区
#   old_limbs    = 抠图alpha ∧ tube_band       旧鱼搭管肢体区（骨架位置基准）
#   zz_letters   = 样张 ZZ 字母区（收窄框，z 序在肢体之上）
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import mask15
import mask16

T = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(T, 'review/reference/swim-ref.jpg')
CUTD = os.path.join(T, 'review/reference/ringcut')
REF_SIZE = (1080, 1920)

CROPS = mask16.CROPS                    # 分区裁片（历代布局不变）
FISHES = mask15.FISHES                  # 旧鱼实测多边形
HOLE_ELL = mask16.HOLE_ELL              # 内孔椭圆（管内缘）
OPEN_ELL = mask16.OPEN_ELL              # 圈口开度椭圆
EXTRA_ELL = mask16.EXTRA_ELL

# —— 圈外缘实测椭圆（issue #35 给定 A/B，宽裁坐标+裁框原点 → 样张坐标；
#     C 未测 → 本脚本 bottom-fit 自测（下半对称弧拟合法，见下），顶缘/左右翼复核吻合；
#     A/B 同法自测与给定值一致性见 issue 报告）——
OUTER_ELL = {
    'fish1': (55 + 256, 835 + 248, 210, 94),
    'fish2': (415 + 308.5, 655 + 228, 208, 105),
    'fish3': (689.0, 1349.4, 238.3, 156.1),   # 自测 b/a=0.655 rms=0.5（避开左下泡沫带）
}

# —— 抠图单元 PNG 在样张中的原点（色中位模板匹配自测，中位色差 ≤18：
#     与 issue「宽裁原点」(55,835)/(415,655) 不同——那是测量宽裁框原点，非 PNG 原点）——
CUTS = {
    'fish1': ('ring-a-cut.png', (122, 880)),
    'fish2': ('ring-b-cut.png', (510, 660)),
    'fish3': ('ring-c-cut.png', (440, 1080)),
}

# —— ZZ 字母收窄区（十六代 z1 二轮实测：x365–545, y1005–1170，仅字母不含鱼身）——
ZZ_BOX = (365, 1005, 545, 1170)


# ============================ 掩膜构造 ============================

def _ell(size, items, scale=2):
    im = Image.new('L', (size[0] * scale, size[1] * scale), 0)
    d = ImageDraw.Draw(im)
    for cx, cy, a, b in items:
        d.ellipse([2 * (cx - a), 2 * (cy - b), 2 * (cx + a), 2 * (cy + b)], fill=255)
    return im.resize(size, Image.LANCZOS)


def outer_mask(fish, size=REF_SIZE):
    cx, cy, a, b = OUTER_ELL[fish]
    return _ell(size, [(cx, cy, a, b)]).point(lambda v: 255 if v > 127 else 0)


def hole_mask(fish, size=REF_SIZE, grow=0):
    cx, cy, a, b = HOLE_ELL[fish]
    return _ell(size, [(cx, cy, a + grow, b + grow)]).point(lambda v: 255 if v > 127 else 0)


def tube_band(fish, size=REF_SIZE, shrink=2):
    """圈管体保护带 = outer ∧ ¬hole(生长 shrink)。shrink>0 给内缘 AA 圈留余地。"""
    o = np.asarray(outer_mask(fish, size)) > 127
    h = np.asarray(hole_mask(fish, size, grow=shrink)) > 127
    return Image.fromarray(((o & ~h) * 255).astype('uint8'))


def front_band(fish, size=REF_SIZE, feather=1.2, shrink=2):
    """前缘带：管体 ∧ y>hole_cy，羽化。贴回样张像素=圈前缘压身。"""
    tb = np.asarray(tube_band(fish, size, shrink=shrink))
    cy = HOLE_ELL[fish][1]
    yy = np.arange(size[1])[:, None]
    m = (tb > 0) & (yy > cy)
    im = Image.fromarray((m * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(feather))
    return im.point(lambda v: 255 if v > 96 else 0)


def footprint(fish, size=REF_SIZE, margin=14):
    """旧鱼足迹 = OPEN 椭圆(∪到管内缘) ∪ 多边形外扩 ∪ EXTRA_ELL（硬芯，不羽化）。"""
    cx, cy, a, b = OPEN_ELL[fish]
    im = Image.new('L', (size[0] * 2, size[1] * 2), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([2 * (cx - a), 2 * (cy - b), 2 * (cx + a), 2 * (cy + b)], fill=255)
    d.polygon([(2 * x, 2 * y) for x, y in mask16._poly(fish, margin)], fill=255)
    for (ex, ey, ea, eb) in EXTRA_ELL.get(fish, []):
        d.ellipse([2 * (ex - ea), 2 * (ey - eb), 2 * (ex + ea), 2 * (ey + eb)], fill=255)
    return im.resize(size, Image.LANCZOS).point(lambda v: 255 if v > 140 else 0)


def zz_letters(size=REF_SIZE, feather=2):
    """ZZ 字母区：收窄框内取样张黄字 chroma（十六代 z1 二轮口径），另含字母周围 6px 缓冲。"""
    ref = Image.open(REF).convert('RGB')
    x0, y0, x1, y1 = ZZ_BOX
    a = np.asarray(ref.crop((x0, y0, x1, y1))).astype(int)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    yl = (r > 170) & (g > 130) & (r - b > 60) & (g - b > 30)
    m = Image.fromarray((yl * 255).astype('uint8')).filter(ImageFilter.MaxFilter(13))
    big = Image.new('L', size, 0)
    big.paste(m, (x0, y0))
    return big.filter(ImageFilter.GaussianBlur(feather)).point(lambda v: 255 if v > 90 else 0)


def cut_rgba(fish):
    im = Image.open(os.path.join(CUTD, CUTS[fish][0])).convert('RGBA')
    return im


def cut_origins():
    return {k: v[1] for k, v in CUTS.items()}


def old_limbs(fish, size=REF_SIZE, dil=0):
    """旧鱼搭在管上的肢体区 = 抠图 alpha(样张坐标) ∧ tube_band。骨架位置基准。"""
    ox, oy = cut_origins()[fish]
    rgba = cut_rgba(fish)
    alpha = rgba.split()[3]
    big = Image.new('L', size, 0)
    big.paste(alpha, (ox, oy))
    a = np.asarray(big) > 127
    tb = np.asarray(tube_band(fish, size)) > 0
    m = (a & tb)
    im = Image.fromarray((m * 255).astype('uint8'))
    if dil:
        im = im.filter(ImageFilter.MaxFilter(2 * dil + 1))
    return im


def erase_mask(fish, size=REF_SIZE, margin=14, feather=5):
    """旧鱼足迹补画区 = 足迹 ∧ (¬前缘带 ∨ 旧肢体压管区) ∧ ¬ZZ。
    前缘管体像素动不得（贴回样张），但「旧肢体压管」的鱼像素（旧鞋/旧鳍/旧手）
    不贴回——并入补画区重绘成管面，杜绝「鱼穿裤子」残留（十七代教训核心）。"""
    fp = np.asarray(footprint(fish, size, margin=margin)) > 127
    fb = np.asarray(front_band(fish, size)) > 0
    limbs = np.asarray(old_limbs(fish, size=size, dil=4)) > 0
    zz = np.asarray(zz_letters(size)) > 0
    m = fp & (~fb | limbs)
    if fish == 'fish1':
        m = m & ~zz
    im = Image.fromarray((m * 255).astype('uint8'))
    if feather:
        im = im.filter(ImageFilter.GaussianBlur(feather))
        im = im.point(lambda v: 255 if v > 64 else int(v * 1.6))
    return im


# ============================ 自测工具 ============================

def _ref_edges():
    ref = Image.open(REF).convert('L')
    a = np.asarray(ref).astype(np.int16)
    ex = np.abs(np.diff(a, axis=1, prepend=a[:, :1]))
    ey = np.abs(np.diff(a, axis=0, prepend=a[:1, :]))
    e = np.maximum(ex, ey)
    return (e > 45).astype(np.uint8)


def measure_outer_ellipse(fish, origin):
    """圈外缘椭圆（长轴水平）自测——下半对称弦宽拟合（与 issue 给定 A/B 同法）：
    逐行取 alpha 最外弦 [xmin,xmax]，下半部行拟合 w² = q2·y² + q1·y + q0（w=半弦宽），
    根 → cy±b，顶点 → a。输出样张坐标 (cx, cy, a, b) 与 rms。"""
    rgba = cut_rgba(fish)
    alpha = np.asarray(rgba.split()[3]) > 127
    ys_, xs_ = np.where(alpha)
    rows = {}
    for y, x in zip(ys_, xs_):
        if y in rows:
            lo, hi = rows[y]
            rows[y] = (min(lo, x), max(hi, x))
        else:
            rows[y] = (x, x)
    ry = sorted(rows)
    wmax = max((rows[y][1] - rows[y][0]) for y in ry)
    ywid = [y for y in ry if (rows[y][1] - rows[y][0]) >= wmax - 2]
    y0 = sum(ywid) / len(ywid)                      # 最宽行 ≈ cy（长轴水平）
    lo_ = [y for y in ry if y >= y0]
    Y = np.array(lo_, dtype=float)
    W = np.array([(rows[y][1] - rows[y][0]) / 2.0 for y in lo_])
    c = np.polyfit(Y, W * W, 2)                     # w² = q(y)
    q2, q1, q0 = c
    cy = -q1 / (2 * q2)
    disc = q1 * q1 - 4 * q2 * q0
    b = (disc ** 0.5) / (2 * abs(q2)) if disc > 0 else float('nan')
    a = float(np.polyval(c, cy) ** 0.5)
    cx = float(np.mean([(rows[y][0] + rows[y][1]) / 2.0 for y in ry if y >= y0]))
    resid = W - np.sqrt(np.clip(np.polyval(c, Y), 0, None))
    rms = float(np.sqrt((resid ** 2).mean()))
    out = (round(cx + origin[0], 1), round(cy + origin[1], 1), round(a, 1), round(b, 1))
    print('measure_outer %s -> %s (b/a=%.3f, rms=%.1f)' % (fish, out, b / a, rms))
    return out


def probe():
    print('origins:', cut_origins())
    print('OUTER_ELL:', OUTER_ELL)
    # 各分区掩膜导出
    os.makedirs('output', exist_ok=True)
    ref = Image.open(REF).convert('RGB')
    for f in CROPS:
        for nm, m in (('tube', tube_band(f)), ('front', front_band(f)),
                      ('erase', erase_mask(f)), ('limbs', old_limbs(f))):
            ov = Image.composite(Image.new('RGB', REF_SIZE, (255, 40, 40)), ref,
                                 m.point(lambda v: int(v * 0.55)))
            ov.save('output/m17-%s-%s.png' % (f, nm))
    zz = zz_letters()
    Image.composite(Image.new('RGB', REF_SIZE, (40, 255, 40)), ref,
                    zz.point(lambda v: int(v * 0.6))).save('output/m17-zz.png')
    print('masks -> output/m17-*.png')


if __name__ == '__main__':
    probe()
