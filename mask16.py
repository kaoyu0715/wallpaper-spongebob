# -*- coding: utf-8 -*-
# 十六代 B 路线掩膜/贴回定义（issue #33）：在十五代基础上返工——
#  1) 贴回统一「紧凑档」：只贴「圈内孔 ∪ 角色多边形外扩」，样张的圈管（褶皱/高光渐变）、
#     水面（星芒/泡沫/波纹/鱼影/心形）像素级保留 —— 十五代 fish1 宽松档抹平这些的问题即此修复；
#  2) 内孔椭圆按样张实测扩大（十五代孔太小 → 内环残「死蓝底」+ 原鱼鳍/脚残留带）：
#     fish1 (356,1044,150,50) / fish2 (720,856,158,94) / fish3 (690,1330,163,115)，
#     均覆盖到泳圈管内缘（样张网格实测，见 output/g16-fish*grid.png）。
# 角色多边形沿用 mask15（样张三鱼实测轮廓）。
import os
from PIL import Image, ImageDraw, ImageFilter

REF_SIZE = (1080, 1920)

FISHES = None  # 运行时由 mask15 注入（十六代沿用十五代多边形）

# —— 内孔椭圆（cx,cy,a,b，样张实测=管圈内缘）——
HOLE_ELL = {
    'fish1': (356, 1044, 150, 50),
    'fish2': (720, 856, 158, 94),
    'fish3': (690, 1330, 163, 115),
}

# —— 样张保留区（参考帧椭圆）：这些区域绝不贴/绝不 inpaint（样张独有元素）——
KEEP_ELL = {
    'fish1': [(420, 1055, 100, 78)],   # ZZ 字母（样张 z序：ZZ 在角色之前）
    'fish2': [],
    'fish3': [],
}

# —— 额外覆盖椭圆：原鱼伸出多边形之外的鳍/残影带（贴+inpaint 都要覆盖）——
EXTRA_ELL = {
    'fish1': [],
    'fish2': [(862, 715, 48, 88)],      # 右缘鱼鳍残条
    'fish3': [(868, 1180, 38, 95)],     # 右缘白头巾/黑带/绿领残留条
}

# —— 圈口开度椭圆（比 HOLE_ELL 略大，覆盖管内缘阴影带；网格实测）——
OPEN_ELL = {
    'fish1': (356, 1044, 178, 68),
    'fish2': (720, 852, 148, 84),
    'fish3': (690, 1330, 178, 125),
}

# —— 分区裁片（同十五代，用户认可布局）——
CROPS = {
    'fish1': (40, 770, 660, 1420),
    'fish2': (470, 560, 1060, 1180),
    'fish3': (380, 1000, 1000, 1600),
}


def _poly(which, margin):
    import mask15
    cx = sum(p[0] for p in mask15.FISHES[which]) / len(mask15.FISHES[which])
    cy = sum(p[1] for p in mask15.FISHES[which]) / len(mask15.FISHES[which])
    out = []
    for x, y in mask15.FISHES[which]:
        dx, dy = x - cx, y - cy
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        out.append((x + margin * dx / n, y + margin * dy / n))
    return out


def paste_mask(which, size=REF_SIZE, feather=11, cg=38, hole_grow=10):
    """十六代统一紧凑贴回掩膜：内孔椭圆(∪到管内缘) ∪ 角色多边形外扩，羽化。"""
    cx, cy, a, b = HOLE_ELL[which]
    big = 2
    im = Image.new('L', (size[0] * big, size[1] * big), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([2 * (cx - a - hole_grow), 2 * (cy - b - hole_grow),
               2 * (cx + a + hole_grow), 2 * (cy + b + hole_grow)], fill=255)
    poly = _poly(which, cg)
    d.polygon([(2 * x, 2 * y) for x, y in poly], fill=255)
    return im.resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(feather))


def paste_back(base, crop_path, which, **kw):
    """crop(模型输出矩形) 等比缩回后按掩膜合成。"""
    x0, y0, x1, y1 = CROPS[which]
    w, h = x1 - x0, y1 - y0
    crop = Image.open(crop_path).convert('RGB').resize((w, h), Image.LANCZOS)
    m = paste_mask(which, **kw).crop((x0, y0, x1, y1))
    out = base.convert('RGB').copy()
    out.paste(crop, (x0, y0), m)
    return out


def paste_hole_only(base, crop_path, which, grow=6, feather=9):
    """只贴内孔（局部重绘用：二次 pass 修内孔底色，不动角色本体）。"""
    cx, cy, a, b = HOLE_ELL[which]
    big = 2
    im = Image.new('L', (REF_SIZE[0] * big, REF_SIZE[1] * big), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([2 * (cx - a - grow), 2 * (cy - b - grow),
               2 * (cx + a + grow), 2 * (cy + b + grow)], fill=255)
    m = im.resize(REF_SIZE, Image.LANCZOS).filter(ImageFilter.GaussianBlur(feather))
    x0, y0, x1, y1 = CROPS[which]
    w, h = x1 - x0, y1 - y0
    crop = Image.open(crop_path).convert('RGB').resize((w, h), Image.LANCZOS)
    out = base.convert('RGB').copy()
    out.paste(crop, (x0, y0), m.crop((x0, y0, x1, y1)))
    return out


def overlay(which, base, alpha=70):
    im = base.convert('RGB').copy()
    m = paste_mask(which)
    red = Image.new('RGB', im.size, (255, 0, 0))
    return Image.composite(red, im, m.point(lambda v: int(v * alpha / 255) if v else 0))


if __name__ == '__main__':
    import os
    import mask15
    T = os.path.dirname(os.path.abspath(__file__))
    REF = Image.open(os.path.join(T, 'review/reference/swim-ref.jpg'))
    for w in mask15.FISHES:
        overlay(w, REF).save(os.path.join(T, f'output/pmask16-{w}.png'))
        paste_mask(w).save(os.path.join(T, f'output/pmask16-{w}-m.png'))
        print(w, 'ok')


# —— 十六代 v2 贴回：轮廓门控（silhouette-gated）——
# 只贴「内孔 ∪ 角色剪影 ∪ extras」：模型重绘的平色圈管/岛/酒等全部丢弃，
# 样张的管面渐变、波纹、星芒、泡沫、ZZ 像素级保留。剪影=角色色 chroma 膨胀。
import numpy as np


def _chroma(crop_rgb, which):
    a = np.asarray(crop_rgb).astype(int)
    r, g, b = a[:,:,0], a[:,:,1], a[:,:,2]
    if which == 'fish1':   # SB：黄身/棕裤/红袜/黑白鞋
        m = (((r>170)&(g>140)&(r-b>60)) | ((r>110)&(r<210)&(g>50)&(g<150)&(b<110)&(r-g>30))
             | ((r>140)&(g<90)&(b<100)) | ((r<95)&(g<95)&(b<95))
             | ((abs(r-g)<14)&(abs(g-b)<14)&(r>120)&(r<215)))
    elif which == 'fish2':  # Patrick：粉肤/绿裤/橙汁/杯白
        m = ((r-b>60)&(r-g>50)&(r>170)) | ((g-r>15)&(g-b>40)) \
            | ((r>200)&(g>90)&(g<190)&(b<110)) | ((r>225)&(g>225)&(b>225))
    else:                   # Sandy：白衣/棕褐肤尾/粉花/黑鞋（白仅在膨胀区内）
        skin = ((r-b>45)&(r>150)&(g>80)&(g<210)) | ((r>100)&(r<200)&(g>55)&(g<150)&(b<120)&(r-b>25))
        pink = (r-b>35)&(r-b<120)&(r-g>35)&(r>190)
        dark = (r<95)&(g<95)&(b<95)
        white = (r>205)&(g>205)&(b>205)
        m = skin | pink | dark
        core = Image.fromarray((m*255).astype('uint8')).filter(ImageFilter.MaxFilter(51))
        m = m | (white & (np.asarray(core) > 0))
    return Image.fromarray((m*255).astype('uint8'))


def zz_keep(size=REF_SIZE, dil=10, feather=4):
    """样张 ZZ 字母本体（chroma 提取 + 膨胀），贴回时保样张字母不被覆盖。"""
    ref = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'review', 'reference', 'swim-ref.jpg')).convert('RGB')
    x0, y0, x1, y1 = CROPS['fish1']
    a = np.asarray(ref.crop((x0, y0, x1, y1))).astype(int)
    r, g, b = a[:,:,0], a[:,:,1], a[:,:,2]
    yl = (r>170)&(g>130)&(r-b>60)&(g-b>30)
    mm = Image.fromarray((yl*255).astype('uint8')).filter(ImageFilter.MaxFilter(2*dil+1))
    mm = mm.filter(ImageFilter.GaussianBlur(2))
    big = Image.new('L', size, 0)
    big.paste(mm, (x0, y0))
    return big.point(lambda v: 255 if v > 100 else 0).filter(ImageFilter.GaussianBlur(feather))


def sil_mask(cand_path, which, dil=10, feather=4, hole=True, hole_grow=6,
             extras=(), size=REF_SIZE, poly_margin=18, keep=None, keep_img=None):
    from PIL import Image as I
    x0, y0, x1, y1 = CROPS[which]
    w, h = x1-x0, y1-y0
    crop = I.open(cand_path).convert('RGB')
    sil = _chroma(crop, which).filter(ImageFilter.MaxFilter(2*dil+1)).filter(ImageFilter.GaussianBlur(2))
    sil = sil.resize((w, h), I.LANCZOS)
    big = 2
    full = Image.new('L', (size[0]*big, size[1]*big), 0)
    if poly_margin is not None:
        for ppx, ppy in _poly(which, poly_margin):
            pass  # polygon drawn below
    d = ImageDraw.Draw(full)
    if hole:
        cx, cy, a, b = OPEN_ELL[which]
        d.ellipse([2*(cx-a-hole_grow), 2*(cy-b-hole_grow), 2*(cx+a+hole_grow), 2*(cy+b+hole_grow)], fill=255)
    for (ex, ey, ea, eb) in list(extras) + EXTRA_ELL.get(which, []):
        d.ellipse([2*(ex-ea), 2*(ey-eb), 2*(ex+ea), 2*(ey+eb)], fill=255)
    if poly_margin is not None:
        ImageDraw.Draw(full).polygon([(2*px, 2*py) for px, py in _poly(which, poly_margin)], fill=255)
    full.paste(255, (2*x0, 2*y0), sil)
    full = full.resize(size, I.LANCZOS).point(lambda v: 255 if v > 140 else 0)
    if keep_img is not None:
        full.paste(0, (0, 0), keep_img)
    return full.filter(ImageFilter.GaussianBlur(feather))


def sil_paste_back(base, crop_path, which, **kw):
    x0, y0, x1, y1 = CROPS[which]
    w, h = x1-x0, y1-y0
    crop = Image.open(crop_path).convert('RGB').resize((w, h), Image.LANCZOS)
    m = sil_mask(crop_path, which, **kw).crop((x0, y0, x1, y1))
    out = base.convert('RGB').copy()
    out.paste(crop, (x0, y0), m)
    return out
