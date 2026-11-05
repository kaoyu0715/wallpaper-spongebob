# 通用锚点测量（issue #25 相对身高重建）：给出各角色官方立绘中
#   full = 顶→脚底 ; frac = (顶→指定锚点) / full
# 锚点：pat=短裤下缘, sq=衬衫下缘, krab=裤下缘, sandy=盔圈下缘(单位图)。
# 半自动：颜色分段 + QA 蒙太奇人工复核。
import sys
import numpy as np
from PIL import Image

def content(im):
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    a = np.asarray(im).astype(np.int16)
    vis = a[..., 3] > 40
    rows = np.nonzero(vis.any(1))[0]
    return a[..., :3], vis, int(rows[0]), int(rows[-1])

def band(mask_rows):
    idx = np.nonzero(mask_rows)[0]
    return int(idx[0]), int(idx[-1]) if len(idx) else (None, None)

def measure_pat(path):
    rgb, vis, top, bot = content(Image.open(path))
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = vis & (G > 150) & (R < 200) & (G - R > 25) & (G - B > 60)
    rc = green.sum(1)
    thr = max(6, int(0.015 * vis.shape[1]))
    rows = np.nonzero(rc[top:bot + 1] >= thr)[0] + top
    if len(rows) == 0:
        return None
    shorts_bot = int(rows[-1])
    return dict(top=top, anchor=shorts_bot, bot=bot,
                frac=round((shorts_bot - top) / (bot - top), 3))

def measure_sq(path):
    rgb, vis, top, bot = content(Image.open(path))
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # 棕色衬衫：R>G>B, R-G>40, 亮度中
    brown = vis & (R > 120) & (R < 220) & (R - G > 40) & (G - B > 15)
    rc = brown.sum(1)
    thr = max(6, int(0.015 * vis.shape[1]))
    rows = np.nonzero(rc[top:bot + 1] >= thr)[0] + top
    if len(rows) == 0:
        return None
    shirt_bot = int(rows[-1])
    return dict(top=top, anchor=shirt_bot, bot=bot,
                frac=round((shirt_bot - top) / (bot - top), 3))

def measure_krab(path):
    rgb, vis, top, bot = content(Image.open(path))
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # 紫灰裤子：B>R, 低饱和, 中低亮度
    pants = vis & (B - R > 15) & (B > 90) & (B < 200) & (G < B) & (R < B)
    rc = pants.sum(1)
    thr = max(6, int(0.015 * vis.shape[1]))
    rows = np.nonzero(rc[top:bot + 1] >= thr)[0] + top
    if len(rows) == 0:
        return None
    pants_bot = int(rows[-1])
    return dict(top=top, anchor=pants_bot, bot=bot,
                frac=round((pants_bot - top) / (bot - top), 3))

if __name__ == '__main__':
    role = sys.argv[1]
    fn = {'pat': measure_pat, 'sq': measure_sq, 'krab': measure_krab}[role]
    for p in sys.argv[2:]:
        print(p.split('/')[-1], fn(p))
