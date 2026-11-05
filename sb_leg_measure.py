# 官方海绵宝宝立绘腿身比自动测量：
# 顶=内容顶行；裤腰=身体中线以下第一条连续棕色(短裤)行；脚底=内容底行。
# 输出每图 顶/裤腰/底/腿身比，供人工用网格图复核。
import sys
import numpy as np
from PIL import Image

def measure(path, min_alpha=40):
    im = Image.open(path)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    a = np.asarray(im).astype(np.int16)
    R, G, B, A = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    vis = A > min_alpha
    rows = np.nonzero(vis.any(1))[0]
    if len(rows) == 0:
        return None
    top, bot = rows[0], rows[-1]
    h = bot - top
    # 棕色短裤：R 130-210, G 80-150, B 20-110, R-G>40, G-B>20
    brown = vis & (R > 130) & (R < 215) & (G > 70) & (G < 160) & (B < 120) & ((R - G) > 35) & ((G - B) > 10)
    w = vis.shape[1]
    rowcount = brown.sum(1)
    # 只看下半身区域（顶以下 45% 处起，短裤必在身体中下部）
    start = top + int(h * 0.45)
    thr = max(8, int(0.02 * vis.shape[1]))
    cand = np.nonzero(rowcount[start:bot + 1] >= thr)[0]
    if len(cand) == 0:
        return dict(top=int(top), bot=int(bot), waist=None, ratio=None, h=int(h))
    waist = int(cand[0] + start)
    legs = bot - waist
    body = waist - top
    return dict(top=int(top), bot=int(bot), waist=waist, h=int(h),
                leg=legs, body=body, ratio=round(legs / body, 3))

if __name__ == '__main__':
    for p in sys.argv[1:]:
        r = measure(p)
        print(p.split('/')[-1], r)
