# 十二代（issue #25）：海绵宝宝腿身比 硬闸C —— 腿(裤腰→脚底)/身(头顶→裤腰) ∈ [0.55, 0.75]。
# 口径依据（见 delivery/recipe.md §4.2）：官方经典立绘实测 0.59/0.64/0.66/0.67/0.68（5 幅），
# 样张海绵宝宝（腿部被报纸遮挡，按站位地面线反推）≈0.62 —— 一致支持目标 0.65±0.10。
# - leg_ratio: 程序测量 RGBA/白底 PNG 单元的腿身比（棕色短裤带=裤腰，内容底行=脚底）。
# - shorten_legs: 确定性手术修正——若选中单元腿偏长，在小腿(袜)区整行裁除并缝合，
#   使比值精确落目标；行带仅含双腿列（臂/领带均在裤腰以上），缝合安全由列簇数断言保证。
import numpy as np
from PIL import Image

LEG_BAND = (0.55, 0.75)
LEG_TARGET = 0.65


def _vis(a):
    """返回可见像素掩膜：优先 alpha；无 alpha 图按四角背景色距离判定。"""
    if a.shape[2] >= 4:
        al = a[..., 3]
        if (al < 250).any():
            return al > 40
    rgb = a[..., :3].astype(np.int16)
    h, w = rgb.shape[:2]
    corners = np.array([rgb[2, 2], rgb[2, w - 3], rgb[h - 3, 2], rgb[h - 3, w - 3]])
    bg = np.median(corners, axis=0)
    d = np.abs(rgb - bg).sum(-1)
    return d > 60


def _brown(a, vis):
    R, G, B = a[..., 0].astype(np.int16), a[..., 1].astype(np.int16), a[..., 2].astype(np.int16)
    return vis & (R > 130) & (R < 215) & (G > 70) & (G < 160) & (B < 120) & ((R - G) > 35) & ((G - B) > 10)


def leg_ratio(img_or_path, verbose=False):
    """(ratio, anchors) anchors=(top, waist, bottom)；无法判定时 (None, None)。"""
    im = Image.open(img_or_path) if isinstance(img_or_path, str) else img_or_path
    a = np.asarray(im.convert('RGBA')).astype(np.int16)
    vis = _vis(a)
    rows = np.nonzero(vis.any(1))[0]
    if len(rows) == 0:
        return None, None
    top, bot = int(rows[0]), int(rows[-1])
    h = bot - top
    brown = _brown(a, vis)
    rc = brown.sum(1)
    maxrc = int(rc.max())
    start = top + int(h * 0.30)   # 裤腰最低位置下限（短裤带在 43% 起，留余量）
    # 相对阈值：腰带是高计数棕带，排除嘴/描边 AA 的零星棕行（计数 <5%峰值）
    thr = max(8, int(0.02 * vis.shape[1]), int(0.45 * maxrc))
    cand = np.nonzero(rc[start:bot + 1] >= thr)[0]
    if len(cand) == 0 or maxrc < 30:
        return None, None
    waist = int(cand[0] + start)
    ratio = (bot - waist) / (waist - top)
    if verbose:
        print('leg_ratio=%.3f (top=%d waist=%d bottom=%d body=%d leg=%d)'
              % (ratio, top, waist, bot, waist - top, bot - waist))
    return ratio, (top, waist, bot)


def shorten_legs(img, target=LEG_TARGET, verbose=True):
    """把腿身比修正到 target：在小腿区整行裁除 N 行后缝合。已达标/无法测量则原样返回。"""
    fig = img.convert('RGBA')
    ratio, (top, waist, bot) = leg_ratio(fig)
    a = np.asarray(fig)
    if ratio is None:
        raise SystemExit('shorten_legs: 无法定位短裤/脚底，拒绝合成')
    if ratio <= target + 0.005:
        return fig
    n = int(round(bot - waist - target * (waist - top)))
    vis = _vis(np.asarray(fig).astype(np.int16))
    # 缝合带：裤腰以下 40 行起，到脚底以上 30 行止（避开描边过渡），选带内
    brown = _brown(np.asarray(fig).astype(np.int16), vis)
    rc = brown.sum(1)
    thr = max(8, int(0.02 * vis.shape[1]))
    shorts_rows = np.nonzero(rc > thr)[0]
    crotch = int(shorts_rows[-1]) if len(shorts_rows) else waist + 80
    y0 = crotch + 12
    y1 = bot - 30
    if y1 - y0 < n:
        y0 = max(crotch + 6, bot - 30 - n)
    # 断言：缝合带每行内容列簇 ≤2（只有两条腿），否则拒绝
    for y in range(y0, y0 + n):
        xs = np.nonzero(vis[y])[0]
        if len(xs) == 0:
            continue
        splits = 1 + int((np.diff(xs) > 8).sum())
        if splits > 2:
            raise SystemExit('shorten_legs: 缝合带第 %d 行有 %d 个列簇（应≤2），拒绝缝合' % (y, splits))
    out = np.concatenate([a[:y0], a[y0 + n:]], axis=0)
    if verbose:
        print('shorten_legs: 缝合 %d..%d（%d 行），腿身比 %.3f → %.3f'
              % (y0, y0 + n, n, ratio, leg_ratio(Image.fromarray(out, 'RGBA'))[0]))
    return Image.fromarray(out, 'RGBA')


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        r, anc = leg_ratio(p, verbose=True)
        band = 'OK' if r is not None and LEG_BAND[0] <= r <= LEG_BAND[1] else 'FAIL'
        print(p.split('/')[-1], 'gate C:', band)
