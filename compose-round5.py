# 五代合成：三角色「追泡泡」→ 1440x3200，路线A官方图案背景。
# 相比四代：角色集群放大约 1.35 倍并整体上移（贴近样张中部集群），泡泡场按样张在全画面散点补充。
# 内置派大星短裤红斑清理（green-enclosed 红色像素回填短裤纯色），仅对 patrick 生效。
# 用法: python3 compose-round5.py --bg <bg.png> --cutdir <g5cut目录> --out <out.png> \
#         --variant v1|v2|v3 [--no-clean]
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1440, 3200

# (高px, cx, ground_y, 倾角°) —— 「追泡泡」：后排中间章鱼哥叉腰嫌弃，
# 前排左海绵宝宝吹泡泡并指向派大星，前排右派大星伸手接泡泡。集群放大、上移。
BASE = {
  'squidward': (1120, 745, 2610, 0.0),
  'patrick':   (980, 1035, 2830, 3.0),
  'spongebob': (900, 415, 2880, -3.0),
}
VARIANTS = {
  'v1': {},
  'v2': {'patrick': (990, 965, 2810, 4.0), 'spongebob': (910, 470, 2860, -4.0),
         'squidward': (1150, 735, 2560, 0.0)},
  'v3': {'patrick': (970, 1010, 2820, 5.0), 'spongebob': (905, 450, 2870, -5.0),
         'squidward': (1130, 720, 2520, 1.5)},
}
ORDER = ['squidward', 'patrick', 'spongebob']  # 后→前

# 泡泡带（x, y, r）：从海绵宝宝嘴边泡泡棒升起，掠过章鱼哥头顶，飘向派大星伸出的手。
BUBBLES = {
  'v1': [(560, 2160, 26), (640, 1850, 46), (800, 1560, 80), (950, 1290, 40), (1040, 1080, 20)],
  'v2': [(600, 2130, 24), (700, 1820, 42), (860, 1520, 74), (990, 1260, 38), (1050, 1060, 18)],
  'v3': [(580, 2120, 28), (660, 1800, 50), (830, 1500, 86), (980, 1230, 44), (1060, 1020, 22)],
}

# 样张式散点泡泡场：全画面大小泡（样张上半部泡泡很多，此前版本偏空）。避开角色区。
FIELD_BUBBLES = [
  (185, 480, 34), (560, 300, 22), (905, 420, 15), (1180, 560, 26), (1330, 940, 18),
  (95, 900, 24), (430, 760, 14), (760, 660, 30), (1080, 800, 16), (1265, 1300, 22),
  (150, 1320, 16), (520, 1180, 12), (1010, 1560, 14), (240, 1700, 26), (1300, 1750, 15),
  (120, 2260, 18), (1330, 2200, 20),
]


def clean_patrick_blotches(fig, radius=9, verbose=True):
    """把被绿色短裤（可含描边）围住的异色孤岛回填为短裤纯色。
    两道判定：1) 紧贴绿区的薄边缘；2) 不接触透明、且邻边多为绿/暗描边的封闭小块。"""
    arr = np.asarray(fig).astype(np.int32)
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    visible = A > 140
    green = visible & (G > R + 5) & (G > B + 30)
    dark = (R < 95) & (G < 95) & (B < 95)
    skin = (R > 190) & (G > 110) & (G < 210) & (B > 100) & (B < 200) & (R > G + 30)
    if green.sum() < 4000:
        if verbose: print('clean: not shorts green, skip')
        return fig
    # 1) 绿缘薄斑（抗锯齿红边等）：邻域绿占比高的薄边缘像素
    ii = np.zeros((A.shape[0] + 1, A.shape[1] + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(green.astype(np.float64), 0), 1)
    y0 = np.clip(np.arange(A.shape[0])[:, None] - radius, 0, None)
    y1 = np.clip(np.arange(A.shape[0])[:, None] + radius + 1, 0, A.shape[0])
    x0 = np.clip(np.arange(A.shape[1])[None, :] - radius, 0, None)
    x1 = np.clip(np.arange(A.shape[1])[None, :] + radius + 1, 0, A.shape[1])
    box = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
    area = (y1 - y0) * (x1 - x0)
    frac = box / np.maximum(area, 1)
    blot = visible & ~green & ~dark & ~skin & (frac > 0.33)
    # 2) 封闭孤岛：可见、非绿非皮肤；整块不接触透明；边界邻像素多为绿/暗描边
    cand = visible & ~green & ~skin & ~dark
    maxn = max(4000, int(0.10 * green.sum()))
    seen = np.zeros_like(cand)
    H_, W_ = cand.shape
    ys, xs = np.where(cand & ~blot)
    nbr = ((1, 0), (-1, 0), (0, 1), (0, -1))
    comp_total = 0
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        touches_alpha = False
        adj = adj_ok = 0
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy, dx in nbr:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H_ and 0 <= nx < W_:
                    if cand[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
                    elif not cand[ny, nx]:
                        adj += 1
                        if A[ny, nx] < 140:
                            touches_alpha = True
                        elif green[ny, nx]:
                            adj_ok += 1
        if len(comp) <= maxn and not touches_alpha and adj and adj_ok / adj >= 0.25:
            for cy, cx in comp:
                blot[cy, cx] = True
            comp_total += len(comp)
    n = int(blot.sum())
    if n == 0:
        if verbose: print('clean: no blot pixels')
        return fig
    if n > 0.03 * int(visible.sum()):
        if verbose: print('clean: blot share abnormal (%d), skip' % n)
        return fig
    gp = arr[green][:, :3]
    med = np.median(gp, axis=0).astype(np.uint8)
    flat = np.zeros_like(arr)
    flat[..., :3] = med
    flat[..., 3] = 255
    fill = Image.fromarray(flat.astype(np.uint8), 'RGBA').filter(ImageFilter.GaussianBlur(2))
    mmask = Image.fromarray((blot * 255).astype(np.uint8), 'L').filter(ImageFilter.MaxFilter(3))
    base = fig.copy()
    base = Image.composite(fill, base, mmask)
    if verbose: print('clean: filled %d px (islands %d) -> %s' % (n, comp_total, tuple(int(v) for v in med)))
    return base


def tighten_alpha(fig, floor=120, erode=2):
    a = np.asarray(fig.getchannel('A')).astype(np.float32)
    out = np.clip((a - floor) * (255.0 / (255 - floor)), 0, 255).astype('uint8')
    fig = fig.copy()
    fig.putalpha(Image.fromarray(out, 'L'))
    if erode > 0:
        fig.putalpha(fig.getchannel('A').filter(ImageFilter.MinFilter(erode * 2 + 1)))
    return fig


def paste_figure(canvas, fig, cx, ground_y, target_h, tilt=0.0):
    if tilt:
        fig = fig.rotate(tilt, expand=True, resample=Image.BICUBIC)
    r = target_h / fig.height
    fw, fh = max(1, int(fig.width * r)), target_h
    f = fig.resize((fw, fh), Image.LANCZOS)
    canvas.alpha_composite(f, (int(cx - fw / 2), int(ground_y - fh)))


def draw_bubbles(canvas, pts):
    d = ImageDraw.Draw(canvas)
    for x, y, r in pts:
        if r <= 12:  # 样张同款小实心白点
            d.ellipse([x - r, y - r, x + r, y + r], fill=(248, 250, 248, 225))
            continue
        # 样张同款线环泡泡：细描边 + 左上高光弧，内部透出底色
        w = max(4, r // 6)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 240, 245, 12),
                  outline=(250, 253, 250, 225), width=w)
        d.arc([x - r + r * 0.22, y - r + r * 0.22, x + r - r * 0.22, y + r - r * 0.22],
              start=135, end=225, fill=(255, 255, 255, 235), width=max(4, r // 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', required=True)
    ap.add_argument('--cutdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='v1')
    ap.add_argument('--no-clean', action='store_true')
    args = ap.parse_args()

    canvas = Image.open(args.bg).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    # 沙地上延（样张沙地更厚，角色脚部落在沙面上）：同参数椭圆加深，盖住原描边
    sd = ImageDraw.Draw(canvas)
    scx, scy, srx, sry = W * 0.5, 3200, 1180, 690
    sd.ellipse([scx - srx, scy - sry, scx + srx, scy + sry], fill=(216, 196, 140, 255),
               outline=(88, 104, 44, 255), width=12)
    # 背景层散点泡泡：先画在角色之后之前，避免环边蹭脏角色
    draw_bubbles(canvas, FIELD_BUBBLES)
    layout = dict(BASE)
    layout.update(VARIANTS.get(args.variant, {}))
    for role in ORDER:
        cands = sorted(glob.glob(os.path.join(args.cutdir, role + '*.png')), key=os.path.getmtime)
        if not cands:
            raise SystemExit('缺少抠图: ' + role + ' in ' + args.cutdir)
        fig = Image.open(cands[-1]).convert('RGBA')
        if role == 'patrick' and not args.no_clean:
            fig = clean_patrick_blotches(fig)
        fig = tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        h, cx, gy, tilt = layout[role]
        paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
        print('贴入', role, os.path.basename(cands[-1]), '->', h, 'tilt', tilt)
    draw_bubbles(canvas, BUBBLES[args.variant])
    canvas.convert('RGB').save(args.out)
    print('合成完成', canvas.size, '→', args.out)


if __name__ == '__main__':
    main()
