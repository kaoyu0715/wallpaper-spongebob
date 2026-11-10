# 八代合成：官方图案背景 + 场景表批量合成（配方见 delivery/recipe.md）。
# 单元后处理与场景规则全部沿用五~七代收敛管线：
#   - 派大星三件套：clean_patrick_blue → clean_patrick_blotches → fix_patrick_outline（顺序不可换）
#   - alpha 收紧(floor=115, erode=2) 去白 halo
#   - 贴入比 = 贴入高/裁后高 ≤ 1.0x（脚本打印核对，禁止放大贴入）
#   - 泡泡带（样张同款线环泡泡+小实心点）串视线；除泡泡外无道具
# 用法: python3 compose-round8.py --bg <bg.png> --scene <id>|all --outdir <目录>
import argparse, glob, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1440, 3200
CUT8 = '/mnt/g/workspace/comfyui/output/g8cut'
CUT7 = '/mnt/g/workspace/comfyui/output/g7cut'


def unit(key):
    if key in _UNITS:
        p = _UNITS[key]
        return p if os.path.isabs(p) else os.path.join(CUT8, p)
    cands = sorted(glob.glob(os.path.join(CUT8, key.replace('_', '-') + '*.png')), key=os.path.getmtime)
    if not cands:
        raise SystemExit('缺少抠图: ' + key)
    return cands[-1]


# 单元表：显式路径（已验收单元固定；新单元取 g8cut 最新）
_UNITS = {
  'sb_r':  os.path.join(CUT7, 'spongebob-c1_00001.png'),  # 七代胜出单元（朝右）
  'pat_l': os.path.join(CUT7, 'patrick-g5.png'),          # 五代胜出单元（朝左上）
  'sq':    os.path.join(CUT7, 'squidward-g5.png'),        # 五代胜出单元（正面叉腰）
}

# 场景表：chars 按 z 序 后→前：(unit, 贴入高, cx, ground_y, 倾角°)
# band: 泡泡带 (x,y,r)；None=不用泡泡带（靠视线与站位构成叙事）
SCENES = {
  # 海绵宝宝+派大星：追泡泡二人组（七代场景双角色版）
  'b01-sb-pat-bubbles': dict(
    title='追泡泡二人组',
    chars=[('sb_r', 800, 430, 2880, -3.0), ('pat_l', 980, 1035, 2830, 3.0)],
    band=[(560, 2160, 26), (640, 1850, 46), (800, 1560, 80), (950, 1290, 40), (1040, 1080, 20)]),
  # 海绵宝宝+章鱼哥+派大星：经典三人追泡泡（七代 v3 布局）
  'b02-sb-sq-pat-bubbles': dict(
    title='三人追泡泡',
    chars=[('sq', 1130, 720, 2520, 1.5), ('sb_r', 805, 455, 2870, -5.0),
           ('pat_l', 970, 1010, 2820, 5.0)],
    band=[(580, 2120, 28), (660, 1800, 50), (830, 1500, 86), (980, 1230, 44), (1060, 1020, 22)]),
  # 珊迪+海绵宝宝：隔空打招呼（对望）
  'b03-sandy-sb-hello': dict(
    title='珊迪与海绵宝宝打招呼',
    chars=[('sb_r', 790, 400, 2870, -3.0), ('sandy_l', 1000, 1050, 2790, 2.0)],
    band=[(600, 2260, 18), (700, 2050, 26), (820, 1830, 14)]),
  # 蟹老板+海绵宝宝：老板员工齐笑
  'b04-krabs-sb-laugh': dict(
    title='蟹老板与海绵宝宝',
    chars=[('krabs', 900, 1020, 2750, 0.0), ('sb_r', 790, 430, 2880, -3.0)],
    band=[(620, 2230, 20), (760, 2010, 30), (880, 1760, 16)]),
  # 小蜗+海绵宝宝：遛蜗牛（海绵宝宝俯视小蜗）
  'b05-gary-sb-pet': dict(
    title='小蜗和海绵宝宝',
    chars=[('sb_r', 800, 500, 2840, -3.0), ('gary', 330, 1020, 2960, 0.0)],
    band=[(700, 2300, 18), (830, 2100, 26)]),
  # 章鱼哥+海绵宝宝：泡泡落在章鱼哥头顶（五代笑点复用）
  'b06-sq-sb-bubbles': dict(
    title='章鱼哥的嫌弃',
    chars=[('sq', 1130, 820, 2600, 0.0), ('sb_r', 790, 400, 2880, -3.0)],
    band=[(560, 2180, 24), (650, 1870, 42), (790, 1560, 64), (880, 1400, 30)]),
  # 珊迪+海绵宝宝+派大星：珊迪后排欢呼
  'b07-sandy-sb-pat-cheer': dict(
    title='珊迪欢呼三人组',
    chars=[('sandy_c', 1010, 720, 2540, 0.0), ('sb_r', 800, 420, 2870, -4.0),
           ('pat_l', 970, 1030, 2820, 4.0)],
    band=[(560, 2150, 26), (660, 1830, 44), (820, 1540, 78), (960, 1280, 38), (1050, 1070, 20)],
    arcs=[(732, 1718, 208, (28, 26, 24), 5, 265, 325)]),
  # 蟹老板+海绵宝宝+派大星：蟹老板后排坐镇
  'b08-krabs-sb-pat': dict(
    title='蟹老板坐镇',
    chars=[('krabs', 900, 730, 2550, 0.0), ('sb_r', 795, 430, 2870, -4.0),
           ('pat_l', 965, 1020, 2820, 4.0)],
    band=[(570, 2140, 26), (670, 1820, 46), (830, 1530, 80), (970, 1270, 40), (1050, 1060, 20)]),
  # 小蜗+海绵宝宝+派大星：小蜗领队
  'b09-gary-sb-pat': dict(
    title='小蜗领队',
    chars=[('sb_r', 800, 470, 2810, -3.0), ('pat_l', 960, 1040, 2780, 4.0),
           ('gary', 330, 740, 2960, 0.0)],
    band=[(600, 2100, 22), (700, 1820, 36), (850, 1560, 58)]),
  # 珊迪+章鱼哥：热情对嫌弃（章鱼哥用八代新单元：龇牙叉腰）
  'b10-sandy-sq': dict(
    title='热情对嫌弃',
    chars=[('squidward-annoyed', 1120, 470, 2610, 0.0), ('sandy_l', 1000, 1040, 2790, 2.0)],
    band=[(700, 2240, 18), (820, 2000, 28), (940, 1760, 14)]),
  # 派大星+蟹老板：朝右版派大星
  'b11-pat-krabs': dict(
    title='派大星和蟹老板',
    chars=[('krabs', 890, 1010, 2760, 0.0), ('patrick-right', 960, 430, 2870, -3.0)],
    band=[(600, 2220, 20), (720, 1990, 30), (850, 1730, 16)]),
  # 珊迪+海绵宝宝+小蜗：全家福小场景
  'b12-sandy-sb-gary': dict(
    title='珊迪海绵宝宝和小蜗',
    chars=[('sandy_l', 1000, 1060, 2780, 2.0), ('sb_r', 790, 450, 2850, -3.0),
           ('gary', 320, 730, 2960, 0.0)],
    band=[(620, 2200, 20), (760, 1960, 32)]),
}

# 样张式散点泡泡场（背景层，先贴角色后被覆盖）；七代同款
FIELD_BUBBLES = [
  (185, 480, 34), (560, 300, 22), (905, 420, 15), (1180, 560, 26), (1330, 940, 18),
  (95, 900, 24), (430, 760, 14), (760, 660, 30), (1080, 800, 16), (1265, 1300, 22),
  (150, 1320, 16), (520, 1180, 12), (1010, 1560, 14), (240, 1700, 26), (1300, 1750, 15),
  (85, 2100, 18), (1330, 2200, 20),
]


def dilate(m, k=1):
    out = m.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def fix_patrick_outline(fig, depth=10, target=(20, 18, 16), verbose=True):
    arr = np.asarray(fig).astype(np.float64)
    A = arr[..., 3]
    R, G = arr[..., 0], arr[..., 1]
    vis = A > 10
    if vis.sum() == 0:
        return fig
    edge = vis & dilate(~vis, 1)
    band = dilate(edge, depth) & vis
    t1 = np.clip((R - G - 60) / 50.0, 0, 1)
    t2 = np.clip((140 - G) / 40.0, 0, 1)
    w = t1 * t2 * band
    n = int(((w > 0.3) & band).sum())
    if n < 2000:
        if verbose: print('outline: red stroke share small (%d), skip' % n)
        return fig
    wm = Image.fromarray((w * 255).astype('uint8'), 'L').filter(ImageFilter.GaussianBlur(1))
    w = np.asarray(wm).astype(np.float64) / 255.0
    out = arr.copy()
    for i, c in enumerate(target):
        out[..., i] = arr[..., i] * (1 - w) + c * w
    out[..., 3] = A
    if verbose: print('outline: recolored %d strong-red px -> %s' % (n, target))
    return Image.fromarray(np.round(out).astype('uint8'), 'RGBA')


def clean_patrick_blotches(fig, radius=9, verbose=True):
    arr = np.asarray(fig).astype(np.int32)
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    visible = A > 140
    green = visible & (G > R + 5) & (G > B + 30)
    dark = (R < 95) & (G < 95) & (B < 95)
    skin = (R > 190) & (G > 110) & (G < 210) & (B > 100) & (B < 200) & (R > G + 30)
    if green.sum() < 4000:
        return fig
    ii = np.zeros((A.shape[0] + 1, A.shape[1] + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(green.astype(np.float64), 0), 0 + 1)
    y0 = np.clip(np.arange(A.shape[0])[:, None] - radius, 0, None)
    y1 = np.clip(np.arange(A.shape[0])[:, None] + radius + 1, 0, A.shape[0])
    x0 = np.clip(np.arange(A.shape[1])[None, :] - radius, 0, None)
    x1 = np.clip(np.arange(A.shape[1])[None, :] + radius + 1, 0, A.shape[1])
    box = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
    area = (y1 - y0) * (x1 - x0)
    frac = box / np.maximum(area, 1)
    blot = visible & ~green & ~dark & ~skin & (frac > 0.33)
    cand = visible & ~green & ~skin & ~dark
    maxn = max(4000, int(0.10 * green.sum()))
    seen = np.zeros_like(cand)
    H_, W_ = cand.shape
    ys, xs = np.where(cand & ~blot)
    nbr = ((1, 0), (-1, 0), (0, 1), (0, -1))
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
    n = int(blot.sum())
    if n == 0:
        return fig
    if n > 0.03 * int(visible.sum()):
        if verbose: print('clean: blot share abnormal (%d), skip' % n)
        return fig
    gp = arr[green][:, :3]
    med = np.median(gp, axis=0).astype(np.uint8)
    flat = np.zeros_like(arr)
    flat[..., :3] = med
    flat[..., 3] = 255
    fill = Image.fromarray(flat.astype('uint8'), 'RGBA').filter(ImageFilter.GaussianBlur(2))
    mmask = Image.fromarray((blot * 255).astype('uint8'), 'L').filter(ImageFilter.MaxFilter(3))
    base = Image.composite(fill, fig.copy(), mmask)
    if verbose: print('clean: filled %d px -> %s' % (n, tuple(int(v) for v in med)))
    return base


def clean_patrick_blue(fig, verbose=True):
    arr = np.asarray(fig).astype(np.int32)
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    visible = A > 140
    green = visible & (G > R + 5) & (G > B + 30)
    blue = visible & (B > R + 30) & (B > G + 20)
    if green.sum() < 4000 or blue.sum() == 0:
        return fig
    keep = np.zeros_like(blue)
    keep_band = np.zeros_like(blue)
    seen = np.zeros_like(blue)
    H_, W_ = blue.shape
    ys, xs = np.where(blue)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H_ and 0 <= nx < W_ and blue[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > 25000:
            continue
        cmask = np.zeros_like(blue)
        cmask[[p[0] for p in comp], [p[1] for p in comp]] = True
        near_green = int((green & dilate(cmask, 10) & ~cmask).sum())
        if near_green >= 300:
            keep |= cmask
            if not (dilate(cmask, 4) & (A < 140)).any():
                keep_band |= cmask
    n = int(keep.sum())
    if n == 0:
        return fig
    dark = visible & (R < 95) & (G < 95) & (B < 95)
    alpha_edge = dilate(A < 140, 1) & visible
    ring = dark & dilate(keep_band, 2) & ~alpha_edge
    keep = keep | ring
    gm = np.median(arr[green][:, :3], axis=0)
    band = dilate(keep_band, 6) & visible & ~alpha_edge
    dev = np.abs(arr[..., :3] - gm[None, None, :]).sum(-1) > 45
    keep = keep | (band & dev)
    n = int(keep.sum())
    panel_zone = green & dilate(keep, 16) & ~dilate(keep, 5)
    panel = np.median(arr[panel_zone][:, :3], axis=0) if int(panel_zone.sum()) >= 500 \
        else np.median(arr[green][:, :3], axis=0)
    ys_, xs_ = np.where(keep)
    y0, y1 = max(ys_.min() - 20, 0), min(ys_.max() + 21, arr.shape[0])
    x0, x1 = max(xs_.min() - 20, 0), min(xs_.max() + 21, arr.shape[1])
    sub = arr[y0:y1, x0:x1, :3].astype(np.float64)
    m = keep[y0:y1, x0:x1]
    sub[m] = panel
    lam = 0.03
    for _ in range(2500):
        upd = (np.roll(sub, 1, 0) + np.roll(sub, -1, 0) +
               np.roll(sub, 1, 1) + np.roll(sub, -1, 1)) / 4.0
        sub[m] = (upd[m] + lam * panel[None, :]) / (1.0 + lam)
    out = np.asarray(fig).copy()
    out[y0:y1, x0:x1, :3] = np.clip(sub, 0, 255).round().astype(np.uint8)
    if verbose: print('blue: inpainted %d px (panel %s)' % (n, tuple(int(v) for v in panel)))
    return Image.fromarray(out, 'RGBA')


def tighten_alpha(fig, floor=115, erode=2):
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
        if r <= 12:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(248, 250, 248, 225))
            continue
        w = max(4, r // 6)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 240, 245, 12),
                  outline=(250, 253, 250, 225), width=w)
        d.arc([x - r + r * 0.22, y - r + r * 0.22, x + r - r * 0.22, y + r - r * 0.22],
              start=135, end=225, fill=(255, 255, 255, 235), width=max(4, r // 8))


def compose_scene(scene_id, bg_path, out_path):
    sc = SCENES[scene_id]
    canvas = Image.open(bg_path).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    sd = ImageDraw.Draw(canvas)
    scx, scy, srx, sry = W * 0.5, 3200, 1180, 690  # 沙地上延（角色脚落沙面）
    sd.ellipse([scx - srx, scy - sry, scx + srx, scy + sry], fill=(216, 196, 140, 255),
               outline=(88, 104, 44, 255), width=12)
    draw_bubbles(canvas, FIELD_BUBBLES)
    report = []
    for key, h, cx, gy, tilt in sc['chars']:
        path = unit(key)
        fig = Image.open(path).convert('RGBA')
        is_pat = key.startswith('pat')
        if is_pat:
            fig = clean_patrick_blue(fig)
            fig = clean_patrick_blotches(fig)
            fig = fix_patrick_outline(fig)
        fig = tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        ratio = h / fig.height
        report.append((key, os.path.basename(path), h, fig.height, round(ratio, 3)))
        paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
        print('贴入', scene_id, key, os.path.basename(path), 'h=%d trim=%d scale=%.2fx'
              % (h, fig.height, ratio))
    for arc in sc.get('arcs', []):
        x, y, r, col, w = arc[:5]
        a0, a1 = (arc[5], arc[6]) if len(arc) > 5 else (0, 360)
        ImageDraw.Draw(canvas).arc([x - r, y - r, x + r, y + r], start=a0, end=a1,
                                   fill=col + (255,), width=w)
    if sc.get('band'):
        draw_bubbles(canvas, sc['band'])
    canvas.convert('RGB').save(out_path)
    print('合成完成', scene_id, canvas.size, '→', out_path)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', required=True)
    ap.add_argument('--scene', default='all')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ids = sorted(SCENES) if args.scene == 'all' else [args.scene]
    for sid in ids:
        compose_scene(sid, args.bg, os.path.join(args.outdir, 'r8-%s.png' % sid))


if __name__ == '__main__':
    main()
