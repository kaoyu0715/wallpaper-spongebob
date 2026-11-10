# 六代合成：延续五代「追泡泡」v3 布局与官方图案背景，仅两处改动：
#   1) 新增派大星外轮廓定向回色 fix_patrick_outline：
#      沿 alpha 外缘 depth 像素内，把「红描边族」颜色（高 R-G、低 G）按权重回正为均匀深色 (20,18,16)；
#      皮肤粉（G≥160）与嘴内深红（在深度带外）不受影响。
#   2) 海绵宝宝抠图改为本轮新种子（手势吹泡泡或合规泡泡棒），章鱼哥/派大星形象沿用五代。
# 沿用五代短裤红斑孤岛清理。
# 用法: python3 compose-round6.py --bg <bg.png> --cutdir <g6cut目录> --out <out.png> \
#         --variant v1|v2|v3 [--no-clean]
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1440, 3200

# (高px, cx, ground_y, 倾角°) —— 与五代 compose 完全一致（追泡泡：后排章鱼哥叉腰嫌弃，
# 前排左海绵宝宝吹泡泡指向派大星，前排右派大星伸手接泡泡）
# 海绵宝宝新形象（seed 1462）头身比更大，同布局等比缩小到 ~760 高以保持官方三人比例
BASE = {
  'squidward': (1120, 745, 2610, 0.0),
  'patrick':   (980, 1035, 2830, 3.0),
  'spongebob': (760, 430, 2880, -3.0),
}
VARIANTS = {
  'v1': {},
  'v2': {'patrick': (990, 965, 2810, 4.0), 'spongebob': (770, 480, 2860, -4.0),
         'squidward': (1150, 735, 2560, 0.0)},
  'v3': {'patrick': (970, 1010, 2820, 5.0), 'spongebob': (765, 455, 2870, -5.0),
         'squidward': (1130, 720, 2520, 1.5)},
}
ORDER = ['squidward', 'spongebob', 'patrick']  # 后→前：海绵宝宝新姿势的指尖伸向派大星，
# 让派大星压在上层，手指从其身后指出，避免两手在躯干上打架（五代 SB 无此伸展故在前的顺序可用）

# 泡泡带（x, y, r）：从海绵宝宝嘴边升起，掠过章鱼哥头顶，飘向派大星伸出的手。
BUBBLES = {
  'v1': [(560, 2160, 26), (640, 1850, 46), (800, 1560, 80), (950, 1290, 40), (1040, 1080, 20)],
  'v2': [(600, 2130, 24), (700, 1820, 42), (860, 1520, 74), (990, 1260, 38), (1050, 1060, 18)],
  'v3': [(580, 2120, 28), (660, 1800, 50), (830, 1500, 86), (980, 1230, 44), (1060, 1020, 22)],
}

# 样张式散点泡泡场：全画面大小泡（样张上半部泡泡很多）。避开角色区。
FIELD_BUBBLES = [
  (185, 480, 34), (560, 300, 22), (905, 420, 15), (1180, 560, 26), (1330, 940, 18),
  (95, 900, 24), (430, 760, 14), (760, 660, 30), (1080, 800, 16), (1265, 1300, 22),
  (150, 1320, 16), (520, 1180, 12), (1010, 1560, 14), (240, 1700, 26), (1300, 1750, 15),
  (120, 2260, 18), (1330, 2200, 20),
]


def dilate(m, k=1):
    out = m.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def fix_patrick_outline(fig, depth=10, target=(20, 18, 16), verbose=True):
    """外轮廓红描边 → 均匀深色。权重 = 红度(R-G) × 低绿度(G)：
    描边核心 (227,85,70) 全量回色；皮肤粉 (243,167,155) 权重 0 不动；
    描边与皮肤之间的 AA 像素按红度部分回色。仅限外缘 depth 像素内，
    嘴内深红与身体内部线条（在带外）保持不变。"""
    arr = np.asarray(fig).astype(np.float64)
    A = arr[..., 3]
    R, G = arr[..., 0], arr[..., 1]
    vis = A > 10
    if vis.sum() == 0:
        if verbose: print('outline: empty, skip')
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
    """把被绿色短裤（可含描边）围住的异色孤岛回填为短裤纯色（五代同款）。"""
    arr = np.asarray(fig).astype(np.int32)
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    visible = A > 140
    green = visible & (G > R + 5) & (G > B + 30)
    dark = (R < 95) & (G < 95) & (B < 95)
    skin = (R > 190) & (G > 110) & (G < 210) & (B > 100) & (B < 200) & (R > G + 30)
    if green.sum() < 4000:
        if verbose: print('clean: not shorts green, skip')
        return fig
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


def clean_patrick_blue(fig, verbose=True):
    """短裤上的蓝色花形图案（Patrick_il 属性泄漏，>4000px 逃出五代孤岛上限）：
    封闭、不接触透明、且绿邻接占比 ≥0.5 的蓝色连通域 → 回填短裤中位色。
    等效五代 PR 中的人工定向矩形清理，此处固化入管线。"""
    arr = np.asarray(fig).astype(np.int32)
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    visible = A > 140
    green = visible & (G > R + 5) & (G > B + 30)
    blue = visible & (B > R + 30) & (B > G + 20)
    if green.sum() < 4000 or blue.sum() == 0:
        if verbose: print('blue: none, skip')
        return fig
    keep = np.zeros_like(blue)
    keep_band = np.zeros_like(blue)
    seen = np.zeros_like(blue)
    total = 0
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
        # 背景花形/彩条泄漏嵌在短裤上：以 10px 邻域内的绿色像素数量判定（花形实测 >2000，
        # 右侧接缝彩条约 374）；蓝色被自身描边/接缝暗线隔离，故须 10px 半径
        cmask = np.zeros_like(blue)
        cmask[[p[0] for p in comp], [p[1] for p in comp]] = True
        near_green = int((green & dilate(cmask, 10) & ~cmask).sum())
        if near_green >= 300:
            keep |= cmask
            total += len(comp)
            # 贴近轮廓的彩条（如右侧接缝条）只做蓝色本体的回填，
            # 不吃暗接缝线/AA 圈，避免冲淡描边
            if not (dilate(cmask, 4) & (A < 140)).any():
                keep_band |= cmask
    n = int(keep.sum())
    if n == 0:
        if verbose: print('blue: no enclosed blue islands')
        return fig
    # 连同花形自身的黑色描边圈（可见、暗、紧贴蓝岛、不接触透明）一并回填，贴回五代纯色短裤
    dark = visible & (R < 95) & (G < 95) & (B < 95)
    alpha_edge = dilate(A < 140, 1) & visible
    ring = dark & dilate(keep_band, 2) & ~alpha_edge
    keep = keep | ring
    # 花形外围的暗橄榄绿 AA 圈（中位 ~(74,128,75)/(104,158,75)，与短裤中位色偏差大）一并回填；
    # 只吃非绿像素（dev>45），真绿永不被覆盖；带宽 6px，够吃掉最外圈 AA
    gm = np.median(arr[green][:, :3], axis=0)
    band = dilate(keep_band, 6) & visible & ~alpha_edge
    dev = np.abs(arr[..., :3] - gm[None, None, :]).sum(-1) > 45
    keep = keep | (band & dev)
    n = int(keep.sum())
    # 调和扩散 inpaint（无 cv2/skimage）：内部向局部面板色（花形外围 5–16px 环带绿色中位，
    # 比全局中位更贴合该处明暗）收敛 + 边界邻域值混合，2500 轮 Jacobi（花形深 ~40px）
    # 结果保留短裤渐变、无平色补丁"鬼影"
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
    base = Image.fromarray(out, 'RGBA')
    if verbose: print('blue: inpainted %d px (panel %s)' % (n, tuple(int(v) for v in panel)))
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
    ap.add_argument('--g5ref', default=None,
                    help='五代成图路径；派大星沿用五代且粘贴参数一致，直接复用其已清理的短裤花形区域（椭圆羽化）')
    args = ap.parse_args()

    canvas = Image.open(args.bg).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    # 沙地上延（样张沙地更厚，角色脚部落在沙面上）
    sd = ImageDraw.Draw(canvas)
    scx, scy, srx, sry = W * 0.5, 3200, 1180, 690
    sd.ellipse([scx - srx, scy - sry, scx + srx, scy + sry], fill=(216, 196, 140, 255),
               outline=(88, 104, 44, 255), width=12)
    draw_bubbles(canvas, FIELD_BUBBLES)
    layout = dict(BASE)
    layout.update(VARIANTS.get(args.variant, {}))
    for role in ORDER:
        cands = sorted(glob.glob(os.path.join(args.cutdir, role + '*.png')), key=os.path.getmtime)
        if not cands:
            raise SystemExit('缺少抠图: ' + role + ' in ' + args.cutdir)
        fig = Image.open(cands[-1]).convert('RGBA')
        if role == 'patrick':
            if not args.no_clean:
                # 先蓝后斑：blue 把花形+暗描边圈+橄榄 AA 圈整体回填；若先跑 blotches，
                # 它会级联填掉蓝花但留下橄榄圈，且 blue 此时看不到蓝岛而空转
                fig = clean_patrick_blue(fig)
                fig = clean_patrick_blotches(fig)
            fig = fix_patrick_outline(fig)
        fig = tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        h, cx, gy, tilt = layout[role]
        paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
        if role == 'patrick' and args.g5ref:
            # 复用五代成图对应区域（同 h/cx/gy/tilt → 像素级对位；椭圆避开左缘轮廓与肚脐线）
            ref = Image.open(args.g5ref).convert('RGBA')
            px0, py0, px1, py1 = 866, 2530, 1080, 2660
            patch = ref.crop((px0, py0, px1, py1))
            pmask = Image.new('L', patch.size, 0)
            pd = ImageDraw.Draw(pmask)
            ecx, ecy = (px0 + px1) // 2, (py0 + py1) // 2
            pd.ellipse([ecx - 100, ecy - 58, ecx + 100, ecy + 58], fill=255)
            pmask = pmask.filter(ImageFilter.GaussianBlur(6))
            canvas.paste(patch, (px0, py0), pmask)
            print('  复用五代花形清理区域', (px0, py0, px1, py1))
        print('贴入', role, os.path.basename(cands[-1]), '->', h, 'tilt', tilt)
    draw_bubbles(canvas, BUBBLES[args.variant])
    canvas.convert('RGB').save(args.out)
    print('合成完成', canvas.size, '→', args.out)


if __name__ == '__main__':
    main()
