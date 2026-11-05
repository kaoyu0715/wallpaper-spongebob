# 十一代合成（issue #23：以样张为唯一标准自进化逼近）。
# 相对十代的四项变化（对症差距清单 G1/G2/G4/G5/G3，见 delivery/batch/gap-analysis-round11.md）：
#   G1 场景表重写：三簇心收拢 + 后排上站台（gy 2470–2560）+ 身体边缘自然叠压 60–160px
#      —— 群像感来自"堆叠"而非"并列"；只叠体侧与手臂，主要肢体保持可辨（C3 不违）。
#   G2 沙地由通栏横带（rx1180）改回样张的"聚拢沙岛"（rx520/ry620，顶缘≈2500，两侧露深绿）。
#   G4 场泡与泡泡带改样张档：数量减、半径升（大圆环 25–90px + 弧高光），聚于角色簇上方。
#   G5 珊迪盔圈左上方程序绘制样张同款粉花（角色标志装饰，非道具；--no-flower 可关）。
#   G3 海绵宝宝单元过 sb_palette 调色（--no-sb-palette 可关，供 A/B）。
# 尺寸档/TIER、硬闸A（蟹老板≤0.85x）、硬闸B（珊迪盔圈裁切）、C4 皮肤必检、TOP_FLOOR(50%)
#   全部沿用十代 compose-round10 同一实现口径。
# 用法: python3 compose-round11.py --bg <bg.png> --scene <id>|all --outdir <目录>
#       [--no-sb-palette] [--no-flower] [--sandy-new]
import argparse, json, os, sys
import importlib.util
import numpy as np
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location(
    'compose_round10', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compose-round10.py'))
c10 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['compose-round10.py']
spec.loader.exec_module(c10)
sys.argv = _argv

m9 = c10.m9
m = c10.m
ps = c10.ps
W, H = c10.W, c10.H
import sb_palette

# 样张档散点场泡（G4）：数量少、直径大、聚在角色簇上方两侧
FIELD11 = [
  (300, 430, 58), (980, 330, 72), (1190, 640, 40), (150, 780, 34), (560, 290, 26),
  (1260, 1150, 64), (215, 1230, 44), (1085, 1425, 30), (1240, 1760, 22), (430, 905, 20),
]

# G5：珊迪盔圈粉花（相对贴入框位置 dx,dy 按贴入高 h 的比例；R 为花半径比例）
# 花坐在盔圈 10–11 点钟弧点上（样张位置）：盔圈圆心 ≈(cx, top+0.25~0.28h)、半径≈0.25h，
# 135° 点 = (cx-0.18h, top+0.075~0.10h)。
FLOWER_ANCHOR = {'sandy_c': (-0.18, 0.100, 0.085), 'sandy_l': (-0.18, 0.075, 0.080)}
PINK = (219, 116, 173)
PINK_EDGE = (158, 66, 122)
ORANGE = (238, 146, 64)


def draw_helmet_flower(canvas, cx, top, h, key):
    fx, fy, fr = FLOWER_ANCHOR[key]
    x, y, R = cx + fx * h, top + fy * h, fr * h
    d = ImageDraw.Draw(canvas)
    import math
    for i in range(5):
        t = -math.pi / 2 + i * 2 * math.pi / 5
        px, py = x + R * 0.62 * math.cos(t), y + R * 0.62 * math.sin(t)
        d.ellipse([px - R * 0.55, py - R * 0.55, px + R * 0.55, py + R * 0.55],
                  fill=PINK + (255,), outline=PINK_EDGE + (255,), width=max(3, int(R * 0.16)))
    d.ellipse([x - R * 0.34, y - R * 0.34, x + R * 0.34, y + R * 0.34],
              fill=ORANGE + (255,), outline=PINK_EDGE + (255,), width=max(2, int(R * 0.12)))


def clean_sandy_ball(fig, verbose=True):
    # 十一代珊迪新单元（g11cut）种子带亮红小球：cheer11 悬在盔圈外、left11 骑在盔圈线上。
    # 处理（局部传播、无需拟合）：亮红小孤岛中，与盔圈淡色像素相邻的部分按最近环带像素逐层
    # 回填（盔圈保持闭合，硬闸B），其余透明化。红球判据收紧为高饱和正红小连通块，
    # 不误伤舌头/口腔粉。
    arr = np.asarray(fig.convert('RGBA')).astype(np.int16)
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    hh = arr.shape[0]
    vis = A > 40
    red = vis & (R > 190) & (G < 95) & (B < 95) & (np.arange(hh)[:, None] < hh * 0.35)
    comps = [c for c in _components(red) if 40 < len(c) < 9000]
    if not comps:
        if verbose:
            print('sandy_ball: 无红球')
        return fig
    pale = vis & (R > 170) & (G > 170) & (B > 180) & (np.abs(R - B) < 70)
    out = arr.copy()
    color = out[..., :3].copy()
    n_fix = n_clear = 0
    for comp in comps:
        ball = np.zeros(fig.size[::-1], bool)
        for yy, xx in comp:
            ball[yy, xx] = True
        assigned = np.zeros_like(ball)
        cur = _dilate_np(ball, 1) & pale & ~ball   # 与球相邻的盔圈环带种子
        for _ in range(7):   # ≈盔圈线宽，回填限制在环带内，防止误涂球冠
            nxt = _dilate_np(cur, 1) & ball & ~assigned
            if not nxt.any():
                break
            for yy, xx in zip(*np.nonzero(nxt)):
                best = None; bd = 9
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = yy + dy, xx + dx
                        if cur[ny, nx]:
                            d = dy * dy + dx * dx
                            if d < bd:
                                bd = d; best = (ny, nx)
                if best is not None:
                    color[yy, xx] = color[best]
            assigned |= nxt
            cur = nxt | cur
            n_fix += int(nxt.sum())
        rest = ball & ~assigned
        ys, xs = np.nonzero(rest)
        out[ys, xs, 3] = 0
        n_clear += int(rest.sum())
    out[..., :3] = color
    if verbose:
        print('sandy_ball: 红球 %d 个，盔圈回填 %d px，透明化 %d px' % (len(comps), n_fix, n_clear))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA')


def _dilate_np(mask, k=1):
    out = mask.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def _components(mask, maxn=60000):
    from collections import deque
    H_, W_ = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        q = deque([(sy, sx)]); seen[sy, sx] = True; comp = []
        while q and len(comp) < maxn:
            yy, xx = q.popleft(); comp.append((yy, xx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < H_ and 0 <= nx < W_ and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; q.append((ny, nx))
        comps.append(comp)
    return comps


# 十一代场景表（G1）：簇心收拢、后排上站台、边缘叠压；后排先贴、前排后贴。
# 视线链与十代同：sb_r 朝右、pat_l 朝左、sandy_l 朝左、patrick_r 朝右、正面角色居中/对视。
SCENES11 = {
  'b01-sb-pat-bubbles': dict(
    title='追泡泡二人组',
    chars=[('pat_l', 745, 860, 2770, 4.0), ('sb_r', 615, 560, 2870, -3.0)],
    band=[(700, 2330, 30), (790, 2060, 52), (860, 1790, 78), (900, 1540, 42), (925, 1390, 24)]),
  'b02-sb-sq-pat-bubbles': dict(
    title='三人追泡泡',
    chars=[('sq_front', 865, 700, 2520, 1.5), ('pat_l', 745, 930, 2790, 5.0),
           ('sb_r', 615, 500, 2870, -5.0)],
    band=[(660, 2280, 32), (760, 2010, 56), (840, 1730, 84), (895, 1470, 46), (930, 1320, 26)]),
  'b03-sandy-sb-hello': dict(
    title='珊迪与海绵宝宝打招呼',
    chars=[('sandy_l', 765, 880, 2700, 2.0), ('sb_r', 610, 560, 2870, -3.0)],
    band=[(640, 2430, 24), (620, 2210, 40), (600, 1980, 58), (640, 1760, 34)]),
  'b04-krabs-sb-laugh': dict(
    title='蟹老板与海绵宝宝',
    chars=[('krabs', 780, 860, 2700, 0.0), ('sb_r', 610, 540, 2875, -3.0)],
    band=[(680, 2320, 24), (770, 2100, 44), (850, 1880, 62), (900, 1660, 32)]),
  'b05-gary-sb-pet': dict(
    title='小蜗和海绵宝宝',
    chars=[('sb_r', 615, 560, 2850, -3.0), ('gary', 245, 800, 2950, 0.0)],
    band=[(740, 2300, 24), (840, 2080, 44), (920, 1860, 60)]),
  'b06-sq-sb-bubbles': dict(
    title='章鱼哥的嫌弃',
    chars=[('sq_front', 865, 800, 2530, 0.0), ('sb_r', 610, 520, 2870, -4.0)],
    band=[(600, 2290, 28), (690, 2040, 50), (780, 1790, 76), (830, 1600, 36)]),
  'b07-sandy-sb-pat-cheer': dict(
    title='珊迪欢呼三人组',
    chars=[('sandy_c', 760, 740, 2500, 0.0), ('pat_l', 745, 940, 2790, 4.0),
           ('sb_r', 615, 520, 2860, -4.0)],
    band=[(660, 2420, 28), (470, 2150, 46), (420, 1900, 70), (460, 1640, 40), (510, 1470, 22)]),
  'b08-krabs-sb-pat': dict(
    title='蟹老板坐镇',
    chars=[('krabs', 780, 730, 2500, 0.0), ('pat_l', 745, 930, 2790, 4.0),
           ('sb_r', 610, 510, 2865, -4.0)],
    band=[(620, 2270, 30), (710, 2010, 56), (800, 1740, 84), (865, 1490, 44), (900, 1340, 24)]),
  'b09-gary-sb-pat': dict(
    title='小蜗领队',
    chars=[('pat_l', 740, 880, 2760, 4.0), ('sb_r', 615, 520, 2855, -3.0),
           ('gary', 245, 690, 2965, 0.0)],
    band=[(640, 2230, 26), (740, 1990, 48), (840, 1750, 70), (895, 1560, 34)]),
  'b10-sandy-sq': dict(
    title='热情对嫌弃',
    chars=[('sq_annoyed', 855, 560, 2530, 0.0), ('sandy_l', 765, 860, 2790, 2.0)],
    band=[(620, 2350, 24), (560, 2130, 44), (540, 1900, 60), (575, 1690, 32)]),
  'b11-pat-krabs': dict(
    title='派大星和蟹老板',
    chars=[('krabs', 780, 880, 2720, 0.0), ('patrick_r', 730, 560, 2865, -3.0)],
    band=[(660, 2330, 26), (760, 2110, 48), (850, 1890, 64), (905, 1680, 32)]),
  'b12-sandy-sb-gary': dict(
    title='珊迪海绵宝宝和小蜗',
    chars=[('sandy_l', 765, 860, 2700, 2.0), ('sb_r', 610, 520, 2860, -3.0),
           ('gary', 240, 740, 2955, 0.0)],
    band=[(590, 2330, 24), (500, 2100, 46), (490, 1880, 64), (530, 1660, 34)]),
}

SB_KEYS = ('sb_r',)
UNITS11 = dict(m9._UNITS9)   # --sandy-new 时替换为十一代珊迪单元


def draw_bubbles11(canvas, pts):
    # 样张档泡泡：透明描边圆环 + 左上弧高光。既有 draw_bubbles 在 RGBA 画布上直接绘制，
    # PIL 是替换像素而非 alpha 混合——内部低透明填充会变成不透明浅盘（十一代大泡泡下暴露）。
    # 本函数画在独立 overlay 层再 alpha_composite，泡泡内部真正透出背景。
    ov = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for x, y, r in pts:
        if r <= 12:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(248, 250, 248, 225))
            continue
        w = max(4, r // 6)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 240, 245, 16),
                  outline=(250, 253, 250, 225), width=w)
        d.arc([x - r + r * 0.22, y - r + r * 0.22, x + r - r * 0.22, y + r - r * 0.22],
              start=135, end=225, fill=(255, 255, 255, 235), width=max(4, r // 8))
    canvas.alpha_composite(ov)


def patrick_postclean(canvas, crop_box, ratio, pmask=None, verbose=True):
    # 十一代 C4 补充（对症 b07）：贴入缩放 LANCZOS 后，派大星描边与"白色衣物"交界处会产生
    # 4–6px 的红色 AA 小岛——被皮肤围合、但不贴黑描线（detect 的免检条件是贴黑线），计为残留。
    # 另一种假阳性（十一代实测）：裁切框罩住珊迪舌头时，口腔红碎块被粉色舌面围合、同样命中
    # 判据。两类统一处理：检测一律限定在**派大星贴入 footprint 掩膜**内（pmask），检出即以其
    # 环带皮肤中位色回填，复检为 0 才放行。固定流程可复现。
    crop = canvas.crop(crop_box).convert('RGB')

    def _det(img):
        found = ps.detect_patrick_specks(img, paste_ratio=ratio, verbose=False)
        if pmask is not None:
            found = [f for f in found if pmask[f[1], f[2]]]
        return found

    found = _det(crop)
    n = 0
    while found:
        a = np.asarray(crop).copy()
        for _area, cy, cx in found:
            ys0, ys1 = max(0, cy - 6), min(a.shape[0], cy + 7)
            xs0, xs1 = max(0, cx - 6), min(a.shape[1], cx + 7)
            win = a[ys0:ys1, xs0:xs1]
            c = win.astype(np.int32)
            skinm = (c[..., 0] > 170) & (c[..., 1] > 90) & (c[..., 1] < 200) & (c[..., 2] > 90) & (c[..., 2] < 210) & ((c[..., 0] - c[..., 1]) > 25)
            ring = win[skinm]
            if len(ring) < 3:
                continue
            med = np.median(ring, 0)
            # 岛 = 窗口内高红度且非皮肤的像素（与 detect 判据同族）
            redm = (c[..., 0] > 150) & ((c[..., 0] - c[..., 1]) > 55) & (c[..., 1] < 140) & ~skinm
            ays, axs = np.nonzero(redm)
            a[ys0 + ays, xs0 + axs] = med.astype(np.uint8)
        crop = Image.fromarray(a, 'RGB')
        n += len(found)
        found = _det(crop)
        if n > 60:
            break
    if n and verbose:
        print('patrick_postclean: 成品级回填红岛 %d 个，复检 %d' % (n, len(found)))
    canvas.paste(crop.convert('RGBA'), crop_box)
    return len(found)


def compose_scene(scene_id, bg_path, out_path, audit_dir, sb_palette_on=True, flower_on=True,
                  sandy_new=False):
    sc = SCENES11[scene_id]
    canvas = Image.open(bg_path).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    sd = ImageDraw.Draw(canvas)
    # G2：样张同款"聚拢沙岛"（扁平椭圆岛：顶缘≈2500、完整下缘、两侧不到画缘）
    scx, scy, srx, sry = W * 0.5, 2830, 520, 330
    sd.ellipse([scx - srx, scy - sry, scx + srx, scy + sry], fill=(216, 196, 140, 255),
               outline=(88, 104, 44, 255), width=12)
    draw_bubbles11(canvas, FIELD11)
    report = []
    for key, h, cx, gy, tilt in sc['chars']:
        fam = m9.tier_of(key)
        lo, hi = m9.TIER[fam]
        if not (lo <= h <= hi):
            raise SystemExit('硬闸 尺寸档: %s 贴入高 %d 不在 §4.2 新档 [%d,%d]' % (key, h, lo, hi))
        path = UNITS11[key] if not sandy_new else UNITS11.get('new_' + key, UNITS11[key])
        fig = Image.open(path).convert('RGBA')
        if key in m9.PATRICK_KEYS:
            fig = m.clean_patrick_blue(fig)
            fig = m.clean_patrick_blotches(fig)
            fig = ps.clean_patrick_skin(fig)
            fig = m.fix_patrick_outline(fig)
        if sb_palette_on and key in SB_KEYS:
            fig = sb_palette.sb_palette(fig)
        if 'g11cut' in path:
            fig = clean_sandy_ball(fig)
        fig = m9.drop_stray_specks(fig)
        fig = m.tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        ratio = h / fig.height
        if key.startswith('krabs') and ratio > m9.KRABS_MAX_RATIO + 1e-6:
            raise SystemExit('硬闸A 蟹老板贴入比 %.3fx > %.2fx ——拒绝合成' % (ratio, m9.KRABS_MAX_RATIO))
        if ratio > 1.0:
            raise SystemExit('C3 贴入比 %.3fx > 1.0x（%s）' % (ratio, key))
        top = gy - h
        if top < m9.TOP_FLOOR:
            raise SystemExit('尺寸自检: %s 最高点 y=%d 高于 %.0f 线' % (key, top, m9.TOP_FLOOR))
        report.append(dict(scene=scene_id, key=key, unit=os.path.basename(path),
                           paste_h=h, trim_h=fig.height, ratio=round(ratio, 3), top=top))
        m.paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
        print('贴入', scene_id, key, os.path.basename(path), 'h=%d trim=%d scale=%.2fx top=%d'
              % (h, fig.height, ratio, top))
        if key in FLOWER_ANCHOR and flower_on:
            draw_helmet_flower(canvas, cx, top, h, key)
            print('盔圈粉花 →', key)
        if key in ('sandy_l', 'sandy_c'):
            os.makedirs(audit_dir, exist_ok=True)
            hx0, hx1 = max(0, int(cx - h * 0.75)), min(W, int(cx + h * 0.75))
            hy0, hy1 = max(0, int(gy - h * 1.10)), int(gy - h * 0.30)
            canvas.crop((hx0, hy0, hx1, hy1)).save(
                os.path.join(audit_dir, '%s-%s-helmet.png' % (scene_id, key)))
        if key in m9.PATRICK_KEYS:
            # 皮肤必检口径与十代一致：贴入后、泡泡/道具绘制前（泡泡 AA 不算皮肤红点）。
            # 检测限定在派大星贴入 footprint 掩膜内（排除裁切框罩住的珊迪舌头等假阳性）；
            # 成品级回填处理贴入缩放产生的描边×白色衣物交界红岛（R4）。
            os.makedirs(audit_dir, exist_ok=True)
            px0, px1 = max(0, int(cx - h * 0.45)), min(W, int(cx + h * 0.45))
            py0, py1 = max(0, int(gy - h)), int(gy)
            crop_box = (px0, py0, px1, py1)
            mc = np.asarray(fig)[..., 3] > 10
            mimg = Image.fromarray((mc * 255).astype(np.uint8), 'L')
            if tilt:
                mimg = mimg.rotate(tilt, expand=True, resample=Image.BILINEAR)
            r_ = h / mimg.height
            fw_, fh_ = max(1, int(mimg.width * r_)), h
            mimg = mimg.resize((fw_, fh_), Image.NEAREST)
            fullmask = Image.new('L', canvas.size, 0)
            fullmask.paste(mimg, (int(cx - fw_ / 2), int(gy - fh_)))
            pm = np.asarray(fullmask.crop(crop_box)) > 100
            pm = c10.ps._dilate(pm, 2) if hasattr(c10, 'ps') else _dilate_np(pm, 2)
            n = patrick_postclean(canvas, crop_box, ratio, pmask=pm)
            canvas.crop(crop_box).save(os.path.join(audit_dir, '%s-%s-skin.png' % (scene_id, key)))
            print('皮肤必检 %s → 残留红点 %d 个' % (scene_id, n))
            report[-1]['skin_specks'] = n
    for arc in sc.get('arcs', []):
        x, y, r, col, w = arc[:5]
        a0, a1 = (arc[5], arc[6]) if len(arc) > 5 else (0, 360)
        ImageDraw.Draw(canvas).arc([x - r, y - r, x + r, y + r], start=a0, end=a1,
                                   fill=col + (255,), width=w)
    if sc.get('band'):
        draw_bubbles11(canvas, sc['band'])
    canvas.convert('RGB').save(out_path)
    print('合成完成', scene_id, canvas.size, '→', out_path)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', required=True)
    ap.add_argument('--scene', default='all')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--no-sb-palette', action='store_true')
    ap.add_argument('--no-flower', action='store_true')
    ap.add_argument('--sandy-new', action='store_true')
    args = ap.parse_args()
    if args.sandy_new:
        g11 = '/mnt/g/workspace/comfyui/output/g11cut'
        UNITS11['new_sandy_c'] = os.path.join(g11, 'sandy-cheer11_00001.png')
        UNITS11['new_sandy_l'] = os.path.join(g11, 'sandy-left11_00001.png')
    os.makedirs(args.outdir, exist_ok=True)
    audit_dir = os.path.join(os.path.dirname(os.path.abspath(args.outdir.rstrip('/'))), 'audit11')
    ids = sorted(SCENES11) if args.scene == 'all' else [args.scene]
    rep = []
    for sid in ids:
        rep.extend(compose_scene(sid, args.bg, os.path.join(args.outdir, 'r11-%s.png' % sid),
                                 audit_dir, sb_palette_on=not args.no_sb_palette,
                                 flower_on=not args.no_flower, sandy_new=args.sandy_new))
    rp = os.path.join(os.path.dirname(os.path.abspath(args.outdir.rstrip('/'))),
                      'paste-report-generation-11.json')
    prev = []
    if os.path.exists(rp):
        prev = json.load(open(rp))
    seen = {r['scene'] for r in rep}
    json.dump(prev + [r for r in rep if r['scene'] not in {p['scene'] for p in prev}],
              open(rp, 'w'), ensure_ascii=False, indent=1)
    print('paste report →', rp)
