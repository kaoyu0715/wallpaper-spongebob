# -*- coding: utf-8 -*-
# 十四代合成（issue #29：实景动漫背景 + 角色坐进泳圈，遮挡路线一「圈层三明治」）。
#
# 路线一实现：背景不重绘。把每个泳圈按拟合椭圆对（outer/hole）切成「前缘层」
#   （outer ∧ ¬hole ∧ 下半平面，取自背景本身，2px 羽化）。合成序（后→前）：
#   背景（含圈后缘与圈内水面/沙面）→ 沙滩圈的圈内接触投影 → 角色单元（坐态=站姿单元
#   下半身被遮）→ 圈前缘层压角色下半身 → 水面接触波纹高光。
#   水圈（蓝/黄）：角色 ∩ hole 椭圆 = 水面以下 → 半透明水色回染 + 轮廓处白色涟漪；
#   沙滩圈（粉）：hole 内先画坐实投影（偏左下，与背景日落逆光投影同向）。
# 蓝圈另加低透明度倒影呼应（左下，与影向一致）。
#
# 场景表（场景表口径=背景原生 768x1664 椭圆 (cx,cy,a,b,rot°)，rot 正=屏幕顺时针；
# 合成画布 = 原生 ×1.875 = 1440x3120，与既有 shallow-anime-2k 档一致）：
#   yellow: outer(254,820,172,112,+4)  hole(262,811,85,44,-5)   水圈-水线边(浅)
#   blue:   outer(497,686,160,79,-3)   hole(494,689,93,46,+6)   水圈-深水
#   pink:   outer(501,1025,216,172,+5) hole(503,1022,108,57,-6) 沙滩圈
# 角色：黄圈=海绵宝宝(sb-right12，向右看蓝圈派大星)；蓝圈=派大星(pat_l，向左看SB)；
#       粉圈=珊迪(new_sandy_l，向左看两人)。视线围合成环。
# 硬口径延续（沿用十一/十二代实现）：单元=现有 LoRA 管线产物；drop_stray_specks；
#   派大星四件套 + 成品级红岛回填 + 皮肤残留=0；珊迪 clean_sandy_ball + 硬闸B 盔圈裁切；
#   SB sb_palette 调色 + shorten_legs 缝合 + 硬闸C 腿身比 [0.55,0.75]；贴入比 ≤1.0x。
# 用法: python3 compose-round14.py --outdir output/round14 [--v v1]
import argparse, importlib.util, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 开发期回退：辅助模块在十三代工作树（本仓库签出后此路径不存在也无妨）
sys.path.insert(0, '/home/linus/.dsh/state/superagent/work/wallpaper-replica-crowd-611327bb99e0')

def _load(mod, fname):
    for d in (HERE, '/home/linus/.dsh/state/superagent/work/wallpaper-replica-crowd-611327bb99e0'):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            break
    spec = importlib.util.spec_from_file_location(mod, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

c12 = _load('compose_round12', 'compose-round12.py')
c11 = c12.c11
m8 = c11.m    # compose_round8: 四件套/tighten/paste
m9 = c11.m9   # compose_round9: drop_stray_specks
ps = c11.ps   # patrick_skin
sb_palette = c11.sb_palette
import sb_proportion

UNITS = c12.UNITS12
# 底图 = shallow-style_00001_.png（原生 768x1664）的 1.875x 模型放大档（与 2K 线同源口径）
BG_SRC = '/mnt/g/workspace/comfyui/output/shallow-anime-2k_00001_.png'
S = 1.875  # 原生 -> 画布
CW, CH = 1440, 3120

RINGS = {  # 原生坐标；rot 度，屏幕顺时针为正
  'yellow': dict(outer=(254, 820, 172, 112, 4),  hole=(262, 811, 85, 44, -5),  water='shallow'),
  'blue':   dict(outer=(497, 686, 160, 79, -3),  hole=(494, 689, 93, 46, 6),   water='deep'),
  'pink':   dict(outer=(501, 1025, 216, 172, 5), hole=(503, 1022, 108, 57, -6), water=None),
}

# 场景表（画布坐标）：坐态锚定法——角色腰部(裤腰/西装腰)置于 hole 中心深度(hole_cy+δ)，
# 胯以下截断(路线二式 cut，截断面藏于圈前缘带内=「水下/圈内」)，圈前缘层压住截断。
# s=贴入比（相对单元原生 bbox 像素，≤1.0）；waist_frac/cut_frac=腰线/截断线在单元
# 原始画布上的行比例（经 bbox 偏移换算）。dx=身体中心修正(画布px)。
SCENE14 = {
  'yellow': dict(unit='sb_r',        s=0.68, waist_y=1529, waist_frac=0.600, cut_frac=0.716, tilt=2,  dx=-40),
  'blue':   dict(unit='pat_l',       s=0.60, waist_y=1300, waist_frac=0.500, cut_frac=0.660, tilt=-3, dx=0),
  'pink':   dict(unit='new_sandy_l', s=0.56, waist_y=1945, waist_frac=0.655, cut_frac=0.846, tilt=2,  dx=0),
}

WARM = (255, 214, 178)


def Sx(v):
    return v * S


def ell_mask(size, ell, ss=4):
    """旋转椭圆掩膜（float32 0..1，4x 超采样抗锯齿）。ell=(cx,cy,a,b,rot°) 画布坐标。"""
    cx, cy, a, b, rot = ell
    W, H = size
    big = Image.new('L', (W * ss, H * ss), 0)
    d = ImageDraw.Draw(big)
    t = math.radians(rot)
    n = 240
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        x, y = a * ss * math.cos(th), b * ss * math.sin(th)
        pts.append(((cx * ss) + x * math.cos(t) - y * math.sin(t),
                    (cy * ss) + x * math.sin(t) + y * math.cos(t)))
    d.polygon(pts, fill=255)
    return np.asarray(big.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0


def feather(mask, px=1.2):
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(px))
    return np.asarray(im, dtype=np.float32) / 255.0


def scaled(ell):
    cx, cy, a, b, rot = ell
    return (cx * S, cy * S, a * S, b * S, rot)


def ring_masks(ring, canvas_size):
    r = RINGS[ring]
    outer = ell_mask(canvas_size, scaled(r['outer']))
    hole = ell_mask(canvas_size, scaled(r['hole']))
    hcx, hcy, ha, hb, hrot = scaled(r['hole'])
    yy, xx = np.mgrid[0:canvas_size[1], 0:canvas_size[0]].astype(np.float32)
    # 前缘 = 下半平面：以 hole 中心沿 rot 方向的割线为界（rot 小，取水平割线）
    below = (yy > hcy).astype(np.float32)
    belowm = feather(below, 1.0)
    front = feather(np.clip(outer * (1.0 - hole), 0, 1) * belowm, 1.2)
    return dict(outer=outer, hole=hole, front=front, hcx=hcx, hcy=hcy, ha=ha, hb=hb, hrot=hrot)


def pat_halo_clean(fig, verbose=True):
    """单位级补充（R4 前置）：clean_patrick_skin 回填后遗留的虚线晕圈，其红度落在
    原阈值(rg_med+12 / G<med-18)盲区——以放宽阈值(rg_med+5 / G<med-6)同机制回填
    （围合判定/环带中位/高斯融合完全沿用 patrick_skin 口径）。产品级 postclean 保留兜底。"""
    im = np.asarray(fig).astype(np.int32)
    visible, skin, cand, skin_med, rg_med = ps._masks(im)
    if skin_med is None:
        return fig
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    cand2 = (visible & (R > 110) & ((R - G) > rg_med + 5)
             & (G < skin_med[1] - 6) & (B > R - 170) & ~skin)
    out = np.asarray(fig).copy()
    total = 0
    for comp in ps._components(cand2):
        if not (3 <= len(comp) <= 400):
            continue
        ok, ring_skin = ps._enclosed(comp, cand2, skin, visible)
        if not ok or len(ring_skin) < 20:
            continue
        ys = [c[0] for c in comp]
        xs = [c[1] for c in comp]
        cm = np.zeros(cand2.shape, bool)
        cm[ys, xs] = True
        patch = ps._dilate(cm, 2)
        fillc = np.median(im[ring_skin[:, 0], ring_skin[:, 1], :3], axis=0).astype(np.uint8)
        y0, y1 = max(min(ys) - 6, 0), min(max(ys) + 7, cand2.shape[0])
        x0, x1 = max(min(xs) - 6, 0), min(max(xs) + 7, cand2.shape[1])
        flat = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        flat[..., :3] = fillc
        flat[..., 3] = 255
        fill = Image.fromarray(flat, 'RGBA').filter(ImageFilter.GaussianBlur(2))
        mmask = Image.fromarray((patch[y0:y1, x0:x1] * 255).astype('uint8'), 'L') \
            .filter(ImageFilter.MaxFilter(3))
        sub = Image.fromarray(out[y0:y1, x0:x1], 'RGBA')
        out[y0:y1, x0:x1] = np.asarray(Image.composite(fill, sub, mmask))
        total += len(comp)
    if verbose:
        print('halo: 放宽阈值回填晕圈 %d px' % total)
    return Image.fromarray(out, 'RGBA')


def load_unit(key, verbose=True):
    """单元入库管线：抠图净化 + 角色固定后处理 + 硬闸，返回 bbox 裁剪后 RGBA。"""
    path = UNITS[key]
    fig = Image.open(path).convert('RGBA')
    fig = m9.drop_stray_specks(fig, verbose=verbose)
    info = {}
    if key.startswith('pat'):
        fig = m8.clean_patrick_blue(fig, verbose=verbose)
        fig = m8.clean_patrick_blotches(fig, verbose=verbose)
        ps.clean_patrick_skin(fig, verbose=verbose)
        fig = pat_halo_clean(fig, verbose=verbose)
        fig = m8.fix_patrick_outline(fig, verbose=verbose)
    if key.startswith('sandy') or key.startswith('new_sandy'):
        fig = c11.clean_sandy_ball(fig, verbose=verbose)
    if key.startswith('sb'):
        fig = c12._sb_palette12(fig)  # 十二代固化：调色 + shorten_legs 缝合 + 硬闸C
        info['sb_leg_ratio'] = sb_proportion.leg_ratio(fig)[0]
    bb = fig.split()[3].getbbox()
    info['canvas'] = fig.size
    return fig.crop(bb), info, bb[1], fig.height


def warm_grade(fig, k=0.085):
    """统一光向的轻暖色：仅作用于单元 alpha 内，向日落暖光轻混。"""
    a = np.asarray(fig.convert('RGBA')).astype(np.float32)
    mix = a[..., 3:4] / 255.0 * k
    w = np.array(WARM, dtype=np.float32)
    out = a[..., :3] * (1 - mix) + w * mix
    o = a.copy(); o[..., :3] = out
    return Image.fromarray(o.astype(np.uint8), 'RGBA')


def paste_layer(canvas, layer, mask_f, box):
    """canvas.paste 的 alpha 版：按 mask_f 将 layer(RGBA/RGB) 合成进 canvas(RGB ndarray)。"""
    arr = np.asarray(canvas, dtype=np.float32)
    lay = np.asarray(layer.convert('RGB'), dtype=np.float32)
    x0, y0 = box
    h, w = mask_f.shape
    region = arr[y0:y0 + h, x0:x0 + w]
    m = mask_f[..., None]
    arr[y0:y0 + h, x0:x0 + w] = region * (1 - m) + lay[:h, :w] * m
    return Image.fromarray(arr.astype(np.uint8), 'RGB')


def residual_speck_fill(canvas, crop_box, ratio, pmask, rounds=3):
    """R4 延伸：detect 命中但 postclean 回填判据(G<140)不匹配的「回填晕圈」残留——
    以检测点 13px 窗口环带肤色中位、高斯圆斑(α≤0.9)直接覆盖，复检归 0。固定流程。"""
    x0, y0 = crop_box[0], crop_box[1]
    for _ in range(rounds):
        crop = canvas.crop(crop_box).convert('RGB')
        det = ps.detect_patrick_specks(crop, paste_ratio=ratio, verbose=False)
        det = [d for d in det if pmask[d[1], d[2]]]
        if not det:
            return 0
        a = np.asarray(crop).copy().astype(np.int32)
        for _area, cy, cx in det:
            ys0, ys1 = max(0, cy - 7), min(a.shape[0], cy + 8)
            xs0, xs1 = max(0, cx - 7), min(a.shape[1], cx + 8)
            win = a[ys0:ys1, xs0:xs1]
            skinm = (win[..., 0] > 190) & (win[..., 1] > 120) & (win[..., 1] < 210) \
                & (win[..., 0] - win[..., 1] > 30) & (win[..., 0] - win[..., 1] < 110)
            if skinm.sum() < 20:
                continue
            med = np.median(win[skinm], axis=0)
            yy2, xx2 = np.mgrid[ys0:ys1, xs0:xs1]
            d2 = np.sqrt((yy2 - cy) ** 2 + (xx2 - cy) ** 2 * 0 + (xx2 - cx) ** 2)
            al = np.clip(1.0 - (d2 - 2.0) / 4.0, 0, 1) * 0.9
            winf = win.astype(np.float32)
            win[..., :][...] = (winf * (1 - al[..., None]) + med[None, None, :] * al[..., None]
                                ).clip(0, 255).astype(np.int32)
        canvas.paste(Image.fromarray(a.astype(np.uint8), 'RGB'), (x0, y0))
    crop = canvas.crop(crop_box).convert('RGB')
    det = [d for d in ps.detect_patrick_specks(crop, paste_ratio=ratio, verbose=False)
           if pmask[d[1], d[2]]]
    return len(det)


# 组合变体：同一场景表下换角色/姿态即量产（参数即全部状态，无需改代码）
VARIANTS = {
  'A': {'pink': dict(unit='new_sandy_c')},   # 换珊迪为欢呼姿
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='output/round14')
    ap.add_argument('--v', default='v1')
    ap.add_argument('--variant', default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.variant:
        for ring, ov in VARIANTS[args.variant].items():
            SCENE14[ring] = dict(SCENE14[ring], **ov)

    bg = Image.open(BG_SRC).convert('RGB').resize((CW, CH), Image.LANCZOS)
    canvas = bg
    report = {'generation': 14, 'version': args.v, 'occlusion': 'ring-layer-sandwich', 'scene': {}}
    audit = os.path.join(args.outdir, 'audit14')
    os.makedirs(audit, exist_ok=True)

    masks = {r: ring_masks(r, (CW, CH)) for r in RINGS}

    # 圈内水体色：从背景 hole 区采样（蓝/黄）
    water_col = {}
    bgarr = np.asarray(bg, dtype=np.float32)
    for r in RINGS:
        if RINGS[r]['water']:
            hm = masks[r]['hole']
            ys, xs = np.nonzero(hm > 0.9)
            # 取 hole 上半部（远离前缘阴影）
            sel = ys < ys.mean()
            col = bgarr[ys[sel], xs[sel]].mean(axis=0)
            water_col[r] = tuple(float(v) for v in col)

    for ring, sc in SCENE14.items():
        mk = masks[ring]
        fig, info, bb_top, canvas_h = load_unit(sc['unit'])
        waist_px = sc['waist_frac'] * canvas_h - bb_top
        cut_px = int(round(sc['cut_frac'] * canvas_h - bb_top))
        crop_h = fig.height
        assert cut_px < crop_h, f'{ring}: cut 超出单元'
        fig = fig.crop((0, 0, fig.width, cut_px))  # 路线二式截断，截断面藏进圈前缘带
        scale = sc['s']
        assert scale <= 1.0, f'{ring}: 贴入比 {scale:.2f}x > 1.0x'
        w = int(round(fig.width * scale)); h = int(round(fig.height * scale))
        fig_s = fig.resize((w, h), Image.LANCZOS)
        fig_s = warm_grade(fig_s)
        top = sc['waist_y'] - waist_px * scale
        x0 = int(round(mk['hcx'] + sc.get('dx', 0) - w / 2))
        if sc.get('tilt'):
            fig_s = fig_s.rotate(-sc['tilt'], expand=True, resample=Image.BICUBIC)
            x0 -= (fig_s.width - w) // 2
            top -= (fig_s.height - h) // 2
        y0 = int(round(top))
        fa = np.asarray(fig_s)[..., 3].astype(np.float32) / 255.0

        # 沙滩圈：圈内坐实投影（偏左下，与背景影向一致）——画在角色之前
        if RINGS[ring]['water'] is None:
            sh = Image.new('L', (CW, CH), 0)
            d = ImageDraw.Draw(sh)
            scx, scy = mk['hcx'] - mk['ha'] * 0.28, mk['hcy'] + mk['hb'] * 0.42
            d.ellipse([scx - mk['ha'] * 0.62, scy - mk['hb'] * 0.55,
                       scx + mk['ha'] * 0.62, scy + mk['hb'] * 0.55], fill=90)
            shm = np.asarray(sh.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32) / 255.0
            shm *= mk['hole']  # 只落圈内沙面
            canvas = paste_layer(canvas, Image.new('RGB', (CW, CH), (40, 28, 24)), shm, (0, 0))

        # 角色贴入（整体），记录 body mask
        body = np.zeros((CH, CW), np.float32)
        body[max(0, y0):y0 + fa.shape[0], max(0, x0):x0 + fa.shape[1]] = \
            fa[:CH - max(0, y0), :CW - max(0, x0)]
        canvas = paste_layer(canvas, fig_s, fa, (x0, y0))
        bodymask = np.zeros((CH, CW), np.float32)
        bodymask[max(0, y0):y0 + fa.shape[0], max(0, x0):x0 + fa.shape[1]] = fa[:CH - max(0, y0), :CW - max(0, x0)]

        # 水圈：水位线=腰部深度（hole 中心）。水下回染自上而下滑入（wl+14 起、20px 渐变）；
        # 接触水波=身体两侧水面上的短弧 + 身体水线处的边缘微光（不横切身体，避免"搁浅"感）
        yy = np.mgrid[0:CH, 0:CW][0].astype(np.float32)
        if RINGS[ring]['water']:
            wl = sc['waist_y']
            ramp = np.clip((yy - (wl + 14)) / 20.0, 0, 1)
            sub = body * mk['hole'] * ramp
            alpha_t = 0.24 if RINGS[ring]['water'] == 'deep' else 0.14
            tint = np.array(water_col[ring], dtype=np.float32)
            canvas = paste_layer(canvas, Image.fromarray(
                np.tile(tint.astype(np.uint8), (CH, CW, 1)), 'RGB'), sub * alpha_t, (0, 0))
            # 圈外下垂肢体（如腕尖）= 没入圈前水面：更重回染
            sub2 = body * (1 - mk['outer']) * np.clip((yy - (mk['hcy'] + mk['hb'] * 0.3)) / 12.0, 0, 1)
            canvas = paste_layer(canvas, Image.fromarray(
                np.tile(tint.astype(np.uint8), (CH, CW, 1)), 'RGB'), sub2 * 0.45, (0, 0))
            # 水线泡沫：只在身体两侧的水面上（避开身体正面）
            foam = (1 - body) * mk['hole'] * feather(np.clip(1 - np.abs(yy - wl) / 8.0, 0, 1), 1.0)
            # 身体水线接触微光：body 边缘带 ∧ y>wl
            bimg = Image.fromarray((body * 255).astype(np.uint8))
            edge = np.clip(np.asarray(bimg.filter(ImageFilter.MaxFilter(7)), np.float32) / 255.0
                           - np.asarray(bimg.filter(ImageFilter.MinFilter(7)), np.float32) / 255.0, 0, 1)
            edge = edge * mk['hole'] * np.clip((yy - wl) / 10.0, 0, 1)
            canvas = paste_layer(canvas, Image.new('RGB', (CW, CH), (244, 251, 253)),
                                 foam * 0.4, (0, 0))
            canvas = paste_layer(canvas, Image.new('RGB', (CW, CH), (240, 250, 252)),
                                 np.asarray(Image.fromarray((edge * 255).astype(np.uint8))
                                            .filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0
                                 * 0.45, (0, 0))

        # 前缘层压回（背景像素，2px 羽化）
        canvas = paste_layer(canvas, bg, mk['front'], (0, 0))

        # 水面接触波纹：前缘内沿泛水高光（沿 hole 下缘内侧细弧，轻、不压身体）
        if RINGS[ring]['water']:
            hm = mk['hole']
            band = hm * feather(np.clip(1 - np.abs(yy - (mk['hcy'] + mk['hb'] * 0.72)) / 10, 0, 1), 1.0) \
                 * (1 - mk['front']) * (1 - body * 0.7)
            band = np.asarray(Image.fromarray((np.clip(band, 0, 1) * 255).astype(np.uint8))
                              .filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255.0
            canvas = paste_layer(canvas, Image.new('RGB', (CW, CH), (245, 252, 254)),
                                 np.clip(band, 0, 1) * 0.4, (0, 0))

        # 派大星成品级红岛回填（十一代 R4 口径：皮肤区检出→环带肤色回填→复检=0）
        if sc['unit'].startswith('pat'):
            x1, y1 = max(0, x0), max(0, y0)
            x2, y2 = min(CW, x0 + fig_s.width), min(CH, y0 + fig_s.height)
            pm = (bodymask[y1:y2, x1:x2] > 0.5) & (mk['front'][y1:y2, x1:x2] < 0.5)
            nres = c11.patrick_postclean(canvas, (x1, y1, x2, y2), scale, pmask=pm)
            if nres:
                nres = residual_speck_fill(canvas, (x1, y1, x2, y2), scale, pm)
            if nres:
                sub = canvas.crop((x1, y1, x2, y2))
                det = ps.detect_patrick_specks(sub.convert('RGB'), paste_ratio=scale, verbose=False)
                det = [d for d in det if pm[d[1], d[2]]]
                print('RESIDUE', det)
                sub.save(os.path.join(audit, 'pat-residue.png'))
            assert nres == 0, f'{ring}: 派大星皮肤红点复检未归零 ({nres})'

        # 蓝圈：倒影呼应（左下低透明度压扁镜像）
        if RINGS[ring]['water'] == 'deep':
            refl = fig_s.transpose(Image.FLIP_TOP_BOTTOM)
            rw, rh = refl.width, int(refl.height * 0.32)
            refl = refl.resize((rw, rh), Image.LANCZOS)
            rarr = np.asarray(refl).astype(np.float32)
            rarr[..., 3] *= 0.20
            refl = Image.fromarray(rarr.astype(np.uint8), 'RGBA')
            rarr2 = np.asarray(refl)
            rx = int(mk['hcx'] - mk['ha'] * 1.05 - rw * 0.4); ry = int(mk['hcy'] + mk['hb'] * 1.5)
            body_r = np.asarray(Image.fromarray(rarr2[..., 3]).filter(ImageFilter.GaussianBlur(5)),
                                np.float32) / 255.0
            # 倒影不叠到别的圈上（只落开放水面）
            ys0 = max(0, ry); ys1 = min(CH, ry + body_r.shape[0])
            xs0 = max(0, rx); xs1 = min(CW, rx + body_r.shape[1])
            clip = (1 - masks['yellow']['outer'][ys0:ys1, xs0:xs1]) \
                 * (1 - masks['pink']['outer'][ys0:ys1, xs0:xs1])
            body_r = body_r[ys0 - ry:ys1 - ry, xs0 - rx:xs1 - rx] * clip
            canvas = paste_layer(canvas, refl, body_r, (xs0, ys0))

        by = y0 + fig_s.height
        report['scene'][ring] = dict(unit=sc['unit'], paste_h=fig_s.height, unit_crop_h=crop_h,
                                     paste_ratio=round(scale, 3), tilt=sc.get('tilt', 0),
                                     bottom_y=by, cx_canvas=round(mk['hcx']), **info)

        # 审计裁切
        if sc['unit'].startswith('pat'):
            x1, y1 = int(mk['hcx'] - w / 2), y0
            canvas.crop((x1, y1, x1 + w, by)).save(
                os.path.join(audit, f'{ring}-{sc["unit"]}-skin.png'))
        if sc['unit'].startswith(('sandy', 'new_sandy')):
            canvas.crop((int(mk['hcx'] - mk['ha'] * 1.4), int(y0 - 10),
                         int(mk['hcx'] + mk['ha'] * 1.4), int(y0 + h * 0.55))).save(
                os.path.join(audit, f'{ring}-{sc["unit"]}-helmet.png'))

    vtag = args.v + (f'-var{args.variant}' if args.variant else '')
    out = os.path.join(args.outdir, f'r14-swim-rings-{vtag}.png')
    canvas.save(out)
    with open(os.path.join(args.outdir, f'paste-report-generation-14-{vtag}.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('saved', out)
    print(json.dumps(report['scene'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
