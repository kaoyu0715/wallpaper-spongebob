# -*- coding: utf-8 -*-
# 十七代合成器（issue #35「保圈换鱼身」单元替换路线，禁整区重绘）：
#   区域四步（顺序= z 序）：
#     ① erase-inpaint：旧鱼足迹(¬前缘带 ¬ZZ) d1.0 inpaint 成同款海水/管后带（掩膜内无鱼像素存活）；
#     ② unit paste：白底 LoRA 单元（RMBG+tighten+固定后处理）按姿态表缩放/倾斜贴入（清晰通道，
#        底图不再二次重绘人物）；
#     ③ front band 贴回样张像素：圈前缘压身（前缘管体像素=样张原像素，动不得）；
#     ④ limb-top：抠图骨架 old_limbs(∧管带) ∩ 新单元肢体 → 新肢体压在管上的区域；ZZ 字母最后贴回。
#   整图链：三区依次 → stage(1080x1920, 样张坐标) → outpaint 1440x3120（历代交付口径）。
# 用法:
#   python3 compose17.py inpaint --fish fish1 --seed 501 [--upscale 1.7] [--stage-in <png>] [--stage-out <png>]
#   python3 compose17.py region  --fish fish1 [--stage-in <png>] [--out <png>]   # ②③④
#   python3 compose17.py chain   --i1 a.png --i2 b.png --i3 c.png --out stage.png
#   python3 compose17.py outpaint --in stage.png --out <png> [--seed 20261108] [--seam 40]
import argparse, os, sys, time
_T0 = time.time()
def _p(msg):
    if os.environ.get('DEBUG17'): print('%.1fs' % (time.time()-_T0), msg, flush=True)
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask17
import mask15
import inp16                      # ComfyUI post/wait/out_of 经 gen16crop
import gen16crop as G

T = os.path.dirname(os.path.abspath(__file__))
REF = mask17.REF
REF_SIZE = mask17.REF_SIZE
IN_DIR = G.IN_DIR
os.makedirs('output', exist_ok=True)

# —— 姿态对位表（样张坐标；全部状态。anchor=单元 alpha 包围盒中心落点，
#     s=贴入比(按收紧后 bbox)，tilt=顺时针°，limb_dil=骨架肢体区外扩）——
POSDIR = '/mnt/g/workspace/comfyui/output/g17cut'
POSE = {
  # 选定单元与对位参数（fit1..3 迭代确定；flip=水平镜像对齐样张头向）
  'fish1': dict(unit=POSDIR + '/sb-ring17f-sb-ring17f-7162_00001__00001-post.png',
                anchor=(318, 1050), s=0.44, tilt=2, flip=True, limb_dil=9,
                shadow=(0.16, 8), warm=0.07),
  'fish2': dict(unit=POSDIR + '/pat-ring17-pat-ring17-7204_00001__00001-post.png',
                anchor=(720, 820), s=0.28, tilt=4, flip=True, limb_dil=9,
                shadow=(0.16, 8), warm=0.07),
  'fish3': dict(unit=POSDIR + '/san-ring17c-san-ring17c-7324_00001__00001-post.png',
                anchor=(665, 1252), s=0.285, tilt=-15, limb_dil=9,
                shadow=(0.16, 8), warm=0.07),
}

WARM = (255, 214, 178)

WATER_CFG = 3.0   # 高 cfg 强化文本先验=必画主体（少女/绿鱼/木板实测），低 cfg 取均值纹理
DONOR_WATER = 'output/g17-fish1-inp.png'   # fish1 水段(d1.0 验收)供体，运行序保证先跑
WATER_OFFSET = {'fish2': (412, -200), 'fish3': (378, 266)}   # 外环心 OUTER_ELL 实测位移
stage_out_donors = {}

INPAINT_TXT = {
  'w': ('calm open teal sea surface with soft ripples and gentle current shading, seamless continuation '
        'of the surrounding water, subtle darker depth shading, no objects, no horizon, no sky',
        'person, human, girl, boy, woman, man, face, hair, eyes, arms, legs, body, skin, bikini, '
        'swimsuit, clothes, fish, fins, donut, swim ring, inflatable ring, torus, circle, tube, '
        'flat solid color, uniform gray, white foam, clouds, text, object'),
  'tube': {
      'fish1': ('the red and white striped inflatable swim ring tube continues naturally behind with '
                'the same curvature and glossy shading, open sea around',
                'person, human, girl, face, hair, limbs, body, fish, fins, second ring, extra ring, '
                'donut floating in water, blue stripes, flat color, text'),
      'fish2': ('the blue and white striped inflatable swim ring tube continues naturally behind with '
                'the same curvature and glossy shading, open sea around',
                'person, human, girl, face, hair, limbs, body, fish, fins, second ring, extra ring, '
                'donut floating in water, red stripes, flat color, text'),
      'fish3': ('the blue and white striped inflatable swim ring tube continues naturally behind with '
                'the same curvature and glossy shading, open sea around',
                'person, human, girl, face, hair, limbs, body, fish, fins, second ring, extra ring, '
                'donut floating in water, red stripes, flat color, text')}}


def _ell_img(mask, box):
    return mask.crop(box)


def _inpaint_pass(fish, seed, tag, base_bin, stage, crop_box, upscale):
    """单次补画：base_bin(1x硬掩膜)→羽化→d1.0→内收贴回 stage，返回新 stage。"""
    x0, y0, x1, y1 = crop_box
    W, H = int((x1 - x0) * upscale) // 8 * 8, int((y1 - y0) * upscale) // 8 * 8
    m = Image.fromarray((base_bin * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(4))
    m = m.point(lambda v: 255 if v > 128 else (0 if v < 20 else v))
    m = m.crop((x0, y0, x1, y1)).resize((W, H), Image.LANCZOS)
    name = 'g17in-%s-%d-%s' % (fish, seed, tag)
    d = os.path.join(G.INPUT, 'g17in')
    os.makedirs(d, exist_ok=True)
    big = stage.crop((x0, y0, x1, y1)).resize((W, H), Image.LANCZOS)
    # d0.95 + 无鱼归一化卷积底色（掩膜外颜色多尺度扩散进掩膜）：
    # 平涂死色（0.8 实测）与 hallucination（1.0 实测）之间的折中。
    DEN = 1.0   # 1.0+强负面（person/donut/ring 全禁）实测优于 0.95（大区域死色）
    barr = np.asarray(big).astype(np.float32)
    wsm = np.asarray(m).astype(np.float32)            # 掩膜内=1 → 权重=0
    wmt = Image.fromarray((255.0 - wsm).astype('uint8'))
    wa = np.asarray(wmt).astype(np.float32)
    fill = barr.copy()
    for r in (15, 40, 100, 240):
        rr = int(r * upscale)
        den = np.asarray(wmt.filter(ImageFilter.GaussianBlur(rr))).astype(np.float32)
        ok = den > 30.0
        den = np.maximum(den, 1.0)
        for i in range(3):
            num = np.asarray(Image.fromarray(np.clip(barr[:, :, i] * wa / 255.0, 0, 255).astype('uint8'))
                             .filter(ImageFilter.GaussianBlur(rr))).astype(np.float32)
            v = np.clip(num * 255.0 / den, 0, 255)
            fill[:, :, i] = np.where(ok, v, fill[:, :, i])
    if DEN < 0.98:
        mhard = m.point(lambda v: 255 if v > 127 else 0)
        big = Image.composite(Image.fromarray(fill.astype('uint8')), big, mhard)
    big.save(os.path.join(d, name + '.png'))
    m.save(os.path.join(d, name + '-mask.png'))
    key = 'tube' if tag == 'tube' else 'w'
    pos, neg = INPAINT_TXT[key]['fish1'] if key == 'tube' else INPAINT_TXT['w']
    if key == 'tube':
        pos, neg = INPAINT_TXT['tube'][fish]
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': pos}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g17in/' + name + '.png'}},
      '7': {'class_type': 'LoadImage', 'inputs': {'image': 'g17in/' + name + '-mask.png'}},
      '7b': {'class_type': 'ImageToMask', 'inputs': {'image': ['7', 0], 'channel': 'red'}},
      '8': {'class_type': 'InpaintModelConditioning', 'inputs': {'positive': ['4', 0], 'negative': ['5', 0],
              'vae': ['1', 2], 'pixels': ['6', 0], 'mask': ['7b', 0], 'noise_mask': True}},
      '10': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['8', 0], 'negative': ['8', 1],
               'latent_image': ['8', 2], 'seed': int(seed), 'steps': 30, 'cfg': WATER_CFG,
               'sampler_name': 'dpmpp_2m', 'karras': True, 'denoise': DEN,
               'scheduler': 'karras'}},
      '11': {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}},
      '12': {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0], 'filename_prefix': 'g17hole/' + name}},
    }
    g['10']['inputs'].pop('karras')
    import gen17unit
    e = G.wait(gen17unit._submit(g))
    p = G.out_of(e, '12')
    print('inpaint', fish, tag, 'seed', seed, '->', p, flush=True)
    im = Image.open(p).convert('RGB').resize((x1 - x0, y1 - y0), Image.LANCZOS)
    pm = m.filter(ImageFilter.MinFilter(9)).resize((x1 - x0, y1 - y0), Image.LANCZOS)
    pm = pm.filter(ImageFilter.GaussianBlur(2))
    pm = pm.point(lambda v: 255 if v > 200 else (0 if v < 20 else v))
    out = stage.convert('RGB').copy()
    out.paste(im, (x0, y0), pm)
    return out


def water_copy_fill(fish, sel, stage, donor_path, offset):
    """海水段=同画平移克隆：以 fish1 已验收海水图为唯一纹理源（同一支笔的水），
    平移到本圈位置、色调增益匹配本地水面、掩膜羽化贴回。
    模型 d1.0 在大面积水域必产 hallucination（动漫少女/木板/小圈——511/521/605 实测），弃用。"""
    import mask17 as _m
    don = np.asarray(Image.open(donor_path).convert('RGB')).astype(np.float64)
    stg = np.asarray(stage.convert('RGB')).astype(np.float64)
    ox, oy = offset
    donor = np.roll(np.roll(don, oy, axis=0), ox, axis=1)
    # 供体失效区 = fish1 管体∪ZZ 平移后（黄字/管纹不可作海水）
    t1 = np.asarray(_m.tube_band('fish1')) > 0
    z1 = np.asarray(_m.zz_letters()) > 0
    inv = np.roll(np.roll(t1 | z1, oy, axis=0), ox, axis=1)
    ok = sel & ~inv
    # 色调增益：sel 外环带（1x，距 sel≤24 的水面像素）
    sd = Image.fromarray((sel * 255).astype('uint8')).filter(ImageFilter.MaxFilter(49))
    near = (np.asarray(sd) > 0) & ~sel & ~t1 if False else (np.asarray(sd) > 0) & ~sel
    t1s = np.roll(np.roll(t1, oy, axis=0), ox, axis=1)
    near = near & ~t1s
    g = np.ones(3)
    if near.sum() > 500:
        tm = np.median(stg[near], axis=0)
        dm = np.median(donor[near], axis=0)
        g = np.clip(tm / np.maximum(dm, 1.0), 0.8, 1.25)
    out = stg.copy()
    out[ok] = np.clip(donor[ok] * g, 0, 255)
    # 残余 sel∧inv：多尺度归一化卷积兜底（周边水色扩散）
    rem = sel & ~ok
    if rem.any():
        barr = out
        wa = 255.0 - np.asarray(Image.fromarray((sel * 255).astype('uint8')).filter(ImageFilter.MaxFilter(3))).astype(np.float32)
        wmt = Image.fromarray(wa.astype('uint8'))
        fill = barr.copy()
        for r in (15, 40, 100, 240):
            den = np.asarray(wmt.filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
            okk = den > 30.0
            den = np.maximum(den, 1.0)
            for i in range(3):
                num = np.asarray(Image.fromarray(np.clip(barr[:, :, i] * wa / 255.0, 0, 255).astype('uint8'))
                                 .filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
                v = np.clip(num * 255.0 / den, 0, 255)
                fill[:, :, i] = np.where(okk, v, fill[:, :, i])
        out[rem] = fill[rem]
    fim = Image.fromarray(out.astype('uint8'))
    mh = Image.fromarray((sel * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(2.5))
    ob = stage.convert('RGB').copy()
    ob.paste(fim, (0, 0), mh)
    return ob


def tube_mirror_patch(fish, sel_t, stage):
    """鱼鳍压管区精修：以同圈同高度的镜像侧管面克隆（泳圈左右条纹同构、
    光照相近），极坐标分箱只兜镜像源不可用的像素。源须干净（管带 ∧ ¬补画域 ∧ ¬ZZ）。"""
    import mask17 as _m
    cx = _m.OUTER_ELL[fish][0]
    tube = np.asarray(_m.tube_band(fish)) > 0
    zz = np.asarray(_m.zz_letters()) > 0 if fish == 'fish1' else np.zeros_like(tube)
    base = inpaint_base(fish)
    stg = np.asarray(stage.convert('RGB')).astype(np.float64)
    x = np.arange(stg.shape[1])
    mir = stg[:, 2 * int(round(cx)) - x, :]      # 水平镜像
    # 镜像源干净判据：REF 该处是管面且非鱼/鳍暖色（红条 G<90、紫暗部 R<150 不误伤）
    ref = np.asarray(Image.open(REF).convert('RGB')).astype(int)
    finref = (ref[:, :, 0] > 150) & (ref[:, :, 1] > 100) & (ref[:, :, 0] - ref[:, :, 2] > 50)
    xidx = 2 * int(round(cx)) - np.arange(stg.shape[1])
    oksrc = np.clip(xidx, 0, stg.shape[1] - 1)
    mir_ok = (tube & ~finref & ~zz)[:, oksrc]
    # 色调增益：目标带周边本侧干净管面 vs 镜像侧
    near = np.asarray(Image.fromarray((sel_t * 255).astype('uint8'))
                      .filter(ImageFilter.MaxFilter(41))) > 0
    near = near & tube & ~base & ~zz
    clean_m = mir_ok & np.roll(near[:, ::-1], 1, axis=1)
    g = np.ones(3)
    if near.sum() > 400 and clean_m.sum() > 400:
        g = np.clip(np.median(stg[near], axis=0) / np.maximum(np.median(mir[near], axis=0), 1.0),
                    0.8, 1.25)
    out = stg.copy()
    tgt = sel_t & mir_ok
    out[tgt] = np.clip(mir[tgt] * g, 0, 255)
    im = Image.fromarray(out.astype('uint8'))
    mh = Image.fromarray((tgt * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(1.6))
    ob = stage.convert('RGB').copy()
    ob.paste(im, (0, 0), mh)
    return ob


def tube_row_med(fish, sel_t, stage):
    """压管鱼鳍区：逐行中位填充——管体明暗主要沿纵向（行内恒定），
    横向中位可保光影、延续同侧条纹色。源=同行管面 ∧ ¬补画域(膨胀9) ∧ ¬ZZ ∧ ¬鱼色。"""
    import mask17 as _m
    tube = np.asarray(_m.tube_band(fish)) > 0
    zz = np.asarray(_m.zz_letters()) > 0 if fish == 'fish1' else np.zeros_like(tube)
    ref = np.asarray(Image.open(REF).convert('RGB')).astype(np.float64)
    finref = (ref[:, :, 0] > 150) & (ref[:, :, 1] > 100) & (ref[:, :, 0] - ref[:, :, 2] > 50)
    wide = np.asarray(Image.fromarray((sel_t * 255).astype('uint8'))
                      .filter(ImageFilter.MaxFilter(19))) > 0
    ok = tube & ~finref & ~zz & ~wide
    stg = np.asarray(stage.convert('RGB')).astype(np.float64)
    out = stg.copy()      # 行源不足的行保持管分箱结果，绝不回退 REF（会贴回鱼鳍）
    rows = np.flatnonzero(sel_t.any(1))
    cnt = ok.sum(1)
    for yy in rows:
        cols = np.flatnonzero(ok[yy])
        if cnt[yy] >= 40:
            med = np.median(ref[yy, cols], axis=0)
            m = sel_t[yy]
            out[yy, m] = med
    im = Image.fromarray(np.clip(out, 0, 255).astype('uint8'))
    mh = Image.fromarray((sel_t * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(1.5))
    ob = stage.convert('RGB').copy()
    ob.paste(im, (0, 0), mh)
    return ob


def inpaint_base(fish):
    """补画基础域：足迹(¬前缘|压管肢体) ∪ 抠图α∧¬前缘（鱼鳍越出足迹，实测教训）∧¬ZZ。"""
    import mask17 as _m
    fp = np.asarray(_m.footprint(fish, margin=14)) > 127
    fb = np.asarray(_m.front_band(fish)) > 0
    limbs = np.asarray(_m.old_limbs(fish, dil=4)) > 0
    base = fp & (~fb | limbs)
    ox, oy = _m.cut_origins()[fish]
    ca = Image.new('L', REF_SIZE, 0)
    ca.paste(_m.cut_rgba(fish).split()[3], (ox, oy))
    add = (np.asarray(ca.filter(ImageFilter.MaxFilter(9))) > 60) & ~fb
    base = base | add
    if fish == 'fish1':
        base = base & ~(np.asarray(_m.zz_letters()) > 0)
    return base


def tube_fill(fish, sel, stage):
    """管体重建=确定性极坐标填色（模型 inpaint 实测总画坏圈条带几何，弃用）：
    以样张自身可见管面按（绕环角 φ × 截面深度 v）分箱取中位色，
    条纹沿角向天然延续、光照逐角正确；仅写入 sel∧管带 像素。"""
    import mask17 as _m
    cx, cy, a, b = _m.OUTER_ELL[fish]
    hx, hy, ha, hb = _m.HOLE_ELL[fish]
    tube = np.asarray(_m.tube_band(fish)) > 0
    zz = np.asarray(_m.zz_letters()) > 0 if fish == 'fish1' else np.zeros_like(tube)
    H, W = tube.shape
    ys, xs = np.mgrid[0:H, 0:W]
    # 归一化空间：外圈→单位圆；沿φ射线求孔椭圆交点 ρ_h，v=截面参数(0孔缘→1外缘)
    X, Y = (xs - cx) / a, (ys - cy) / b
    rho = np.sqrt(X * X + Y * Y)
    phi = np.arctan2(Y, X) % (2 * np.pi)
    cf, sf = np.cos(phi), np.sin(phi)
    oxx, oyy = (cx - hx) / ha, (cy - hy) / hb
    A = (a * cf / ha) ** 2 + (b * sf / hb) ** 2
    B = 2 * (oxx * a * cf / ha + oyy * b * sf / hb)
    C = oxx * oxx + oyy * oyy - 1.0
    disc = np.maximum(B * B - 4 * A * C, 0.0)
    rho_h = (-B + np.sqrt(disc)) / (2 * A)      # 外心在孔内，取正根=前向交点
    v = (rho - rho_h) / np.maximum(1.0 - rho_h, 1e-3)
    NB, NV = 240, 20
    ub = np.clip((phi / (2 * np.pi) * NB).astype(int), 0, NB - 1)
    vb = np.clip((v * NV).astype(int), 0, NV - 1)
    # 可见管面 = 管带 ∧ ¬ZZ——遮挡体不可几何确定（ringcut α 含整圈：fish2 实测
    # 肢体∧管=全部管；足迹 margin0 也吞圈），改由统计法剔除鱼染箱（下方离群判定）。
    vis = tube & ~zz
    # 只允许条纹色进分箱源——鱼鳍/花环/肤色 AA 一律排除；
    # 仅用于 bin 清洗，非圈保护掩膜（issue① 禁的是 chroma 保圈掩膜）。
    arr0 = np.asarray(stage).astype(int)
    Rr, Gg, Bb = arr0[:, :, 0], arr0[:, :, 1], arr0[:, :, 2]
    white = (Rr > 190) & (Gg > 170) & (Bb > 170)
    if fish == 'fish1':
        stripe = ((Rr > 140) & (Rr - Gg > 80) & (Bb < 100)) | white
    else:
        stripe = ((Bb > 140) & (Bb - Rr > 40)) | white
    vis = vis & stripe
    arr = np.asarray(stage).astype(np.float64)
    cells = np.full((NB, NV, 3), np.nan)
    idx = ub[vis] * NV + vb[vis]
    pix = arr[vis]
    order = np.argsort(idx)
    idx_s, pix_s = idx[order], pix[order]
    bnd = np.flatnonzero(np.diff(idx_s)) + 1
    for lo, hi in zip(np.r_[0, bnd], np.r_[bnd, len(idx_s)]):
        g = pix_s[lo:hi]
        cells[idx_s[lo] // NV, idx_s[lo] % NV] = np.median(g, axis=0)
    # 空格填充：波前最近邻复制（φ 环形优先同 v 行，其次 v±1 列）——
    # 加权平均会把红白条纹混成灰紫（实测），最近邻保证条纹按角向干净延续。
    # 离群剔除：鱼身覆盖箱的中位色与同 v 行角向邻域(±1,±2)差距>180(Σ3通道)则弃；
    # 邻域用双侧取小，条纹跨界不误杀。两轮迭代（先清明显鱼染，再清残余）。
    for _ in range(2):
        filled0 = ~np.isnan(cells[:, :, 0])
        far = np.full((NB, NV), 1e9)
        for k in (-2, -1, 1, 2):
            r = np.roll(cells, k, axis=0)
            r2 = np.roll(cells, 2 * k, axis=0)
            pair = np.fmin(np.abs(cells - r), np.abs(cells - r2)).sum(-1)
            far = np.minimum(far, pair)
        bad = filled0 & (np.nan_to_num(far, nan=0.0) < 1e9) & (np.nan_to_num(far) > 180)
        if not bad.any():
            break
        cells[bad] = np.nan
    filled = ~np.isnan(cells[:, :, 0])
    for _ in range(NB + NV):
        if filled.all():
            break
        cands = [np.roll(cells, (k, 0), axis=(0, 1)) for k in (1, -1)] + [
            np.concatenate([cells[:, 1:], np.full((NB, 1, 3), np.nan)], 1),
            np.concatenate([np.full((NB, 1, 3), np.nan), cells[:, :-1]], 1)]

        newc = np.where(filled[:, :, None], cells, np.nan)
        nf = filled.copy()
        for cand in cands:
            ok = np.isnan(newc[:, :, 0]) & ~np.isnan(cand[:, :, 0])
            newc = np.where(ok[:, :, None], cand, newc)
            nf = nf | ok
            if nf.all():
                break
        if not nf.any() or (nf == filled).all() and not filled.any():
            break
        cells, filled = newc, nf
    if __debug__ and os.environ.get('T17DUMP'):
        Image.fromarray(np.where(np.isnan(cells), 0, cells).astype('uint8')).resize((960, 400), Image.NEAREST).save(os.environ['T17DUMP'])
    gmed = np.nanmedian(cells.reshape(-1, 3), axis=0)
    cells = np.where(np.isnan(cells), gmed[None, None, :], cells)
    fillimg = cells[ub, vb]
    tgt = sel & tube
    out = np.asarray(stage).copy()
    out[tgt] = np.clip(fillimg[tgt], 0, 255).astype('uint8')
    sm = np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(1.5)))
    out[tgt] = sm[tgt]
    im = Image.fromarray(out)
    mh = Image.fromarray((sel * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(2))
    outb = stage.convert('RGB').copy()
    outb.paste(im, (0, 0), mh)
    return outb


def cut_alpha_mask(fish, dil=7):
    """抠图全轮廓（含墨镜/发/压圈发丝）贴回样张坐标系的布尔掩膜。"""
    _cut = mask17.cut_rgba(fish).split()[3]
    _ox, _oy = mask17.cut_origins()[fish]
    shp = np.asarray(mask17.tube_band(fish)).shape
    _big = Image.new('L', shp[::-1], 0)
    _big.paste(_cut, (_ox, _oy))
    return np.asarray(_big.filter(ImageFilter.MaxFilter(dil))) > 0


def inpaint_region(fish, seed, upscale=1.7, stage_in=None, stage_out=None, mask_only=False):
    """① 两段 erase-inpaint（均 d1.0，无预填充——预填充实测要么留鱼鬼影要么死色平涂）：
       先海水段（足迹¬管带），后管体段（足迹∧管带，圈条带自然延续）。"""
    import mask17 as _m
    stage = Image.open(stage_in) if stage_in else Image.open(REF).convert('RGB')
    box = mask17.CROPS[fish]
    tube = np.asarray(_m.tube_band(fish)) > 0
    base = inpaint_base(fish)
    # 水与背景一律禁模型补底（用户裁决 2026-11-09 二轮）：模型 inpaint=低频糊块
    # （灰绿块/白抹痕/圈外灰雾同症状），达不到样张清晰度。擦除空缺=样张自身干净
    # 水面「平移复制克隆」：涟漪方向连续的竖直偏移优先、光斑密度天然一致、边界无缝羽化。
    refimg = Image.open(REF).convert('RGB')
    refc = np.asarray(refimg).astype(np.int16)
    stg = np.asarray(stage.convert('RGB')).astype(np.int16)
    # 最小补底域：样张里**鱼实际盖住**的像素（非水相，或暗相=鳍/深影）+5px 膨胀。
    # 圈外水、圈缘影、孔内鱼旁水面一律保留样张原像素——最彻底的水连续性。
    W = base & ~tube                                       # 全部非管面空缺
    finref = (refc[:, :, 0] > 150) & (refc[:, :, 1] > 100) & (refc[:, :, 0] - refc[:, :, 2] > 50)
    zz = np.asarray(_m.zz_letters()) > 0 if fish == 'fish1' else np.zeros_like(tube)
    tubeE = np.asarray(Image.fromarray((tube * 255).astype('uint8'))
                       .filter(ImageFilter.MaxFilter(13))) > 0   # 圈缘红白晕不作源
    srcok = ~tubeE & ~(np.asarray(mask17.footprint(fish, margin=14)) > 127) & ~zz & ~finref
    # 样张「水相」掩膜：仅用于选择「周边水面」参考带（源有效性用 srccool 冷调口径）。
    rmax = refc.max(2).astype(int); rmin = refc.min(2).astype(int)
    waterph = ~tubeE & ~zz & (rmax < 252) & (rmax - rmin < 95) & \
        ~((refc[:, :, 0] > 150) & (refc[:, :, 0] - refc[:, :, 2] > 55))
    srccool = srcok & (refc[:, :, 0] - refc[:, :, 2] < 20)       # 样张水主色(含白亮涟漪/星芒)
    # 「干净水面」硬矩形：样张左侧/中部开阔水区。别的泳者(蓝白圈/果汁)、暗鱼、
    # 星芒都冷调、会被 srccool 误收为“水”→克隆出块状异体内容（实测教训）。
    clean = np.zeros(tube.shape, dtype=bool)
    clean[150:780, 60:1420] = True      # 上方开阔水
    clean[150:1420, 60:420] = True      # 左侧水带（200px 环带需要近处取源）
    clean[1330:1560, 560:1050] = True   # 下中带（暗鱼影由 fshad 排除）
    cy, cx, cha, chb = 235, 458, 110, 75                       # 星芒强闪排除
    yy0, xx0 = np.mgrid[0:tube.shape[0], 0:tube.shape[1]]
    clean &= ~(((yy0 - cy) / chb) ** 2 + ((xx0 - cx) / cha) ** 2 <= 1.0)
    srccoolc = srccool & clean
    # 最小补底域=实测旧鱼多边形+4px（mask15 标定）。圈外水/圈缘影/孔内鱼旁水面
    # 一律保留样张原像素——最彻底的水连续；色相启发式会漏鱼裤(水相色)不可用。
    tube9 = np.asarray(Image.fromarray((tube * 255).astype('uint8'))
                       .filter(ImageFilter.MaxFilter(9))) > 0   # 圈可见像素不可刷没（右缘实测）
    tube5 = np.asarray(Image.fromarray((tube * 255).astype('uint8'))
                       .filter(ImageFilter.MaxFilter(5))) > 0
    _cutm = cut_alpha_mask(fish)                 # 抠图全轮廓(含墨镜/发/压圈发丝)
    zz3 = np.asarray(Image.fromarray((zz * 255).astype('uint8'))
                     .filter(ImageFilter.MaxFilter(7))) > 0 if fish == 'fish1' else np.zeros_like(tube)
    W = ((np.asarray(mask15.fish_mask(fish, margin=4)) > 127) | _cutm) & (base | _cutm) \
        & ~tube & ~zz3    # 仅护管面核心；5px 领圈鱼色像素须擦(tube5 护领=黄弧实测教训)
    # 圈条纹像素（鱼鳍/发压在管面上处）= 独立克隆道（水克隆刷圈=条码感根因，实测）：
    # 源=样张自身管面条纹，沿角向小偏移平移复制；残差交管体极坐标填色兜底。
    _Rr, _Gg, _Bb = refc[:, :, 0], refc[:, :, 1], refc[:, :, 2]
    _redst = (_Rr - _Gg > 50) & (_Rr - _Bb > 60) & (_Rr > 90)
    _whst = (_Rr > 190) & (_Rr - _Gg > -18) & (_Rr - _Gg < 40) & (_Bb > 140)
    tube15 = np.asarray(Image.fromarray((tube * 255).astype('uint8'))
                        .filter(ImageFilter.MaxFilter(31))) > 0
    # 白圈白与眼白同色相——仅管带邻域的白算圈白（眼白在管内上方,实测教训）。
    refstripe = _redst | (_whst & tube15)
    _ringzone = np.asarray(Image.fromarray((tube * 255).astype('uint8'))
                           .filter(ImageFilter.MaxFilter(27))) > 0    # 管+13px：内缘 AA/领圈带
    _nonwat = ~(((refc[:, :, 1] - refc[:, :, 0] > 12) & (refc.max(2) < 225)) |
                (refc.max(2) < 110))     # REF 非水相(圈色/AA/压圈鱼体)
    ringW = ((np.asarray(mask15.fish_mask(fish, margin=4)) > 127) | _cutm) & ~zz3 & ~tube & \
        _ringzone & _nonwat              # 几何判据(色相判据漏 AA 半像素,实测)
    W = W & ~ringW

    HH, WW = tube.shape

    def _shift(a, dx, dy):
        sy = np.clip(np.arange(HH) + dy, 0, HH - 1)
        sx = np.clip(np.arange(WW) + dx, 0, WW - 1)
        return a[np.ix_(sy, sx)]

    def _ix(o):
        sy = np.clip(np.arange(HH) + o[1], 0, HH - 1)
        sx = np.clip(np.arange(WW) + o[0], 0, WW - 1)
        return np.ix_(sy, sx)

    def _clone(todo, oksrc, offs_):
        r2 = todo.copy()
        for (dx, dy) in offs_:
            if not r2.any():
                break
            use = r2 & (_shift(oksrc.astype(np.uint8), dx, dy) > 0)
            if use.any():
                stg[use] = refc[_ix((dx, dy))][use]
                r2 = r2 & ~use
        return r2

    def _dil(a, k):
        return np.asarray(Image.fromarray((a * 255).astype('uint8'))
                          .filter(ImageFilter.MaxFilter(2 * k + 1))) > 0

    # R3v2（用户裁决 2026-11-09 issue #35 三条口径）：水补全收口——
    # ① 取源=擦除域周边 200px 近处干净水面（无圈体、无鱼影、无气泡、无角色；
    #    禁止远端天际线取源——上下光照渐变不同必出色差）；
    # ② 平移偏移只做一次全局均值匹配（对目标域四周 24px 环带均值），禁逐块匹配；
    # ③ 贴入后边界仅 1px 羽化；④ 水过关后才贴人物（顺序=region 四步天然保证）。
    fp26 = (np.asarray(mask17.footprint(fish, margin=26)) > 127)   # margin10 源近=灰杂点增多(实测回滚)
    bubble = (refc.min(2) > 218)                     # 仅纯白气泡核；涟漪亮痕保留（纹理一致）
    _sat = refc[:, :, 1] - refc[:, :, 0]
    fshad = (refc.max(2) < 110) | ((_sat < 25) & (refc.max(2) < 170))
    # 排除=深暗鱼体 与 低饱和暗影痕；饱和暗水(圈影 G−R 大)可作暗区源（底色带实测）。
    cleanband = srccool & clean & ~fp26 & ~bubble & ~fshad \
        & _dil(W, 200) & ~_dil(W, 0)
    # 圈条纹像素（鱼鳍/发压在管面处）：沿角向小偏移复制样张自身条纹；残差极坐标兜底。
    srcstripe = refstripe & tubeE & ~zz & ~finref
    offsR = tuple((dx, dy) for dx in (0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10, 12, -12)
                  for dy in (0, 2, -2, 4, -4))          # 条纹相位保真：偏移≤12（红梳齿=大偏移实测）
    ringres = _clone(ringW, srcstripe, offsR)
    k = 0
    while ringres.any() and k < 300:                     # 最近邻条纹链（不极坐标，避免弧线外溢）
        moved = False
        for dy_, dx_ in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src_ok = _shift(srcstripe.astype(np.uint8), dx_, dy_) > 0
            from_ok = np.roll(ringW & ~ringres, (-dy_, -dx_), axis=(0, 1))
            u2 = ringres & src_ok & from_ok
            if u2.any():
                stg[u2] = refc[_ix((dx_, dy_))][u2]
                ringres = ringres & ~u2
                moved = True
        k += 1
        if not moved:
            break
    if os.environ.get('DEBUG17'):
        print('ring clone px:', int((ringW & ~ringres).sum()), 'residual:', int(ringres.sum()), flush=True)
    W = W | ringres        # 条纹残差回水车道（留在原地=REF 鱼黄弧/紫唇残留，实测教训）


    def _fillzone(Wz):
        W = Wz
        nonlocal stg
        ringb = _dil(W, 24) & ~W & ~tubeE & ~zz & ~fp26 & ~fshad
        tmr = refc[ringb].mean(0) if ringb.sum() >= 64 else refc[_dil(W, 0)].mean(0)
        _dyv = (4, 8, 16, 34, 60, 90, 132, 176, 220, 260)
        offs = tuple((dx, s * dy) for dy in _dyv for s in (-1, 1)
                     for dx in (-8, -4, 0, 4, 8)) + \
            tuple((s * dx, dy) for dx in (34, 68, 100, 140, 180, 220, 260) for s in (-1, 1)
                  for dy in (-16, -8, 0, 8, 16))
        nw = max(int(W.sum()), 1)
        # 光斑密度基准（一轮裁决①「对齐光斑密度」）：环带水 vs 候选源高频能量
        _lgr = np.asarray(Image.open(REF).convert('L')).astype(np.float32)
        _hpr = _lgr - np.asarray(Image.fromarray(np.clip(_lgr, 0, 255).astype('uint8'))
                                 .filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
        thf = float(np.sqrt((_hpr[ringb] ** 2).mean()))
        bs, best = 1e18, None
        for o in offs:
            vm = (_shift(cleanband.astype(np.uint8), *o) > 0) & W
            cov = float(vm.sum()) / nw
            if cov < 0.30:
                continue
            sm = refc[_ix(o)][vm].mean(0)
            shf = float(np.sqrt((_hpr[vm] ** 2).mean()))
            s = float(np.abs(sm - tmr).mean()) + 0.004 * (abs(o[0]) + abs(o[1])) \
                + 40.0 * (1 - cov) + 0.8 * abs(shf - thf)   # 光斑密度权重需小(实测:3.0→5/11 更差)
            if s < bs:
                bs, best = s, o
        print('global off =', best, 'score=%.2f' % bs, flush=True)
        if best is not None:
            valid = (_shift(cleanband.astype(np.uint8), *best) > 0) & W
            stg[valid] = refc[_ix(best)][valid]
        rem = W & ~valid if best is not None else W.copy()
        # 全局偏移无效像素（源落圈体上）：沿**同一方向**取近处干净水（k 缩放），
        # 不做逐块配色；仍无效才进最近邻链（糊带=4-connected 蔓延实测）。
        if best is not None and rem.any():
            for kf in (0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 0.85, 1.2, 1.5, 2.0, 3.0):
                if not rem.any():
                    break
                ok_ = (round(best[0] * kf), round(best[1] * kf))
                vk = (_shift(cleanband.astype(np.uint8), *ok_) > 0) & rem
                if vk.any():
                    stg[vk] = refc[_ix(ok_)][vk]
                    rem = rem & ~vk
        if rem.any():
            rem = _clone(rem, srcok, offs)   # 最后兜底：全域有效 REF 平移（少量孤立像素）
        k = 0
        while rem.any() and k < 400:
            filled_m = W & ~rem
            k += 1
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                use = rem & np.roll(filled_m, (-dy, -dx), axis=(0, 1))
                if use.any():
                    stg[use] = np.roll(stg, (-dy, -dx), axis=(0, 1))[use]
                    filled_m = filled_m | use
                    rem = W & ~filled_m
            if k > 300 and rem.sum() == getattr(_dil, '_prev', -1):
                break
            _dil._prev = rem.sum()
        if rem.any():
            print('water rem after chain:', int(rem.sum()), flush=True)


    _holeD = np.asarray(mask17.hole_mask(fish, grow=25)) > 0
    _cyh = mask17.HOLE_ELL[fish][1]
    _yy0 = np.arange(tube.shape[0])[:, None]
    _Wbot = W & ~_holeD & (_yy0 > _cyh)          # 圈下影水（暗区）
    W = W & ~_Wbot                               # 其余（孔内+圈上）先补
    _fillzone(W)
    W = W | _Wbot
    _fillzone(_Wbot)                             # 影水单独全局平移（亮度=自身周边环带匹配）
    Wfull = (W | ringW).copy()
    # 边界仅 1px 羽化（裁决口径）：收缩 1px + 0.6σ 微羽化合成回 REF。
    _fm = Image.fromarray((Wfull * 255).astype('uint8')).filter(ImageFilter.MinFilter(3)) \
              .filter(ImageFilter.GaussianBlur(0.6))
    _fmm = np.asarray(_fm) > 100
    _out = np.asarray(refimg).copy()
    _out[_fmm] = np.clip(stg, 0, 255).astype('uint8')[_fmm]
    stage = Image.fromarray(_out)
    _p('water clone-filled')
    stage = tube_fill(fish, (base & tube) | (ringres & tube9), stage)
    _p('tube filled')
    stage = tube_row_med(fish, base & tube, stage)
    dst = stage_out or 'output/stage17-%s-inp.png' % fish
    stage.save(dst)
    print('stage ->', dst)
    return dst


def _unit_layer(fish, p):
    """单元 RGBA → 全幅 RGBA 层（缩放/倾斜/落位 + 贴入比打印）。"""
    cfg = POSE[fish]
    unit = Image.open(cfg['unit']).convert('RGBA')
    if cfg.get('flip'):
        unit = unit.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = unit.size
    s = cfg['s']
    Wu, Hu = max(1, int(round(w * s))), max(1, int(round(h * s)))
    u = unit.resize((Wu, Hu), Image.LANCZOS).rotate(-cfg['tilt'], expand=True, resample=Image.BICUBIC)
    # 裁决④：单元贴入前 alpha 边缘收缩 1px 再 0.6 羽化（消白晕）。
    _al = u.split()[3].filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
    u = Image.merge('RGBA', list(u.split()[:3]) + [_al])
    # bbox（旋转后）对齐 anchor
    bb = u.split()[3].getbbox()
    u = u.crop(bb)
    ax, ay = cfg['anchor']
    px, py = int(ax - u.size[0] / 2), int(ay - u.size[1] / 2)
    layer = Image.new('RGBA', REF_SIZE, (0, 0, 0, 0))
    layer.paste(u, (px, py))
    print('unit paste ratio: %s %.1fpx -> %.1fpx = %.2fx (≤1.0 规则按贴入高/单元高)'
          % (fish, h, u.size[1], s))
    return layer, u, (px, py)


def warm_mix(u, k):
    if k <= 0:
        return u
    a = np.asarray(u).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    tr, tg, tb = WARM
    m = a[:, :, 3] > 0
    for i, t in enumerate((tr, tg, tb)):
        ch = a[:, :, i]
        ch = (ch + (t - ch) * k).astype(np.int16)
        a[:, :, i] = np.where(m, ch, a[:, :, i])
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))


def region_composite(fish, stage_in=None, out=None, inpainted=None):
    """②unit ③front-band ④limb-top（+ZZ）。stage_in 若缺省=已含该区 inpaint 的 stage。"""
    base = Image.open(inpainted or stage_in).convert('RGB')
    ref = Image.open(REF).convert('RGB')
    cfg = POSE[fish]
    layer, u, (px, py) = _unit_layer(fish, cfg)
    ua = np.asarray(layer.split()[3])
    # 接触影：body 掩膜下移模糊压暗（紧贴身体，不拉雾带）
    sh_alpha, sh_dy = cfg['shadow']
    sh = Image.fromarray((ua > 40).astype('uint8') * 255).filter(ImageFilter.GaussianBlur(7))
    sh = sh.transform(REF_SIZE, Image.AFFINE, (1, 0, 0, 0, 1, -sh_dy))  # 下移
    base_a = np.asarray(base).astype(np.int16)
    wf = (np.asarray(sh).astype(np.float32) / 255.0) * sh_alpha
    wf[wf < 0.01] = 0.0   # 影缘量化：阈值外精确零改动
    base_a = (base_a * (1 - wf[:, :, None])).astype(np.uint8)
    out_im = Image.fromarray(base_a).copy()
    # ② 贴入单元（warm 混 + alpha 合成，边缘仅 1.5px 羽化=「必要时低强度修缝」）
    warmu = warm_mix(u, cfg['warm'])
    whole = Image.new('RGBA', REF_SIZE, (0, 0, 0, 0))
    whole.paste(warmu, (px, py))
    a = whole.split()[3].filter(ImageFilter.GaussianBlur(1.5))
    out_im = Image.composite(whole.convert('RGB'), out_im, a)
    # ③ 前缘带贴回样张像素（圈前缘压身）——但「旧肢体压管」区不贴回
    #   （旧鞋/旧鳍是鱼像素，已由 erase_mask 送补画；贴回会复发「鱼穿裤子」）
    fb = mask17.front_band(fish)
    _fbn = np.asarray(fb).copy()
    _fbn[np.asarray(mask17.old_limbs(fish, dil=3)) > 0] = 0
    fb = Image.fromarray(_fbn).filter(ImageFilter.GaussianBlur(1.5))
    out_im = Image.composite(ref, out_im, fb)
    # ④ 骨架肢体区：旧肢体 ∧ 新单元 → 压在管上
    old_limbs = mask17.old_limbs(fish, dil=cfg['limb_dil'])
    E = Image.composite(old_limbs, Image.new('L', REF_SIZE, 0), a.point(lambda v: 255 if v > 60 else 0))
    out_im = Image.composite(whole.convert('RGB'), out_im, E)
    # ③b 前缘带锁定：带内（除旧肢体压管区与新单元本体）硬性还原样张像素，
    #     杜绝任何羽化残差——「圈前缘像素动不得」的机器级保证。
    fb_raw = np.asarray(mask17.front_band(fish)) > 0
    _lk = fb_raw & (np.asarray(mask17.old_limbs(fish, dil=4)) == 0)
    _body = np.asarray(Image.composite(Image.new('L', REF_SIZE, 255),
              Image.new('L', REF_SIZE, 0), whole.split()[3].point(lambda v: 255 if v > 60 else 0))) > 0
    _lk = _lk & ~_body
    o_arr = np.asarray(out_im).copy()
    o_arr[_lk] = np.asarray(ref)[_lk]
    out_im = Image.fromarray(o_arr)
    # ZZ 字母（样张 z 序：字母在最前）
    if fish == 'fish1':
        zz = mask17.zz_letters()
        out_im = Image.composite(ref, out_im, zz)
    dst = out or 'output/stage17-%s-reg.png' % fish
    out_im.save(dst)
    print('region ->', dst)
    # 肢体区/掩膜调试图
    dbg = Image.composite(Image.new('RGB', REF_SIZE, (255, 255, 0)), out_im,
                          E.point(lambda v: int(v * 0.6)))
    dbg.save('output/g17-%s-limbs.png' % fish)
    return dst


def fit_preview(fish, unit, s, anchor, tilt, limb_dil=None, tag='fit'):
    """姿态对位预览：区域②③④直接叠在样张上（无补画），输出 2.4x 并排。"""
    import sbs16
    cfg = POSE[fish]
    cfg.update(unit=unit, s=s, tilt=tilt)
    cfg['anchor'] = tuple(anchor)
    if limb_dil is not None:
        cfg['limb_dil'] = limb_dil
    out = 'output/g17-%s-%s-reg.png' % (fish, tag)
    region_composite(fish, stage_in=REF, out=out)
    p = 'output/g17-%s-%s-sbs.png' % (fish, tag)
    sbs16.sbs(out, mask17.CROPS[fish], p, zoom=2.4, la='样张同区', lb='%s 姿态预览 %s' % (fish, tag))
    print('sbs ->', p)
    return p


def main_chain(i1, i2, i3, out):
    """三区合成依次叠加：以样张为底，逐区贴回区域完成图（羽化带覆盖）。"""
    ref = Image.open(REF).convert('RGB')
    stage = ref.copy()
    for f, p in (('fish1', i1), ('fish2', i2), ('fish3', i3)):
        reg = Image.open(p).convert('RGB')
        x0, y0, x1, y1 = mask17.CROPS[f]
        m = mask17.footprint(f, margin=26).filter(ImageFilter.GaussianBlur(6))
        # 前缘带外沿 + 肢体骨架 + ZZ 也要带过去 → 掩膜并上这些区
        m = Image.composite(Image.new('L', REF_SIZE, 255), m, mask17.front_band(f))
        if f == 'fish1':
            m = Image.composite(Image.new('L', REF_SIZE, 255), m, mask17.zz_letters())
        stage.paste(reg.crop((x0, y0, x1, y1)), (x0, y0), m.crop((x0, y0, x1, y1)))
    stage.save(out)
    print('stage ->', out)
    return out


def outpaint(inp, out, seed=20261108, seam=40):
    """stage(1080x1920) → LANCZOS 1.333x → 上下 280px pad inpaint → 1440x3120 + 接缝平滑。
       （同 compose16.outpaint 口径；此处直接复用其实现）"""
    import compose16
    compose16.outpaint(inp, out, seed=seed, seam=seam)


def pipeline(seeds, out='output/stage17.png', tag='p'):
    """全链：三区依次 inpaint → region（串联 stage）。"""
    stage = REF
    for i, f in enumerate(('fish1', 'fish2', 'fish3')):
        st = inpaint_region(f, seeds[i], stage_in=stage,
                            stage_out='output/g17-%s-%s-inp.png' % (tag, f))
        stage = region_composite(f, stage_in=st, out='output/g17-%s-%s-reg.png' % (tag, f))
    import shutil
    shutil.copy(stage, out)
    print('pipeline ->', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a0 = sub.add_parser('inpaint')
    a0.add_argument('--fish', required=True); a0.add_argument('--seed', type=int, required=True)
    a0.add_argument('--upscale', type=float, default=1.7)
    a0.add_argument('--stage-in'); a0.add_argument('--stage-out')
    a1 = sub.add_parser('region')
    a1.add_argument('--fish', required=True)
    a1.add_argument('--stage-in', required=True); a1.add_argument('--out')
    a2 = sub.add_parser('chain')
    a2.add_argument('--i1', required=True); a2.add_argument('--i2', required=True)
    a2.add_argument('--i3', required=True); a2.add_argument('--out', required=True)
    a3 = sub.add_parser('outpaint')
    a3.add_argument('--in', dest='inp', required=True); a3.add_argument('--out', required=True)
    a3.add_argument('--seed', type=int, default=20261108); a3.add_argument('--seam', type=int, default=40)
    a5 = sub.add_parser('iter')
    a5.add_argument('--fish', required=True); a5.add_argument('--unit', required=True)
    a5.add_argument('--s', type=float, required=True)
    a5.add_argument('--anchor', required=True); a5.add_argument('--tilt', type=float, default=0.0)
    a5.add_argument('--limb-dil', type=int, default=None); a5.add_argument('--tag', default='fit')
    a5.add_argument('--flip', type=int, default=None)
    a4 = sub.add_parser('maskprobe')
    a = ap.parse_args()
    if a.cmd == 'inpaint':
        inpaint_region(a.fish, a.seed, a.upscale, a.stage_in, a.stage_out)
    elif a.cmd == 'region':
        region_composite(a.fish, stage_in=a.stage_in, out=a.out)
    elif a.cmd == 'chain':
        main_chain(a.i1, a.i2, a.i3, a.out)
    elif a.cmd == 'outpaint':
        outpaint(a.inp, a.out, a.seed, a.seam)
    elif a.cmd == 'iter':
        x, y = (int(v) for v in a.anchor.split(','))
        if a.flip is not None:
            compose17_flip = POSE[a.fish]
            compose17_flip['flip'] = bool(a.flip)
        fit_preview(a.fish, a.unit, a.s, (x, y), a.tilt, a.limb_dil, a.tag)
    elif a.cmd == 'maskprobe':
        for f in mask17.CROPS:
            fp = mask17.footprint(f)
            Image.composite(Image.new('RGB', REF_SIZE, (255, 40, 40)),
                            Image.open(REF).convert('RGB'),
                            fp.point(lambda v: int(v * 0.5))).save('output/m17-%s-fp.png' % f)
        print('ok')
