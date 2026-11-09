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
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask17
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
  'fish1': dict(unit=POSDIR + '/sb-ring17f-sb-ring17f-7153_00001__00001-post.png',
                anchor=(272, 1062), s=0.235, tilt=2, flip=True, limb_dil=9,
                shadow=(0.16, 8), warm=0.07),
  'fish2': dict(unit=POSDIR + '/pat-ring17-pat-ring17-7204_00001__00001-post.png',
                anchor=(720, 820), s=0.28, tilt=4, flip=True, limb_dil=9,
                shadow=(0.16, 8), warm=0.07),
  'fish3': dict(unit=POSDIR + '/san-ring17c-san-ring17c-7322_00001__00001-post.png',
                anchor=(665, 1252), s=0.285, tilt=-15, limb_dil=9,
                shadow=(0.16, 8), warm=0.07),
}

WARM = (255, 214, 178)

INPAINT_POS = ('empty swim ring floating on calm teal sea with nobody inside, empty ring hole filled '
               'with calm teal sea water and soft ripples, gentle current shading, subtle darker '
               'shadow near where a body was, wherever the body was in front of the far ring tube '
               'the striped swim ring tube continues naturally, open sea surface, no horizon, no sky')
INPAINT_NEG = ('flat solid color, solid fill, white foam, clouds, body parts, arm, leg, hand, foot, '
               'fish, fins, character, donut, ring tube in front, object, person, text, '
               'person, human, girl, boy, woman, man, child, face, hair, eyes, nose, mouth, '
               'skin, neck, shoulders, chest, collarbone, bikini, swimsuit, swimwear, shirt, clothes')


def _ell_img(mask, box):
    return mask.crop(box)


def inpaint_region(fish, seed, upscale=1.7, stage_in=None, stage_out=None, mask_only=False):
    """① erase-inpaint：在 stage 裁片上对 erase 掩膜 d1.0 重绘海水/管后带。"""
    stage = Image.open(stage_in) if stage_in else Image.open(REF).convert('RGB')
    x0, y0, x1, y1 = mask17.CROPS[fish]
    crop = stage.crop((x0, y0, x1, y1))
    W, H = int((x1 - x0) * upscale) // 8 * 8, int((y1 - y0) * upscale) // 8 * 8
    m = mask17.erase_mask(fish).crop((x0, y0, x1, y1)).resize((W, H), Image.LANCZOS)
    name = 'g17in-%s-%d' % (fish, seed)
    d = os.path.join(G.INPUT, 'g17in')
    os.makedirs(d, exist_ok=True)
    big = crop.resize((W, H), Image.LANCZOS)
    # Fooocus 式预填充——归一化卷积：颜色只从掩膜外（水/管）扩散进掩膜，
    # 旧鱼色彩不参与（blur30 直接模糊含鱼图会留鱼形鬼影——512/522 号实测教训），
    # 随后 d0.8 在该无鱼底色上重建（d1.0 大掩膜又会画出新人/新圈——501/511/521 教训）。
    mh = m.point(lambda v: 255 if v > 127 else 0)
    barr = np.asarray(big).astype(np.float32)
    wa = 255.0 - np.asarray(m).astype(np.float32)          # 软权重：掩膜内=0
    fill = barr.copy()
    mh_b = np.asarray(mh) > 127
    for r in (15, 40, 100, 240):          # 多尺度传播，直到掩膜深处也有颜色
        r = int(r * upscale)
        den = np.asarray(Image.fromarray(wa.astype('uint8')).filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
        ok = den > 30.0
        den = np.maximum(den, 1.0)
        for i in range(3):
            num = np.asarray(Image.fromarray(np.clip(barr[:, :, i] * wa / 255.0, 0, 255).astype('uint8'))
                             .filter(ImageFilter.GaussianBlur(r))).astype(np.float32)
            v = np.clip(num * 255.0 / den, 0, 255)
            fill[:, :, i] = np.where(ok, v, fill[:, :, i])
    pre = Image.fromarray(fill.astype('uint8'))
    big = Image.composite(pre, big, mh)
    big.save(os.path.join(d, name + '.png'))
    m.save(os.path.join(d, name + '-mask.png'))
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': INPAINT_POS}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': INPAINT_NEG}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g17in/' + name + '.png'}},
      '7': {'class_type': 'LoadImage', 'inputs': {'image': 'g17in/' + name + '-mask.png'}},
      '7b': {'class_type': 'ImageToMask', 'inputs': {'image': ['7', 0], 'channel': 'red'}},
      '8': {'class_type': 'InpaintModelConditioning', 'inputs': {'positive': ['4', 0], 'negative': ['5', 0],
              'vae': ['1', 2], 'pixels': ['6', 0], 'mask': ['7b', 0], 'noise_mask': True}},
      '10': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['8', 0], 'negative': ['8', 1],
               'latent_image': ['8', 2], 'seed': int(seed), 'steps': 30, 'cfg': 5.5,
               'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 0.8}},
      '11': {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}},
      '12': {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0], 'filename_prefix': 'g17hole/' + name}},
    }
    import gen17unit
    e = G.wait(gen17unit._submit(g))
    p = G.out_of(e, '12')
    print('inpaint', fish, 'seed', seed, '->', p, flush=True)
    if mask_only:
        return p
    im = Image.open(p).convert('RGB').resize((x1 - x0, y1 - y0), Image.LANCZOS)
    out = stage.convert('RGB').copy()
    pm = m.resize((x1 - x0, y1 - y0), Image.LANCZOS).filter(ImageFilter.MinFilter(9))  # 内收4px
    pm = pm.point(lambda v: 255 if v > 200 else (0 if v < 20 else v))
    pm = pm.filter(ImageFilter.GaussianBlur(2))
    out.paste(im, (x0, y0), pm)
    dst = stage_out or 'output/stage17-%s-inp.png' % fish
    out.save(dst)
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
