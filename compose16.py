# -*- coding: utf-8 -*-
# 十六代终装（issue #33）：链式贴回（统一紧凑档）+ 出画（1440x3120）+ 接缝平滑。
# 用法:
#   python3 compose16.py chain --c1 <crop1> --c2 <crop2> --c3 <crop3> --out output/stage-final.png
#   python3 compose16.py outpaint --in output/stage-final.png --out <out.png> [--seed 20261108] [--seam 25]
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageFilter
import mask16

T = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(T, 'review/reference/swim-ref.jpg')


def chain(c1, c2, c3, out):
    img = mask16.sil_paste_back(Image.open(REF), c1, 'fish1',
                                keep_img=mask16.zz_keep(dil=5, feather=3))
    img = mask16.sil_paste_back(img, c2, 'fish2')
    img = mask16.sil_paste_back(img, c3, 'fish3', hole=False)
    img.save(out)
    print('chained ->', out)
    return img


def outpaint(inp, out, seed=20261108, seam=0):
    """1080x1920 → LANCZOS 1.333x → 上下扩展 latent → 分区 img2img（pad-only 噪声）。"""
    import shutil
    from gen16crop import post, wait, out_of, INPUT
    img = Image.open(inp).convert('RGB')
    W, H = 1440, 2560
    img = img.resize((W, H), Image.LANCZOS)
    stage = os.path.join(INPUT, 'g16in')
    os.makedirs(stage, exist_ok=True)
    name = 'g16out-' + os.path.basename(out).replace('.png', '')
    img.save(os.path.join(stage, name + '.png'))
    PAD_T, PAD_B = 280, 280
    pos = ('sunset sky with soft clouds, warm golden light, calm sea and sandy beach with sea foam, '
           'painterly flat 2d cartoon background, no people, no characters, no text, no watermark')
    neg = ('text, watermark, signature, people, characters, person, swim ring, lowres, worst quality, '
           '3d, photorealistic')
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': pos}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g16in/' + name + '.png'}},
      '7': {'class_type': 'ImagePadForOutpaint', 'inputs': {'image': ['6', 0], 'left': 0, 'top': PAD_T,
              'right': 0, 'bottom': PAD_B, 'feathering': 64}},
      '9': {'class_type': 'InpaintModelConditioning', 'inputs': {'positive': ['4', 0], 'negative': ['5', 0],
              'vae': ['1', 2], 'pixels': ['7', 0], 'mask': ['7', 1], 'noise_mask': True}},
      '10': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['9', 0], 'negative': ['9', 1],
               'latent_image': ['9', 2], 'seed': seed, 'steps': 30, 'cfg': 5.5, 'sampler_name': 'dpmpp_2m',
               'scheduler': 'karras', 'denoise': 1.0}},
      '11': {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}},
      '12': {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0], 'filename_prefix': 'g16out/' + name}},
    }
    e = wait(post('/prompt', {'prompt': g})['prompt_id'])
    p = out_of(e, '12')
    im = Image.open(p).convert('RGB')
    print('outpaint raw', im.size)
    im = im.resize((1440, 3120), Image.LANCZOS)
    if seam:
        im = smooth_seam(im, y=2840, half=seam)
        print('seam smoothed at y=2840 +/-', seam)
    im.save(out)
    print('outpaint ->', out, im.size)


def smooth_seam(im, y=2840, half=25, radius=6):
    """上下出画带与保留带交界：三角窗纵向高斯混光，消横缝（十五代成品此处可见接缝）。"""
    blurred = im.filter(ImageFilter.GaussianBlur(radius))
    out = im.copy()
    px, bx, ox = im.load(), blurred.load(), out.load()
    for dy in range(-half, half + 1):
        yy = y + dy
        if not 0 <= yy < im.height:
            continue
        w = 1.0 - abs(dy) / half
        for x in range(im.width):
            r1, g1, b1 = px[x, yy]
            r2, g2, b2 = bx[x, yy]
            ox[x, yy] = (int(r1 + (r2 - r1) * w), int(g1 + (g2 - g1) * w), int(b1 + (b2 - b1) * w))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('chain')
    a1.add_argument('--c1', required=True); a1.add_argument('--c2', required=True)
    a1.add_argument('--c3', required=True); a1.add_argument('--out', required=True)
    a2 = sub.add_parser('outpaint')
    a2.add_argument('--in', dest='inp', required=True); a2.add_argument('--out', required=True)
    a2.add_argument('--seed', type=int, default=20261108)
    a2.add_argument('--seam', type=int, default=25)
    a = ap.parse_args()
    if a.cmd == 'chain':
        chain(a.c1, a.c2, a.c3, a.out)
    else:
        outpaint(a.inp, a.out, a.seed, a.seam)
