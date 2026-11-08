# -*- coding: utf-8 -*-
# 内圈水体补画（十六代·issue #33「派大星内圈死蓝」根治）：
# 在 crop 内，mask = 内孔椭圆 − 角色剪影（chroma：粉肤/绿裤/橙汁/白杯，膨胀12），
# InpaintModelConditioning d1.0 重绘为周围同款海水（波纹+星芒，非平色）。
import sys, os, json
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import gen16crop as G
import mask16

IN_DIR = G.IN_DIR


def silhouette(arr):
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    pink = (r > 170) & (r - b > 45) & (r - g > 35)                      # 粉肤/粉脚
    green = (g - r > 15) & (g - b > 40)                                 # 绿短裤
    orange = (r > 200) & (g > 90) & (g < 190) & (b < 110)               # 橙汁/橙片
    white = (r > 225) & (g > 225) & (b > 225)                           # 杯体白色
    m = pink | green | orange | white
    return Image.fromarray((m * 255).astype('uint8'))


def hole_mask(crop, fish, feather=3, dil=14):
    w, h = crop.size
    x0, y0, x1, y1 = mask16.CROPS[fish]
    sc = w / (x1 - x0)
    cx, cy, a, b = mask16.OPEN_ELL[fish]
    big = Image.new('L', (w*2, h*2), 0)
    d = ImageDraw.Draw(big)
    d.ellipse([2*((cx-a-x0)*sc - dil), 2*((cy-b-y0)*sc - dil),
               2*((cx+a-x0)*sc + dil), 2*((cy+b-y0)*sc + dil)], fill=255)
    d.polygon([(2*(p[0]-x0)*sc, 2*(p[1]-y0)*sc) for p in mask16._poly(fish, 18)], fill=255)
    for (ex, ey, ea, eb) in mask16.EXTRA_ELL.get(fish, []):
        d.ellipse([2*(ex-ea-x0)*sc, 2*(ey-eb-y0)*sc, 2*(ex+ea-x0)*sc, 2*(ey+eb-y0)*sc], fill=255)
    for (kx, ky, ka, kb) in mask16.KEEP_ELL.get(fish, []):
        ImageDraw.Draw(big).ellipse([2*(kx-x0)*sc, 2*(ky-y0)*sc, 2*(kx+ka-x0)*sc, 2*(ky+kb-y0)*sc], fill=0)
    m = big.resize((w, h), Image.LANCZOS)
    sil = mask16._chroma(crop, fish).filter(ImageFilter.MaxFilter(2*dil+1))
    sil = sil.filter(ImageFilter.GaussianBlur(2))
    m = Image.composite(Image.new('L', (w, h), 0), m, sil)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return m


def region_mask(crop, fish, ellipses, feather=4, dil=0):
    """ellipses: [(cx,cy,a,b)] 参考帧坐标 → crop 分辨率并集 mask。"""
    w, h = crop.size
    x0, y0, x1, y1 = mask16.CROPS[fish]
    sc = w / (x1 - x0)
    m = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(m)
    for cx, cy, a, b in ellipses:
        d.ellipse([(cx-a-x0)*sc, (cy-b-y0)*sc, (cx+a-x0)*sc, (cy+b-y0)*sc], fill=255)
    if dil:
        m = m.filter(ImageFilter.MaxFilter(2*dil+1))
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return m


def reg_inpaint(crop_path, fish, ellipses, pos, neg, seed, denoise=1.0, tag='r1', name=None):
    crop = Image.open(crop_path).convert('RGB')
    base = (name or (tag + '-' + os.path.basename(crop_path).replace('.png', '')))
    crop.save(os.path.join(IN_DIR, base + '.png'))
    base = 'g16in/' + base
    m = region_mask(crop, fish, ellipses)
    mname = base + '-rmask.png'
    m.save(os.path.join(IN_DIR, os.path.basename(mname)))
    m.save('output/%s-rmask.png' % os.path.basename(base))
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': pos}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': base + '.png'}},
      '7': {'class_type': 'LoadImage', 'inputs': {'image': mname}},
      '7b': {'class_type': 'ImageToMask', 'inputs': {'image': ['7', 0], 'channel': 'red'}},
      '8': {'class_type': 'InpaintModelConditioning',
            'inputs': {'positive': ['4', 0], 'negative': ['5', 0], 'vae': ['1', 2],
                       'pixels': ['6', 0], 'mask': ['7b', 0], 'noise_mask': True}},
      '10': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['8', 0],
              'negative': ['8', 1], 'latent_image': ['8', 2], 'seed': int(seed), 'steps': 30,
              'cfg': 5.5, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': denoise}},
      '11': {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}},
      '12': {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
              'filename_prefix': f'g16hole/{base.replace("/", "-")}-{seed}'}},
    }
    e = G.wait(G.post('/prompt', {'prompt': g})['prompt_id'])
    p = G.out_of(e, '12')
    print('reg_inpaint', fish, 'seed', seed, '->', p, flush=True)
    return p


def inpaint(crop_path, fish, seed, denoise=1.0, tag='h1', prompt_key='water', pos=None, neg=None):
    PROMPTS = {
      'water': ('calm teal sea water with soft ripples and subtle white star sparkles, gentle '
                'shadow around body edges, open sea surface, no horizon, no sky'),
    }
    crop = Image.open(crop_path).convert('RGB')
    w, h = crop.size
    base = os.path.basename(crop_path).replace('.png', '')
    name = tag + '-' + base
    crop.save(os.path.join(IN_DIR, name + '.png'))
    name = 'g16in/' + name + '.png'
    m = hole_mask(crop, fish)
    mname = name[:-4] + '-mask.png'
    m.save(os.path.join(IN_DIR, os.path.basename(mname)))
    m.save('output/%s-holemask.png' % os.path.basename(name))
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': pos or PROMPTS[prompt_key]}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0],
              'text': neg or ('flat solid color, solid fill, white foam, clouds, body parts, arm, leg, hand, '
                              'foot, donut, ring tube, character, object')}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': name}},
      '7': {'class_type': 'LoadImage', 'inputs': {'image': mname}},
      '7b': {'class_type': 'ImageToMask', 'inputs': {'image': ['7', 0], 'channel': 'red'}},
      '8': {'class_type': 'InpaintModelConditioning',
            'inputs': {'positive': ['4', 0], 'negative': ['5', 0], 'vae': ['1', 2],
                       'pixels': ['6', 0], 'mask': ['7b', 0], 'noise_mask': True}},
      '10': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['8', 0],
              'negative': ['8', 1], 'latent_image': ['8', 2], 'seed': int(seed), 'steps': 30,
              'cfg': 5.5, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': denoise}},
      '11': {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}},
      '12': {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
              'filename_prefix': 'g16hole/' + tag + '-' + fish + '-' + str(seed)}},
    }
    e = G.wait(G.post('/prompt', {'prompt': g})['prompt_id'])
    p = G.out_of(e, '12')
    print('inpaint', fish, 'seed', seed, '->', p, flush=True)
    return p


if __name__ == '__main__':
    a = sys.argv
    inpaint(a[1], a[2], int(a[3]), float(a[4]) if len(a) > 4 else 1.0,
            a[5] if len(a) > 5 else 'h1')
