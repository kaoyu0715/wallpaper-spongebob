# -*- coding: utf-8 -*-
# 十五代 B 路线·分区法（issue #31）：整图分区重绘的落地形式 = 以泳圈为中心的方形裁片，
# 放大后整片 img2img（denoise 分区控制），再等比缩回、羽化贴回样张底。
# 与「掩膜 inpaint」的差别（实测）：掩膜法把重绘限制在鱼形孔内→姿态被压成「脸贴圈」；
# 裁片法让模型拥有完整小场景先验→能出「全身瘫坐入圈、腿压前缘」。
# 用法: python3 gen15crop.py --fish fish1 --seeds 22,44 [--denoise 0.85] [--scale 1.7]
import argparse, json, os, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask15
from gen15b import post, wait, out_of, CHARS, prepare

T = os.path.dirname(os.path.abspath(__file__))

# 分区裁片（swim-ref 坐标）：以圈为主、四邻留水；fish1 右上含少量蓝圈左缘——
# 低 denoise 下相邻圈大体原位，羽化贴回吸收接缝。
CROPS = {
  'fish1': (40, 770, 660, 1420),    # x0,y0,x1,y1 620x650
  'fish2': (470, 560, 1060, 1180),  # 590x620
  'fish3': (380, 1000, 1000, 1600), # 620x600
}


def run(fish, seeds, denoise=0.85, scale=1.7, tag='crop', base='g15in/swimref.png', cfg=5.5, mask=None):
    c = CHARS[fish]
    x0, y0, x1, y1 = CROPS[fish]
    w, h = x1 - x0, y1 - y0
    W, H = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': c['lora'], 'strength_model': c['strength'], 'strength_clip': c['strength']}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': c['pos']}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': c['neg']}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g15in/swimref.png'}},
      '7': {'class_type': 'ImageCrop', 'inputs': {'image': ['6', 0], 'x': x0, 'y': y0,
              'width': w, 'height': h}},
      '8': {'class_type': 'ImageScale', 'inputs': {'image': ['7', 0], 'upscale_method': 'lanczos',
              'width': W, 'height': H, 'crop': 'center'}},
    }
    if mask:
        g['85'] = {'class_type': 'LoadImage', 'inputs': {'image': mask}}
        g['86'] = {'class_type': 'ImageToMask', 'inputs': {'image': ['85', 0], 'channel': 'red'}}
        g['9'] = {'class_type': 'InpaintModelConditioning', 'inputs': {'positive': ['4', 0],
                'negative': ['5', 0], 'vae': ['1', 2], 'pixels': ['8', 0], 'mask': ['86', 0],
                'noise_mask': True}}
    else:
        g['9'] = {'class_type': 'VAEEncode', 'inputs': {'pixels': ['8', 0], 'vae': ['1', 2]}}
    outs = []
    pos_ref, neg_ref, lat_ref = (['4', 0], ['5', 0], ['9', 0]) if not mask else (['9', 0], ['9', 1], ['9', 2])
    for s in seeds:
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': pos_ref,
                    'negative': neg_ref, 'latent_image': lat_ref, 'seed': int(s), 'steps': 30,
                    'cfg': cfg, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': denoise}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
                    'filename_prefix': f'g15crop/{tag}-{fish}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        p = out_of(e, '12')
        print('crop', fish, 'seed', s, '->', p, flush=True)
        outs.append(p)
    return outs


def paste_back(base_path, crop_path, fish, feather=28):
    """缩回原尺寸、圆角羽化贴回。返回 PIL 图。"""
    x0, y0, x1, y1 = CROPS[fish]
    w, h = x1 - x0, y1 - y0
    base = Image.open(base_path).convert('RGB')
    crop = Image.open(crop_path).convert('RGB').resize((w, h), Image.LANCZOS)
    m = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([feather, feather, w - feather, h - feather], radius=48, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(feather * 0.75))
    base.paste(crop, (x0, y0), m)
    return base


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fish', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--denoise', type=float, default=0.85)
    ap.add_argument('--scale', type=float, default=1.7)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--base', default='g15in/swimref.png')
    ap.add_argument('--cfg', type=float, default=5.5)
    ap.add_argument('--mask', default=None)
    a = ap.parse_args()
    # CHARS 无 neg 键时补基线负面
    for k, v in CHARS.items():
        v.setdefault('neg', __import__('gen15b').NEG_BASE + ', ' + v['neg_extra'])
    prepare()
    run(a.fish, [int(s) for s in a.seeds.split(',')], a.denoise, a.scale, a.tag, a.base, a.cfg, a.mask)
