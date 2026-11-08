# -*- coding: utf-8 -*-
# 十六代 B 路线·分区生成器（issue #33）：十五代返工——
#  ① 分辨率提档：裁片放大 2.6x（十五代 1.7x）后整片 img2img，缩回贴回即等效 hires，
#     角色/内孔水面细节密度上到样张档；
#  ② 提示词按样张同区补「细节词」：内孔水面=同周围海水+星芒（派大星死蓝底修复项）、
#     珊迪头盔=细白线圆圈全透明、花环=垂在锁骨；
#  ③ --mask 支持局部高 denoise 重绘（内孔二次细化 pass）。
# 管线口径沿用 recipe：illustriousXL, CLIP -2, 30 步 dpmpp_2m karras cfg 5.5, 角色 LoRA。
# 用法: python3 gen16crop.py --fish fish2 --seeds 12,34 [--denoise 0.88] [--scale 2.6]
#        [--cfg 5.5] [--steps 35] [--tag v1] [--base g16in/swimref.png] [--mask g16in/xx.png]
import argparse, json, os, sys, time, shutil, urllib.request
from PIL import Image

API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'
T = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(INPUT, 'g16in')

import mask16

NEG_BASE = ('watermark, signature, lowres, worst quality, 3d, photorealistic, realistic, '
            'fish, fins, tail fin, gills, scales, two characters, multiple characters, extra characters, '
            'extra eyes, missing arm, missing hands, missing leg, missing feet, extra limbs, extra arms, '
            'deformed hands, blob hands, malformed hands, cropped, cut off, out of frame, partial body, '
            'blurry, jpeg artifacts')

STYLE = 'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo'

# 十六代 v2 实测教训（seed 1234/2345/3456 等，scale2.6/d0.88 + star sparkles 词）：
# 高倍放大+闪光词会让模型重解构图（圈缩小移位、海面变泡沫云）。
# 一轮=沿用十五代构图锚定档（1.7x d0.88）+ 明确左右方位词；细节密度由 refine() 二次
# hires 低 denoise pass 提档（issue #33「二次细化 pass」指定手段）。
INNER_WATER = ('wherever his body does not cover it the ring opening is filled with calm teal sea '
               'water blending continuously with the surrounding sea, no flat solid color fill, ')
FOAM_NEG = 'sea foam, foam bubbles, splashes, white clouds, '

CHARS = {
  'fish1': dict(lora='sbpilot.safetensors', strength=0.85, out='sb', cfg=5.5,
    pos=('spongebob squarepants, yellow porous sponge, square body, official cartoon proportions, '
         'big blue eyes, buck teeth, white shirt, red tie, brown shorts, red striped socks, black shoes, '
         'black sunglasses resting pushed up on forehead, relaxed lazy smile, half-lidded sleepy eyes, '
         'sitting deep inside a red and white striped inflatable swim ring on calm sea, '
         'hips in the ring, leaning back lazily, '
         'head and torso on the left side of the ring, legs bent with ankles crossed resting on the '
         'front tube toward the right, feet up on the tube not hanging down, '
         'one arm draped over the left tube, '
         'large yellow cartoon ZZZ snoring letters painted on the front of the ring tube, '
         'surrounded by calm open sea surface, no horizon, no sky, ' + STYLE),
    neg_extra=('standing, waving hand, wide open mouth, tongue, sunglasses over eyes, '
               'legs dangling in water, legs hanging below ring, horizon, sky')),
  'fish2': dict(lora='Patrick_il.safetensors', strength=1.0, out='pat', cfg=6.0,
    pos=('p4tr1ckst4r, official style, starfish, pink skin, green shorts, '
         'content expression, half-closed eyes, small closed-mouth smile, '
         'reclining deep inside a blue and white striped inflatable swim ring, bold blue and white '
         'stripes on the ring tube, on calm sea, '
         'leaning back lazily with his head low toward the upper right of the ring, torso slouched deep '
         'into the ring opening, '
         'one hand holding up a tall glass half full of orange juice with ice and a straw and an orange '
         'slice raised at the left side, other arm draped relaxed over the right tube, '
         'legs stretched toward the lower left resting on the inner front tube with pink feet on the '
         'tube, feet on the tube not dangling, ' + INNER_WATER +
         'surrounded by calm open sea surface, no horizon, no sky, ' + STYLE),
    neg_extra=('standing, sitting upright, upright torso, head high above the ring, waving, '
               'wide open mouth, missing glass, empty hand, legs dangling in water, horizon, sky, '
               'beach, sand, shore, plain blue ring, solid color ring, aqua ring, '
               'flat solid blue inside the ring, empty blue ring center, solid blue fill behind body, '
               'teal feet, green feet, webbed feet, fin on head')),
  'fish3': dict(lora='Sandy_il.safetensors', strength=1.0, out='sandy', cfg=5.5,
    pos=('sandycheeks, squirrel in white astronaut suit, pink and white flower lei garland worn '
         'low around her neck hanging on her collarbone, '
         'round transparent glass space helmet over her head, drawn as a thin white circle outline only, '
         'helmet interior perfectly transparent with the sea background clearly visible through it, '
         'standard squirrel face, buck teeth, gentle relaxed smile, calm half-lidded eyes, '
         'sitting deep inside a blue and white striped inflatable swim ring on calm sea, '
         'head toward the upper right of the ring, leaning back relaxed, '
         'one hand holding a whole brown coconut drink with a bendy straw at her lap on the left side, '
         'other arm draped relaxed over the right tube, '
         'legs crossed stretched toward the lower left resting on top of the front tube of the ring, '
         'feet on the tube not dangling, ' + INNER_WATER +
         'surrounded by calm blue-green open sea water with ripples, no beach, no sand, no horizon, no sky, ' + STYLE),
    neg_extra=('flower on the ring, flowers floating in water, flower near the mouth, flowers in front of '
               'the face, flower on the helmet, grey helmet, opaque helmet, thick helmet rim, closed helmet '
               'cover, headscarf, turban, headband, black strap, helmet glare, standing, '
               'legs dangling in water, horizon, sky, cocktail, cocktail glass, pineapple slice, '
               'lemon slice, paper umbrella, drink on the ring')),
}
# 去花环变体：负向词加花环禁令，正向删 lei 句（宁可无，不可错位——见 issue #33）
NO_LEI_POS = ('pink and white flower lei garland worn low around her neck hanging on her collarbone, ')
NO_LEI_NEG = 'flower lei, lei garland, necklace, flowers around neck, flowers, '


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def wait(pid, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(urllib.request.urlopen(API + f'/history/{pid}', timeout=10).read())
            e = h.get(pid)
            if e and e.get('outputs'):
                return e
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(pid)


def out_of(e, node):
    im = e['outputs'].get(node, {}).get('images', [None])[0]
    return os.path.join(OUTDIR, im['subfolder'], im['filename']) if im else None


def prepare():
    os.makedirs(IN_DIR, exist_ok=True)
    ref = os.path.join(T, 'review/reference/swim-ref.jpg')
    Image.open(ref).convert('RGB').save(os.path.join(IN_DIR, 'swimref.png'))


def upload_mask(name, pil_img):
    p = os.path.join(IN_DIR, name)
    pil_img.save(p)
    return 'g16in/' + name


def run(fish, seeds, denoise=0.88, scale=1.7, tag='v1', base='g16in/swimref.png',
        cfg=None, steps=30, mask=None, lei=True):
    c = CHARS[fish]
    x0, y0, x1, y1 = mask16.CROPS[fish]
    w, h = x1 - x0, y1 - y0
    W, H = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    pos, neg = c['pos'], NEG_BASE + ', ' + FOAM_NEG + c['neg_extra']
    if fish == 'fish3' and not lei:
        pos = pos.replace(NO_LEI_POS, '')
        neg = NO_LEI_NEG + neg
    cfg = cfg if cfg is not None else c.get('cfg', 5.5)
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': c['lora'], 'strength_model': c['strength'], 'strength_clip': c['strength']}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': pos}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': base}},
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
        pos_ref, neg_ref, lat_ref = ['9', 0], ['9', 1], ['9', 2]
    else:
        g['9'] = {'class_type': 'VAEEncode', 'inputs': {'pixels': ['8', 0], 'vae': ['1', 2]}}
        pos_ref, neg_ref, lat_ref = ['4', 0], ['5', 0], ['9', 0]
    outs = []
    for s in seeds:
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': pos_ref,
                    'negative': neg_ref, 'latent_image': lat_ref, 'seed': int(s), 'steps': steps,
                    'cfg': cfg, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': denoise}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
                    'filename_prefix': f'g16crop/{tag}-{fish}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        p = out_of(e, '12')
        print('crop', fish, 'seed', s, '->', p, flush=True)
        outs.append(p)
    return outs


def rework(src_img, fish, seeds, denoise=0.5, outscale=2.35, tag='w1', steps=25,
           pos_extra='', neg_extra='', lei=True):
    """返工 pass（十六代主打）：以十五代定型 crop 为底、中等 denoise img2img——
    构图锚定保留，修形体/头盔/杂物。src_img 可为 crop 文件路径或 'g16in/...'。"""
    c = CHARS[fish]
    x0, y0, x1, y1 = mask16.CROPS[fish]
    w, h = x1 - x0, y1 - y0
    W, H = int(w * outscale) // 8 * 8, int(h * outscale) // 8 * 8
    if os.path.isabs(src_img):
        stage = 'g16in/rework-' + os.path.basename(src_img)
        shutil.copy(src_img, os.path.join(IN_DIR, os.path.basename(stage)))
    else:
        stage = src_img
    pos = c['pos'] + (', ' + pos_extra if pos_extra else '')
    neg = NEG_BASE + ', ' + FOAM_NEG + c['neg_extra'] + (', ' + neg_extra if neg_extra else '')
    if fish == 'fish3' and not lei:
        pos = pos.replace(NO_LEI_POS, '')
        neg = NO_LEI_NEG + neg
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': c['lora'], 'strength_model': c['strength'], 'strength_clip': c['strength']}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': pos}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': stage}},
      '8': {'class_type': 'ImageScale', 'inputs': {'image': ['6', 0], 'upscale_method': 'lanczos',
              'width': W, 'height': H, 'crop': 'center'}},
      '9': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['8', 0], 'vae': ['1', 2]}},
    }
    outs = []
    for s in seeds:
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['9', 0], 'seed': int(s), 'steps': steps,
                    'cfg': c.get('cfg', 5.5), 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras',
                    'denoise': denoise}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
                    'filename_prefix': f'g16refine/{tag}-{fish}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        p = out_of(e, '12')
        print('rework', fish, 'seed', s, '->', p, flush=True)
        outs.append(p)
    return outs


def refine(crop_path, fish, seeds, denoise=0.30, outscale=2.35, tag='r1', steps=25, lei=True):
    """二次细化 hires pass（issue #33 手段③）：一轮候选放大到高分辨率后低 denoise 重绘——
    提细节密度（描边/褶皱/高光渐变/水波星芒），构图形体锁定不动。"""
    c = CHARS[fish]
    x0, y0, x1, y1 = mask16.CROPS[fish]
    w, h = x1 - x0, y1 - y0
    W, H = int(w * outscale) // 8 * 8, int(h * outscale) // 8 * 8
    stage = 'g16in/refine-' + os.path.basename(crop_path)
    shutil.copy(crop_path, os.path.join(IN_DIR, os.path.basename(stage)))
    pos = ('detailed flat 2d cartoon, crisp clean lineart, striped inflatable swim ring tube with '
           'folds and soft highlight gradients, calm sea water with ripple shading and tiny white '
           'sparkles, high detail, sharp focus, ' + c['pos'])
    neg = NEG_BASE + ', ' + FOAM_NEG + c['neg_extra']
    if fish == 'fish3' and not lei:
        pos = pos.replace(NO_LEI_POS, '')
        neg = NO_LEI_NEG + neg
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': c['lora'], 'strength_model': c['strength'] * 0.7,
              'strength_clip': c['strength']}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': pos}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': stage}},
      '8': {'class_type': 'ImageScale', 'inputs': {'image': ['6', 0], 'upscale_method': 'lanczos',
              'width': W, 'height': H, 'crop': 'center'}},
      '9': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['8', 0], 'vae': ['1', 2]}},
    }
    outs = []
    for s in seeds:
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['9', 0], 'seed': int(s), 'steps': steps,
                    'cfg': c.get('cfg', 5.5), 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras',
                    'denoise': denoise}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
                    'filename_prefix': f'g16refine/{tag}-{fish}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        p = out_of(e, '12')
        print('refine', fish, 'seed', s, '->', p, flush=True)
        outs.append(p)
    return outs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fish', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--denoise', type=float, default=0.88)
    ap.add_argument('--scale', type=float, default=1.7)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--base', default='g16in/swimref.png')
    ap.add_argument('--cfg', type=float, default=None)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--mask', default=None)
    ap.add_argument('--lei', type=int, default=1)
    ap.add_argument('--rework-from', default=None, help='返工底 crop 路径（中等 denoise 修形体）')
    ap.add_argument('--refine-from', default=None, help='给定一轮候选 crop 路径则做二次 hires 细化')
    ap.add_argument('--denoise2', type=float, default=0.30)
    ap.add_argument('--outscale', type=float, default=2.35)
    ap.add_argument('--pos-extra', default='')
    ap.add_argument('--neg-extra', default='')
    a = ap.parse_args()
    prepare()
    if a.rework_from:
        rework(a.rework_from, a.fish, [int(s) for s in a.seeds.split(',')],
               a.denoise, a.outscale, a.tag, 25, a.pos_extra, a.neg_extra, bool(a.lei))
    elif a.refine_from:
        refine(a.refine_from, a.fish, [int(s) for s in a.seeds.split(',')],
               a.denoise2, a.outscale, a.tag, bool(a.lei))
    else:
        run(a.fish, [int(s) for s in a.seeds.split(',')], a.denoise, a.scale, a.tag, a.base,
            a.cfg, a.steps, a.mask, bool(a.lei))
