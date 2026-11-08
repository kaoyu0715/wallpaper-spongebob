# -*- coding: utf-8 -*-
# 十五代 B 路线生成器（issue #31）：整图分区重绘——底=样张 swim-ref.jpg（背景像素保留），
# 三条鱼逐区 inpaint 换成海绵宝宝/派大星/珊迪（单角色单次，避免多角色互串）。
# 管线口径沿用 recipe：illustriousXL, CLIP -2, 30 步 dpmpp_2m karras cfg 5.5, 角色 LoRA。
# 用法: python3 gen15b.py pass --fish fish1 --seeds 11,22 [--stage out/x.png] [--denoise f]
#       （--stage 指定底图：链式第二/三区在上一区成品上重绘）
import json, os, sys, time, shutil, urllib.request
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask15

API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'
T = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(INPUT, 'g15in')

NEG_BASE = ('text, watermark, signature, lowres, worst quality, 3d, photorealistic, realistic, '
            'fish, fins, tail fin, gills, scales, two characters, multiple characters, extra characters, '
            'extra eyes, missing arm, missing hands, missing leg, missing feet, extra limbs, extra arms, '
            'deformed hands, blob hands, malformed hands, cropped, cut off, out of frame, partial body, '
            'blurry, jpeg artifacts')

STYLE = 'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo'

CHARS = {
  'fish1': dict(lora='sbpilot.safetensors', strength=0.85, out='sb',
    pos=('spongebob squarepants, yellow porous sponge, square body, official cartoon proportions, '
         'big blue eyes, buck teeth, white shirt, red tie, brown shorts, red striped socks, black shoes, '
         'black sunglasses resting pushed up on forehead, relaxed happy smile, '
         'sitting deep inside a red and white striped inflatable swim ring on calm sea, '
         'hips in the ring, leaning back lazily, one arm draped over the ring tube, '
         'legs bent up casually, ankles crossed resting high on the front tube of the ring close to his body, feet up on the tube not hanging down, '
         'surrounded by calm open sea surface, no horizon, no sky, ' + STYLE),
    neg_extra='standing, waving hand, wide open mouth, tongue, sunglasses over eyes, legs dangling in water, legs hanging below ring, horizon, sky'),
  'fish2': dict(lora='Patrick_il.safetensors', strength=1.0, out='pat',
    pos=('p4tr1ckst4r, official style, starfish, pink skin, green shorts, '
         'content expression, half-closed eyes, small closed-mouth smile, '
         'sitting deep inside a blue and white striped inflatable swim ring, bold blue and white stripes on the ring tube, on calm sea, '
         'lying back lazily in the ring with his whole body inside it, '
         'one hand holding up a tall glass half full of orange juice with ice, '
         'other arm draped relaxed over the ring tube, '
         'legs stretched forward resting horizontally on top of the front tube of the ring, feet on the tube not dangling, '
         'surrounded by calm open sea surface, no horizon, no sky, ' + STYLE),
    neg_extra='standing, waving, wide open mouth, missing glass, empty hand, legs dangling in water, horizon, sky, beach, sand, shore, plain blue ring, solid color ring, aqua ring'),
  'fish3': dict(lora='Sandy_il.safetensors', strength=1.0, out='sandy',
    pos=('sandycheeks, squirrel in white astronaut suit, pink and white flower lei garland worn around her neck, '
         'round transparent glass space helmet over her head, helmet interior fully transparent, '
         'gentle relaxed smile, '
         'sitting deep inside a blue and white striped inflatable swim ring on calm sea, '
         'one hand holding a whole brown coconut drink with a bendy straw on her lap, '
         'other arm draped relaxed over the ring tube, '
         'legs crossed resting on top of the front tube of the ring, feet on the tube not dangling, '
         'surrounded by calm blue-green open sea water with ripples, no beach, no sand, no horizon, no sky, ' + STYLE),
    neg_extra='flower on the ring, flowers floating in water, no necklace, grey helmet, opaque helmet, closed helmet cover, headscarf, turban, standing, legs dangling in water, horizon, sky'),
}


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
    shutil.copy(ref, os.path.join(IN_DIR, 'swimref.png')) if False else \
        Image.open(ref).convert('RGB').save(os.path.join(IN_DIR, 'swimref.png'))
    for w in mask15.FISHES:
        m = mask15.fish_mask(w)
        rgb = Image.merge('RGB', [m, Image.new('L', m.size, 0), Image.new('L', m.size, 0)])
        rgb.save(os.path.join(IN_DIR, f'mask-{w}.png'))  # red channel = mask


def run_pass(fish, seeds, stage=None, denoise=1.0):
    c = CHARS[fish]
    base = stage or 'g15in/swimref.png'
    neg = NEG_BASE + ', ' + c['neg_extra']
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': c['lora'], 'strength_model': c['strength'], 'strength_clip': c['strength']}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': c['pos']}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': base}},
      '7': {'class_type': 'LoadImage', 'inputs': {'image': f'g15in/mask-{fish}.png'}},
      '8': {'class_type': 'ImageToMask', 'inputs': {'image': ['7', 0], 'channel': 'red'}},
      '9': {'class_type': 'InpaintModelConditioning', 'inputs': {'positive': ['4', 0], 'negative': ['5', 0],
              'vae': ['1', 2], 'pixels': ['6', 0], 'mask': ['8', 0], 'noise_mask': True}},
    }
    outs = []
    for s in seeds:
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['9', 0],
                    'negative': ['9', 1], 'latent_image': ['9', 2], 'seed': int(s), 'steps': 30,
                    'cfg': 5.5, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': denoise}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
                    'filename_prefix': f'g15test/{c["out"]}-{fish}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        p = out_of(e, '12')
        print('pass', fish, 'seed', s, '->', p, flush=True)
        outs.append(p)
    return outs


if __name__ == '__main__':
    args = sys.argv[1:]
    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    prepare()
    if args[0] == 'pass':
        run_pass(a('fish'), [int(x) for x in a('seeds').split(',')],
                 stage=a('stage'), denoise=float(a('denoise', '1.0')))
