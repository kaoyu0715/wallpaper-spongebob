# 四代出图驱动：三角色互动构图（海绵宝宝+派大星+章鱼哥），双线对比
# 用法:
#   python3 gen-round4.py char --role spongebob --seeds 20261211,20261212,20261213
#   python3 gen-round4.py bg-real --seeds 20261201,20261202     # Krea2DarkBeast3.0 写实海底
#   python3 gen-round4.py rmbg --role spongebob                 # 对该角色最新一张抠图
import json, os, sys, time, urllib.request, shutil

API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'

NEG_CHAR = ('text, watermark, signature, lowres, worst quality, 3d, photorealistic, realistic, '
            'outline, greyscale, monochrome, lineart, extra limbs, extra eyes, extra characters, '
            'two characters, multiple views, duplicate character, cropped, partial body, out of frame, '
            'flower, plants, shadow, bubble, wand')

# 互动场景「吹泡泡」：海绵宝宝前排吹泡泡并抬手指向泡泡、派大星欢呼看泡泡、章鱼哥抱臂嫌弃
ROLES = {
  'spongebob': {'lora': 'sbp-r7.safetensors', 'pos':
    'spongebob squarepants, yellow porous sponge, square body, big blue eyes, buck teeth, white shirt, red tie, brown shorts, black shoes, '
    'holding a round bubble wand up to his mouth with one hand blowing a big soap bubble, other arm raised pointing up, '
    'excited open mouth grin, looking up, '
    'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, full body, solo, simple white background'},
  'patrick':   {'lora': 'Patrick_il.safetensors', 'pos':
    'p4tr1ckst4r, starfish, star-shaped body, pink skin, black eyes, black eyebrows, dots on body, green shorts, '
    'both arms raised high cheering, mouth wide open happy laugh, looking up, '
    'flat 2d cartoon, tv cartoon style, clean lineart, full body, solo, simple white background'},
  'squidward': {'lora': 'squidward-v3.safetensors', 'pos':
    'squidward tentacles, teal skinnish squid person, bald, huge droopy nose, half-closed sleepy eyes, long face, wearing brown short sleeve shirt, '
    'arms crossed, annoyed grumpy frown, unimpressed, '
    'flat 2d cartoon style, thick outlines, bright primary colors, full body, standing, solo, simple white background'},
}

BG_REAL_PROMPT = ('Photorealistic underwater seascape: a wide open rippled white sand seabed across the lower half of the frame, '
  'crystal clear turquoise water above, golden sunlight rays beaming down from the surface high above, tiny air bubbles rising, '
  'a few faded soft coral and seaweed silhouettes far away at the left and right edges, gentle blue gradient into depth. '
  'Underwater fine-art photography, volumetric god rays, high detail, natural color grading. '
  'Vertical composition: open empty sand in the lower middle third, vast bright open water above. '
  'Empty scene: no fish, no sea creatures, no animals of any kind, no live coral reef, no text, no watermark.')
BG_REAL_NEG = ('characters, people, animals, fish, sea creatures, cartoon, anime, illustration, drawing, text, watermark, signature, blurry, low quality')


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API+path, data=data, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def wait(pid, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(urllib.request.urlopen(API + f'/history/{pid}', timeout=10).read())
            e = h.get(pid)
            if e:
                if e.get('status', {}).get('status_str') == 'error':
                    raise RuntimeError(json.dumps(e)[:500])
                if e.get('status', {}).get('completed'):
                    return e
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(pid)


def out_of(entry, node):
    img = entry.get('outputs', {}).get(node, {}).get('images', [{}])[0]
    return os.path.join(OUTDIR, img.get('subfolder', ''), img.get('filename', ''))


def char(role, seeds, lora=None, strength=0.9):
    pos = ROLES[role]['pos']
    lora = lora if lora is not None else ROLES[role]['lora']
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': NEG_CHAR}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 832, 'height': 1216, 'batch_size': 1}},
    }
    clip_out, model_out = ('3', 0), ('1', 0)
    if lora:
        g['2'] = {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0], 'lora_name': lora,
                    'strength_model': strength, 'strength_clip': strength}}
        clip_out, model_out = ('2', 1), ('2', 0)
    for s in seeds:
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': [clip_out[0], clip_out[1]], 'text': pos}}
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': [model_out[0], model_out[1]], 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['6', 0], 'seed': int(s), 'steps': 30, 'cfg': 7.0,
                    'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['1', 2]}}
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g4raw/{role}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        print('char', role, s, out_of(e, '9'), flush=True)


def bg_real(seeds):
    g = {
      '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'Krea2DarkBeast3.0-bf16-ComfyUI.safetensors', 'weight_dtype': 'default'}},
      '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_4b_bf16.safetensors', 'type': 'krea2'}},
      '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'wan_2.1_vae.safetensors'}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': BG_REAL_PROMPT}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': BG_REAL_NEG}},
      '7': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1440, 'height': 3200, 'batch_size': 1}},
      '8': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['5', 0],
              'latent_image': ['7', 0], 'seed': 0, 'steps': 28, 'cfg': 1.0, 'sampler_name': 'euler',
              'scheduler': 'simple', 'denoise': 1.0}},
      '9': {'class_type': 'VAEDecode', 'inputs': {'samples': ['8', 0], 'vae': ['3', 0]}},
    }
    for s in seeds:
        g['8']['inputs']['seed'] = int(s)
        g['10'] = {'class_type': 'SaveImage', 'inputs': {'images': ['9', 0], 'filename_prefix': f'g4bg/real-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=1800)
        print('bg-real', s, out_of(e, '10'), flush=True)


def rmbg_latest(role, keep=None):
    d = os.path.join(OUTDIR, 'g4raw')
    files = sorted([f for f in os.listdir(d) if f.startswith(role + '-') and f.endswith('.png')],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    f = keep or files[-1]
    dst = os.path.join(INPUT, 'g4in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g4in/' + role + '.png'}},
      '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': 'RMBG-2.0', 'process_res': 1024,
              'sensitivity': 1.0, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
              'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}},
      '3': {'class_type': 'InvertMask', 'inputs': {'mask': ['2', 1]}},
      '4': {'class_type': 'SaveImageWithAlpha', 'inputs': {'images': ['2', 0], 'mask': ['3', 0],
              'filename_prefix': f'g4cut/{role}'}},
    }
    e = wait(post('/prompt', {'prompt': g})['prompt_id'])
    print('rmbg', role, 'src', f, '->', out_of(e, '4'), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    seeds = (a('seeds') or '20261201').split(',')
    if cmd == 'char':
        char(a('role'), seeds, a('lora'), float(a('strength', '0.9') or 0.9))
    elif cmd == 'bg-real':
        bg_real(seeds)
    elif cmd == 'rmbg':
        rmbg_latest(a('role'), a('keep'))
