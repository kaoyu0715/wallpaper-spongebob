# 五代出图驱动：三角色互动构图「追泡泡」（海绵宝宝+派大星+章鱼哥），锁定路线A官方图案背景。
# 用法:
#   python3 gen-round5.py char --role spongebob --seeds 20261311,20261312
#   python3 gen-round5.py rmbg --role patrick [--keep patrick-20261322_00001_.png]
import json, os, sys, time, urllib.request, shutil

API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'

NEG_CHAR = ('text, watermark, signature, lowres, worst quality, 3d, photorealistic, realistic, '
            'outline, greyscale, monochrome, lineart, extra limbs, extra eyes, extra characters, '
            'two characters, multiple views, duplicate character, cropped, partial body, out of frame, '
            'flower, plants, shadow')

# 互动场景「追泡泡」：海绵宝宝左前吹泡泡并侧手指向派大星，派大星右前伸手去接泡泡，
# 章鱼哥后排叉腰嫌弃，泡泡带把三人视线串成一个场景。
# 海绵宝宝按 issue #11 指定用 sbpilot（触发词 spongebob squarepants）。
ROLES = {
  'spongebob': {'lora': 'sbpilot.safetensors',
    'neg': NEG_CHAR,  # 允许泡泡棒入画
    'pos':
    'spongebob squarepants, yellow porous sponge, square body, big blue eyes, buck teeth, white shirt, red tie, brown shorts, black shoes, '
    'holding a bubble wand up to his mouth blowing a bubble, cheeks puffed, other arm stretched sideways pointing to the right, '
    'happy smile, looking to the right, '
    'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, full body, solo, simple white background'},
  'patrick':   {'lora': 'Patrick_il.safetensors',
    'neg': NEG_CHAR + ', bubble, wand, pattern on shorts, floral shorts',
    'pos':
    'p4tr1ckst4r, starfish, star-shaped body, pink skin, black eyes, black eyebrows, dots on body, '
    'solid plain green shorts, '
    'both arms stretched forward and up reaching for bubbles above, mouth wide open happy laugh, head tilted looking up, '
    'flat 2d cartoon, tv cartoon style, clean lineart, full body, solo, simple white background'},
  'squidward': {'lora': 'squidward-v3.safetensors',
    'neg': NEG_CHAR + ', bubble, wand',
    'pos':
    'squidward tentacles, teal skinnish squid person, bald, huge droopy nose, half-closed sleepy eyes, long face, wearing brown short sleeve shirt, '
    'standing with hands on hips, annoyed grumpy frown, unimpressed, '
    'flat 2d cartoon style, thick outlines, bright primary colors, full body, standing, solo, simple white background'},
}


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
    neg = ROLES[role]['neg']
    lora = lora if lora is not None else ROLES[role]['lora']
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
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
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g5raw/{role}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        print('char', role, s, out_of(e, '9'), flush=True)


def rmbg_latest(role, keep=None):
    d = os.path.join(OUTDIR, 'g5raw')
    files = sorted([f for f in os.listdir(d) if f.startswith(role + '-') and f.endswith('.png')],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    f = keep or files[-1]
    dst = os.path.join(INPUT, 'g5in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g5in/' + role + '.png'}},
      '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': 'RMBG-2.0', 'process_res': 1024,
              'sensitivity': 1.0, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
              'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}},
      '3': {'class_type': 'InvertMask', 'inputs': {'mask': ['2', 1]}},
      '4': {'class_type': 'SaveImageWithAlpha', 'inputs': {'images': ['2', 0], 'mask': ['3', 0],
              'filename_prefix': f'g5cut/{role}'}},
    }
    e = wait(post('/prompt', {'prompt': g})['prompt_id'])
    print('rmbg', role, 'src', f, '->', out_of(e, '4'), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    seeds = (a('seeds') or '20261301').split(',')
    if cmd == 'char':
        char(a('role'), seeds, a('lora'), float(a('strength', '0.9') or 0.9))
    elif cmd == 'rmbg':
        rmbg_latest(a('role'), a('keep'))
