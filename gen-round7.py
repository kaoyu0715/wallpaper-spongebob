# 七代出图驱动：硬门槛=海绵宝宝肢体完整（双臂、双手手指齐全、双腿）。
# 六代教训：sbp-r7 手势姿势出「团块拳」+蓝色袖泄漏；放大 2.1x 加剧发软。
# 本轮：出图分辨率提到 1024x1536（裁剪后贴图比例≈0.6-1.0x，不再放大），
#       姿态改为双臂外张、双手张开五指可辨；对比 sbp-r7 与 sbpilot（单角色，无属性互串）。
# 用法:
#   python3 gen-round7.py char --role spongebob_r7   --seeds 20261501,20261502,20261503
#   python3 gen-round7.py char --role spongebob_pilot --seeds 20261511,20261512,20261513 [--strength 0.8]
#   python3 gen-round7.py rmbg --role <role> [--keep <file>]
import json, os, sys, time, urllib.request, shutil

API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'

# 肢体完整负面清单：缺件/融合/遮挡/裁切全排除
LIMB_NEG = ('missing arm, missing hands, missing leg, missing fingers, extra finger, extra arms, extra hands, '
            'extra legs, fused fingers, mitten hands, deformed hands, blob hands, malformed hands, clenched fist, '
            'hands hidden, arms behind back, arms crossed, hands in pockets, holding object, cropped, cut off, out of frame, partial body')
NEG_CHAR = ('text, watermark, signature, lowres, worst quality, 3d, photorealistic, realistic, '
            'outline, greyscale, monochrome, lineart, extra eyes, extra characters, '
            'two characters, multiple views, duplicate character, flower, plants, shadow')

# 姿态：双臂离开躯干向两侧张开、双手手掌张开手指可辨；双腿直立齐全；
# 头与视线向右（指向派大星方向，保持五/六代叙事视线链），无道具。
POSE = ('standing upright with both arms extended outward and away from the body, '
        'left arm raised waving with an open palm, fingers clearly spread and separated, '
        'right arm lowered and angled away from the body with an open hand, fingers clearly spread and separated, '
        'both legs straight and fully visible down to black shoes, '
        'happy open smile, looking to the right, '
        'full body from head to shoes, generous margin around the whole body nothing cropped')

FULLBODY_TAGS = 'yellow porous sponge, square body, big blue eyes, buck teeth, white shirt, red tie, brown shorts, black shoes'

ROLES = {
  # 六代同款 LoRA，新姿态 + 权重 0.85（压低属性泄漏，蓝色袖）
  'spongebob_r7': {'lora': 'sbp-r7.safetensors', 'strength': 0.85,
    'pos': 'spongebob squarepants, ' + FULLBODY_TAGS + ', ' + POSE +
           ', flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo, simple white background'},
  # 单角色 pilot LoRA 对比（训练 caption 即上述完整特征词，无跨角色互串）
  'spongebob_pilot': {'lora': 'sbpilot2.safetensors', 'strength': 0.9,
    'pos': 'spongebob squarepants, ' + FULLBODY_TAGS + ', ' + POSE +
           ', flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo, simple white background'},
  # 第二轮：精简提示词 + 强调双脚踩地、鞋只有两只；cfg 5.5 减 artifacts
  'spongebob_c': {'lora': 'sbpilot2.safetensors', 'strength': 0.9, 'cfg': 5.5,
    'pos': 'spongebob squarepants, yellow porous sponge, square body, big blue eyes, buck teeth, '
           'white shirt, red tie, brown shorts, red striped socks, black shoes, '
           'standing on the ground, both feet on the ground, exactly two shoes, '
           'arms relaxed slightly away from body, open hands with four spread fingers, '
           'happy smile, looking to the right, '
           'full body visible with margin around, nothing cropped, '
           'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo, simple white background'},
  'spongebob_c1': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': 'spongebob squarepants, yellow porous sponge, square body, big blue eyes, buck teeth, '
           'white shirt, red tie, brown shorts, red striped socks, black shoes, '
           'standing on the ground, both feet on the ground, exactly two shoes, '
           'arms relaxed slightly away from body, open hands with four spread fingers, '
           'happy smile, looking to the right, '
           'full body visible with margin around, nothing cropped, '
           'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo, simple white background'},
}

NEG = {r: NEG_CHAR + ', ' + LIMB_NEG + ', three shoes, floating shoe, detached shoe, third shoe'
       for r in ROLES}


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API+path, data=data, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def wait(pid, timeout=1800):
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


def char(role, seeds, strength=None):
    pos = ROLES[role]['pos']
    neg = NEG[role]
    lora = ROLES[role]['lora']
    strength = ROLES[role]['strength'] if strength is None else strength
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1024, 'height': 1536, 'batch_size': 1}},
    }
    clip_out, model_out = ('3', 0), ('1', 0)
    if lora:
        g['2'] = {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0], 'lora_name': lora,
                    'strength_model': strength, 'strength_clip': strength}}
        clip_out, model_out = ('2', 1), ('2', 0)
    prefix = 'g7raw/' + role.replace('_', '-')
    for s in seeds:
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': [clip_out[0], clip_out[1]], 'text': pos}}
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': [model_out[0], model_out[1]], 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['6', 0], 'seed': int(s), 'steps': 30,
                    'cfg': ROLES[role].get('cfg', 7.0),
                    'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['1', 2]}}
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'{prefix}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        print('char', role, s, out_of(e, '9'), flush=True)


def rmbg_latest(role, keep=None):
    d = os.path.join(OUTDIR, 'g7raw')
    files = sorted([f for f in os.listdir(d) if f.startswith(role.replace('_', '-') + '-') and f.endswith('.png')],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    f = keep or files[-1]
    dst = os.path.join(INPUT, 'g7in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g7in/' + role + '.png'}},
      '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': 'RMBG-2.0', 'process_res': 1024,
              'sensitivity': 1.0, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
              'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}},
      '3': {'class_type': 'InvertMask', 'inputs': {'mask': ['2', 1]}},
      '4': {'class_type': 'SaveImageWithAlpha', 'inputs': {'images': ['2', 0], 'mask': ['3', 0],
              'filename_prefix': f'g7cut/{role}'}},
    }
    e = wait(post('/prompt', {'prompt': g})['prompt_id'])
    print('rmbg', role, 'src', f, '->', out_of(e, '4'), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    seeds = (a('seeds') or '20261501').split(',')
    if cmd == 'char':
        st = a('strength')
        char(a('role'), seeds, None if st is None else float(st))
    elif cmd == 'rmbg':
        rmbg_latest(a('role'), a('keep'))
