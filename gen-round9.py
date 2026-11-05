# 九代批量出图驱动：配方 = 五~七代收敛管线（见 delivery/recipe.md）。
# 角色单元逐个生成：illustriousXL + 角色专属 LoRA，单人白底，1024x1536，
# 30 步 dpmpp_2m karras cfg 5.5（七代胜出参数），肢体完整负面清单。
# 用法:
#   python3 gen-round9.py char --role <role> --seeds s1,s2,... [--strength f] [--cfg f]
#   python3 gen-round8.py rmbg --role <role> [--keep <file>]
import json, os, sys, time, urllib.request, shutil

API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'

# 肢体完整负面清单（七代版，全角色通用）
LIMB_NEG = ('missing arm, missing hands, missing leg, missing fingers, extra finger, extra arms, extra hands, '
            'extra legs, fused fingers, mitten hands, deformed hands, blob hands, malformed hands, clenched fist, '
            'hands hidden, arms behind back, arms crossed, hands in pockets, holding object, cropped, cut off, out of frame, partial body')
NEG_CHAR = ('text, watermark, signature, lowres, worst quality, 3d, photorealistic, realistic, '
            'outline, greyscale, monochrome, lineart, extra eyes, extra characters, '
            'two characters, multiple views, duplicate character, flower, plants, shadow')

# 通用风格尾缀（官方 2D + 单人白底）
STYLE = ('flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo, simple white background')

SB_TAGS = ('yellow porous sponge, square body, big blue eyes, buck teeth, white shirt, red tie, '
           'brown shorts, red striped socks, black shoes')
# 双臂离身、手掌张开手指可辨、双脚落地恰好两只鞋（七代胜出姿态模板，方向可换）
SB_POSE_L = ('standing on the ground, both feet on the ground, exactly two shoes, '
             'arms relaxed slightly away from body, open hands with four spread fingers, '
             'happy smile, looking to the left, full body visible with margin around, nothing cropped')
SB_POSE_R = SB_POSE_L.replace('looking to the left', 'looking to the right')

ROLES = {
  # 海绵宝宝朝左版（用于海绵宝宝在画面右侧的场景）；sbpilot @0.85 cfg 5.5 = 七代胜出配方
  'sb-left': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': 'spongebob squarepants, ' + SB_TAGS + ', ' + SB_POSE_L + ', ' + STYLE,
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe'},
  # 海绵宝宝朝右版同模板（新种子补充，七代胜出单元不重出）
  'sb-right': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': 'spongebob squarepants, ' + SB_TAGS + ', ' + SB_POSE_R + ', ' + STYLE,
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe'},
  # 派大星朝右（用于派大星在画面左侧的场景）；Patrick_il 触发词 + 官方风格词
  # 八代教训①：短裤词写 purple 出紫色短裤（官方为绿色）；改为 green shorts，负面排紫/花
  'patrick-right': {'lora': 'Patrick_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': 'p4tr1ckst4r, official style, starfish, pink skin, green shorts, '
           'big happy open mouth smile, arms spread open and slightly raised, both arms fully visible, '
           'looking to the right, standing on the ground, full body with margin around, nothing cropped, ' + STYLE,
    'neg_extra': 'purple shorts, blue shorts, clothes pattern, missing arm, three arms'},
  # 珊迪：宇航服松鼠，双臂上举欢呼（正面，任意布局可用）
  'sandy-cheer': {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': 'sandycheeks, squirrel in white astronaut suit with round glass helmet dome, brown fur and tail, '
           'big happy smile, standing, both arms raised cheerfully, arms fully visible away from body, '
           'looking at viewer, standing on the ground, full body with margin around, nothing cropped, ' + STYLE,
    'neg_extra': 'missing tail, net, rope, lasso, camping gear, pink flower, helmet cracked'},
  # 珊迪朝左（与海绵宝宝右侧对望）
  'sandy-left': {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': 'sandycheeks, squirrel in white astronaut suit with round glass helmet dome, brown fur and tail, '
           'big happy smile, standing, one arm raised waving with open hand, other arm relaxed away from body, '
           'looking to the left, standing on the ground, full body with margin around, nothing cropped, ' + STYLE,
    'neg_extra': 'missing tail, net, rope, lasso, camping gear, pink flower, helmet cracked, three arms'},
  # 蟹老板：大钳张开笑（正面），负面排除钱袋/美元符号道具
  'krabs-open': {'lora': 'krabs-v3.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': 'eugene krabs, red crab, big claws spread open in front, big smile showing teeth, '
           'blue shirt, dark trousers, standing on the ground, full body with margin around, nothing cropped, ' + STYLE,
    'neg_extra': 'money, dollar sign, coins, money bag, register, three claws, extra claw'},
  # 小蜗：地面前景（正面），眼柄恰好两根
  'gary-front': {'lora': 'gary-v3.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': 'gary the snail, pink shell with purple spots, teal body, two eye stalks with big red eyes, '
           'happy closed smile, crawling on the ground, full body with margin around, nothing cropped, ' + STYLE,
    'neg_extra': 'three eye stalks, missing eye stalk, extra eyes, shell cracked'},
  # 章鱼哥新种子：叉腰嫌弃（正面），配合不同场景方向
  'squidward-annoyed': {'lora': 'squidward-v3.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': 'squidward tentacles, teal octopus, long nose, half-closed eyes, unimpressed frown, '
           'brown short sleeve shirt, hands on hips, arms bent with elbows out, '
           'standing on the ground, full body with margin around, nothing cropped, ' + STYLE,
    'neg_extra': 'clarinet, instrument, missing arm, extra arm'},
}


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


def char(role, seeds, strength=None, cfg=None):
    r = ROLES[role]
    pos = r['pos']
    neg = NEG_CHAR + ', ' + LIMB_NEG + ', ' + r.get('neg_extra', '')
    strength = r['strength'] if strength is None else strength
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1024, 'height': 1536, 'batch_size': 1}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0], 'lora_name': r['lora'],
              'strength_model': strength, 'strength_clip': strength}},
    }
    for s in seeds:
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': pos}}
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['6', 0], 'seed': int(s), 'steps': 30,
                    'cfg': r['cfg'] if cfg is None else cfg,
                    'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['1', 2]}}
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g9raw/{role}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        print('char', role, s, out_of(e, '9'), flush=True)


def rmbg(role, keep=None):
    d = os.path.join(OUTDIR, 'g9raw')
    files = sorted([f for f in os.listdir(d) if f.startswith(role + '-') and f.endswith('.png')],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    f = keep or files[-1]
    dst = os.path.join(INPUT, 'g9in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g9in/' + role + '.png'}},
      '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': 'RMBG-2.0', 'process_res': 1024,
              'sensitivity': 1.0, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
              'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}},
      '3': {'class_type': 'InvertMask', 'inputs': {'mask': ['2', 1]}},
      '4': {'class_type': 'SaveImageWithAlpha', 'inputs': {'images': ['2', 0], 'mask': ['3', 0],
              'filename_prefix': f'g9cut/{role}'}},
    }
    e = wait(post('/prompt', {'prompt': g})['prompt_id'])
    print('rmbg', role, 'src', f, '->', out_of(e, '4'), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    seeds = (a('seeds') or '').split(',')
    st, cfg = a('strength'), a('cfg')
    if cmd == 'char':
        char(a('role'), seeds, None if st is None else float(st), None if cfg is None else float(cfg))
    elif cmd == 'rmbg':
        rmbg(a('role'), a('keep'))
