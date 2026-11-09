# -*- coding: utf-8 -*-
# 十八代·香蕉船出海（issue #37）：整图生成路线。
# 实测教训（自查记录见 PR）：三角色 LoRA 同挂在本底模上系统性属性互串——
#   v1 全局拼接 0.85/1.0/1.0 → 三 LoRA 融成单个混合体（seed 18001/18005/18006）；
#   v2/v3 ConditioningSetArea 分区条件 → 底景稀释/跨框染色（18002/18003/18004_00002_）；
#   v4 仅挂 sbpilot 整图 → 单元档 LoRA 在长场景 prompt 下漂移成动漫风（18101/18102）。
# v8 终局路线 =「Qwen-Image-2.1 整图直出 +（对症）单 LoRA 角色区域二次出」：
#   Illustrious 自查记录：三 LoRA 同挂必融合（v1 单元档 / v6 降档）、分区条件染色（v2/v3）、
#   无角色 LoRA 不识角色（v5 萌系动物）、无主体底景必崩（v7 粉肠船/空白/书法屏风）。
#   Qwen-Image-2.1 = README 路线1（全员合影已验证），多人整图 + 质感贴 swim-ref 风格锚；
#   角色若糊则 fix() 用该角色 LoRA 做区域 d1.0 inpaint（issue 允许的分区二次出）。
#   Illustrious 单元口径保留 = recipe §1：CLIP -2、30 步 dpmpp_2m karras cfg 5.5。
#   高清/交付：Qwen 原生 1088x1920 → 居中裁 1080x1920（qfinal）；硬闸自查=逐角色 2.4x 裁剪。
# 用法:
#   python3 gen18.py qgen --seeds 18401,...            Qwen 整图 → ComfyUI g18q/
#   python3 gen18.py qfinal <q-file>...                居中裁 1080x1920 → output/g18final/
#   python3 gen18.py fix <file> [--to sb|pat|sandy] [--sb-box x,y,w,h] [--pat-box ...] [--sandy-box ...]
#   python3 gen18.py gen --seeds ... / up <file>       （留档：Illustrious 整图底 v7 已否 / Remacri 高清化）
#   python3 gen18.py crop <final> --char sb --box x,y,w,h   2.4x 角色裁剪 → output/g18gate/
#   python3 gen18.py sbs  <final>                      与样张并排 → output/g18gate/
import json, os, shutil, sys, time, urllib.request
from PIL import Image, ImageFilter

T = os.path.dirname(os.path.abspath(__file__))
API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'
FINAL = os.path.join(T, 'output', 'g18final')
GATE = os.path.join(T, 'output', 'g18gate')
REF = os.path.join(T, 'review', 'reference', 'swim-ref.jpg')
FW, FH = 1080, 1920  # 交付尺寸

os.makedirs(FINAL, exist_ok=True)
os.makedirs(GATE, exist_ok=True)

GEN_W, GEN_H = 768, 1344
STEPS, CFG = 30, 5.5

# —— 风格锚：swim-ref.jpg（绿松石海面+清晰波纹+太阳光斑、上方夕阳霞光天际线、卡通描边干净）——
# 风格词面 = recipe §1.1 胜出词（v4 教训：自造风格词漂移成插画/动漫风）
STYLE = 'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, vibrant colors'

SEA = ('turquoise ocean water, clear water ripples, sparkling sun glitter reflections, '
       'straight clear horizon line one third down from the top, '
       'sunset sky occupying the top third of the image, sea occupying the lower two thirds, '
       'soft orange peach sunset glow above the horizon, sunset light reflection on the water')

BOAT = ('a long yellow inflatable banana boat floating diagonally on the turquoise sea '
        'in the lower middle of the image, water splashes and reflection under the hull')

# pass1 整图底：空香蕉船场景，不挂角色 LoRA、无角色词。
# 实测终局（自查记录见 PR）：
#   单元档同挂（v1 0.85/1/1）、分区条件（v2/v3）、降档同挂（v6 0.6/0.7/0.7）→ 一律融合成单个穿衬衫混合体；
#   不挂角色 LoRA + 角色词（v5）→ 底模不认识海绵宝宝系角色，画成萌系动物。
#   结论：三角色必须由各自 LoRA 单独成区（十六代 A 路线 inpaint 原语已验证）。
POS_P1 = (STYLE + ', ' + SEA + ', ' + BOAT + ', '
  'nobody on the boat, empty banana boat, no characters, no people, '
  'a small yellow starfish floating on the water nearby, '
  'clean simple composition, no text, no watermark')

# pass2/3/4 区域二次出（掩码内=角色本体 ∪ 羽化带；prompt 须覆盖整掩码域；词面=recipe §1.1 单元模板）
_UNIT = 'flat 2d cartoon, tv cartoon style, clean lineart, simple shading'
_SB_TAGS = ('yellow porous sponge, square body, big blue eyes, buck teeth, white shirt, red tie, '
            'brown shorts, red striped socks, black shoes')
_HA = ('round transparent glass space helmet, helmet interior fully transparent, '
       'only a plain thin white helmet ring outline, no grey glass fill')

FIXES = {
  'sb': {'lora': ('sbpilot.safetensors', 0.85), 'next': 'pat',
    'box': None, 'feather': 18,
    'pos': ('spongebob squarepants, ' + _SB_TAGS + ', happy relaxed smile, '
            'sitting on the yellow banana boat, legs resting forward on the banana surface, '
            'both arms fully visible away from body, open hands, '
            'the yellow banana boat surface under him, '
            'calm turquoise ocean water and soft sunset light around, ' + _UNIT)},
  'pat': {'lora': ('Patrick_il.safetensors', 1.0), 'next': 'sandy',
    'box': None, 'feather': 18,
    'pos': ('p4tr1ckst4r, official style, starfish, pink skin, green shorts, '
            'lying on his back on the yellow banana boat, relaxed dopey happy open mouth smile, '
            'both stubby arms and legs fully visible, '
            'the yellow banana boat surface under his body, '
            'calm turquoise ocean water and soft sunset light around, ' + _UNIT)},
  'sandy': {'lora': ('Sandy_il.safetensors', 1.0), 'next': None,
    'box': None, 'feather': 18,
    'pos': ('sandycheeks, squirrel in white astronaut suit with ' + _HA + ', brown fur, '
            'friendly smile, sitting on the yellow banana boat, '
            'holding a brown coconut drink with a straw in both hands in front of her chest, '
            'the yellow banana boat surface under her, '
            'calm turquoise ocean water and soft sunset light around, ' + _UNIT)},
}

NEG = (
  'text, watermark, signature, logo, lowres, worst quality, jpeg artifacts, '
  'photorealistic, realistic photo, blurry, blur, depth of field, pastel washed out, crayon, sketch, '
  'anime style, chibi, big eyed anime girl, '
  'extra characters, four characters, five characters, duplicated character, duplicate person, '
  'fused characters, merged characters, hybrid creature, conjoined bodies, '
  'crowd, group of people, extra heads, missing head, '
  'missing arm, missing hands, missing leg, missing fingers, extra finger, extra arms, extra hands, '
  'extra legs, fused fingers, mitten hands, deformed hands, blob hands, '
  'hands hidden, arms behind back, cropped, cut off, out of frame, partial body, '
  'patrick wearing shirt, patrick red skin, patrick brown skin, brown starfish, '
  'grey opaque helmet, mustache on sandy, '
  'three shoes, floating shoe, human face on character, night, dark sky, rainbow, '
  'whales, boats with sails, motor boat'
)


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError('ComfyUI %s: %s' % (e.code, e.read().decode()[:800]))


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


def to_input(src, name):
    d = os.path.join(INPUT, 'g18in')
    os.makedirs(d, exist_ok=True)
    shutil.copy(src, os.path.join(d, name))
    return 'g18in/' + name


def ckpt_lora(g, loras):
    """底模 + CLIP-2 + 单角色 LoRA 链，返回 (model, clip)。"""
    g['1'] = {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}}
    g['3'] = {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}}
    m, c = ['1', 0], ['3', 0]
    for i, (name, s) in enumerate(loras):
        nid = str(20 + i)
        g[nid] = {'class_type': 'LoraLoader', 'inputs': {'model': m, 'clip': c, 'lora_name': name,
                  'strength_model': s, 'strength_clip': s}}
        m, c = [nid, 0], [nid, 1]
    return m, c


def gen(seeds):
    """pass1：整图底，空香蕉船场景，不挂角色 LoRA（见 POS_P1 终局结论）。"""
    for s in seeds:
        g = {}
        m, c = ckpt_lora(g, [])
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': c, 'text': POS_P1}}
        g['5'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': c, 'text': NEG}}
        g['6'] = {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': GEN_W, 'height': GEN_H, 'batch_size': 1}}
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': m, 'positive': ['4', 0], 'negative': ['5', 0],
                    'latent_image': ['6', 0], 'seed': int(s), 'steps': STEPS, 'cfg': CFG,
                    'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['1', 2]}}
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g18p1/wp-banana-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        print('p1', s, out_of(e, '9'), flush=True)


def fix(src, first='sb', boxes=None, size=None):
    """pass2/3/4：对 src 依次做 sb → pat → sandy 单 LoRA 区域 d1.0 二次出（latent-mask inpaint）。
    boxes: {'sb':(x,y,w,h), 'pat':..., 'sandy':...} 按该 seed 实际站位给出（必填，src 像素系）。
    size: (w,h) src 尺寸；None=按图自动。"""
    cur = src
    tag = ''.join(ch for ch in os.path.basename(src) if ch.isdigit())[:5]
    step = first
    while step:
        f = FIXES[step]
        name = f'wp-banana-{tag}-{step}in.png'
        img_rel = to_input(cur, name)
        W, H = size or Image.open(cur).size
        box = (boxes or {}).get(step) or f['box']
        if not box:
            raise SystemExit('缺少 --%s-box x,y,w,h（该 seed 实际站位）' % step)
        x, y, w, h = box
        m = Image.new('L', (W, H), 0)
        m.paste(Image.new('L', (w, h), 255), (x, y))
        m = m.filter(ImageFilter.GaussianBlur(f['feather']))
        mpath = os.path.join(INPUT, 'g18in', name.replace('.png', '-mask.png'))
        m.save(mpath)
        g = {}
        lm, lc = ckpt_lora(g, [f['lora']])
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': lc, 'text': f['pos']}}
        g['5'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': lc, 'text': NEG}}
        g['6'] = {'class_type': 'LoadImage', 'inputs': {'image': img_rel}}
        g['7'] = {'class_type': 'VAEEncode', 'inputs': {'pixels': ['6', 0], 'vae': ['1', 2]}}
        g['8'] = {'class_type': 'LoadMask', 'inputs': {'mask': os.path.relpath(mpath, INPUT).replace('\\', '/')}}
        g['9'] = {'class_type': 'SetLatentNoiseMask', 'inputs': {'samples': ['7', 0], 'mask': ['8', 0]}}
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': lm, 'positive': ['4', 0], 'negative': ['5', 0],
                     'latent_image': ['9', 0], 'seed': 700 + int(tag), 'steps': STEPS, 'cfg': CFG,
                     'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0],
                    'filename_prefix': f'g18p2/wp-banana-{tag}-{step}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        cur = out_of(e, '12')
        print(step, cur, flush=True)
        step = f['next']
    return cur


def up(raws):
    for raw in raws:
        seed = ''.join(ch for ch in os.path.basename(raw) if ch.isdigit())[:5]
        name = 'wp-banana-' + seed + '.png'
        g = {
          '1': {'class_type': 'LoadImage', 'inputs': {'image': to_input(raw, name)}},
          '2': {'class_type': 'UpscaleModelLoader', 'inputs': {'model_name': '4x_foolhardy_Remacri.pth'}},
          '3': {'class_type': 'UpscaleImage', 'inputs': {'image': ['1', 0], 'upscale_model': ['2', 0]}},
          '4': {'class_type': 'SaveImage', 'inputs': {'images': ['3', 0], 'filename_prefix': 'g18up/' + name}},
        }
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        big = Image.open(out_of(e, '4')).convert('RGB')
        w, h = big.size
        tw = min(w, int(h * FW / FH))
        x0 = (w - tw) // 2
        img = big.crop((x0, 0, x0 + tw, h)).resize((FW, FH), Image.LANCZOS)
        dst = os.path.join(FINAL, name)
        img.save(dst)
        print('up ', dst, flush=True)


def crop(final, char, box, zoom=2.4):
    img = Image.open(final).convert('RGB')
    x, y, w, h = box
    c = img.crop((x, y, x + w, y + h)).resize((int(w * zoom), int(h * zoom)), Image.LANCZOS)
    base = os.path.splitext(os.path.basename(final))[0]
    dst = os.path.join(GATE, f'{base}-{char}@{zoom}x.png')
    c.save(dst)
    print(dst, flush=True)


def sbs(final):
    ref = Image.open(REF).convert('RGB').resize((FW, FH))
    cand = Image.open(final).convert('RGB')
    sheet = Image.new('RGB', (FW * 2 + 24, FH), (255, 255, 255))
    sheet.paste(ref, (0, 0))
    sheet.paste(cand, (FW + 24, 0))
    base = os.path.splitext(os.path.basename(final))[0]
    dst = os.path.join(GATE, f'{base}-sbs-vs-ref.png')
    sheet.save(dst, optimize=True)
    print(dst, flush=True)


# —— Qwen-Image-2.1 整图路线（README 路线1「已验证成功，全员合影版」；gen-bikini.cjs 同构）——
# v7 终局：Illustrious 无主体底景必崩（粉肠船/米色空白），多角色同框必融合；
# Qwen 文本编码认识海绵宝宝三角色，多人整图 + 绘画质感也更贴 swim-ref 风格锚。
# 若单角色糊 → fix()（单 LoRA 区域 d1.0 二次出）对症修，issue 口径允许。
QW_W, QW_H = 1088, 1920
POS_QWEN = (
  'Vertical phone wallpaper, no text, no watermark, no border. '
  'SpongeBob SquarePants, Patrick Star and Sandy Cheeks from the SpongeBob SquarePants cartoon, '
  'official 2D cartoon style with clean outlines, riding a big yellow inflatable banana-shaped boat '
  'floating on a calm turquoise sea. The banana boat lies diagonally in the lower middle of the frame. '
  'SpongeBob, the yellow sponge in his white shirt, red tie and brown shorts, sits relaxed at the left end '
  'of the banana boat with a happy smile. Patrick, the pink starfish in green shorts, lies lazily on his '
  'back in the middle with a dopey grin. Sandy, the squirrel in her white space suit with a round glass '
  'helmet, sits at the right end holding a coconut drink with a straw. '
  'Turquoise sea with clear gentle ripples and sparkling sun glitter. A straight horizon line in the '
  'upper third, above it a soft orange and peach sunset sky with a warm glow. A small yellow starfish '
  'floats on the water nearby. Relaxed summer vacation mood, bright saturated colors, crisp details.'
)


def qgen(seeds):
    """整图 txt2img（Qwen-Image-2.1，1088x1920，euler/simple/cfg1/40 步 = gen-bikini 口径）。"""
    res = round((QW_W * QW_H) ** 0.5 / 32) * 32
    for s in seeds:
        g = {
          '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2.1_bf16.safetensors', 'weight_dtype': 'default'}},
          '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_image_2.1_text_encoder.safetensors', 'type': 'qwen_image'}},
          '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_2.1_vae.safetensors'}},
          '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'prompt': POS_QWEN,
                    'negative_prompt': 'text, watermark, signature, logo, border, frame, lowres, worst quality, blurry, extra characters',
                    'resolution': min(res, 4096)}},
          '6': {'class_type': 'EmptyLatentImage', 'inputs': {'width': QW_W, 'height': QW_H, 'batch_size': 1}},
          '7': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['4', 1],
                    'latent_image': ['6', 0], 'seed': int(s), 'steps': 40, 'cfg': 1.0,
                    'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
          '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['3', 0]}},
          '9': {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g18q/wp-banana-{s}'}},
        }
        e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=3600)
        print('q1', s, out_of(e, '9'), flush=True)


def qfinal(raws):
    """Qwen 原图居中裁 1080x1920 → FINAL（无放大，原生分辨率交付=最锐）。"""
    for raw in raws:
        src = raw if os.path.isabs(raw) else os.path.join(OUTDIR, 'g18q', raw)
        seed = ''.join(ch for ch in os.path.basename(src) if ch.isdigit())[:5]
        img = Image.open(src).convert('RGB')
        w, h = img.size
        tw, th = int(h * FW / FH + 0.5), h
        if w > tw:
            x0 = (w - tw) // 2
            img = img.crop((x0, 0, x0 + tw, th))
        img = img.resize((FW, FH), Image.LANCZOS)
        dst = os.path.join(FINAL, 'wp-banana-' + seed + '.png')
        img.save(dst)
        print('qf ', dst, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'gen':
        gen(sys.argv[sys.argv.index('--seeds') + 1].split(','))
    elif cmd == 'fix':
        boxes = {}
        for key in ('sb', 'pat', 'sandy'):
            flag = '--%s-box' % key
            if flag in sys.argv:
                boxes[key] = tuple(int(v) for v in sys.argv[sys.argv.index(flag) + 1].split(','))
        fix(sys.argv[2], sys.argv[sys.argv.index('--to') + 1] if '--to' in sys.argv else 'sb', boxes)
    elif cmd == 'qgen':
        qgen(sys.argv[sys.argv.index('--seeds') + 1].split(','))
    elif cmd == 'qfinal':
        qfinal(sys.argv[2:])
    elif cmd == 'up':
        up(sys.argv[2:])
    elif cmd == 'crop':
        crop(sys.argv[2], sys.argv[sys.argv.index('--char') + 1],
             tuple(int(v) for v in sys.argv[sys.argv.index('--box') + 1].split(',')))
    elif cmd == 'sbs':
        sbs(sys.argv[2])
    else:
        print('unknown cmd', cmd)
        sys.exit(1)
