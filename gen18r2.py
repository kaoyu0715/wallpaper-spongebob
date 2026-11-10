# -*- coding: utf-8 -*-
# 十八代·香蕉船出海 二轮路线（issue #37 用户 2026-11-10 指定，替代整图从零生成）：
#   步骤一 去圈：rings-photoref.jpg 实拍海景，框四个泳圈(+投影/水波环)做掩膜，
#           FLUX.1-Fill 周边海面补全 → 无圈干净实拍海景（保持实拍质感，不提前动漫化）。
#   步骤二 素材：黄香蕉船+SB/派/珊一张出。用户口径=角色 LoRA 同挂；实测 Illustrious
#           三 LoRA 同挂融合（见 gen18.py 注释 v1-v7），故双轨：a) 照用户口径单挂尝试；
#           b) Qwen 整图素材兜底（一代已证角色可辨）。白底 → 抠图成透明素材。
#   步骤三 统一动漫化：素材贴到去圈海景 → 整图 img2img（d≈0.5，挂全部角色 LoRA 防脸崩），
#           海面动漫化但保留光斑带/霞光构图，船与人融入；出图 ≥1440x3136。
#   硬闸沿用一代 + 手指完整 + 贴图边缘无雾。交物=2 seed 初版 + 逐角色 2.4x 裁剪。
# 用法:
#   python3 gen18r2.py dering            步骤一 → output/r2dering.png（+ ComfyUI g18r2/）
#   python3 gen18r2.py ground            去圈图 Remacri x2 → 覆盖裁 1440x3136 → output/r2ground.png
#   python3 gen18r2.py asset --route il|qw [--seeds a,b]   素材 → ComfyUI g18r2/
#   python3 gen18r2.py key <png>         白底抠图 → 同名-key.png（RGBA）
#   python3 gen18r2.py paste <key> [--pos x,y --w px]      贴到 r2ground → output/r2comp_<tag>.png
#   python3 gen18r2.py final <comp> --seeds a,b            统一动漫化 → ComfyUI g18r2/
#   python3 gen18r2.py crop <png> --char sb --box x,y,w,h  2.4x 自查裁剪 → output/r2gate/
#   python3 gen18r2.py sbs <png>         与 swim-ref 并排（同高缩半）→ output/r2gate/
import json, os, shutil, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

T = os.path.dirname(os.path.abspath(__file__))
API = 'http://127.0.0.1:8188'
OUTDIR = '/mnt/g/workspace/comfyui/output'
INPUT = '/mnt/g/workspace/comfyui/input'
O = os.path.join(T, 'output')
GATE = os.path.join(O, 'r2gate')
REF = os.path.join(T, 'review', 'reference', 'rings-photoref.jpg')
SWIMREF = os.path.join(T, 'review', 'reference', 'swim-ref.jpg')
DERING = os.path.join(O, 'r2dering.png')
GROUND = os.path.join(O, 'r2ground.png')
FW, FH = 1440, 3136  # 交付下限（issue：不低于 1440x3136）
os.makedirs(GATE, exist_ok=True)

# 四个泳圈框（photoref 1100x2383 像素系；含圈体+投影/水波环，用户口径「检/框四个泳圈」）
RING_BOXES = [(185, 485, 470, 300), (530, 645, 530, 430), (55, 830, 615, 395), (230, 1190, 870, 830)]
# 放大地面后同比例框（Remacri x2 → cover 裁到 1440x3136 的仿射：s=max, 再居中裁）


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError('ComfyUI %s: %s' % (e.code, e.read().decode()[:800]))


def wait(pid, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.loads(urllib.request.urlopen(API + '/history/' + pid, timeout=30).read())
        if pid in h:
            e = h[pid]
            st = e.get('status', {})
            if st.get('status_str') == 'error':
                raise RuntimeError('ComfyUI exec error: %s' % json.dumps(e.get('outputs'))[:600])
            if st.get('completed'):
                return e
        time.sleep(5)
    raise TimeoutError(pid)


def out_of(e, node):
    im = e['outputs'][node]['images'][0]
    return os.path.join(OUTDIR, im['subfolder'], im['filename']) if im.get('subfolder') else os.path.join(OUTDIR, im['filename'])


def to_input(path, name):
    d = os.path.join(INPUT, 'g18in')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, name)
    shutil.copy(path, dst)
    return 'g18in/' + name


def dering():
    """步骤一：四圈掩膜 → FLUX.1-Fill 补全，保持实拍质感。"""
    img = Image.open(REF).convert('RGB')
    W, H = img.size
    m = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(m)
    for (x, y, w, h) in RING_BOXES:
        dr.ellipse((x, y, x + w, y + h), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(10))
    # 本机实测：此 ComfyUI 的 LoadImage→SetLatentNoiseMask 组合按「0=重绘」生效（首次跑验证），故反相。
    m = Image.eval(m, lambda v: 255 - v)
    rgba = Image.merge('RGBA', [Image.new('L', (W, H), 0)] * 3 + [m])  # alpha=掩膜，LoadImage 输出1 取回
    d1 = os.path.join(INPUT, 'g18in')
    os.makedirs(d1, exist_ok=True)
    shutil.copy(REF, os.path.join(d1, 'rings-photoref.jpg'))
    rgba.save(os.path.join(d1, 'r2ringmask.png'))
    g = {
      '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'F.1-Fill-fp16_Inpaint_Outpaint_1.0.safetensors', 'weight_dtype': 'default'}},
      '2': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['8', 0], 'text':
            'calm open ocean water surface, gentle ripples, sparkling sun glitter trail, photorealistic, '
            'seamless continuation of the sea, no swim ring, no inflatable ring, no objects, no shadow patch'}},
      '3': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['8', 0], 'text': 'swim ring, inflatable ring, shadow'}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g18in/rings-photoref.jpg'}},
      '7': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'ae.safetensors'}},
      '8': {'class_type': 'DualCLIPLoader', 'inputs': {'clip_name1': 'clip_l.safetensors', 'clip_name2': 't5xxl_fp8_e4m3fn.safetensors', 'type': 'flux'}},
      '10': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['6', 0], 'vae': ['7', 0]}},
      '16': {'class_type': 'LoadImage', 'inputs': {'image': 'g18in/r2ringmask.png'}},
      '13': {'class_type': 'SetLatentNoiseMask', 'inputs': {'samples': ['10', 0], 'mask': ['16', 1]}},
      '11': {'class_type': 'FluxGuidance', 'inputs': {'conditioning': ['2', 0], 'guidance': 30.0}},
      '12': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['11', 0], 'negative': ['3', 0],
                'latent_image': ['13', 0], 'seed': 518, 'steps': 28, 'cfg': 1.0,
                'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
      '14': {'class_type': 'VAEDecode', 'inputs': {'samples': ['12', 0], 'vae': ['7', 0]}},
      '15': {'class_type': 'SaveImage', 'inputs': {'images': ['14', 0], 'filename_prefix': 'g18r2/dering'}},
    }
    print('dering queued…', flush=True)
    e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=3600)
    shutil.copy(out_of(e, '15'), DERING)
    print('dering ->', DERING, flush=True)


UNIT_STYLE = 'flat 2d cartoon, tv cartoon style, clean lineart, simple shading, vibrant colors'
LORAS = [('sbpilot.safetensors', 0.85), ('Patrick_il.safetensors', 1.0), ('Sandy_il.safetensors', 1.0)]
POS_IL = (UNIT_STYLE + ', solo, white background, '
  'one long yellow inflatable banana boat viewed from side, with three cartoon characters on the boat: '
  'spongebob squarepants, yellow porous square sponge, white shirt red tie brown shorts, sitting upright at the left end, '
  'a pink starfish in green shorts lying on his back in the middle, '
  'a squirrel in white astronaut suit with round glass helmet sitting at the right end holding a brown coconut drink with a straw, '
  'full body of the boat and all three characters, no sea, no water, no horizon, no text, no watermark')
NEG = ('text, watermark, signature, lowres, worst quality, blurry, 3d, photorealistic, realistic, '
       'extra characters, four characters, two characters, duplicate character, fused characters, merged, hybrid, conjoined, '
       'extra arms, extra legs, missing limbs, deformed hands, extra fingers, missing fingers, '
       'red spots, blemishes, skin specks, cropped, cut off')
POS_QW = ('Sticker illustration on a plain pure white background, no sea, no water, no horizon, no shadow on background, '
  'no text, no watermark. The whole yellow inflatable banana boat with SpongeBob SquarePants, Patrick Star and '
  'Sandy Cheeks from the SpongeBob SquarePants cartoon, official 2D cartoon style with clean outlines. '
  'SpongeBob, the yellow sponge in white shirt, red tie and brown shorts, sits upright at the left end of the '
  'banana boat waving happily. Patrick, the pink starfish in green shorts with purple flowers, lies relaxed on '
  'his back in the middle. Sandy, the squirrel in her white space suit with a round transparent glass helmet, '
  'sits at the right end holding a coconut drink with a straw. Bright saturated colors, crisp clean edges, '
  'complete boat with both dark tips visible.')


def asset(route, seeds):
    """步骤二素材：船+三角色一张出。il=用户口径三 LoRA 同挂（验证用）；qw=Qwen 兜底。"""
    for s in seeds:
        if route == 'il':
            g = {}
            g['1'] = {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}}
            g['2'] = {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['1', 1],
                       'lora_name': LORAS[0][0], 'strength_model': LORAS[0][1], 'strength_clip': LORAS[0][1]}}
            g['3'] = {'class_type': 'LoraLoader', 'inputs': {'model': ['2', 0], 'clip': ['2', 1],
                       'lora_name': LORAS[1][0], 'strength_model': LORAS[1][1], 'strength_clip': LORAS[1][1]}}
            g['4'] = {'class_type': 'LoraLoader', 'inputs': {'model': ['3', 0], 'clip': ['3', 1],
                       'lora_name': LORAS[2][0], 'strength_model': LORAS[2][1], 'strength_clip': LORAS[2][1]}}
            g['5'] = {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['4', 1], 'stop_at_clip_layer': -2}}
            g['6'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['5', 0], 'text': POS_IL}}
            g['7'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['5', 0], 'text': NEG}}
            g['8'] = {'class_type': 'EmptyLatentImage', 'inputs': {'width': 1536, 'height': 1024, 'batch_size': 1}}
            g['9'] = {'class_type': 'KSampler', 'inputs': {'model': ['4', 0], 'positive': ['6', 0], 'negative': ['7', 0],
                      'latent_image': ['8', 0], 'seed': int(s), 'steps': 30, 'cfg': 5.5,
                      'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
            g['10'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['9', 0], 'vae': ['1', 2]}}
            g['11'] = {'class_type': 'SaveImage', 'inputs': {'images': ['10', 0], 'filename_prefix': f'g18r2/asset-il-{s}'}}
        else:
            res = round((1536 * 1024) ** 0.5 / 32) * 32
            g = {
              '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2.1_bf16.safetensors', 'weight_dtype': 'default'}},
              '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_8b_bf16.safetensors', 'type': 'qwen_image', 'device': 'default'}},
              '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_2.1_vae_bf16.safetensors'}},
              '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'prompt': POS_QW,
                        'negative_prompt': 'text, watermark, signature, blurry, lowres, extra characters, background scenery',
                        'resolution': min(res, 4096)}},
              '6': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 1536, 'height': 1024, 'batch_size': 1}},
              '7': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['4', 1],
                        'latent_image': ['6', 0], 'seed': int(s), 'steps': 20, 'cfg': 1.0,
                        'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
              '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['3', 0]}},
              '9': {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g18r2/asset-qw-{s}'}},
            }
        print('asset', route, s, '…', flush=True)
        e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=3600)
        print('asset', route, s, out_of(e, '11' if route == 'il' else '9'), flush=True)


def key(png):
    """白底抠图 → RGBA（阈值+去孤岛+1px alpha 羽化）。"""
    src = png if os.path.isabs(png) else os.path.join(OUTDIR, 'g18r2', png)
    im = np.asarray(Image.open(src).convert('RGB')).astype(np.int16)
    near = 245 - im.max(axis=2)          # 距白的程度
    alpha = np.clip(near * 6, 0, 255).astype(np.uint8)
    a = Image.fromarray(alpha)
    a = a.point(lambda v: 0 if v < 40 else v)
    a = a.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1))
    out = Image.open(src).convert('RGB').copy()
    out.putalpha(a)
    bbox = a.getbbox()
    out = out.crop(bbox)
    dst = os.path.splitext(src)[0] + '-key.png'
    dst_local = os.path.join(O, os.path.basename(dst))
    out.save(dst_local)
    print('key ->', dst_local, out.size, flush=True)


EDIT_PROMPT = ('去掉海面上四个彩色充气泳圈以及它们的投影和水波圈，用周围相同的海水自然补全，'
               '保持照片其余部分完全不变：同机位、同海面、同夕阳反光带、同天空。真实照片质感。')


def dering2(seed=701, denoise=0.65):
    """步骤一改用 Qwen-Image-Edit 2511：FLUX Fill 版掩膜区变磨砂灰板（见 g18r2/dering*），改指令编辑。"""
    d1 = os.path.join(INPUT, 'g18in')
    os.makedirs(d1, exist_ok=True)
    Image.open(REF).convert('RGB').resize((1104, 2384), Image.LANCZOS).save(os.path.join(d1, 'rings-photoref-edit.png'))  # Qwen 需 /16 尺寸
    g = {
      '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_edit_2511_bf16.safetensors', 'weight_dtype': 'default'}},
      '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_8b_bf16.safetensors', 'type': 'qwen_image', 'device': 'default'}},
      '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g18in/rings-photoref-edit.png'}},
      '4': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['2', 0], 'prompt': EDIT_PROMPT,
              'vae': ['3', 0], 'image1': ['6', 0]}},
      '4b': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['2', 0], 'prompt': '',
              'vae': ['3', 0], 'image1': ['6', 0]}},
      '5': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['6', 0], 'vae': ['3', 0]}},
      '7': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['4b', 0],
              'latent_image': ['5', 0], 'seed': seed, 'steps': 20, 'cfg': 1.0,
              'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': denoise}},
      '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['3', 0]}},
      '9': {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': 'g18r2/dering2'}},
    }
    print('dering2 queued…', flush=True)
    e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=3600)
    shutil.copy(out_of(e, '9'), DERING)
    print('dering2 ->', DERING, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dering':
        dering()
    elif cmd == 'dering2':
        dering2()
    elif cmd == 'asset':
        asset(sys.argv[sys.argv.index('--route') + 1], sys.argv[sys.argv.index('--seeds') + 1].split(','))
    elif cmd == 'key':
        key(sys.argv[2])
    else:
        print('unknown cmd', cmd)
        sys.exit(1)
