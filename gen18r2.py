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
RING_BOXES = [(165, 465, 510, 345), (510, 625, 565, 565), (20, 810, 675, 550), (210, 1165, 890, 1010)]
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
    """白底抠图 → RGBA。旧版「近白即透明」会误删高亮彩色（黄船 R=255→透明）——
    改为从画面边缘洪泛填充白底，只去与背景连通的白色（角色白衬衫等内部白色保留）。"""
    src = png if os.path.isabs(png) else os.path.join(OUTDIR, 'g18r2', png)
    im = Image.open(src).convert('RGB')
    W, H = im.size
    MARK = (255, 0, 255)
    px = im.load()
    seeds = [(x, 0) for x in range(0, W, 4)] + [(x, H - 1) for x in range(0, W, 4)] \
          + [(0, y) for y in range(0, H, 4)] + [(W - 1, y) for y in range(0, H, 4)]
    for (x, y) in seeds:
        r, g, b = px[x, y]
        if min(r, g, b) > 235 and (r, g, b) != MARK:
            ImageDraw.floodfill(im, (x, y), MARK, thresh=12)
    arr = np.asarray(im).astype(np.int16)
    is_bg = (np.abs(arr - np.array(MARK)).max(axis=2)) == 0
    alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    a = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.6))  # 边缘微羽化防锯齿
    out = Image.open(src).convert('RGB')
    out.putalpha(a)
    bbox = a.getbbox()
    out = out.crop(bbox)
    dst_local = os.path.join(O, os.path.splitext(os.path.basename(src))[0] + '-key.png')
    out.save(dst_local)
    print('key ->', dst_local, out.size, flush=True)


EDIT_PROMPT = ('去掉海面上四个彩色充气泳圈以及它们的投影和水波圈，用周围相同的海水自然补全，'
               '保持照片其余部分完全不变：同机位、同海面、同夕阳反光带、同天空。真实照片质感。')


def dering2(seed=701, denoise=1.0):
    """步骤一改用 Qwen-Image-Edit 2511：FLUX Fill 版掩膜区变磨砂灰板（见 g18r2/dering*），改指令编辑。
    注意：edit-2511 配 qwen_2.5_vl_7b（3584 hidden）与 16ch qwen_image_vae；2.1 组合不通用。
    Edit 模型按「全噪声+参考图」训练，denoise 必须 1.0（0.65 实测全黑，seed 701）。"""
    d1 = os.path.join(INPUT, 'g18in')
    os.makedirs(d1, exist_ok=True)
    Image.open(REF).convert('RGB').resize((1104, 2384), Image.LANCZOS).save(os.path.join(d1, 'rings-photoref-edit.png'))  # Qwen 需 /16 尺寸
    g = {
      '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_edit_2511_bf16.safetensors', 'weight_dtype': 'default'}},
      '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'type': 'qwen_image', 'device': 'default'}},
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


def dering3(seed=733):
    """步骤三选一：Qwen-Image-2.1 已验证栈做掩膜 inpaint 去圈（Edit-2511 两次全黑，fp8 clip 存疑；
    FLUX Fill 磨砂灰板弃用）。掩膜语义实测：LoadImage 的 MASK=1-alpha，SetLatentNoiseMask 在 1 处加噪
    → 圈区 alpha 需=0。提示词写真海面补全，保持照片感。"""
    img = Image.open(REF).convert('RGB')
    W0, H0 = img.size
    img = img.resize((1104, 2384), Image.LANCZOS)  # /16
    W, H = img.size
    m = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(m)
    sx, sy = W / W0, H / H0
    for (x, y, w, h) in RING_BOXES:  # 圆角矩形保证整圈覆盖（椭圆实测漏圈底弧）
        dr.rounded_rectangle((int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)),
                             radius=int(min(w, h) * sx * 0.22), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(10))
    m = Image.eval(m, lambda v: 255 - v)  # 圈区 alpha=0 → MASK=1 → 重绘
    d1 = os.path.join(INPUT, 'g18in')
    os.makedirs(d1, exist_ok=True)
    img.save(os.path.join(d1, 'rings-photoref-edit.png'))
    Image.merge('RGBA', [Image.new('L', (W, H), 0)] * 3 + [m]).save(os.path.join(d1, 'r2ringmask.png'))
    g = {
      '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2.1_bf16.safetensors', 'weight_dtype': 'default'}},
      '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_8b_bf16.safetensors', 'type': 'qwen_image', 'device': 'default'}},
      '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_2.1_vae_bf16.safetensors'}},
      '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'prompt':
            'realistic photograph of a calm open ocean sea surface, gentle ripples and sparkling sun glitter trail, '
            'the colorful inflatable swim rings and their shadows are completely removed, seamless natural water everywhere',
            'negative_prompt': 'swim ring, inflatable ring, shadow, object, text, watermark, cartoon', 'resolution': 1024}},
      '6': {'class_type': 'LoadImage', 'inputs': {'image': 'g18in/rings-photoref-edit.png'}},
      '10': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['6', 0], 'vae': ['3', 0]}},
      '16': {'class_type': 'LoadImage', 'inputs': {'image': 'g18in/r2ringmask.png'}},
      '13': {'class_type': 'SetLatentNoiseMask', 'inputs': {'samples': ['10', 0], 'mask': ['16', 1]}},
      '7': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['4', 1],
              'latent_image': ['13', 0], 'seed': seed, 'steps': 20, 'cfg': 1.0,
              'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
      '8': {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['3', 0]}},
      '9': {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': 'g18r2/dering3'}},
    }
    print('dering3 queued…', flush=True)
    e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=3600)
    shutil.copy(out_of(e, '9'), DERING)
    print('dering3 ->', DERING, flush=True)


def _up2(path, tag):
    """Remacri x2（ComfyUI）→ 输出路径。"""
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': to_input(path, tag + '.png')}},
      '2': {'class_type': 'UpscaleModelLoader', 'inputs': {'model_name': '4x_foolhardy_Remacri.pth'}},
      '3': {'class_type': 'ImageUpscaleWithModel', 'inputs': {'upscale_model': ['2', 0], 'image': ['1', 0]}},
      '4': {'class_type': 'SaveImage', 'inputs': {'images': ['3', 0], 'filename_prefix': 'g18r2/' + tag}},
    }
    e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=1800)
    return out_of(e, '4')


def ground():
    """去圈图 Remacri x2 → cover 1440x3136（居中裁）→ r2ground.png。"""
    big = _up2(DERING, 'r2up')
    im = Image.open(big).convert('RGB')
    w, h = im.size
    s = max(FW / w, FH / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    im = im.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - FW) // 2, (nh - FH) // 2
    im.crop((x0, y0, x0 + FW, y0 + FH)).save(GROUND)
    print('ground ->', GROUND, im.size, flush=True)


def paste(key_png, pos=None, width=None, tag='a'):
    """素材（RGBA）贴到 r2ground 下部中部。默认宽=画面 92%，中心 y≈60%。"""
    comp = Image.open(GROUND).convert('RGB')
    a = Image.open(key_png if os.path.isabs(key_png) else os.path.join(O, key_png)).convert('RGBA')
    w = width or int(FW * 0.92)
    a = a.resize((w, int(a.height * w / a.width)), Image.LANCZOS)
    cx, cy = pos or (FW // 2, int(FH * 0.60))
    px, py = cx - w // 2, cy - a.height // 2
    comp.paste(a, (px, py), a)
    dst = os.path.join(O, 'r2comp_%s.png' % tag)
    comp.save(dst)
    print('comp ->', dst, flush=True)


POS_FIN = (UNIT_STYLE + ', a long yellow inflatable banana boat floating on a calm turquoise sea, '
  'three cartoon characters on the banana boat: yellow porous square sponge in white shirt red tie brown shorts '
  'sitting at the left end, pink starfish in green shorts lying on his back in the middle, '
  'squirrel in white astronaut suit with round glass helmet holding a coconut drink at the right end, '
  'calm ocean water with gentle ripples, sparkling sun glitter trail on the water, '
  'soft sunset glow near the horizon in the upper third, relaxed summer vacation mood, '
  'exactly three separate characters, no text, no watermark')


def final(comp, seeds, denoise=0.5):
    """步骤三：整图 img2img（d≈0.5，挂全部角色 LoRA 防脸崩）→ 统一动漫化。"""
    for s in seeds:
        g = {}
        g['1'] = {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}}
        prev = '1'
        for i, (ln, st) in enumerate(LORAS, start=2):
            g[str(i)] = {'class_type': 'LoraLoader', 'inputs': {'model': [prev, 0], 'clip': [prev, 1],
                          'lora_name': ln, 'strength_model': st, 'strength_clip': st}}
            prev = str(i)
        g['5'] = {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['4', 1], 'stop_at_clip_layer': -2}}
        g['6'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['5', 0], 'text': POS_FIN}}
        g['7'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['5', 0], 'text': NEG}}
        g['8'] = {'class_type': 'LoadImage', 'inputs': {'image': to_input(comp, 'r2comp.png')}}
        g['9'] = {'class_type': 'VAEEncode', 'inputs': {'pixels': ['8', 0], 'vae': ['1', 2]}}
        g['10'] = {'class_type': 'KSampler', 'inputs': {'model': ['4', 0], 'positive': ['6', 0], 'negative': ['7', 0],
                   'latent_image': ['9', 0], 'seed': int(s), 'steps': 30, 'cfg': 5.5,
                   'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': float(denoise)}}
        g['11'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['10', 0], 'vae': ['1', 2]}}
        g['12'] = {'class_type': 'SaveImage', 'inputs': {'images': ['11', 0], 'filename_prefix': f'g18r2/final-{s}'}}
        print('final', s, '…', flush=True)
        e = wait(post('/prompt', {'prompt': g})['prompt_id'], timeout=5400)
        print('final', s, out_of(e, '12'), flush=True)


def crop(png, char, box, zoom=2.4):
    im = Image.open(png if os.path.isabs(png) else os.path.join(O, png)).convert('RGB')
    x, y, w, h = box
    c = im.crop((x, y, x + w, y + h)).resize((int(w * zoom), int(h * zoom)), Image.LANCZOS)
    base = os.path.splitext(os.path.basename(png))[0]
    dst = os.path.join(GATE, '%s-%s@2.4x.png' % (base, char))
    c.save(dst)
    print(dst, flush=True)


def sbs(png):
    ref = Image.open(SWIMREF).convert('RGB')
    th = 1568
    ref = ref.resize((int(ref.width * th / ref.height), th), Image.LANCZOS)
    cand = Image.open(png if os.path.isabs(png) else os.path.join(O, png)).convert('RGB')
    cand = cand.resize((int(cand.width * th / cand.height), th), Image.LANCZOS)
    sheet = Image.new('RGB', (ref.width + cand.width + 16, th), (255, 255, 255))
    sheet.paste(ref, (0, 0))
    sheet.paste(cand, (ref.width + 16, 0))
    base = os.path.splitext(os.path.basename(png))[0]
    dst = os.path.join(GATE, base + '-sbs-vs-swimref.png')
    sheet.save(dst, optimize=True)
    print(dst, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dering':
        dering()
    elif cmd == 'dering2':
        dering2()
    elif cmd == 'dering3':
        dering3()
    elif cmd == 'ground':
        ground()
    elif cmd == 'paste':
        a = sys.argv[2:]
        pos = None; width = None; tag = 'a'
        if '--pos' in a:
            xy = a[a.index('--pos') + 1].split(','); pos = (int(xy[0]), int(xy[1]))
        if '--w' in a:
            width = int(a[a.index('--w') + 1])
        if '--tag' in a:
            tag = a[a.index('--tag') + 1]
        paste(a[0], pos, width, tag)
    elif cmd == 'final':
        a = sys.argv[2:]
        d = float(a[a.index('--d') + 1]) if '--d' in a else 0.5
        final(a[0], a[a.index('--seeds') + 1].split(','), d)
    elif cmd == 'crop':
        a = sys.argv[2:]
        crop(a[0], a[a.index('--char') + 1], tuple(int(v) for v in a[a.index('--box') + 1].split(',')))
    elif cmd == 'sbs':
        sbs(sys.argv[2])
    elif cmd == 'asset':
        asset(sys.argv[sys.argv.index('--route') + 1], sys.argv[sys.argv.index('--seeds') + 1].split(','))
    elif cmd == 'key':
        key(sys.argv[2])
    else:
        print('unknown cmd', cmd)
        sys.exit(1)
