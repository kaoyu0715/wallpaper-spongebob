# -*- coding: utf-8 -*-
# 十七代单元出图（issue #35「保圈换鱼身」清晰通道）：白底 txt2img + 角色 LoRA →
#   RMBG-2.0 抠图 → alpha 收紧 → 历代固定后处理（SB 调色 / 派大星四件套+晕圈 / 珊迪红球清理）。
# 管线口径 = recipe §1/§2/§3（illustriousXL, CLIP -2, 30 步 dpmpp_2m karras cfg 5.5）。
# 出图 1216x1216（坐姿/仰躺为斜向构图，方形留量均衡；贴入比 ≤1.0x 天然满足）。
# 用法: python3 gen17unit.py char --role sb-ring17 --seeds 7101,7102,...
#       python3 gen17unit.py rmbg --role sb-ring17 --keep <file.png>
import importlib.util, os, sys, shutil


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace('-', '_'), os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    m = importlib.util.module_from_spec(spec)
    _argv = sys.argv
    sys.argv = [name + '.py', 'noop']
    spec.loader.exec_module(m)
    sys.argv = _argv
    return m


g9 = _load('gen-round9')

RAW = 'g17raw'
CUT = 'g17cut'
RES = 1216

_HELMET = ('round transparent glass space helmet, helmet interior fully transparent showing the '
           'background through it, only a plain thin white helmet ring outline, no grey glass fill, '
           'no big white reflection patches, no glass highlights')
LEI = ('pink and white flower lei garland worn around her neck resting on her chest, '
       'flowers strung on a cord, ')
_STYLE = g9.STYLE

# LIMB_NEG 中 'holding object' 对持杯角色不适用 → 定制负面时剔除
_LIMB = g9.LIMB_NEG.replace('holding object, ', '')

ROLES = {
  # 海绵宝宝：翘腿交叠前伸 + 墨镜推额头 + 松弛半闭眼（样张黄鱼姿态锚，白底版）
  'sb-ring17': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': ('spongebob squarepants, yellow porous sponge, tall rectangular sponge body, '
            'big blue eyes, buck teeth, white shirt, red tie, brown shorts, red striped socks, '
            'black shoes, black sunglasses resting pushed up on his forehead, half-lidded lazy eyes, '
            'lazy relaxed smile, sitting on the ground leaning back lazily, '
            'upper body reclined supported by one arm resting behind him, '
            'legs stretched forward with knees slightly bent and ankles crossed, '
            'shoes together resting in front, both legs and exactly two shoes fully visible, '
            'looking to the right, full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe, long legs, big feet, '
                 'standing, walking, sunglasses over eyes, arms crossed, crossed arms, shirtless'},
  # 海绵宝宝·二轮（对症一轮失败：7101 蓝毛衣+土色底；7102 无头+墨镜悬浮）：
  #   服装词前置（十六代实测教训）+ 姿态短语简化直叙 + 负面排毛衣/悬浮墨镜。
  'sb-ring17b': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': ('spongebob squarepants, white shirt, red tie, brown shorts, red striped socks, black shoes, '
            'yellow porous sponge, tall rectangular sponge body, big blue eyes, buck teeth, '
            'black sunglasses resting on his forehead above his eyes, half-lidded lazy eyes, lazy smile, '
            'sitting on the ground, legs stretched forward with ankles crossed, shoes together, '
            'both arms down, hands resting on the ground beside his hips, leaning back slightly, '
            'relaxed pose, looking to the right, full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe, long legs, big feet, '
                 'standing, walking, sunglasses over eyes, floating sunglasses, detached sunglasses, '
                 'arms crossed, crossed arms, shirtless, blue sweater, sweater, hoodie, hat, '
                 'headless, missing head, missing face, tan background, beige background'},
  # 海绵宝宝·三轮（对症二轮全废：坐姿模板在该 LoRA 上双主体/多头怪象）：
  #   改用十二~十五代已验证「站姿交叠踝」模板（贴圈前缘=翘腿压圈），墨镜推额保留。
  'sb-ring17c': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': ('spongebob squarepants, ' + g9.SB_TAGS + ', black sunglasses resting pushed up on forehead, '
            'relaxed lazy smile, half-lidded sleepy eyes, '
            'standing on the ground, both feet on the ground, ankles crossed, one shoe over the other, '
            'exactly two shoes, arms relaxed slightly away from body, open hands with four spread fingers, '
            'short thin legs, proportionally small legs, tall rectangular sponge body, '
            'looking to the right, full body visible with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe, long legs, long thin legs, '
                 'big feet, sitting, lying, sunglasses over eyes, floating sunglasses, '
                 'wide open mouth, tongue, two people, duplicated character'},
  # 海绵宝宝·四轮（对症三轮：墨镜糊在眼上/断鞋/迈步）：墨镜顶头顶、眼部完整露出词面直给。
  'sb-ring17d': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': ('spongebob squarepants, ' + g9.SB_TAGS + ', '
            'black sunglasses resting on the very top of his head pushed up high above his eyes, '
            'both eyes fully visible and half-lidded, relaxed lazy smile with closed mouth, '
            'standing on the ground, both feet on the ground, ankles crossed, '
            'one shoe crossed over the other shoe, exactly two shoes, '
            'arms relaxed at his sides, open hands with four spread fingers, '
            'short thin legs, proportionally small legs, tall rectangular sponge body, '
            'looking to the right, full body visible with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe, long legs, long thin legs, '
                 'big feet, sitting, lying, walking, sunglasses over eyes, sunglasses on eyes, '
                 'eyes hidden, black paint over face, black hair, wide open mouth, tongue, '
                 'two people, duplicated character'},
  # 珊迪·三轮（对症二轮：站姿/方盒椰/罐装）：坐姿词前置 + 椰子形状负面排除。
  'san-ring17c': {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': ('sandycheeks, sitting on the ground with legs stretched forward, brown shoes on her feet, '
            'squirrel in white astronaut suit, ' + _HELMET + ', brown fur and tail, '
            'standard squirrel face with buck teeth, calm relaxed smile, half-lidded eyes, '
            'hugging a whole round brown coconut with a bendy straw to her chest with both paws, '
            'the straw touches her lips, round hairy coconut, not a box, '
            'looking at viewer, full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'missing tail, net, rope, lasso, camping gear, grey helmet, opaque helmet, '
                 'thick helmet rim, helmet cracked, headscarf, turban, black strap, cocktail glass, '
                 'cocktail, wine glass, glass of juice, tin can, blue can, box, rectangle, carton, '
                 'standing, walking, three arms, flowers floating, flower on helmet, snacks, food'},
  # 海绵宝宝·五轮（四轮仍未出「墨镜推额」：7131 悬浮墨镜、7132/7133 戴眼上）：
  #   墨镜词改「pushed up onto his forehead」直给额头位，LoRA 强度提至 0.9。
  'sb-ring17e': {'lora': 'sbpilot.safetensors', 'strength': 0.9, 'cfg': 5.5,
    'pos': ('spongebob squarepants, ' + g9.SB_TAGS + ', '
            'black sunglasses pushed up onto his forehead resting above his eyebrows, '
            'big blue half-lidded eyes fully visible under the sunglasses, '
            'relaxed lazy smile with closed mouth, '
            'standing on the ground, both feet on the ground, ankles crossed, '
            'one shoe crossed over the other shoe, exactly two shoes, '
            'arms relaxed at his sides, open hands with four spread fingers, '
            'short thin legs, proportionally small legs, tall rectangular sponge body, '
            'looking to the right, full body visible with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe, long legs, long thin legs, '
                 'big feet, sitting, lying, walking, sunglasses over eyes, glasses over eyes, '
                 'sunglasses covering eyes, sunglasses on eyes, floating sunglasses, hat, '
                 'eyes hidden, black paint over face, black hair, wide open mouth, tongue, '
                 'two people, duplicated character, missing eyes'},
  # 海绵宝宝·六轮（五轮仍未中：多主体/墨镜位置漂移）。
  #   回退到十六代裁片路线「已验证词面」（当时同款 LoRA 出过墨镜推额+交叠踝头身），
  #   仅把「坐在泳圈里」替换为「坐在地上」，其余措辞原样保留。
  'sb-ring17f': {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
    'pos': ('spongebob squarepants, 1boy, solo, single character only, yellow porous sponge, square body, official cartoon proportions, '
            'big blue eyes, buck teeth, solid filled white collared shirt with bold black outlines, '
            'red tie, brown shorts, red striped socks, black shoes, fully colored opaque torso, '
            'no sketch lines, no translucent areas, flat cel colors, '
            'black sunglasses resting pushed up on forehead, relaxed lazy smile, half-lidded sleepy eyes, '
            'sitting relaxed with legs crossed at the ankles stretched toward viewer, '
            'both arms resting at his sides, '
            'full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'standing, waving hand, wide open mouth, tongue, sunglasses over eyes, '
                 'legs dangling, horizon, sky, two people, duplicated character'},
  # 派大星：仰躺肘撑 + 单手举橙汁杯（含吸管/冰块/橙片）+ 双腿前伸（样张绿鱼姿态锚）
  'pat-ring17': {'lora': 'Patrick_il.safetensors', 'strength': 1.0, 'cfg': 6.0,
    'pos': ('p4tr1ckst4r, official style, starfish, pink skin, green shorts, '
            'reclining on the ground, upper body leaning back propped on one elbow behind him, '
            'head relaxed tilted back, one hand holding up a tall glass of orange juice with ice '
            'cubes and orange slice and a drinking straw in front of his chest, '
            'legs stretched forward on the ground crossed at the ankles, feet visible, '
            'content expression, half-closed eyes, small closed-mouth smile, '
            'full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'purple shorts, blue shorts, clothes pattern, missing glass, empty hand, '
                 'standing, sitting upright, frowning, teeth'},
  # 珊迪·二轮（对症一轮：7301 椰子落地旁置、其余站姿/不透明盔/杯装鸡尾）：
  #   「双手捧椰子于胸前用吸管喝」直叙；花环本轮取下（宁可无不可错位，一轮亦未出花环）。
  'san-ring17b': {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': ('sandycheeks, squirrel in white astronaut suit, ' + _HELMET + ', brown fur and tail, '
            'standard squirrel face with buck teeth, calm relaxed smile, half-lidded eyes, '
            'sitting on the ground with legs stretched forward, brown shoes on her feet, '
            'holding a whole brown coconut drink with both hands in front of her chest, '
            'drinking through a bendy straw, straw at her mouth, '
            'looking at viewer, full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'missing tail, net, rope, lasso, camping gear, grey helmet, opaque helmet, '
                 'thick helmet rim, helmet cracked, headscarf, turban, black strap, cocktail glass, '
                 'cocktail, wine glass, glass of juice, standing, walking, three arms, '
                 'flowers floating, flower on helmet, flower on head, snacks, food, plate'},
  # 珊迪：坐姿抱椰子饮 + 透明盔（细白圈）+ 花环围颈（样张白袍鱼姿态锚）
  'san-ring17': {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
    'pos': ('sandycheeks, squirrel in white astronaut suit, ' + _HELMET + ', brown fur and tail, '
            'standard squirrel face with buck teeth, calm relaxed smile, half-lidded eyes, '
            'sitting on the ground with legs stretched forward crossed at the ankles, '
            'brown shoes on her feet, ' + LEI +
            'one hand holding a whole brown coconut drink with a bendy straw at her chest, '
            'straw pointing up toward her face, other arm resting relaxed beside her hip, '
            'looking at viewer, full body with margin around, nothing cropped, ' + _STYLE),
    'neg_extra': 'missing tail, net, rope, lasso, camping gear, grey helmet, opaque helmet, '
                 'thick helmet rim, helmet cracked, headscarf, turban, black strap, cocktail glass, '
                 'flowers on the ground, lei held away from neck, standing, three arms'},
}
# 花环变体（宁可无，不可错位——去花环档）
SAN_NOLEI_POS = LEI
SAN_NOLEI_NEG = 'flower lei, lei garland, necklace, flowers around neck, flowers, garland, '


def _submit(g):
    import time, json, urllib.error, urllib.request
    # 服务器忙时 POST 返回 500：先等队列清空再提交，失败重试
    # 服务器忙（他人作业在跑）时 POST 返回 500：等到队列空闲再提交，失败退避重试
    last = None
    for k in range(60):
        try:
            q = json.loads(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=10).read())
            busy = bool(q['queue_running'] or q['queue_pending'])
        except Exception:
            busy = True
        if busy:
            if last != 'busy':
                print('queue busy, waiting…', flush=True)
                last = 'busy'
            time.sleep(10)
            continue
        try:
            return g9.post('/prompt', {'prompt': g})['prompt_id']
        except urllib.error.HTTPError as e:
            print('POST retry %d: HTTP %s' % (k + 1, e.code), flush=True)
            last = 'post'
            time.sleep(min(30, 5 * (k + 1)))
        except Exception as e:
            print('POST retry %d: %s' % (k + 1, e), flush=True)
            last = 'err'
            time.sleep(10)
    raise RuntimeError('ComfyUI POST failed')


def char(role, seeds, lei=True):
    r = ROLES[role]
    pos = r['pos']
    neg = g9.NEG_CHAR + ', ' + _LIMB + ', ' + r['neg_extra']
    if role == 'san-ring17' and not lei:
        pos = pos.replace(SAN_NOLEI_POS, '')
        neg = SAN_NOLEI_NEG + neg
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': RES, 'height': RES, 'batch_size': 1}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': r['lora'], 'strength_model': r['strength'], 'strength_clip': r['strength']}},
    }
    # 一次提交整角色多种子（服务器忙时 POST 频繁 500，单次提交出全套，摊薄失败成本）
    g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': pos}}
    for i, s in enumerate(seeds):
        g['7-%d' % i] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
                        'negative': ['5', 0], 'latent_image': ['6', 0], 'seed': int(s), 'steps': 30,
                        'cfg': r['cfg'], 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8-%d' % i] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7-%d' % i, 0], 'vae': ['1', 2]}}
        g['9-%d' % i] = {'class_type': 'SaveImage', 'inputs': {'images': ['8-%d' % i, 0],
                        'filename_prefix': '%s/%s-%s' % (RAW, role, s)}}
    e = g9.wait(_submit(g), timeout=7200)
    for i, s in enumerate(seeds):
        print('char', role, s, g9.out_of(e, '9-%d' % i), flush=True)


def rmbg(role, keep=None, lei=True):
    d = os.path.join(g9.OUTDIR, RAW)
    files = sorted([f for f in os.listdir(d) if f.startswith(role + '-') and f.endswith('.png')],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    f = keep or files[-1]
    dst = os.path.join(g9.INPUT, 'g17in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g17in/' + role + '.png'}},
      '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': 'RMBG-2.0', 'process_res': 1216,
              'sensitivity': 1.0, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
              'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}},
      '3': {'class_type': 'InvertMask', 'inputs': {'mask': ['2', 1]}},
      '4': {'class_type': 'SaveImageWithAlpha', 'inputs': {'images': ['2', 0], 'mask': ['3', 0],
              'filename_prefix': '%s/%s' % (CUT, role + '-' + os.path.basename(f).replace('.png', ''))}},
    }
    e = g9.wait(_submit(g))
    print('rmbg', role, 'src', f, '->', g9.out_of(e, '4'), flush=True)
    return g9.out_of(e, '4')


def post(cut_path, out_path=None):
    """固定后处理链（顺序不可换，recipe §4.3/§4.3a/§4.4 历代口径）→ bbox 裁剪存档 RGBA。"""
    from PIL import Image
    c8 = _load('compose-round8')
    c11 = _load('compose-round11')
    c14 = _load('compose-round14')
    import patrick_skin
    import sb_palette as SBP
    im = Image.open(cut_path).convert('RGBA')
    role = os.path.basename(cut_path)
    if role.startswith('sb'):
        # 坐姿单元：0.68 站姿口径「缝合手术」不适用（issue #35：坐姿以样张姿态为锚如实记录）
        import sb_proportion as SP
        c12 = _load('compose-round12')
        r0 = SP.leg_ratio(im)[0]
        print('SB 坐姿投影腿身比（如实记录，不做缝合）: %s' % (r0,))
        SBP_raw = _load('sb_palette')    # 全新实例=未打补丁的原版调色（坐姿不做缝合手术）
        im = SBP_raw.sb_palette(im)
    if role.startswith('pat'):
        im = c8.clean_patrick_blue(im)
        im = c8.clean_patrick_blotches(im)
        im = patrick_skin.clean_patrick_skin(im)
        im = c8.fix_patrick_outline(im)
        im = c14.pat_halo_clean(im)
    if role.startswith('san'):
        im = c11.clean_sandy_ball(im)
    im = c8.tighten_alpha(im)
    bb = im.split()[3].getbbox()
    im = im.crop(bb)
    out = out_path or cut_path.replace('.png', '-post.png')
    im.save(out)
    print('post ->', out, im.size)
    return out


if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    if cmd == 'char':
        char(a('role'), (a('seeds') or '').split(','), bool(int(a('lei', '1'))))
    elif cmd == 'rmbg':
        rmbg(a('role'), a('keep'), bool(int(a('lei', '1'))))
    elif cmd == 'post':
        post(a('in'))
