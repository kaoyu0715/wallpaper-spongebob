# 十三代单元出图（issue #27：直接复刻样张 密集群像+寻宝图共同焦点）。
# 管线沿用 gen-round9/11（illustriousXL + 角色 LoRA，1024x1536，30 步 dpmpp_2m karras cfg 5.5，
# 肢体完整负面清单，RMBG-2.0 抠图）；十三代按样张里每个角色的姿态/表情新增角色定义，
# 六位主角带专属 LoRA（海绵宝宝 sbpilot 0.85、其余 1.0）；背景鱼/泡泡呆鱼彩蛋无 LoRA（底模+风格词）。
# 输出 g13raw / g13cut。
# 用法: python3 gen-round13.py char --role <role> --seeds s1,s2,...
#       python3 gen-round13.py rmbg --role <role> [--keep <file>]
import importlib.util, os, sys, json, shutil

spec = importlib.util.spec_from_file_location(
    'gen_round9', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen-round9.py'))
g9 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['gen-round9.py', 'noop']
spec.loader.exec_module(g9)
sys.argv = _argv

RAW = 'g13raw'
CUT = 'g13cut'
ROLES = {}

_HELMET = ('round transparent glass space helmet, helmet interior fully transparent showing the '
           'background through it, only a plain thin white helmet ring outline, no grey glass fill, '
           'no big white reflection patches, no glass highlights')
_LAUGH = ('wide open laughing mouth, big joyful open smile showing two upper buck teeth and pink tongue')

# —— 六位主角（带 LoRA）按样张姿态/表情 ——
ROLES['sandy-top13'] = {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
  'pos': ('sandycheeks, squirrel in white astronaut suit with ' + _HELMET + ', brown fur, big fluffy brown tail, ' + _LAUGH +
          ', looking at viewer, standing, arms relaxed down away from body, arms fully visible, '
          'standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE),
  'neg_extra': 'missing tail, net, rope, lasso, camping gear, pink flower, helmet cracked, '
               'grey helmet, opaque helmet, closed mouth, smile with closed lips, three arms, gloves raised'}
ROLES['patrick-laugh13'] = {'lora': 'Patrick_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
  'pos': 'p4tr1ckst4r, official style, starfish, pink skin, green shorts, head tilted to the side, '
         'huge joyful open mouth laugh, big round eyes looking down, arms relaxed down at sides of body, '
         'arms fully visible away from body, standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE,
  'neg_extra': 'purple shorts, blue shorts, clothes pattern, missing arm, three arms, arms raised, mouth closed'}
# 二批：首批 6 种子多为“惊讶张嘴”而非样张的“开怀大笑”，显式 joyful + 笑弯眉眼
ROLES['patrick-laugh13b'] = {'lora': 'Patrick_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
  'pos': 'p4tr1ckst4r, official style, starfish, pink skin, green shorts, head tilted to the side, '
         'laughing out loud happily, huge open smiling mouth, big round joyful eyes looking down, '
         'happy raised eyebrows, arms relaxed down at sides of body, '
         'arms fully visible away from body, standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE,
  'neg_extra': 'purple shorts, blue shorts, clothes pattern, missing arm, three arms, arms raised, '
               'mouth closed, sad, worried, surprised, teeth, teeth visible, tear'}
ROLES['krabs-laugh13'] = {'lora': 'krabs-v3.safetensors', 'strength': 1.0, 'cfg': 5.5,
  'pos': 'eugene krabs, red crab, tall eye stalks, wide happy grin showing big white teeth, '
         'blue shirt, purple trousers, black belt, claws held low in front of body, '
         'standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE,
  'neg_extra': 'money, dollar sign, coins, money bag, register, three claws, extra claw, '
               'claws raised, claws spread'}

# —— 无 LoRA：底模 + 风格词（背景鱼、泡泡呆鱼彩蛋、痞老板）——
_PLAIN = {'lora': None, 'strength': 1.0, 'cfg': 5.5}
ROLES['plankton-happy13'] = dict(_PLAIN, pos=(
    'tiny green one-eyed plankton creature, single big round yellow eye with red pupil, two long thin antennae, '
    'big open happy grin showing teeth, both thin arms raised cheerfully, small happy jump pose, '
    'full body with margin around, nothing cropped, ' + g9.STYLE),
    neg_extra='two eyes, missing antenna, extra arms, realistic')
ROLES['fish-left13'] = dict(_PLAIN, pos=(
    'cartoon fish man, olive yellow-green skin, bald head with small purple fin on the back, '
    'big oval eyes looking sideways, thick purple lips, angry short eyebrows, light purple short sleeve shirt, '
    'blue shorts, long droopy fin arms, standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE),
    neg_extra='realistic fish, scales, seaweed, extra arms, human hands')
ROLES['fish-right13'] = dict(_PLAIN, pos=(
    'cartoon fish man, olive skin, big round eyes, wide smug smile with thick tan lips, '
    'light purple collared shirt, purple fin on the back of head, long droopy fin arms, '
    'standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE),
    neg_extra='realistic fish, scales, seaweed, extra arms, human hands')
ROLES['fish-tie13'] = dict(_PLAIN, pos=(
    'cartoon fish man, tan orange skin, round eyes looking sideways, teal thick lips, '
    'light blue shirt, teal necktie, grey trousers, dark fin on the back of head, '
    'long droopy fin arms, standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE),
    neg_extra='realistic fish, scales, seaweed, extra arms, human hands, holding object')
ROLES['fish-egg13'] = dict(_PLAIN, pos=(
    'tiny cartoon fish creature, light blue skin, big round silly eyes with red rings, goofy open mouth, '
    'two thin dark antennae on head, floating pose, full body with margin around, nothing cropped, ' + g9.STYLE),
    neg_extra='realistic fish, scales, bubbles, extra fins, human hands')


def char(role, seeds):
    r = ROLES[role]
    pos = r['pos']
    neg = g9.NEG_CHAR + ', ' + g9.LIMB_NEG + ', ' + r.get('neg_extra', '')
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1024, 'height': 1536, 'batch_size': 1}},
    }
    if r['lora']:
        g['2'] = {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
                  'lora_name': r['lora'], 'strength_model': r['strength'], 'strength_clip': r['strength']}}
        model, clip = ['2', 0], ['2', 1]
    else:
        model, clip = ['1', 0], ['3', 0]
    for s in seeds:
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': clip, 'text': pos}}
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': model, 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['6', 0], 'seed': int(s), 'steps': 30,
                    'cfg': r['cfg'], 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['1', 2]}}
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'{RAW}/{role}-{s}'}}
        e = g9.wait(g9.post('/prompt', {'prompt': g})['prompt_id'])
        print('char', role, s, g9.out_of(e, '9'), flush=True)


def rmbg(role, keep=None):
    d = os.path.join(g9.OUTDIR, RAW)
    files = sorted([f for f in os.listdir(d) if f.startswith(role + '-') and f.endswith('.png')],
                   key=lambda f: os.path.getmtime(os.path.join(d, f)))
    f = keep or files[-1]
    dst = os.path.join(g9.INPUT, 'g13in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g13in/' + role + '.png'}},
      '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': 'RMBG-2.0', 'process_res': 1024,
              'sensitivity': 1.0, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
              'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}},
      '3': {'class_type': 'InvertMask', 'inputs': {'mask': ['2', 1]}},
      '4': {'class_type': 'SaveImageWithAlpha', 'inputs': {'images': ['2', 0], 'mask': ['3', 0],
              'filename_prefix': f'{CUT}/{role}'}},
    }
    e = g9.wait(g9.post('/prompt', {'prompt': g})['prompt_id'])
    print('rmbg', role, 'src', f, '->', g9.out_of(e, '4'), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    if cmd == 'char':
        char(a('role'), (a('seeds') or '').split(','))
    elif cmd == 'rmbg':
        rmbg(a('role'), a('keep'))
