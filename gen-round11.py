# 十一代单元出图（issue #23，自进化逼近样张）：
#   管线与 gen-round9 完全相同（illustriousXL + 角色 LoRA，1024x1536，30 步 dpmpp_2m karras cfg 5.5），
#   变化只有角色定义——对症十一代差距清单：
#   ① 珊迪：样张头盔为「扁平透明罩 + 白圈」，现有单元是灰半透明填充+大白高光块（玻璃感）；
#      且样张珊迪为大张嘴露齿露舌大笑。新 prompt 显式要求透明罩、仅白圈、张嘴露舌。
#   ② 海绵宝宝：样张嘴内可见舌头；现胜出单元嘴内深色无舌。补充新种子（保险，若后处理调色后
#      现有单元已够接近样张则不启用）。
# 输出前缀 g11raw / g11cut，其余与 gen-round9 一致。
# 用法: python3 gen-round11.py char --role <role> --seeds s1,s2,...
#       python3 gen-round11.py rmbg --role <role> [--keep <file>]
import importlib.util, os, sys, json, shutil

spec = importlib.util.spec_from_file_location(
    'gen_round9', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen-round9.py'))
g9 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['gen-round9.py', 'noop']
spec.loader.exec_module(g9)
sys.argv = _argv

RAW = 'g11raw'
CUT = 'g11cut'
ROLES = dict(g9.ROLES)

# 珊迪欢呼·十一代版：透明头盔（仅白圈）+ 大张嘴露齿露舌
_HELMET = ('round transparent glass space helmet, helmet interior fully transparent showing the '
           'background through it, only a plain thin white helmet ring outline, no grey glass fill, '
           'no big white reflection patches, no glass highlights')
_LAUGH = ('wide open laughing mouth, big joyful open smile showing two upper buck teeth and pink tongue')
ROLES['sandy-cheer11'] = {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
  'pos': ('sandycheeks, squirrel in white astronaut suit with ' + _HELMET + ', brown fur and tail, ' + _LAUGH +
          ', standing, both arms raised cheerfully, arms fully visible away from body, '
          'looking at viewer, standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE),
  'neg_extra': 'missing tail, net, rope, lasso, camping gear, pink flower, helmet cracked, '
               'grey helmet, opaque helmet, closed mouth, smile with closed lips'}
# 珊迪挥手·十一代版
ROLES['sandy-left11'] = {'lora': 'Sandy_il.safetensors', 'strength': 1.0, 'cfg': 5.5,
  'pos': ('sandycheeks, squirrel in white astronaut suit with ' + _HELMET + ', brown fur and tail, ' + _LAUGH +
          ', one arm raised waving with open hand, other arm relaxed away from body, '
          'looking to the left, standing on the ground, full body with margin around, nothing cropped, ' + g9.STYLE),
  'neg_extra': 'missing tail, net, rope, lasso, camping gear, pink flower, helmet cracked, '
               'grey helmet, opaque helmet, closed mouth, smile with closed lips, three arms'}
# 海绵宝宝·十一代版：露舌
ROLES['sb-right11'] = {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
  'pos': ('spongebob squarepants, ' + g9.SB_TAGS +
          ', standing on the ground, both feet on the ground, exactly two shoes, '
          'arms relaxed slightly away from body, open hands with four spread fingers, '
          'happy open mouth smile with two buck teeth and tongue visible, looking to the right, '
          'full body visible with margin around, nothing cropped, ' + g9.STYLE),
  'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe'}


def char(role, seeds):
    r = ROLES[role]
    pos, neg = r['pos'], g9.NEG_CHAR + ', ' + g9.LIMB_NEG + ', ' + r.get('neg_extra', '')
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': neg}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1024, 'height': 1536, 'batch_size': 1}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0], 'lora_name': r['lora'],
              'strength_model': r['strength'], 'strength_clip': r['strength']}},
    }
    for s in seeds:
        g['4'] = {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': pos}}
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
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
    dst = os.path.join(g9.INPUT, 'g11in')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(d, f), os.path.join(dst, role + '.png'))
    g = {
      '1': {'class_type': 'LoadImage', 'inputs': {'image': 'g11in/' + role + '.png'}},
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
