# -*- coding: utf-8 -*-
# 十五代 A 路线探针（issue #31）：全身瘫坐「坐地前伸腿」白底单元（配方管线口径），
# 供 A/B 对比章节取证：证明 A 路线单元可达全身瘫姿，但光影统一仍受贴入天花板限制。
# 用法: python3 gen15a.py unit --role sb-sit --seeds 11,22
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen15b import post, wait, out_of, STYLE, NEG_BASE

NEG_CHAR = NEG_BASE + ', shadow, floor, ground, chair, object, drink, glass'
UNITS = {
  'sb-sit': dict(lora='sbpilot.safetensors', strength=0.85,
    pos=('spongebob squarepants, yellow porous sponge, square body, big blue eyes, buck teeth, '
         'white shirt, red tie, brown shorts, red striped socks, black shoes, '
         'sitting on the ground, legs stretched forward and crossed at the ankles, '
         'leaning back on both hands behind hips, relaxed closed-mouth smile, looking to the right, '
         'full body visible with margin around, nothing cropped, ' + STYLE)),
  'pat-sit': dict(lora='Patrick_il.safetensors', strength=1.0,
    pos=('p4tr1ckst4r, official style, starfish, pink skin, green shorts, '
         'sitting on the ground, legs stretched forward slightly bent, leaning back on both hands, '
         'content half-closed eyes, closed-mouth smile, looking to the left, '
         'full body visible with margin around, nothing cropped, ' + STYLE)),
}

def unit(role, seeds):
    u = UNITS[role]
    g = {
      '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'illustriousXL_v01.safetensors'}},
      '3': {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['1', 1], 'stop_at_clip_layer': -2}},
      '2': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['3', 0],
              'lora_name': u['lora'], 'strength_model': u['strength'], 'strength_clip': u['strength']}},
      '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 1], 'text': u['pos']}},
      '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['3', 0], 'text': NEG_CHAR}},
      '6': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1024, 'height': 1536, 'batch_size': 1}},
    }
    for s in seeds:
        g['7'] = {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
                    'negative': ['5', 0], 'latent_image': ['6', 0], 'seed': int(s), 'steps': 30,
                    'cfg': 5.5, 'sampler_name': 'dpmpp_2m', 'scheduler': 'karras', 'denoise': 1.0}}
        g['8'] = {'class_type': 'VAEDecode', 'inputs': {'samples': ['7', 0], 'vae': ['1', 2]}}
        g['9'] = {'class_type': 'SaveImage', 'inputs': {'images': ['8', 0], 'filename_prefix': f'g15raw/{role}-{s}'}}
        e = wait(post('/prompt', {'prompt': g})['prompt_id'])
        print('unit', role, s, out_of(e, '9'), flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    unit(a('role'), [int(x) for x in a('seeds').split(',')])
