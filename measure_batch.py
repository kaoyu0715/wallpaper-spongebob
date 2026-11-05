# 十二代测量（issue #25）：按 compose-round11 的贴入几何**精确重建**每张 r11 成品中
# 各角色在画面里的实际可见包围盒（后贴角色遮挡先贴角色），得到"实测相对身高"。
# 可见高 = 贴入 footprint 掩膜 减去 之后贴入角色掩膜 后的内容上下缘差。
# 用法: python3 measure_batch.py
import json, os, sys
import importlib.util
import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('compose_round11', os.path.join(here, 'compose-round11.py'))
c11 = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ['compose-round11.py', '--bg', 'x', '--outdir', 'x']
spec.loader.exec_module(c11)
sys.argv = _argv

m9 = c10 = c11.m9
W, H = c11.W, c11.H
# 复现 --sandy-new：十一代成品用的是 g11cut 新珊迪单元
g11 = '/mnt/g/workspace/comfyui/output/g11cut'
c11.UNITS11['new_sandy_c'] = os.path.join(g11, 'sandy-cheer11_00001.png')
c11.UNITS11['new_sandy_l'] = os.path.join(g11, 'sandy-left11_00001.png')

def char_masks(scene_id):
    """返回 [(key, mask bool HxW)]，贴入顺序（后→前）。掩膜=alpha>40 经旋转/缩放后的 footprint。"""
    sc = c11.SCENES11[scene_id]
    masks = []
    for key, h, cx, gy, tilt in sc['chars']:
        path = c11.UNITS11.get('new_' + key, c11.UNITS11[key]) if key.startswith('sandy') else c11.UNITS11[key]
        fig = Image.open(path).convert('RGBA')
        a = np.asarray(fig)[..., 3]
        a = np.where(a >= 115, 255, 0)           # tighten_alpha floor
        ys, xs = np.nonzero(a > 0)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = Image.fromarray(a.astype(np.uint8), 'L')
        if tilt:
            img = img.rotate(tilt, expand=True, resample=Image.BILINEAR)
        r = h / img.height
        fw = max(1, int(img.width * r))
        img = img.resize((fw, h), Image.NEAREST)
        full = np.zeros((H, W), bool)
        full[gy - h:gy - h + h, int(cx - fw / 2):int(cx - fw / 2) + fw] = np.asarray(img) > 40
        masks.append((key, full))
    return masks

def measure(scene_id):
    masks = char_masks(scene_id)
    out = []
    for i, (key, mk) in enumerate(masks):
        later = np.zeros_like(mk)
        for _, mk2 in masks[i + 1:]:
            later |= mk2
        vis = mk & ~later
        ys = np.nonzero(vis.any(1))[0]
        xs = np.nonzero(vis.any(0))[0]
        out.append(dict(key=key, top=int(ys[0]), bottom=int(ys[-1]),
                        vis_h=int(ys[-1] - ys[0] + 1),
                        paste_h=int([h for k, h, *_ in c11.SCENES11[scene_id]['chars'] if k == key][0])))
    return out

if __name__ == '__main__':
    res = {}
    for sid in sorted(c11.SCENES11):
        res[sid] = measure(sid)
    json.dump(res, open('output/measure/r11-visible-heights.json', 'w'), ensure_ascii=False, indent=1)
    for sid, rows in res.items():
        sb = [r for r in rows if r['key'] == 'sb_r']
        sbh = sb[0]['vis_h'] if sb else None
        print(sid)
        for r in rows:
            rel = ('rel=%.2f' % (r['vis_h'] / sbh)) if sbh else ''
            print('  %-10s paste=%d vis_h=%d top=%d %s' % (r['key'], r['paste_h'], r['vis_h'], r['top'], rel))
