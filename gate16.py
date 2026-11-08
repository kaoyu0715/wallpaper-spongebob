# -*- coding: utf-8 -*-
# 十六代分区过闸自检（issue #33 硬性流程）：单候选贴回预览 + 样张同区 2-3x 并排。
# 用法: python3 gate16.py --fish fish2 --cand <crop.png> --name p1-1234 [--zoom 2.4]
# 输出: output/gate/<fish>-<name>-paste.png（贴回预览）/ output/gate/<fish>-<name>-sbs.png（并排）
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
import mask16, sbs16

T = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(T, 'review/reference/swim-ref.jpg')
OUTD = os.path.join(T, 'output', 'gate')
os.makedirs(OUTD, exist_ok=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fish', required=True)
    ap.add_argument('--cand', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--zoom', type=float, default=2.4)
    ap.add_argument('--sil', type=int, default=1, help='1=轮廓门控贴回（十六代 v2），0=旧紧凑档')
    ap.add_argument('--dil', type=int, default=14)
    ap.add_argument('--feather', type=int, default=4)
    ap.add_argument('--extras', default='', help='如 "740,1150,108,108;..."（参考帧椭圆）')
    ap.add_argument('--hole', type=int, default=1)
    a = ap.parse_args()
    extras = tuple(tuple(float(v) for v in e.split(',')) for e in a.extras.split(';') if e)
    ki = mask16.zz_keep(dil=5, feather=3) if a.fish == 'fish1' and a.sil else None
    if a.sil:
        prev = mask16.sil_paste_back(Image.open(REF), a.cand, a.fish, dil=a.dil,
                                     feather=a.feather, extras=extras, hole=bool(a.hole),
                                     keep_img=ki)
    else:
        prev = mask16.paste_back(Image.open(REF), a.cand, a.fish)
    p1 = os.path.join(OUTD, f'{a.fish}-{a.name}-paste.png')
    prev.save(p1)
    print('paste ->', p1)
    box = mask16.CROPS[a.fish]
    p2 = os.path.join(OUTD, f'{a.fish}-{a.name}-sbs.png')
    sbs16.sbs(p1, box, p2, zoom=a.zoom, la='样张同区', lb=f'候选 {a.name} 贴回')
    # 内孔放大单独自检（死蓝/连续性/边界）
    cx, cy, ea, eb = mask16.HOLE_ELL[a.fish]
    hb = (int(cx - ea - 30), int(cy - eb - 30), int(cx + ea + 30), int(cy + eb + 30))
    p3 = os.path.join(OUTD, f'{a.fish}-{a.name}-hole.png')
    sbs16.sbs(p1, hb, p3, zoom=3.0, la='样张内孔', lb='候选内孔')
