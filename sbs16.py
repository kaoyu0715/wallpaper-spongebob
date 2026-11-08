# -*- coding: utf-8 -*-
# 十六代并排自检工具（issue #33）：样张同区 vs 候选图同区，2-3x 放大并排。
# 用法: python3 sbs16.py --cand <img> --box x0,y0,x1,y1 --zoom 2.5 --out out.png [--la 标签] [--lb 标签]
import argparse
from PIL import Image, ImageDraw

REF = 'review/reference/swim-ref.jpg'


def side(a, b, path, la, lb, zoom=2.5):
    w, h = a.size
    wa, ha = int(w * zoom), int(h * zoom)
    ca, cb = a.resize((wa, ha), Image.LANCZOS), b.resize((wa, ha), Image.LANCZOS)
    pad = 12
    canvas = Image.new('RGB', (wa * 2 + pad * 3, ha + 36), (255, 255, 255))
    canvas.paste(ca, (pad, 36))
    canvas.paste(cb, (wa + pad * 2, 36))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 10), la, fill=(0, 0, 0))
    d.text((wa + pad * 2, 10), lb, fill=(0, 0, 0))
    canvas.save(path)
    print(path, canvas.size)


def sbs(cand_path, box, out, zoom=2.5, la='样张', lb='候选'):
    x0, y0, x1, y1 = box
    ref = Image.open(REF).convert('RGB').crop(box)
    cand = Image.open(cand_path).convert('RGB').crop(box)
    side(ref, cand, out, la, lb, zoom)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=None, help='替代基准图（默认样张）')
    ap.add_argument('--cand', required=True)
    ap.add_argument('--box', required=True)
    ap.add_argument('--zoom', type=float, default=2.5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--la', default='样张')
    ap.add_argument('--lb', default='候选')
    a = ap.parse_args()
    x0, y0, x1, y1 = map(int, a.box.split(','))
    ref = Image.open(a.ref or REF).convert('RGB').crop((x0, y0, x1, y1))
    cand = Image.open(a.cand).convert('RGB').crop((x0, y0, x1, y1))
    side(ref, cand, a.out, a.la, a.lb, a.zoom)
