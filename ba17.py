# -*- coding: utf-8 -*-
# 十七代整图 Before/After 对照（交付要求：基线=G16 v1 整图；同为 1440x3120）。
# 用法: python3 ba17.py --before <before.png> --after <after.png> --out <out.png> [--w 560]
import argparse
from PIL import Image, ImageDraw


def ba(before, after, out, w=560):
    a = Image.open(before).convert('RGB')
    b = Image.open(after).convert('RGB')
    h = int(a.size[1] * w / a.size[0])
    a = a.resize((w, h)); b = b.resize((w, h))
    pad, lab = 24, 56
    c = Image.new('RGB', (w * 2 + pad * 3, h + lab + pad * 2), (245, 245, 245))
    c.paste(a, (pad, pad + lab)); c.paste(b, (w + pad * 2, pad + lab))
    d = ImageDraw.Draw(c)
    d.text((pad, 12), 'Before · G16 v1（整区重绘路线，人物糊）', fill=(30, 30, 30))
    d.text((w + pad * 2, 12), 'After · G17 保圈换鱼身（单元替换）', fill=(30, 30, 30))
    c.save(out)
    print('->', out, c.size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True); ap.add_argument('--after', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--w', type=int, default=560)
    a = ap.parse_args()
    ba(a.before, a.after, a.out, a.w)
