# -*- coding: utf-8 -*-
# 「海绵宝宝坐泳圈」单元素材输出（用户返工第一步交付）：
#   原位合成版(整画布) + 透明版(RGBA：圈体像素级 + ZZ + 角色，其余透明)
#   + 与样张同区 2.4x 并排自检。
# 用法: python3 mkunit17.py --reg output/g17-fish1-reg.png --out-dir delivery/batch/unit17
import argparse, os, sys
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask17, compose17, sbs16

T = os.path.dirname(os.path.abspath(__file__))
REF = mask17.REF
FISH = 'fish1'


def main(reg, outd):
    os.makedirs(outd, exist_ok=True)
    reg_im = Image.open(reg).convert('RGB')
    # 1) 原位整画布版
    reg_im.save(os.path.join(outd, 'unit-sb-ring-canvas.png'))
    # 2) 透明版：alpha = 圈几何(ring-a-cut α) ∪ 角色贴入α(羽化) ∪ ZZ ∪ 接触影
    ox, oy = mask17.cut_origins()[FISH]
    ca = mask17.cut_rgba(FISH).split()[3]
    alpha = Image.new('L', reg_im.size, 0)
    alpha.paste(ca, (ox, oy))
    # 只取圈管几何（旧鱼所占孔内区从 cutα 剔除，避免把水面带进素材）。
    tb = mask17.tube_band(FISH).filter(ImageFilter.MaxFilter(7))
    alpha = Image.composite(alpha, Image.new('L', alpha.size, 0), tb)
    layer, u, _ = compose17._unit_layer(FISH, compose17.POSE[FISH])
    ua = layer.split()[3]
    alpha = Image.composite(Image.new('L', alpha.size, 255), alpha,
                            ua.point(lambda v: 255 if v > 120 else 0))
    zz = mask17.zz_letters()
    alpha = Image.composite(Image.new('L', alpha.size, 255), alpha, zz)
    rgba = reg_im.convert('RGBA')
    rgba.putalpha(alpha)
    rgba.save(os.path.join(outd, 'unit-sb-ring-alpha.png'))
    # 3) 同区 2.4x 并排
    sbs16.sbs(reg, mask17.CROPS[FISH], os.path.join(outd, 'unit-sb-ring-sbs24.png'),
              zoom=2.4, la='样张同区', lb='G17 重做·海绵宝宝坐圈')
    print('->', outd)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reg', required=True)
    ap.add_argument('--out-dir', default='delivery/batch/unit17')
    a = ap.parse_args()
    main(a.reg, a.out_dir)
