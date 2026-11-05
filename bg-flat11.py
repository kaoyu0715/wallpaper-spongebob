# 十一代背景（issue #23 差距 G8）：官方图案背景，样式延续 bg-flat.py（颜色/布局全部不变），
# 唯一变化 = 花朵描边加粗（R*0.085 → R*0.12，alpha 235→250）向样张的粗笔触轮廓靠拢。
# 沙地与散点小圈仍由 compose-round11 统一绘制（沙岛形状在合成层改，见 G2）。
# 用法: python3 bg-flat11.py --out /path/bg.png [--w 1440 --h 3200]
import argparse, math, random
from PIL import Image, ImageDraw

import importlib.util, os
spec = importlib.util.spec_from_file_location(
    'bg_flat', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bg-flat.py'))
bf = importlib.util.module_from_spec(spec)
_argv = __import__('sys').argv
__import__('sys').argv = ['bg-flat.py', '--out', '/tmp/.bgflat-probe.png']
spec.loader.exec_module(bf)
__import__('sys').argv = _argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    W_, H_ = bf.W, bf.H
    random.seed(420715)
    img = Image.new('RGBA', (W_, H_), bf.GREEN + (255,))
    ov = Image.new('RGBA', (W_, H_), (0, 0, 0, 0))
    ImageDraw.Draw(ov)

    # 花朵：位置/颜色与十代一致，描边加粗（G8）
    for x, y, R, ci, a, lob in bf_flowers():
        bf.draw_flower(ov, x, y, R, bf.FLOWER_COLORS[ci], width=max(9, int(R * 0.12)),
                       alpha=250, phase=random.uniform(0, math.pi), lobes=lob)

    img = Image.alpha_composite(img, ov)
    img.convert('RGB').save(args.out)
    print('flat bg11 ->', args.out)


def bf_flowers():
    # (x, y, R, color_idx, alpha, lobes) 与 bg-flat.py 相同的位置表
    return [(700, 540, 150, 0, 235, 5), (1180, 720, 190, 2, 235, 6),
            (200, 690, 130, 1, 225, 6), (420, 1030, 95, 3, 235, 5),
            (95, 1520, 115, 0, 210, 6), (1360, 1480, 110, 4, 225, 5),
            (1240, 1950, 80, 0, 225, 5), (300, 1990, 75, 3, 220, 6),
            (270, 2820, 105, 5, 225, 6), (1150, 2900, 125, 4, 235, 5),
            (880, 3060, 60, 1, 200, 5)]


if __name__ == '__main__':
    main()
