# 十三代复刻背景（issue #27）：官方图案背景，花朵/场泡位置与尺寸直接按样张 target-ref.jpg 网格实测复刻。
# 样式实现（粗描边花形、颜色、泡泡透明描边+左上弧高光）沿用 bg-flat11/bg-flat 与 compose-round11 口径。
# 沙岛/角色脚下元素在合成层（compose-round13.py）绘制。
# 用法: python3 bg-flat13.py --out /path/bg.png
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

# 样张实测（crop(256,356,1065,2093) 809x1737 → 1440x3200：x*1.78, y*1.842）
# (x, y, R, color_idx, alpha, lobes)
FLOWERS13 = [
    (721, 258, 160, 0, 208, 5),    # 顶部大橙
    (1344, 460, 165, 2, 208, 6),   # 右上大黄（半出右缘）
    (-14, 442, 135, 1, 208, 6),    # 左上紫（半出左缘）
    (214, 884, 105, 2, 208, 5),    # 左中黄
    (-14, 1326, 63, 0, 208, 5),    # 左缘橙小（半切）
    (1412, 1634, 95, 0, 208, 5),   # 右缘橙（半切，Hi泡泡下方）
    (196, 3021, 122, 5, 208, 6),   # 左下亮绿
    (1290, 3011, 90, 4, 208, 5),   # 右下粉
]

# 样张实测大圆环场泡 (x, y, r)
FIELD13 = [
    (27, 55, 52), (225, 193, 74), (730, 125, 38), (961, 212, 95), (1210, 368, 108),
    (819, 552, 66), (1246, 553, 40), (1104, 553, 32), (107, 1299, 60), (329, 1419, 68),
    (1094, 727, 62), (1157, 1290, 80), (1356, 1366, 70), (1305, 1109, 52),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    random.seed(130001)
    img = Image.new('RGBA', (bf.W, bf.H), bf.GREEN + (255,))
    ov = Image.new('RGBA', (bf.W, bf.H), (0, 0, 0, 0))
    for x, y, R, ci, a, lob in FLOWERS13:
        bf.draw_flower(ov, x, y, R, bf.FLOWER_COLORS[ci], width=max(10, int(R * 0.13)),
                       alpha=a, phase=random.uniform(0, math.pi), lobes=lob)
    # 样张档大圆环场泡（透明内芯 + 左上弧高光，独立 overlay alpha 混合）
    bub = Image.new('RGBA', (bf.W, bf.H), (0, 0, 0, 0))
    d = ImageDraw.Draw(bub)
    for x, y, r in FIELD13:
        w = max(5, r // 6)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 240, 245, 14),
                  outline=(250, 253, 250, 230), width=w)
        d.arc([x - r + r * 0.22, y - r + r * 0.22, x + r - r * 0.22, y + r - r * 0.22],
              start=135, end=225, fill=(255, 255, 255, 240), width=max(4, r // 8))
    img = Image.alpha_composite(img, ov)
    img = Image.alpha_composite(img, bub)
    img.convert('RGB').save(args.out)
    print('flat bg13 ->', args.out)


if __name__ == '__main__':
    main()
