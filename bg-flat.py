# 路线A背景：官方图案背景——纯色(样张同款深绿) + 花朵轮廓图案 + 底部沙地，程序绘制，非 Krea2。
# 颜色采样自 review/reference/target-ref.jpg。
# 用法: python3 bg-flat.py --out /path/g4-bg-flat.png [--w 1440 --h 3200]
import argparse, math, random
from PIL import Image, ImageDraw, ImageFilter, ImageChops

W, H = 1440, 3200
GREEN = (64, 99, 43)          # 样张底色
SAND = (216, 196, 140)        # 沙地
SAND_EDGE = (88, 104, 44)     # 沙地描边
FLOWER_COLORS = [
    (205, 130, 55),   # 橙
    (150, 95, 165),   # 紫
    (208, 200, 45),   # 黄
    (215, 225, 115),  # 淡黄绿
    (225, 140, 170),  # 粉
    (120, 160, 70),   # 亮绿
]


def draw_flower(img, cx, cy, R, color, width, alpha=235, phase=0.0, lobes=5):
    # 经典花形云：N 个圆瓣围绕中心的并集轮廓（花瓣圆鼓、瓣间有收腰）
    Rp = R * 0.38          # 花瓣半径：尖端在 ~R
    D = R * 0.62           # 花瓣中心距：瓣间收腰在 ~0.6R，圆鼓花瓣
    m = Image.new('L', (int(2 * R) + 20, int(2 * R) + 20), 0)
    md = ImageDraw.Draw(m)
    ox = oy = int(R) + 10
    for i in range(lobes):
        t = 2 * math.pi * i / lobes + phase
        px, py = ox + D * math.cos(t), oy + D * math.sin(t)
        md.ellipse([px - Rp, py - Rp, px + Rp, py + Rp], fill=255)
    md.ellipse([ox - D * 0.5, oy - D * 0.5, ox + D * 0.5, oy + D * 0.5], fill=255)
    sm = m.filter(ImageFilter.GaussianBlur(width * 0.9))
    er = sm.point(lambda v: 255 if v > 225 else 0)   # 高斯腐蚀，凹角不产生伪影
    outline = Image.eval(ImageChops.subtract(m, er), lambda v: 255 if v > 60 else 0)
    tint = Image.new('RGBA', m.size, color + (alpha,))
    tint.putalpha(ImageChops.multiply(outline, Image.new('L', m.size, alpha)))
    img.paste(tint, (int(cx - R) - 10, int(cy - R) - 10), tint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--w', type=int, default=W)
    ap.add_argument('--h', type=int, default=H)
    args = ap.parse_args()
    W_, H_ = args.w, args.h
    random.seed(420715)
    img = Image.new('RGBA', (W_, H_), GREEN + (255,))
    ov = Image.new('RGBA', (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    # 花朵轮廓图案：沿边缘与上半部分布（对照样张位置），中部角色区留空
    flowers = [
        # (x, y, R, color_idx, alpha, lobes)   坐标为 1440x3200 画布参考
        (700, 540, 150, 0, 235, 5),    # 顶部橙（样张顶部同款位置）
        (1180, 720, 190, 2, 235, 6),   # 右上大黄
        (200, 690, 130, 1, 225, 6),    # 左上紫
        (420, 1030, 95, 3, 235, 5),    # 左中淡黄
        (95, 1520, 115, 0, 210, 6),    # 左缘橙（半切）
        (1360, 1480, 110, 4, 225, 5),  # 右缘粉（半切）
        (1240, 1950, 80, 0, 225, 5),   # 右中橙小
        (300, 1990, 75, 3, 220, 6),    # 左中黄绿小
        (270, 2820, 105, 5, 225, 6),   # 左下亮绿
        (1150, 2900, 125, 4, 235, 5),  # 右下粉
        (880, 3060, 60, 1, 200, 5),    # 底部紫小
    ]
    for x, y, R, ci, a, lob in flowers:
        draw_flower(ov, x, y, R, FLOWER_COLORS[ci], width=max(6, int(R * 0.085)), alpha=a,
                    phase=random.uniform(0, math.pi), lobes=lob)

    # 少量白色小气泡圈（样张散布的小圈，装饰用，远离中央叙事泡泡带）
    for x, y, r in [(300, 400, 22), (980, 330, 16), (1120, 1180, 26), (350, 1350, 18),
                    (160, 2350, 20), (1290, 2400, 24), (640, 250, 14)]:
        d.ellipse([x - r, y - r, x + r, y + r], outline=(240, 248, 240, 170), width=4)
        d.arc([x - r + 4, y - r + 4, x + r - 4, y + r - 4], start=150, end=230,
              fill=(245, 250, 245, 200), width=4)

    # 底部沙地（样张同款：一大片浅沙 + 深色描边，向下出血）
    cx, cy, rx, ry = W_ * 0.5, 3200, 1180, 640
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=SAND + (255,),
              outline=SAND_EDGE + (255,), width=12)

    img = Image.alpha_composite(img, ov)
    img.convert('RGB').save(args.out)
    print('flat bg ->', args.out)


if __name__ == '__main__':
    main()
