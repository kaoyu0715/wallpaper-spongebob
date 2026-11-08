# -*- coding: utf-8 -*-
# 十五代 B 路线掩膜定义（issue #31）：样张三鱼区（含随身道具），排除泳圈管与水面。
# 掩膜像素=255 的区域允许重绘；生图=样张底，逐角色分区 inpaint。
from PIL import Image, ImageDraw, ImageFilter

REF_SIZE = (1080, 1920)

# —— 手工实测多边形（swim-ref.jpg 像素坐标）——
# 黄鱼+墨镜+翘腿(蓝脚)+搭管手臂+尾鳍。ZZ 状态符号保留(允许)。
FISH1 = [
    (225, 872), (330, 878), (372, 940), (398, 1005),          # 头顶/墨镜/右侧
    (462, 1000), (478, 1090), (455, 1125),                     # 尾鳍
    (478, 1160), (472, 1228), (392, 1238), (318, 1232),        # 翘腿/下身
    (238, 1196), (152, 1128), (128, 1076), (168, 1050),        # 搭管手臂
    (202, 1052), (196, 985), (180, 922),
]
# 绿鱼+橙汁杯(含吸管/橙片)+前伸双腿+搭右管手臂。
FISH2 = [
    (700, 646), (790, 660), (845, 730), (862, 825),            # 头/右身
    (886, 862), (872, 912), (812, 898),                        # 搭右管手臂
    (788, 942), (700, 968), (612, 952),                        # 前伸双腿/鞋
    (556, 936), (540, 880), (566, 862),                        # 左脚
    (582, 838), (560, 822), (568, 748), (600, 700),            # 橙汁杯/吸管
    (648, 700), (664, 712), (672, 676),
]
# 白袍鱼+头箍+粉花环+椰子(吸管)+交叠双腿黑鞋+搭管手臂。
FISH3 = [
    (700, 1068), (782, 1078), (830, 1132), (826, 1210),        # 头/头箍/右肩
    (862, 1282), (838, 1348), (812, 1408), (748, 1412),        # 搭管手臂
    (700, 1392), (688, 1442), (612, 1452), (538, 1442),        # 交叠腿/黑鞋
    (448, 1402), (442, 1338), (482, 1322),                     # 左脚
    (468, 1258), (498, 1196), (556, 1186),                     # 椰子/吸管
    (618, 1196), (648, 1142), (662, 1096),
]
FISHES = {'fish1': FISH1, 'fish2': FISH2, 'fish3': FISH3}


def _expand(poly, margin):
    """多边形沿质心外扩 margin px（给新角色留出越出旧轮廓的余量，如脚踩在圈管上）。"""
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    out = []
    for x, y in poly:
        dx, dy = x - cx, y - cy
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        out.append((x + margin * dx / n, y + margin * dy / n))
    return out


def fish_mask(which, size=REF_SIZE, blur=6, margin=26):
    """多边形掩膜（float 0..1 灰度 L 图），质心外扩 margin px 后边缘羽化 blur px。"""
    big = 2
    im = Image.new('L', (size[0] * big, size[1] * big), 0)
    d = ImageDraw.Draw(im)
    d.polygon([(x * big, y * big) for x, y in _expand(FISHES[which], margin)], fill=255)
    im = im.resize(size, Image.LANCZOS)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    return im


def overlay(which, base, alpha=70):
    im = base.convert('RGB').copy()
    m = fish_mask(which)
    red = Image.new('RGB', im.size, (255, 0, 0))
    return Image.composite(red, im, m.point(lambda v: int(v * alpha / 255) if v else 0))


if __name__ == '__main__':
    import os
    T = os.path.dirname(os.path.abspath(__file__))
    REF = Image.open(os.path.join(T, 'review/reference/swim-ref.jpg'))
    for w in FISHES:
        overlay(w, REF).save(os.path.join(T, f'output/maskv2-{w}.png'))
        m = fish_mask(w)
        m.save(os.path.join(T, f'output/maskv2-{w}-m.png'))
        print(w, 'ok')
