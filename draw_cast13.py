# 十三代复刻配角落绘图组（issue #27）：背景鱼×3、彩蛋呆鱼、痞老板。
# 依据：样张中这些角色本身是纯几何扁平卡通（椭圆头、白圆眼、粗唇、衬衫块）；
# 底模无 LoRA 出图风格全部偏出官方 2D（g13raw fish/plankton 批审 25 张，无一官方风，见差距清单），
# 程序绘制=风格 100% 可控且零伪影，与背景线同理（“样式靠程序保证”）。
# 3x 超采样绘制 → LANCZOS 缩到目标尺寸，黑描边 3-4px。输出 output/units13d/*.png。
import os
import numpy as np
from PIL import Image, ImageDraw

S = 3  # 超采样


def _cv(x, y):
    return x * S, y * S


def _ell(d, cx, cy, rx, ry, fill=None, outline=None, w=4):
    cx, cy, rx, ry = cx * S, cy * S, rx * S, ry * S
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=outline, width=w * S)


def _poly(d, pts, fill=None, outline=None, w=4, closed=True):
    p = [_cv(x, y) for x, y in pts]
    if fill and not (outline and closed):
        d.polygon(p, fill=fill)
    if outline:
        d.line(p + ([p[0]] if closed and len(p) > 2 else []), fill=outline, width=w * S, joint='curve')
    if fill and outline and closed:
        d.polygon(p, fill=fill)
        d.line(p + [p[0]], fill=outline, width=w * S, joint='curve')


def _line(d, pts, fill, w=4):
    d.line([_cv(*p) for p in pts], fill=fill, width=w * S, joint='curve')


def _finish(img, tw, th):
    return img.resize((tw, th), Image.LANCZOS)


BLK = (25, 22, 20, 255)

# 样张取色
OLIVE = (170, 172, 84, 255)        # 左鱼头身
OLIVE_D = (150, 148, 62, 255)
FIN_PALE = (205, 208, 122, 255)    # 鱼鳍手臂
LIP_LAV = (168, 148, 198, 255)
SHIRT_LAV = (170, 168, 210, 255)
SHORTS_BLUE = (60, 122, 190, 255)
HAIR_PUR = (112, 84, 152, 255)
OLIVE2 = (164, 162, 92, 255)       # 右鱼
LIP_TAN = (224, 200, 142, 255)
TAN = (206, 154, 92, 255)          # 领带鱼头
LIP_TEAL = (118, 160, 170, 255)
SHIRT_BLUE = (176, 214, 234, 255)
TIE_TEAL = (60, 110, 140, 255)
GREY_BLUE = (158, 182, 196, 255)
FIN_BLU = (70, 96, 150, 255)
BLUE_EGG = (122, 152, 192, 255)    # 彩蛋呆鱼
EYE_YEL = (242, 216, 96, 255)
IRIS_RED = (216, 64, 48, 255)
MOUTH_DK = (142, 44, 44, 255)
PLK_GREEN = (134, 158, 128, 255)   # 痞老板
ANT_DK = (86, 92, 62, 255)


def fish_left(tw=560, th=1250):
    # 样张实测：长蛋形脸略倾、眼下厚薄唇低位、紫色小鬃、裸肚+蓝短裤、超长垂臂
    img = Image.new('RGBA', (tw * S, th * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 手臂（后层）：画面左长垂至膝 / 画面右屈向胸前
    _poly(d, [(158, 470), (108, 640), (96, 800), (138, 798), (152, 645), (196, 505)], FIN_PALE, BLK, 4)
    _ell(d, 116, 842, 30, 58, FIN_PALE, BLK, 4)
    _poly(d, [(382, 470), (432, 540), (428, 640), (386, 626), (372, 545)], FIN_PALE, BLK, 4)
    _ell(d, 408, 668, 30, 54, FIN_PALE, BLK, 4)
    # 腿（短）+ 鳍脚 + 左胫创可贴
    _line(d, [(252, 905), (244, 1015)], OLIVE, 17)
    _line(d, [(330, 905), (338, 1015)], OLIVE, 17)
    _ell(d, 218, 1042, 40, 24, FIN_PALE, BLK, 4)
    _ell(d, 362, 1042, 40, 24, FIN_PALE, BLK, 4)
    _poly(d, [(236, 940), (268, 936), (272, 962), (240, 966)], (238, 206, 120, 255), BLK, 2)
    _line(d, [(246, 951), (262, 949)], BLK, 2)
    # 短裤
    _poly(d, [(192, 772), (388, 772), (380, 905), (312, 905), (290, 852), (268, 905), (199, 905)],
          SHORTS_BLUE, BLK, 4)
    # 裸肚（大，衬衫下缘外露）+ 脐
    _ell(d, 290, 700, 105, 92, OLIVE, BLK, 4)
    _line(d, [(298, 692), (292, 712)], OLIVE_D, 4)
    # 衬衫（短，仅到胸下）
    _poly(d, [(172, 435), (235, 418), (288, 452), (341, 418), (404, 435), (412, 612), (164, 612)],
          SHIRT_LAV, BLK, 4)
    _line(d, [(288, 452), (288, 498)], BLK, 4)
    # 头（长蛋形）+ 紫色小鬃（左缘三圆鼓）
    for cy0, r0 in [(215, 34), (300, 36), (385, 32)]:
        _ell(d, 168, cy0, r0, r0, HAIR_PUR, BLK, 4)
    _ell(d, 292, 275, 122, 205, OLIVE, BLK, 4)
    # 眉
    _line(d, [(222, 158), (262, 146)], BLK, 6)
    _line(d, [(325, 146), (362, 160)], BLK, 6)
    # 眼（大，看向右）
    _ell(d, 252, 215, 40, 50, (255, 255, 255, 255), BLK, 4)
    _ell(d, 332, 212, 38, 48, (255, 255, 255, 255), BLK, 4)
    _ell(d, 266, 228, 14, 17, BLK)
    _ell(d, 344, 224, 13, 16, BLK)
    # 低位薄唇（宽、扁）+ 颊侧小褶
    _ell(d, 296, 366, 100, 27, LIP_LAV, BLK, 4)
    _line(d, [(200, 366), (392, 366)], BLK, 3)
    _line(d, [(196, 398), (214, 412)], OLIVE_D, 4)
    return _finish(img, tw, th)


def fish_right(tw=520, th=1100):
    img = Image.new('RGBA', (tw * S, th * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 手臂下垂
    _poly(d, [(130, 470), (85, 640), (92, 760), (138, 752), (140, 640), (175, 500)], FIN_PALE, BLK, 4)
    _poly(d, [(390, 470), (432, 640), (425, 760), (380, 752), (380, 640), (345, 500)], FIN_PALE, BLK, 4)
    # 下半（会被前排挡住，仍完整画：蓝短裤+腿+鳍脚）
    _poly(d, [(175, 800), (345, 800), (338, 920), (280, 920), (260, 872), (240, 920), (182, 920)],
          SHORTS_BLUE, BLK, 4)
    _line(d, [(225, 920), (218, 1030)], OLIVE, 16)
    _line(d, [(300, 920), (306, 1030)], OLIVE, 16)
    _ell(d, 200, 1052, 38, 24, FIN_PALE, BLK, 4)
    _ell(d, 322, 1052, 38, 24, FIN_PALE, BLK, 4)
    # polo 衫
    _poly(d, [(150, 430), (215, 410), (260, 448), (305, 410), (370, 430), (382, 812), (140, 812)],
          SHIRT_LAV, BLK, 4)
    _line(d, [(260, 448), (260, 505)], BLK, 4)
    # 头后发鬣（右后侧紫尖）
    _poly(d, [(392, 140), (428, 190), (396, 215), (438, 265), (398, 290), (430, 350), (388, 352)],
          HAIR_PUR, BLK, 4)
    # 头
    _ell(d, 258, 244, 140, 185, OLIVE2, BLK, 4)
    # 眼（看向左）
    _ell(d, 198, 196, 47, 47, (255, 255, 255, 255), BLK, 4)
    _ell(d, 312, 190, 44, 44, (255, 255, 255, 255), BLK, 4)
    _ell(d, 184, 205, 15, 17, BLK)
    _ell(d, 296, 198, 14, 16, BLK)
    # 宽笑唇（tan 色香蕉弧）
    _poly(d, [(150, 292), (230, 322), (330, 312), (378, 280), (382, 302), (330, 344), (225, 354), (152, 318)],
          LIP_TAN, BLK, 4)
    _line(d, [(158, 306), (378, 292)], BLK, 3)
    return _finish(img, tw, th)


def fish_tie(tw=430, th=770):
    img = Image.new('RGBA', (tw * S, th * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 手臂（灰蓝鳍）
    _poly(d, [(92, 400), (62, 500), (68, 585), (104, 575), (108, 500), (128, 420)], GREY_BLUE, BLK, 4)
    _poly(d, [(338, 400), (368, 500), (362, 585), (326, 575), (322, 500), (302, 420)], GREY_BLUE, BLK, 4)
    # 腿脚
    _line(d, [(185, 690), (178, 752)], GREY_BLUE, 16)
    _line(d, [(252, 690), (258, 752)], GREY_BLUE, 16)
    _ell(d, 158, 758, 36, 18, GREY_BLUE, BLK, 3)
    _ell(d, 280, 758, 36, 18, GREY_BLUE, BLK, 3)
    # 长裤 + 腰带
    _poly(d, [(128, 560), (312, 560), (306, 700), (240, 700), (222, 640), (204, 700), (134, 700)],
          GREY_BLUE, BLK, 4)
    _poly(d, [(126, 545), (314, 545), (314, 575), (126, 575)], (40, 44, 48, 255), BLK, 3)
    _poly(d, [(200, 548), (238, 548), (238, 572), (200, 572)], (232, 196, 64, 255), BLK, 3)
    # 衬衫
    _poly(d, [(118, 330), (180, 315), (220, 352), (258, 315), (322, 330), (318, 552), (122, 552)],
          SHIRT_BLUE, BLK, 4)
    # 领 + 领带
    _poly(d, [(180, 315), (220, 352), (196, 372), (170, 335)], (210, 232, 244, 255), BLK, 3)
    _poly(d, [(260, 315), (220, 352), (244, 372), (270, 335)], (210, 232, 244, 255), BLK, 3)
    _poly(d, [(206, 360), (234, 360), (240, 392), (228, 500), (212, 500), (200, 392)], TIE_TEAL, BLK, 3)
    # 头后发鬣
    _poly(d, [(318, 88), (346, 140), (318, 162), (352, 210), (318, 232), (344, 282), (310, 280)],
          FIN_BLU, BLK, 4)
    # 头
    _ell(d, 212, 178, 116, 152, TAN, BLK, 4)
    # 眼（看向左）
    _ell(d, 165, 148, 42, 42, (255, 255, 255, 255), BLK, 4)
    _ell(d, 262, 148, 40, 40, (255, 255, 255, 255), BLK, 4)
    _ell(d, 152, 158, 13, 15, BLK)
    _ell(d, 248, 156, 12, 14, BLK)
    # 青唇（薄一些）
    _ell(d, 212, 246, 88, 22, LIP_TEAL, BLK, 4)
    _line(d, [(124, 240), (300, 240)], BLK, 3)
    return _finish(img, tw, th)


def fish_egg(tw=200, th=200):
    # 样张：圆头小身呆鱼，两根弯曲深色触须后甩，举手鳍、小短腿、张嘴露上齿
    img = Image.new('RGBA', (tw * S, th * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 触须（先画，头后）
    _line(d, [(86, 36), (60, 22), (40, 40), (36, 66)], ANT_DK, 4)
    _line(d, [(104, 32), (92, 14), (72, 12), (58, 24)], ANT_DK, 4)
    # 身（小）+ 举起的鳍手 + 短腿
    _poly(d, [(76, 118), (50, 100), (44, 116), (66, 130)], BLUE_EGG, BLK, 3)
    _poly(d, [(124, 116), (150, 96), (158, 110), (134, 128)], BLUE_EGG, BLK, 3)
    _ell(d, 98, 138, 26, 30, BLUE_EGG, BLK, 4)
    _ell(d, 88, 168, 10, 8, BLUE_EGG, BLK, 3)
    _ell(d, 110, 168, 10, 8, BLUE_EGG, BLK, 3)
    # 圆头
    _ell(d, 98, 74, 38, 38, BLUE_EGG, BLK, 4)
    _ell(d, 98, 96, 22, 16, (168, 190, 216, 255))  # 浅下颌
    # 独眼
    _ell(d, 106, 62, 21, 21, EYE_YEL, BLK, 4)
    _ell(d, 108, 63, 10.5, 10.5, IRIS_RED)
    _ell(d, 109, 64, 5, 5, BLK)
    # 张嘴：白上齿排 + 深腔 + 下唇
    _poly(d, [(88, 90), (120, 90), (122, 116), (86, 116)], MOUTH_DK, BLK, 3)
    _poly(d, [(90, 91), (119, 91), (118, 99), (91, 99)], (245, 245, 240, 255), None)
    _line(d, [(99, 91), (99, 99)], BLK, 2)
    _line(d, [(109, 91), (109, 99)], BLK, 2)
    return _finish(img, tw, th)


def plankton(tw=170, th=230):
    img = Image.new('RGBA', (tw * S, th * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 触角（分节：沿杆画小鼓点）
    for p0 in ([62, 60], [108, 58]):
        base = p0
        tip = (28, 10) if p0[0] < 80 else (142, 8)
        mid = ((base[0] + tip[0]) / 2 + (-10 if base[0] < 80 else 10), (base[1] + tip[1]) / 2 - 8)
        _line(d, [base, mid, tip], ANT_DK, 4)
        for t in (0.35, 0.65):
            x = base[0] + (tip[0] - base[0]) * t; y = base[1] + (tip[1] - base[1]) * t
            _ell(d, x, y, 3.4, 3.4, ANT_DK)
    # 举起的手臂 V
    _line(d, [(40, 132), (10, 92)], PLK_GREEN, 7)
    _line(d, [(130, 132), (160, 92)], PLK_GREEN, 7)
    for hx, hy, sgn in [(10, 92, -1), (160, 92, 1)]:
        for a in (-0.5, 0.0, 0.5):
            _line(d, [(hx, hy), (hx + sgn * 10 * np.cos(a), hy - 12 - 8 * np.sin(a))], PLK_GREEN, 4)
    # 腿
    _line(d, [(70, 196), (64, 216)], PLK_GREEN, 7)
    _line(d, [(100, 196), (106, 216)], PLK_GREEN, 7)
    _ell(d, 58, 220, 14, 7, PLK_GREEN, BLK, 2)
    _ell(d, 112, 220, 14, 7, PLK_GREEN, BLK, 2)
    # 蛋形身体
    _ell(d, 85, 132, 46, 68, PLK_GREEN, BLK, 4)
    # 独眼
    _ell(d, 85, 100, 26, 26, (250, 250, 246, 255), BLK, 4)
    _ell(d, 86, 101, 14, 14, (240, 172, 64, 255))
    _ell(d, 87, 102, 10, 10, IRIS_RED)
    _ell(d, 87, 102, 4.5, 4.5, BLK)
    # 大笑嘴（白齿排+深腔）
    _ell(d, 85, 146, 24, 15, MOUTH_DK, BLK, 3)
    _poly(d, [(63, 138), (107, 138), (104, 147), (66, 147)], (245, 245, 240, 255), None)
    for xx in (74, 85, 96):
        _line(d, [(xx, 138), (xx, 146)], BLK, 2)
    return _finish(img, tw, th)


DRAW = {'fish_left': fish_left, 'fish_right': fish_right, 'fish_tie': fish_tie,
        'fish_egg': fish_egg, 'plankton': plankton}


def draw_unit(key):
    return DRAW[key]()


if __name__ == '__main__':
    out = 'output/units13d'
    os.makedirs(out, exist_ok=True)
    for k in DRAW:
        u = draw_unit(k)
        u.save(os.path.join(out, k + '.png'))
        print('drawn', k, u.size)
