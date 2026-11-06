# 十三代合成（issue #27：直接复刻样张 target-ref.jpg，密集叠压群像 + 寻宝图共同焦点）。
# 复刻模式：不走 SCENES11 ≤3 角色场景表，按样张实测坐标与显式 z 序（后→前）一次成图。
# 尺寸口径：贴入高按样张网格实测（crop→canvas x*1.78/y*1.842），C5 尺寸档不适用（样张角色簇
#   顶部在 50% 线以上，验收标准=样张本身）；硬口径延续：肢体完整、硬闸C（SB 腿身比）、
#   皮肤无红点（C4 派大星必检+成品级回填）、贴入比≤1.0x、描边干净（RMBG+tighten_alpha+回色）。
# 复用十一/十二代实现：派大星四件套、SB 调色+腿手术+硬闸C、珊迪红球清理、盔圈粉花、
#   drop_stray_specks、tighten_alpha、paste_figure、场泡 alpha 口径（bg-flat13 已绘）。
# 道具全部程序绘制（样张档）：摊开寻宝图（伪字形/虚线/X）、宝箱、小铲、Hi! 圆点泡泡、
#   彩蛋呆鱼泡泡。锚点表 ANCHORS 由样张逐区网格读数换算，迭代时按并排对照微调。
# 用法: python3 compose-round13.py --bg bg13.png --outdir output/round13 [--v v1]
import argparse, json, os, sys
import importlib.util
import numpy as np
from PIL import Image, ImageDraw, ImageChops

spec = importlib.util.spec_from_file_location(
    'compose_round12', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compose-round12.py'))
c12 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['compose-round12.py']
spec.loader.exec_module(c12)
sys.argv = _argv

c11 = c12.c11
m9 = c12.m9
m = c11.m
ps = c11.ps
import sb_palette  # c12 已把 sb_palette.sb_palette 替换为硬闸C注入版（_sb_palette12）
_sb_gate_palette = sb_palette.sb_palette

G13 = '/mnt/g/workspace/comfyui/output/g13cut'
G11 = '/mnt/g/workspace/comfyui/output/g11cut'
G8 = '/mnt/g/workspace/comfyui/output/g8cut'

W, H = 1440, 3200

# 单元表（种子选优后填写）
D13 = 'output/units13d'   # 程序绘制配角落绘图组（draw_cast13.py，底模无 LoRA 25 张风格全部偏出官方 2D）
UNITS = {
  'fish_left': os.environ.get('U_FISH_LEFT',  D13 + '/fish_left.png'),
  'fish_right': os.environ.get('U_FISH_RIGHT', D13 + '/fish_right.png'),
  'fish_tie': os.environ.get('U_FISH_TIE',    D13 + '/fish_tie.png'),
  'fish_egg': os.environ.get('U_FISH_EGG',    D13 + '/fish_egg.png'),
  'sandy': os.environ.get('U_SANDY', G13 + '/sandy-top13_00001.png'),
  'patrick': os.environ.get('U_PATRICK', G13 + '/patrick-laugh13b_00001.png'),
  'squidward': os.environ.get('U_SQUID', G8 + '/squidward-annoyed_00001.png'),
  'krabs': os.environ.get('U_KRABS', G13 + '/krabs-laugh13_00001.png'),
  'sb': os.environ.get('U_SB', G11 + '/sb-right12_00001.png'),
  'plankton': os.environ.get('U_PLANKTON', D13 + '/plankton.png'),
  'gary': os.environ.get('U_GARY', G8 + '/gary-front_00001.png'),
}

# z 序后→前：(key, 贴入高h, cx, top_y, tilt°)。top_y=gy-h；样张网格实测换算，迭代微调。
# v2（样张 z 序修正）：Krabs 躯体/爪在地图前；SB 在地图后但黄手在地图前（程序手部）；
# 小铲在 tie 鱼前（其头压杆）；Gary 宽度受控（样张眼柄短）。
STACK = [
  ('fish_left',   1050,  196, 1560, 0.0),
  ('fish_right',  1050, 1280, 1560, 0.0),
  ('sandy',       1184,  820,  682, 0.0),
  ('_helmet', None, None, None, None),
  ('patrick',     1380,  630, 1080, 10.0),
  ('squidward',   1240, 1030, 1216, 0.0),
  ('plankton',     200, 1060, 1690, 0.0),
  ('sb',          1006,  997, 1943, 0.0),      # SB 躯体在地图后（样张）
  ('_chest_back', None, None, None, None),
  ('_map',   None, None, None, None),          # 地图压 SB/Squidward 下半身
  ('_chest_front', None, None, None, None),    # 前框压地图下缘
  ('krabs',       1020,  400, 1700, 0.0),      # 样张：蟹老板躯体在地图前
  ('_hands', None, None, None, None),          # 黄手捏角（地图+SB 之上、tie 鱼之前）
  ('_shovel', None, None, None, None),
  ('fish_tie',     770, 1205, 2174, 0.0),
  ('_hi',    None, None, None, None),
  ('gary',         390,  690, 2700, 0.0, 440),
]

SAND = (216, 196, 140)
SAND_EDGE = (88, 104, 44)
WOOD = (166, 122, 66)
WOOD_DARK = (92, 62, 30)
WOOD_LIGHT = (206, 168, 112)
MAP_TAN = (211, 193, 149)
MAP_SHADE = (184, 168, 124)
MAP_LINE = (98, 76, 42)
WHITE = (250, 253, 250)
SB_YELLOW = (240, 224, 64)
EGG_RIB = (245, 210, 40)

SAND_ISLAND = dict(cx=726, cy=3010, rx=570, ry=250)

# 寻宝图（canvas 实测角点，顶边带撕口）
MAP_PTS = [(358, 2390), (430, 2340), (560, 2362), (700, 2330), (860, 2372), (1010, 2338),
           (1148, 2388), (1156, 2560), (1110, 2740), (1092, 2870), (860, 2905), (600, 2895),
           (400, 2870), (366, 2700), (344, 2540)]
CHEST_FRONT = dict(x0=180, x1=1188, y0=2745, y1=2990)  # 前排木板横带（下缘被沙岛/小蜗压住）


def clean_sandy_dome(fig):
    """g13 珊迪单元盔内残留 tan 底圆板：从盘顶泛洪去底（白圈/脸/耳停住），
    达成样张「仅细白圈、罩内全透明」口径。"""
    arr = np.asarray(fig).copy().astype(np.int32)
    h, w = arr.shape[:2]
    seed = None
    for y in range(int(h * 0.12), int(h * 0.42)):
        row = arr[y]
        for x in range(int(w * 0.3), int(w * 0.7)):
            r, g, b, a = row[x]
            if a > 200 and abs(r - 210) < 22 and abs(g - 174) < 24 and abs(b - 142) < 26:
                seed = (y, x); break
        if seed:
            break
    if seed is None:
        print('sandy_dome: 未见残留圆板')
        return fig
    from collections import deque
    target = arr[seed[0], seed[1], :3]
    seen = np.zeros((h, w), bool)
    q = deque([seed]); seen[seed] = True
    n = 0
    while q:
        y, x = q.popleft(); n += 1
        arr[y, x, 3] = 0
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                r, g, b, a = arr[ny, nx]
                if a > 30 and abs(int(r) - int(target[0])) < 30 and abs(int(g) - int(target[1])) < 32 and abs(int(b) - int(target[2])) < 34:
                    seen[ny, nx] = True
                    q.append((ny, nx))
    print('sandy_dome: 泛洪去底 %d px' % n)
    return Image.fromarray(arr.astype('uint8'), 'RGBA')


def _prep(key):
    path = UNITS[key]
    if not os.path.exists(path):
        raise SystemExit('缺少单元抠图: ' + path)
    fig = Image.open(path).convert('RGBA')
    if 'units13d' in path:
        return fig, path   # 程序绘制组已按目标尺寸绘制，无需清洗
    if key == 'patrick':
        fig = m.clean_patrick_blue(fig)
        fig = m.clean_patrick_blotches(fig)
        fig = ps.clean_patrick_skin(fig)
        fig = m.fix_patrick_outline(fig)
    if key == 'sb':
        fig = _sb_gate_palette(fig)   # 硬闸C + 调色
    if key == 'sandy':
        fig = c11.clean_sandy_ball(fig)
        fig = clean_sandy_dome(fig)
        fig = fig.transpose(Image.FLIP_LEFT_RIGHT)  # 样张：尾蓬在画面左（g13 单元在右，镜像）
    fig = m9.drop_stray_specks(fig)
    fig = m.tighten_alpha(fig, floor=115, erode=2)
    bb = fig.getbbox()
    if bb:
        fig = fig.crop(bb)
    return fig, path


def _paste(canvas, key, h, cx, top, tilt, report, pw=None):
    fig, path = _prep(key)
    if pw:
        r = h / fig.height
        if fig.width * r > pw:
            fig = fig.resize((max(1, int(pw / r)), fig.height))
    ratio = h / fig.height
    if ratio > 1.0:
        # 复刻模式：越 1.0x 即降贴入高到 1.0x（等比顶缘下移），报出偏差
        print('贴入比≤1.0x 收敛: %s %.3fx → %.3fx（贴入高 %d→%d）' % (key, ratio, 1.0, h, fig.height))
        h = fig.height
    gy = top + h
    report.append(dict(key=key, unit=os.path.basename(path), paste_h=h, trim_h=fig.height,
                       ratio=round(h / fig.height, 3), top=top, gy=gy))
    m.paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
    print('贴入', key, os.path.basename(path), 'h=%d trim=%d scale=%.2fx top=%d' % (h, fig.height, h / fig.height, top))
    return fig, cx, top, h, gy


def draw_sand_island(canvas):
    ov = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    s = SAND_ISLAND
    d.ellipse([s['cx'] - s['rx'], s['cy'] - s['ry'], s['cx'] + s['rx'], s['cy'] + s['ry']],
              fill=SAND + (255,), outline=SAND_EDGE + (255,), width=12)
    rng = np.random.default_rng(13)
    for _ in range(260):
        t = rng.uniform(0, 2 * np.pi); rr = rng.random() ** 0.5
        x = s['cx'] + s['rx'] * 0.94 * rr * np.cos(t)
        y = s['cy'] + s['ry'] * 0.94 * rr * np.sin(t)
        c = tuple(int(v) for v in rng.choice([(176, 150, 96), (196, 174, 118)]))
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=c + (200,))
    canvas.alpha_composite(ov)


def draw_chest(canvas, front=True, back=True):
    """样张档宝箱：盖内衬后缘 + 前排木板横带（伪木纹+竖分割）。back 在地图后、front 在地图前。"""
    d = ImageDraw.Draw(canvas)
    if back:
        # 后排：箱体内沿（左/右露出的木框与蓝衬）
        for x0, x1, y0, y1 in [(150, 350, 2400, 2780), (1040, 1240, 2360, 2760)]:
            d.rectangle([x0, y0, x1, y1], fill=WOOD + (255,), outline=WOOD_DARK + (255,), width=6)
            for i in range(3):
                yy = y0 + (i + 1) * (y1 - y0) // 4
                d.line([(x0, yy), (x1, yy)], fill=WOOD_DARK + (255,), width=3)
        d.rectangle([196, 2470, 300, 2580], fill=(90, 130, 190, 255), outline=WOOD_DARK + (255,), width=5)
        d.rectangle([1050, 2440, 1180, 2560], fill=(90, 130, 190, 255), outline=WOOD_DARK + (255,), width=5)
    if front:
        # 前排木板横带
        c = CHEST_FRONT
        d.rectangle([c['x0'], c['y0'], c['x1'], c['y1']], fill=(176, 134, 80) + (255,),
                    outline=WOOD_DARK + (255,), width=7)
        for i in (1, 2):
            yy = c['y0'] + i * (c['y1'] - c['y0']) // 3
            d.line([(c['x0'], yy), (c['x1'], yy)], fill=WOOD_DARK + (255,), width=5)
        for xx in (c['x0'] + 360, c['x0'] + 700):
            d.line([(xx, c['y0']), (xx, c['y0'] + (c['y1'] - c['y0']) // 3)], fill=WOOD_DARK + (255,), width=4)


def draw_map(canvas):
    """样张档寻宝图：撕边羊皮纸 + 折痕 + 虚线路线 + X + 伪字形（非文字）。"""
    ov = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.polygon(MAP_PTS, fill=MAP_TAN + (255,), outline=MAP_LINE + (255,))
    # 撕边小三角
    pts = MAP_PTS
    for i in range(0, len(pts), 2):
        x, y = pts[i]
        d.polygon([(x - 14, y - 8), (x + 16, y + 4), (x - 4, y + 18)], fill=MAP_TAN + (255,))
    # 折痕
    for x0, y0, x1, y1 in [(620, 2350, 640, 2890), (880, 2345, 900, 2900),
                           (355, 2560, 1150, 2600)]:
        d.line([(x0, y0), (x1, y1)], fill=MAP_SHADE + (230,), width=6)
    # 虚线路线（两条 + X + 岛形）
    def dashed(p0, p1, n=26):
        a0, a1 = np.array(p0, float), np.array(p1, float)
        for i in range(n):
            if i % 2:
                continue
            t0, t1 = i / n, (i + 1) / n
            d.line([tuple(a0 + (a1 - a0) * t0), tuple(a0 + (a1 - a0) * t1)], fill=MAP_LINE + (220,), width=5)
    # 样张样式的曲径：贝塞尔近似拆成小段
    import math as _m
    def bez(p0, c, p1, n=30):
        out = []
        for i in range(n + 1):
            t = i / n
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1]
            out.append((x, y))
        return out
    def dashed_path(p0, c, p1, n=40):
        p = bez(p0, c, p1, n)
        for i in range(len(p) - 1):
            if i % 2 == 0:
                d.line([p[i], p[i + 1]], fill=MAP_LINE + (230,), width=5)
    dashed_path((430, 2760), (620, 2500), (800, 2790))
    dashed_path((830, 2500), (1000, 2470), (1060, 2700))
    dashed((700, 2820), (760, 2700))
    # X 记号
    for (x, y) in [(1010, 2700), (560, 2800)]:
        d.line([(x - 14, y - 14), (x + 14, y + 14)], fill=MAP_LINE + (240,), width=6)
        d.line([(x - 14, y + 14), (x + 14, y - 14)], fill=MAP_LINE + (240,), width=6)
    # 小岛屿形勾线
    d.arc([470, 2430, 620, 2540], 200, 340, fill=MAP_LINE + (220,), width=5)
    d.arc([880, 2760, 1010, 2860], 180, 320, fill=MAP_LINE + (220,), width=5)
    # 伪字形（描边笔画簇，非真实文字；无文字口径）
    def glyph(cx, cy, s, seed):
        rng = np.random.default_rng(seed)
        d2 = ImageDraw.Draw(ov)
        for _ in range(int(3 + rng.random() * 3)):
            x0 = cx + rng.uniform(-s, s); y0 = cy + rng.uniform(-s * 0.5, s * 0.5)
            x1 = x0 + rng.uniform(-s * 0.6, s * 0.6); y1 = y0 + rng.uniform(-s * 0.8, s * 0.8)
            d2.line([(x0, y0), (x1, y1)], fill=MAP_LINE + (235,), width=6)
    for gx, gy, gs, sd in [(500, 2430, 22, 3), (610, 2480, 26, 7), (610, 2560, 20, 11),
                           (500, 2590, 20, 17), (760, 2400, 30, 23), (770, 2470, 18, 29)]:
        glyph(gx, gy, gs, sd)
    canvas.alpha_composite(ov)


def draw_hands13(canvas):
    """样张档：地图两角海绵宝宝黄手捏角（右角手带短前臂、左角四指握）。"""
    d = ImageDraw.Draw(canvas)
    # 右角手：手背在上、四指扣过纸面
    gx, gy = 1080, 2418
    d.line([(gx + 66, gy - 40), (gx + 8, gy - 6)], fill=SB_YELLOW + (255,), width=34)
    d.ellipse([gx - 40, gy - 34, gx + 36, gy + 16], fill=SB_YELLOW + (255,), outline=(20, 18, 16, 255), width=5)
    for i in range(4):
        d.ellipse([gx - 34 + i * 17, gy + 2, gx - 18 + i * 17, gy + 40], fill=SB_YELLOW + (255,),
                  outline=(20, 18, 16, 255), width=4)
    # 左角四指握
    hx, hy = 452, 2445
    for i in range(4):
        d.ellipse([hx - 28, hy + i * 16, hx + 26, hy + i * 16 + 22], fill=SB_YELLOW + (255,),
                  outline=(20, 18, 16, 255), width=4)
    d.ellipse([hx - 10, hy - 24, hx + 40, hy + 8], fill=SB_YELLOW + (255,), outline=(20, 18, 16, 255), width=4)


def draw_shovel13(canvas):
    """样张档小铲：白铲头朝上、黄杆红带，斜倚于 tie 鱼肩后。"""
    d = ImageDraw.Draw(canvas)
    sx, sy = 1348, 2300
    d.line([(sx, sy + 6), (sx - 34, sy + 350)], fill=(232, 212, 96, 255), width=15)
    d.line([(sx, sy + 6), (sx - 34, sy + 350)], fill=(20, 18, 16, 255), width=4)
    d.line([(sx - 24, sy + 268), (sx - 31, sy + 316)], fill=(214, 60, 48, 255), width=15)
    d.polygon([(sx, sy - 56), (sx + 32, sy - 4), (sx, sy + 18), (sx - 32, sy - 4)],
              fill=(250, 253, 250, 255), outline=(20, 18, 16, 255), width=0)
    d.polygon([(sx, sy - 56), (sx + 32, sy - 4), (sx, sy + 18), (sx - 32, sy - 4), (sx, sy - 56)],
              outline=(20, 18, 16, 255))


def draw_helmet_ring13(canvas, ring):
    """样张档盔圈：细白圈 + 右上大椭圆高光 + 左上 10 点位粉花（样张权威口径，硬闸B 复核）。"""
    cx, cy, r = ring[0]
    ov = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    w = max(9, int(r * 0.055))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(252, 253, 250, 242), width=w)
    # 1-2 点高光：沿圈缘细长贴片（样张位置，不压脸）
    hl = Image.new('RGBA', (int(r * 1.1), int(r * 0.6)), (0, 0, 0, 0))
    dh = ImageDraw.Draw(hl)
    dh.ellipse([r * 0.2, r * 0.16, r * 0.9, r * 0.44], fill=(252, 253, 250, 235))
    hl = hl.rotate(-38, expand=True, resample=Image.BICUBIC)
    ov.alpha_composite(hl, (int(cx + r * 0.60 - hl.width / 2), int(cy - r * 0.55 - hl.height / 2)))
    # 粉花：10 点位压圈线（样张：瓣圆鼓、深粉描边、橙心）
    fx, fy = cx - r * 0.88, cy - r * 0.62
    R = r * 0.42
    pet = (216, 128, 186, 255)
    edge = (158, 76, 122, 255)
    import math as _m
    for i in range(6):
        t = _m.pi / 6 + i * _m.pi / 3
        px, py = fx + R * 0.52 * _m.cos(t), fy + R * 0.52 * _m.sin(t)
        d.ellipse([px - R * 0.44, py - R * 0.44, px + R * 0.44, py + R * 0.44], fill=pet, outline=edge, width=max(4, int(R * 0.08)))
    d.ellipse([fx - R * 0.22, fy - R * 0.22, fx + R * 0.22, fy + R * 0.22], fill=(240, 152, 64, 255), outline=edge, width=4)
    canvas.alpha_composite(ov)


def draw_hi_bubble(canvas):
    """样张档：奶油云形对话框（mask 并集干净轮廓）+ 浅橙圆点 + 左下尖尾 + 蓝色 Hi!。"""
    cx, cy = 1250, 1600
    msk = Image.new('L', canvas.size, 0)
    dm = ImageDraw.Draw(msk)
    lobes = [(0, -6, 88), (-82, -26, 56), (76, -34, 54), (-34, -74, 50), (46, -72, 48),
             (-62, 36, 46), (58, 38, 46)]
    for dx, dy, r in lobes:
        dm.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r], fill=255)
    dm.polygon([(cx - 120, cy + 40), (cx - 198, cy + 104), (cx - 74, cy + 76)], fill=255)
    edge = msk.point(lambda v: 255 if v > 128 else 0)
    inner = edge.filter(__import__('PIL.ImageFilter', fromlist=['x']).MaxFilter(1))
    import PIL.ImageFilter as F
    inner = edge.filter(F.MinFilter(13))
    outline = Image.eval(ImageChops.subtract(edge, inner), lambda v: 255 if v > 40 else 0)
    fillc = Image.new('RGBA', canvas.size, (250, 243, 216, 255))
    fillc.putalpha(ImageChops.multiply(msk, Image.new('L', canvas.size, 255)))
    canvas.alpha_composite(fillc)
    oc = Image.new('RGBA', canvas.size, (58, 48, 38, 255))
    oc.putalpha(outline)
    canvas.alpha_composite(oc)
    # 圆点
    d = ImageDraw.Draw(canvas)
    rng = np.random.default_rng(9)
    arr = np.asarray(msk)
    for _ in range(60):
        dx, dy = rng.uniform(-120, 120), rng.uniform(-95, 60)
        x, y = int(cx + dx), int(cy + dy)
        if 0 <= y < H and 0 <= x < W and arr[y, x] > 200:
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(244, 190, 150, 255))
    # Hi!（描线笔画，无 TTF 环境下程序绘制；样张蓝字深描）
    def stroke(pts, w=16):
        d.line(pts, fill=(20, 18, 16, 255), width=w + 8, joint='curve')
        d.line(pts, fill=(120, 178, 240, 255), width=w, joint='curve')
    bx, by = cx - 58, cy - 44
    stroke([(bx, by), (bx, by + 88)]); stroke([(bx + 44, by), (bx + 44, by + 88)])
    stroke([(bx + 2, by + 44), (bx + 42, by + 44)], 14)
    stroke([(bx + 76, by + 26), (bx + 76, by + 88)]); d.ellipse([bx + 66, by, bx + 86, by + 20], fill=(120, 178, 240, 255), outline=(20, 18, 16, 255), width=5)
    stroke([(bx + 108, by - 2), (bx + 104, by + 62)], 14)
    d.ellipse([bx + 98, by + 72, bx + 112, by + 86], fill=(120, 178, 240, 255), outline=(20, 18, 16, 255), width=5)


def draw_egg_bubble(canvas, fig):
    """样张档彩蛋：左侧白色圆环泡泡内的呆鱼（图钉尾在右下）。"""
    cx, cy, r = 294, 1529, 152
    ov = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(225, 240, 245, 20),
              outline=(250, 253, 250, 235), width=6)
    d.arc([cx - r + r * 0.26, cy - r + r * 0.26, cx + r - r * 0.26, cy + r - r * 0.26],
          start=150, end=205, fill=(255, 255, 255, 235), width=5)
    canvas.alpha_composite(ov)
    # 呆鱼贴入（泡泡内，中心对齐）
    h = int(r * 1.28)
    ratio = min(1.0, h / fig.height)
    h = int(fig.height * ratio)
    f = fig.resize((max(1, int(fig.width * ratio)), h), Image.LANCZOS)
    canvas.alpha_composite(f, (int(cx - f.width / 2), int(cy - f.height / 2)))
    # 泡泡外圈再压一层，保证鱼在泡泡“里”
    ov2 = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    d2 = ImageDraw.Draw(ov2)
    d2.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(250, 253, 250, 235), width=5)
    canvas.alpha_composite(ov2)
    return dict(key='fish_egg', ratio=round(ratio, 3), paste_h=h, trim_h=fig.height)


def compose_replica(bg_path, out_path, audit_dir):
    canvas = Image.open(bg_path).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    report = []
    helmet_ring = [None]
    draw_sand_island(canvas)

    for row in STACK:
        key, h, cx, top, tilt = row[:5]
        pw = row[5] if len(row) > 5 else None
        if key == '_helmet':
            draw_helmet_ring13(canvas, helmet_ring)
            continue
        if key == '_hands':
            draw_hands13(canvas)
            continue
        if key == '_shovel':
            draw_shovel13(canvas)
            continue
        if key == '_chest_back':
            draw_chest(canvas, front=False, back=True)
            continue
        if key == '_chest_front':
            draw_chest(canvas, front=True, back=False)
            continue
        if key == '_map':
            draw_map(canvas)
            continue
        if key == '_hi':
            draw_hi_bubble(canvas)
            continue
        if key.startswith('_'):
            continue
        if key == '_chest_back':
            draw_chest(canvas, front=False, back=True)
            continue
        if key == '_chest_front':
            draw_chest(canvas, front=True, back=False)
            continue
        if key == '_map':
            draw_map(canvas)
            continue
        if key == '_grips':
            draw_grips(canvas)
            continue
        if key == '_hi':
            draw_hi_bubble(canvas)
            continue
        fig, cx_, top_, h_, gy = _paste(canvas, key, h, cx, top, tilt, report, pw=pw)
        if key == 'sandy':
            # 样张盔圈：细白圈+右上大高光+左上粉花。单元泛洪后圈线残缺 → 程序重画（硬闸B 口径）。
            # 单元几何：trim bbox x32..978（镜像后），盔圈顶点=trim 顶缘，r=258 单元像素。
            sc = h_ / 1189.0
            ring_cx = cx - 0.007 * (946 * sc)
            ring_cy = top + 258 * sc
            helmet_ring[0] = (ring_cx, ring_cy, 258 * sc)
            os.makedirs(audit_dir, exist_ok=True)
            canvas.crop((max(0, int(cx - h_ * 0.8)), max(0, top - 40),
                         min(W, int(cx + h_ * 0.8)), int(top + h_ * 0.55))).save(
                os.path.join(audit_dir, 'replica-sandy-helmet.png'))
        if key == 'patrick':
            px0, px1 = max(0, int(cx - h_ * 0.45)), min(W, int(cx + h_ * 0.45))
            py0, py1 = max(0, int(gy - h_)), int(gy)
            crop_box = (px0, py0, px1, py1)
            mc = np.asarray(fig)[..., 3] > 10
            mimg = Image.fromarray((mc * 255).astype('uint8'), 'L')
            if tilt:
                mimg = mimg.rotate(tilt, expand=True, resample=Image.BILINEAR)
            r_ = h_ / mimg.height
            fw_, fh_ = max(1, int(mimg.width * r_)), h_
            mimg = mimg.resize((fw_, fh_), Image.NEAREST)
            fullmask = Image.new('L', canvas.size, 0)
            fullmask.paste(mimg, (int(cx - fw_ / 2), int(gy - fh_)))
            pm = np.asarray(fullmask.crop(crop_box)) > 100
            pm = c11._dilate_np(pm, 2)
            n = c11.patrick_postclean(canvas, crop_box, h_ / fig.height, pmask=pm)
            canvas.crop(crop_box).save(os.path.join(audit_dir, 'replica-patrick-skin.png'))
            print('皮肤必检 → 残留红点 %d 个' % n)
            report[-1]['skin_specks'] = n

    # 彩蛋泡泡呆鱼（背景区独立元素，最后画不影响角色）
    egg_path = UNITS['fish_egg']
    if os.path.exists(egg_path):
        ef = Image.open(egg_path).convert('RGBA')
        ef = m9.drop_stray_specks(ef)
        ef = m.tighten_alpha(ef, floor=115, erode=2)
        bb = ef.getbbox()
        if bb:
            ef = ef.crop(bb)
        report.append(draw_egg_bubble(canvas, ef))

    # 前景小蜗之上不再叠物。
    canvas.convert('RGB').save(out_path)
    print('复刻合成完成 →', out_path)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', default='output/bg13.png')
    ap.add_argument('--outdir', default='output/round13')
    ap.add_argument('--v', default='v1')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    audit = os.path.join(args.outdir, 'audit13')
    rep = compose_replica(args.bg, os.path.join(args.outdir, 'r13-replica-%s.png' % args.v), audit)
    rp = os.path.join(args.outdir, 'paste-report-generation-13-%s.json' % args.v)
    json.dump(rep, open(rp, 'w'), ensure_ascii=False, indent=1)
    print('paste report →', rp)
