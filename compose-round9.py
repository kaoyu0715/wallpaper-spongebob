# 九代合成：官方图案背景 + 场景表批量合成（配方见 delivery/recipe.md §4.2 九代新尺寸档）。
# 相对八代的两点变化（issue #19）：
#   1) 人物整体缩小：全部角色在八代官方比例基础上统一下调一档（约 x0.76），
#      各角色间维持官方相对比例，上方与四周留出更多官方图案背景空间；
#   2) 两条硬闸进代码强制：
#      硬闸A 蟹老板贴入比强制 ≤0.85x（超闸直接终止合成，不允许"差一点"）；
#      硬闸B 珊迪头盔完整性必检：每张含珊迪的合成自动导出头盔区 100% 裁切
#            到 output/audit9/<scene>-<unit>-helmet.png，逐张人工复核盔圈闭合。
# 尺寸自检：所有贴入高必须落在 §4.2 新档区间内，角色最高点 ≥ 画面 50% 高度线以下，
# 逐项打印贴入比报告并写 output/round9/paste-report.json 供审计核对。
# 用法: python3 compose-round9.py --bg <bg.png> --scene <id>|all --outdir <目录>
import argparse, json, os, sys
import importlib.util
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location(
    'compose_round8', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compose-round8.py'))
m = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['compose-round8.py']
spec.loader.exec_module(m)
sys.argv = _argv

W, H = m.W, m.H
CUT9 = '/mnt/g/workspace/comfyui/output/g9cut'

# 新尺寸档（九代，§4.2）：角色全高 px 允许区间；贴入高必须落在区间内
TIER = {
  'sb': (600, 630),      # 海绵宝宝（基准 1.00）
  'sandy': (750, 790),   # 珊迪（含头盔，≈1.25）
  'patrick': (730, 765), # 派大星（≈1.22）
  'krabs': (770, 790),   # 蟹老板（≈1.29；另受硬闸A 0.85x 上限约束）
  'squidward': (845, 885),  # 章鱼哥（≈1.41）
  'gary': (230, 255),    # 小蜗（前景地面）
}
KRABS_MAX_RATIO = 0.85   # 硬闸A
TOP_FLOOR = 0.50 * H     # 角色最高点不得低于此线（含头盔/眼柄）；新档下留白 ≥52% 高度

FAMILY = {'sb_r': 'sb', 'pat_l': 'patrick', 'patrick_r': 'patrick',
          'sq_front': 'squidward', 'sq_annoyed': 'squidward',
          'sandy_l': 'sandy', 'sandy_c': 'sandy', 'krabs': 'krabs', 'gary': 'gary'}
PATRICK_KEYS = ('pat_l', 'patrick_r')
CUT8 = m.CUT8
CUT7 = m.CUT7

def tier_of(key):
    return FAMILY[key]

# 单元表：全部显式固定路径——已验收单元复用（七/八代胜出种子），
# sandy_c 用九代过闸新种子 20261902（盔圈闭合，满足硬闸B 的单元级必检）
_UNITS9 = {
  'sb_r':       os.path.join(CUT7, 'spongebob-c1_00001.png'),
  'pat_l':      os.path.join(CUT7, 'patrick-g5.png'),
  'sq_front':   os.path.join(CUT7, 'squidward-g5.png'),
  'sandy_l':    os.path.join(CUT8, 'sandy-left_00001.png'),
  'sandy_c':    os.path.join(CUT9, 'sandy-cheer_00001.png'),
  'patrick_r':  os.path.join(CUT8, 'patrick-right_00001.png'),
  'krabs':      os.path.join(CUT8, 'krabs-open_00001.png'),
  'gary':       os.path.join(CUT8, 'gary-front_00001.png'),
  'sq_annoyed': os.path.join(CUT8, 'squidward-annoyed_00001.png'),
}

def unit9(key):
    p = _UNITS9[key]
    if not os.path.exists(p):
        raise SystemExit('缺少单元抠图: ' + p)
    return p


def drop_stray_specks(fig, min_area=900, verbose=True):
    """C4 防漂浮碎线/怪点：alpha 连通域只保留主域，孤立小碎块（游离细弧线、
    飘浮色块，如九代珊迪欢呼单元手套上方的细弧残线）直接透明化。"""
    a = np.asarray(fig.getchannel('A'))
    vis = a > 10
    lab = -np.ones(vis.shape, np.int32)
    areas = []
    n = 0
    H_, W_ = vis.shape
    ys, xs = np.where(vis)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0] >= 0:
            continue
        q = deque([(y0, x0)])
        lab[y0, x0] = n
        cnt = 0
        while q:
            cy, cx = q.popleft()
            cnt += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H_ and 0 <= nx < W_ and vis[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = n
                    q.append((ny, nx))
        areas.append(cnt)
        n += 1
    if n <= 1:
        return fig
    main = max(areas)
    keep = np.zeros(n, bool)
    for i, ar in enumerate(areas):
        if ar == main or (ar >= min_area and ar >= 0.05 * main):
            keep[i] = True
    drop = vis & ~keep[lab]
    if drop.sum() == 0:
        return fig
    out = np.asarray(fig).copy()
    out[..., 3] = np.where(drop, 0, out[..., 3])
    if verbose:
        print('specks: 清除孤立碎块 %d px（连通域 %d 个，主域 %d px）' % (int(drop.sum()), n, main))
    return Image.fromarray(out, 'RGBA')

# 场景表（九代）：chars 按 z 序 后→前：(unit, 贴入高, cx, ground_y, 倾角°)
# 高度=新档；站位在八代基础上向簇心收拢 ~15%，泡泡带随手部/头顶位置下调重排。
SCENES = {
  'b01-sb-pat-bubbles': dict(
    title='追泡泡二人组',
    chars=[('sb_r', 615, 460, 2875, -3.0), ('pat_l', 745, 1000, 2830, 3.0)],
    band=[(620, 2380, 18), (720, 2120, 34), (840, 1860, 60), (940, 1620, 32), (1000, 1470, 18)]),
  'b02-sb-sq-pat-bubbles': dict(
    title='三人追泡泡',
    chars=[('sq_front', 865, 715, 2530, 1.5), ('sb_r', 615, 470, 2865, -5.0),
           ('pat_l', 745, 990, 2825, 5.0)],
    band=[(600, 2330, 20), (700, 2060, 40), (840, 1790, 70), (960, 1530, 38), (1020, 1380, 20)]),
  'b03-sandy-sb-hello': dict(
    title='珊迪与海绵宝宝打招呼',
    chars=[('sb_r', 610, 440, 2870, -3.0), ('sandy_l', 765, 1020, 2795, 2.0)],
    band=[(640, 2420, 14), (740, 2280, 20), (850, 2130, 12)]),
  'b04-krabs-sb-laugh': dict(
    title='蟹老板与海绵宝宝',
    chars=[('krabs', 780, 1005, 2755, 0.0), ('sb_r', 610, 440, 2875, -3.0)],
    band=[(640, 2400, 16), (760, 2240, 24), (870, 2060, 14)]),
  'b05-gary-sb-pet': dict(
    title='小蜗和海绵宝宝',
    chars=[('sb_r', 615, 505, 2845, -3.0), ('gary', 245, 1015, 2955, 0.0)],
    band=[(700, 2330, 14), (830, 2170, 20)]),
  'b06-sq-sb-bubbles': dict(
    title='章鱼哥的嫌弃',
    chars=[('sq_front', 865, 815, 2600, 0.0), ('sb_r', 610, 410, 2875, -3.0)],
    band=[(570, 2340, 18), (660, 2090, 34), (770, 1840, 52), (818, 1680, 26)]),
  'b07-sandy-sb-pat-cheer': dict(
    title='珊迪欢呼三人组',
    chars=[('sandy_c', 760, 720, 2545, 0.0), ('sb_r', 615, 430, 2865, -4.0),
           ('pat_l', 745, 1005, 2825, 4.0)],
    band=[(590, 2320, 20), (690, 2070, 38), (830, 1810, 66), (950, 1560, 36), (1010, 1410, 18)]),
  'b08-krabs-sb-pat': dict(
    title='蟹老板坐镇',
    chars=[('krabs', 780, 730, 2555, 0.0), ('sb_r', 610, 440, 2865, -4.0),
           ('pat_l', 745, 1000, 2825, 4.0)],
    band=[(590, 2310, 20), (690, 2050, 40), (830, 1790, 68), (950, 1540, 36), (1010, 1390, 18)]),
  'b09-gary-sb-pat': dict(
    title='小蜗领队',
    chars=[('sb_r', 615, 480, 2815, -3.0), ('pat_l', 740, 1020, 2785, 4.0),
           ('gary', 245, 745, 2955, 0.0)],
    band=[(610, 2250, 18), (700, 2020, 30), (840, 1790, 48)]),
  'b10-sandy-sq': dict(
    title='热情对嫌弃',
    chars=[('sq_annoyed', 855, 480, 2610, 0.0), ('sandy_l', 765, 1030, 2795, 2.0)],
    band=[(700, 2380, 14), (820, 2220, 22), (940, 2050, 12)]),
  'b11-pat-krabs': dict(
    title='派大星和蟹老板',
    chars=[('krabs', 780, 1005, 2760, 0.0), ('patrick_r', 740, 450, 2865, -3.0)],
    band=[(620, 2390, 16), (740, 2220, 26), (860, 2020, 14)]),
  'b12-sandy-sb-gary': dict(
    title='珊迪海绵宝宝和小蜗',
    chars=[('sandy_l', 765, 1040, 2790, 2.0), ('sb_r', 610, 460, 2850, -3.0),
           ('gary', 240, 735, 2955, 0.0)],
    band=[(630, 2360, 16), (760, 2200, 24)]),
}


def compose_scene(scene_id, bg_path, out_path, audit_dir):
    sc = SCENES[scene_id]
    canvas = Image.open(bg_path).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    sd = ImageDraw.Draw(canvas)
    scx, scy, srx, sry = W * 0.5, 3200, 1180, 690
    sd.ellipse([scx - srx, scy - sry, scx + srx, scy + sry], fill=(216, 196, 140, 255),
               outline=(88, 104, 44, 255), width=12)
    draw_bubbles = m.draw_bubbles
    draw_bubbles(canvas, m.FIELD_BUBBLES)
    report = []
    for key, h, cx, gy, tilt in sc['chars']:
        fam = tier_of(key)
        lo, hi = TIER[fam]
        if not (lo <= h <= hi):
            raise SystemExit('硬闸 尺寸档: %s 贴入高 %d 不在 §4.2 新档 [%d,%d]' % (key, h, lo, hi))
        path = unit9(key)
        fig = Image.open(path).convert('RGBA')
        if key in PATRICK_KEYS:
            fig = m.clean_patrick_blue(fig)
            fig = m.clean_patrick_blotches(fig)
            fig = m.fix_patrick_outline(fig)
        fig = drop_stray_specks(fig)
        fig = m.tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        ratio = h / fig.height
        # 硬闸A：蟹老板贴入比强制 ≤0.85x
        if key.startswith('krabs') and ratio > KRABS_MAX_RATIO + 1e-6:
            raise SystemExit('硬闸A 蟹老板贴入比 %.3fx > %.2fx（贴入 %d / 裁后 %d）——拒绝合成，需更小单元或降低贴入高'
                             % (ratio, KRABS_MAX_RATIO, h, fig.height))
        # 通用不放大规则（C3）
        if ratio > 1.0:
            raise SystemExit('C3 贴入比 %.3fx > 1.0x（%s）——禁止放大贴入' % (ratio, key))
        top = gy - h
        if top < TOP_FLOOR:
            raise SystemExit('尺寸自检: %s 最高点 y=%d 高于 %.0f 线，留白不足' % (key, top, TOP_FLOOR))
        report.append(dict(scene=scene_id, key=key, unit=os.path.basename(path),
                           paste_h=h, trim_h=fig.height, ratio=round(ratio, 3), top=top))
        m.paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
        print('贴入', scene_id, key, os.path.basename(path), 'h=%d trim=%d scale=%.2fx top=%d'
              % (h, fig.height, ratio, top))
        # 硬闸B：珊迪头盔 100% 裁切导出，供逐张复核盔圈闭合
        if key in ('sandy_l', 'sandy_c'):
            os.makedirs(audit_dir, exist_ok=True)
            hx0, hx1 = max(0, int(cx - h * 0.75)), min(W, int(cx + h * 0.75))
            hy0, hy1 = max(0, int(gy - h * 1.10)), int(gy - h * 0.30)
            canvas.crop((hx0, hy0, hx1, hy1)).save(
                os.path.join(audit_dir, '%s-%s-helmet.png' % (scene_id, key)))
            print('硬闸B 头盔裁切 →', os.path.join(audit_dir, '%s-%s-helmet.png' % (scene_id, key)))
    for arc in sc.get('arcs', []):
        x, y, r, col, w = arc[:5]
        a0, a1 = (arc[5], arc[6]) if len(arc) > 5 else (0, 360)
        ImageDraw.Draw(canvas).arc([x - r, y - r, x + r, y + r], start=a0, end=a1,
                                   fill=col + (255,), width=w)
    if sc.get('band'):
        draw_bubbles(canvas, sc['band'])
    canvas.convert('RGB').save(out_path)
    print('合成完成', scene_id, canvas.size, '→', out_path)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', required=True)
    ap.add_argument('--scene', default='all')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--sandy-c', default=None, help='九代过闸珊迪欢呼单元抠图文件路径')
    args = ap.parse_args()
    if args.sandy_c:
        _UNITS9['sandy_c'] = args.sandy_c
    os.makedirs(args.outdir, exist_ok=True)
    audit_dir = os.path.join(os.path.dirname(os.path.abspath(args.outdir.rstrip('/'))), 'audit9')
    ids = sorted(SCENES) if args.scene == 'all' else [args.scene]
    all_report = []
    for sid in ids:
        all_report.extend(compose_scene(sid, args.bg, os.path.join(args.outdir, 'r9-%s.png' % sid), audit_dir))
    rp = os.path.join(args.outdir, 'paste-report.json')
    with open(rp, 'w') as f:
        json.dump(all_report, f, ensure_ascii=False, indent=1)
    mx = {}
    for r in all_report:
        mx[r['key']] = max(mx.get(r['key'], 0), r['ratio'])
    print('贴入比上限（新档）:', mx)
    print('报告 →', rp)


if __name__ == '__main__':
    main()
