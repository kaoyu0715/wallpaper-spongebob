# 十代合成（issue #21）：管线 = 九代（尺寸档 ×0.76、硬闸A/B、C5 留白自检、场景表全部沿用），
# 唯一变化是 §4.3 派大星固定后处理由三件套扩为**四件套**：
#   clean_patrick_blue → clean_patrick_blotches → clean_patrick_skin(新增) → fix_patrick_outline。
#   clean_patrick_skin 回填皮肤上的暗红散点（patrick_skin.py），置于 outline 修复之前，
#   轮廓带内红点先回肤色、不留近黑残点；描边核心线面积大不受影响。
# 成品审计新增皮肤必检（§5 C4）：每张含派大星的合成自动导出皮肤区 100% 裁切
#   到 output/audit10/<scene>-<unit>-skin.png（供人工 100% 复核），并用
#   patrick_skin.detect_patrick_specks 在合成前先对单元做程序化残留计数（应为 0）。
# 用法: python3 compose-round10.py --bg <bg.png> --scene <id>|all --outdir <目录>
import argparse, json, os, sys
import importlib.util
from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location(
    'compose_round9', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compose-round9.py'))
m9 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['compose-round9.py']
spec.loader.exec_module(m9)
sys.argv = _argv

m = m9.m  # compose-round8 模块（既有 alpha/贴入/泡泡等基础函数）
W, H = m9.W, m9.H

_spec10 = importlib.util.spec_from_file_location(
    'patrick_skin', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patrick_skin.py'))
ps = importlib.util.module_from_spec(_spec10)
_spec10.loader.exec_module(ps)

SCENES = m9.SCENES
PATRICK_KEYS = m9.PATRICK_KEYS


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
        fam = m9.tier_of(key)
        lo, hi = m9.TIER[fam]
        if not (lo <= h <= hi):
            raise SystemExit('硬闸 尺寸档: %s 贴入高 %d 不在 §4.2 新档 [%d,%d]' % (key, h, lo, hi))
        path = m9.unit9(key)
        fig = Image.open(path).convert('RGBA')
        if key in PATRICK_KEYS:
            # §4.3 四件套（十代）：顺序不可换，skin 在 outline 之前
            fig = m.clean_patrick_blue(fig)
            fig = m.clean_patrick_blotches(fig)
            fig = ps.clean_patrick_skin(fig)
            fig = m.fix_patrick_outline(fig)
        fig = m9.drop_stray_specks(fig)
        fig = m.tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        ratio = h / fig.height
        if key.startswith('krabs') and ratio > m9.KRABS_MAX_RATIO + 1e-6:
            raise SystemExit('硬闸A 蟹老板贴入比 %.3fx > %.2fx（贴入 %d / 裁后 %d）——拒绝合成，需更小单元或降低贴入高'
                             % (ratio, m9.KRABS_MAX_RATIO, h, fig.height))
        if ratio > 1.0:
            raise SystemExit('C3 贴入比 %.3fx > 1.0x（%s）——禁止放大贴入' % (ratio, key))
        top = gy - h
        if top < m9.TOP_FLOOR:
            raise SystemExit('尺寸自检: %s 最高点 y=%d 高于 %.0f 线，留白不足' % (key, top, m9.TOP_FLOOR))
        report.append(dict(scene=scene_id, key=key, unit=os.path.basename(path),
                           paste_h=h, trim_h=fig.height, ratio=round(ratio, 3), top=top))
        m.paste_figure(canvas, fig, cx, gy, h, tilt=tilt)
        print('贴入', scene_id, key, os.path.basename(path), 'h=%d trim=%d scale=%.2fx top=%d'
              % (h, fig.height, ratio, top))
        if key in ('sandy_l', 'sandy_c'):
            os.makedirs(audit_dir, exist_ok=True)
            hx0, hx1 = max(0, int(cx - h * 0.75)), min(W, int(cx + h * 0.75))
            hy0, hy1 = max(0, int(gy - h * 1.10)), int(gy - h * 0.30)
            canvas.crop((hx0, hy0, hx1, hy1)).save(
                os.path.join(audit_dir, '%s-%s-helmet.png' % (scene_id, key)))
            print('硬闸B 头盔裁切 →', os.path.join(audit_dir, '%s-%s-helmet.png' % (scene_id, key)))
        if key in PATRICK_KEYS:
            os.makedirs(audit_dir, exist_ok=True)
            px0, px1 = max(0, int(cx - h * 0.45)), min(W, int(cx + h * 0.45))
            py0, py1 = max(0, int(gy - h)), int(gy)
            crop_box = (px0, py0, px1, py1)
            canvas.crop(crop_box).save(
                os.path.join(audit_dir, '%s-%s-skin.png' % (scene_id, key)))
            n = len(ps.detect_patrick_specks(canvas.crop(crop_box).convert('RGB'),
                                             paste_ratio=ratio))
            print('皮肤必检 %s → 残留红点 %d 个，裁切 → %s' %
                  (scene_id, n, os.path.join(audit_dir, '%s-%s-skin.png' % (scene_id, key))))
            report[-1]['skin_specks'] = n
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
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    audit_dir = os.path.join(os.path.dirname(os.path.abspath(args.outdir.rstrip('/'))), 'audit10')
    ids = sorted(SCENES) if args.scene == 'all' else [args.scene]
    all_report = []
    for sid in ids:
        all_report.extend(compose_scene(sid, args.bg, os.path.join(args.outdir, 'r10-%s.png' % sid), audit_dir))
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
