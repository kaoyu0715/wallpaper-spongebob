# 四代合成：三角色互动构图「吹泡泡」→ 1440x3200
# 同一套角色抠图 + 布局分别合成到 路线A官方图案背景 / 路线B Krea2 实景海底。
# flat 风格：无阴影无暖调（保持官方平面感）；real 风格：接触阴影+微倒影+轻暖统一。
# 用法: python3 compose-round4.py --bg <bg.png> --cutdir <g4cut目录> --out <out.png> \
#         --style flat|real --variant v1|v2|v3
import argparse, glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1440, 3200

# (高px, cx, ground_y, 倾角°)  —— 三角色站位互相关联：章鱼哥后排中间抱臂嫌弃，
# 海绵宝宝前排左吹泡泡抬手指向泡泡，派大星前排右欢呼看泡泡，泡泡弧线把三者串成一个场景。
BASE = {
  'squidward': (830, 720, 2570, 0.0),
  'patrick':   (755, 1015, 2755, 3.0),
  'spongebob': (705, 425, 2805, -3.0),
}
VARIANTS = {
  'v1': {},
  'v2': {'patrick': (765, 975, 2740, 4.0), 'spongebob': (725, 445, 2800, -4.0)},
  'v3': {'squidward': (810, 690, 2555, 2.0), 'patrick': (745, 1050, 2765, 2.0)},
}
ORDER = ['squidward', 'patrick', 'spongebob']  # 后→前

# 泡泡带（x, y, r）：从海绵宝宝上举的指尖升起，掠过章鱼哥头顶，飘向派大星举起的手
BUBBLES = {
  'v1': [(555, 2130, 22), (585, 1870, 38), (740, 1630, 66), (890, 1360, 34), (975, 1140, 17)],
  'v2': [(580, 2140, 24), (655, 1870, 44), (810, 1590, 76), (955, 1320, 38), (1040, 1100, 19)],
  'v3': [(540, 2150, 20), (555, 1890, 32), (705, 1660, 56), (655, 1380, 42), (600, 1120, 20)],
}


def tighten_alpha(fig, floor=120, erode=2):
    a = np.asarray(fig.getchannel('A')).astype(np.float32)
    out = np.clip((a - floor) * (255.0 / (255 - floor)), 0, 255).astype('uint8')
    fig = fig.copy()
    fig.putalpha(Image.fromarray(out, 'L'))
    if erode > 0:  # 再收 1-2px，去掉白底渲染的白边
        fig.putalpha(fig.getchannel('A').filter(ImageFilter.MinFilter(erode * 2 + 1)))
    return fig


def paste_figure(canvas, fig, cx, ground_y, target_h, tilt=0.0, style='real'):
    if tilt:
        fig = fig.rotate(tilt, expand=True, resample=Image.BICUBIC)
    r = target_h / fig.height
    fw, fh = max(1, int(fig.width * r)), target_h
    f = fig.resize((fw, fh), Image.LANCZOS)
    x0, y0 = int(cx - fw / 2), int(ground_y - fh)
    if style == 'real':
        sh = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sw = int(fw * 0.62)
        sd.ellipse([cx - sw // 2, ground_y - int(fh * 0.04), cx + sw // 2, ground_y + int(fh * 0.07)],
                   fill=(30, 22, 18, 100))
        canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(int(max(5, fh * 0.02)))))
        ref = f.transpose(Image.FLIP_TOP_BOTTOM).resize((fw, max(1, int(fh * 0.22))), Image.LANCZOS)
        ref = ref.filter(ImageFilter.GaussianBlur(6))
        ra = np.asarray(ref.getchannel('A')).astype(float) * 0.10
        ref.putalpha(Image.fromarray(ra.astype('uint8'), 'L'))
        canvas.alpha_composite(ref, (x0, ground_y))
        fa = np.asarray(f.getchannel('A')).astype(np.float32)
        warm = Image.new('RGBA', f.size, (255, 176, 110, 255))
        warm.putalpha(Image.fromarray((fa * 0.08).astype('uint8'), 'L'))
        f = Image.alpha_composite(f, warm)
    canvas.alpha_composite(f, (x0, y0))


def draw_bubbles(canvas, pts):
    d = ImageDraw.Draw(canvas)
    for x, y, r in pts:
        w = max(5, r // 4)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 240, 245, 14),
                  outline=(250, 253, 250, 235), width=w)
        d.arc([x - r + r * 0.22, y - r + r * 0.22, x + r - r * 0.22, y + r - r * 0.22],
              start=135, end=230, fill=(255, 255, 255, 245), width=max(4, r // 7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', required=True)
    ap.add_argument('--cutdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--style', default='real', choices=['flat', 'real'])
    ap.add_argument('--variant', default='v1')
    args = ap.parse_args()

    canvas = Image.open(args.bg).convert('RGBA')
    if canvas.size != (W, H):
        canvas = canvas.resize((W, H), Image.LANCZOS)
    layout = dict(BASE)
    layout.update(VARIANTS.get(args.variant, {}))
    for role in ORDER:
        cands = sorted(glob.glob(os.path.join(args.cutdir, role + '*.png')), key=os.path.getmtime)
        if not cands:
            raise SystemExit('缺少抠图: ' + role + ' in ' + args.cutdir)
        fig = Image.open(cands[-1]).convert('RGBA')
        fig = tighten_alpha(fig, floor=115, erode=2)
        bb = fig.getbbox()
        if bb:
            fig = fig.crop(bb)
        h, cx, gy, tilt = layout[role]
        paste_figure(canvas, fig, cx, gy, h, tilt=tilt, style=args.style)
        print('贴入', role, os.path.basename(cands[-1]), '->', h, 'tilt', tilt)
    draw_bubbles(canvas, BUBBLES[args.variant])
    canvas.convert('RGB').save(args.out)
    print('合成完成', canvas.size, '→', args.out)


if __name__ == '__main__':
    main()
