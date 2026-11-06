# 十三代自检对照（issue #27 验收=样张本身）：整幅并排 + 关键区域 2.5x 放大裁切对。
# 用法: python3 compare13.py output/round13/r13-replica-v1.png ref/target-crop.png out-prefix
import sys
from PIL import Image, ImageDraw

CANVAS = (1440, 3200)


def pair(replica_path, sample_path, prefix, regions=None):
    rep = Image.open(replica_path).convert('RGB')
    if rep.size != CANVAS:
        rep = rep.resize(CANVAS, Image.LANCZOS)
    sam = Image.open(sample_path).convert('RGB')  # 样张壁纸裁区 809x1737
    # 全幅并排（同高缩放到 1100）
    H_ = 1100
    r1 = rep.resize((int(rep.width * H_ / rep.height), H_))
    r2 = sam.resize((int(sam.width * H_ / sam.height), H_))
    out = Image.new('RGB', (r1.width + r2.width + 24, H_ + 26), (235, 235, 235))
    d = ImageDraw.Draw(out)
    out.paste(r2, (8, 24)); out.paste(r1, (r2.width + 16, 24))
    d.text((12, 6), 'sample', fill=(0, 0, 0)); d.text((r2.width + 24, 6), 'replica', fill=(0, 0, 0))
    out.save(prefix + '-full.png')

    # 关键区域放大对（canvas 坐标 → 样张 crop 坐标: x/1.78, y/1.842）
    REG = regions or [
        ('faces', (300, 1100, 1250, 2150)),
        ('map', (300, 2300, 1250, 3000)),
        ('helmet', (500, 600, 1150, 1350)),
        ('gary', (400, 2550, 1050, 3150)),
        ('leftfish', (0, 1300, 620, 2600)),
        ('egg', (120, 1300, 480, 1750)),
    ]
    for name, (x0, y0, x1, y1) in REG:
        rc = rep.crop((x0, y0, x1, y1))
        sc = (int(x0 / 1.78), int(y0 / 1.842), int(x1 / 1.78), int(y1 / 1.842))
        sm = sam.crop(sc)
        TH_ = 640
        rr = rc.resize((max(1, int(rc.width * TH_ / rc.height)), TH_))
        ss = sm.resize((max(1, int(sm.width * TH_ / sm.height)), TH_))
        o = Image.new('RGB', (rr.width + ss.width + 20, TH_ + 24), (235, 235, 235))
        dd = ImageDraw.Draw(o)
        o.paste(ss, (6, 22)); o.paste(rr, (ss.width + 14, 22))
        dd.text((10, 4), name + ' sample', fill=(0, 0, 0))
        dd.text((ss.width + 22, 4), 'replica', fill=(0, 0, 0))
        o.save('%s-%s.png' % (prefix, name))
        print('pair', name)
    print('saved', prefix + '-*.png')


if __name__ == '__main__':
    pair(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'ref/target-crop.png', sys.argv[3])
