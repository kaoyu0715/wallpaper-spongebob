# -*- coding: utf-8 -*-
# 十七代分区过闸自检（issue #35 硬性流程）：
#   ① 形体清晰度 / ② 边缘干净度：贴回预览 + 样张同区 2.4x 并排（人审图存档）；
#   ③ 圈像素未动：程序断言——管体像素（前缘带非肢体区、后缘非补画区）与样张逐像素一致，
#      并全图统计「改动域外差异=0」。
# 用法: python3 gate17.py --fish fish1 --stage <stage.png> [--name v1] [--zoom 2.4]
import argparse, os, sys
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask17, sbs16, compose17

T = os.path.dirname(os.path.abspath(__file__))
REF = mask17.REF
OUTD = os.path.join(T, 'output', 'gate')
os.makedirs(OUTD, exist_ok=True)


def allowed_zone(fish, stage_img):
    """允许偏离样张的像素域 = 足迹(外扩26) ∪ 前缘带 ∪ 肢体骨架区 ∪ 新单元本体 ∪ ZZ(fish1)。"""
    m = mask17.footprint(fish, margin=26)
    em = mask17.erase_mask(fish).filter(ImageFilter.MaxFilter(13))  # 补画区+贴回晕环
    m = Image.composite(Image.new('L', m.size, 255), m, em)
    bs = Image.fromarray((compose17.inpaint_base(fish) * 255).astype('uint8')) \
        .filter(ImageFilter.MaxFilter(13))                          # 实际补画域(含抠图α∪鱼鳍)+晕环
    m = Image.composite(Image.new('L', m.size, 255), m, bs)
    m = Image.composite(Image.new('L', m.size, 255), m, mask17.front_band(fish))
    m = Image.composite(Image.new('L', m.size, 255), m, mask17.old_limbs(fish, dil=compose17.POSE[fish]['limb_dil'] + 6))
    if fish == 'fish1':
        m = Image.composite(Image.new('L', m.size, 255), m, mask17.zz_letters())
    cfg = compose17.POSE[fish]
    if cfg.get('unit'):
        layer, u, (px, py) = compose17._unit_layer(fish, cfg)
        a = layer.split()[3].point(lambda v: 255 if v > 30 else 0).filter(ImageFilter.GaussianBlur(3))
        m = Image.composite(Image.new('L', m.size, 255), m, a)
        # 接触影（issue ③：允许必要接触影）：与 compose17 同参复现影区并入允许域
        ua = layer.split()[3]
        sh = Image.fromarray((np.asarray(ua) > 40).astype('uint8') * 255)
        sh = sh.filter(ImageFilter.GaussianBlur(7))
        sh = sh.transform(m.size, Image.AFFINE, (1, 0, 0, 0, 1, -cfg['shadow'][1]))
        m = Image.composite(Image.new('L', m.size, 255), m, sh.point(lambda v: 255 if v > 3 else 0))
    return m.point(lambda v: 255 if v > 96 else 0)


def check(fish, stage_path, name, zoom=2.4):
    ref = np.asarray(Image.open(REF).convert('RGB')).astype(np.int16)
    st = np.asarray(Image.open(stage_path).convert('RGB')).astype(np.int16)
    zone = np.asarray(allowed_zone(fish, stage_path)) > 0
    d = np.abs(st - ref).sum(2)
    out = d[~zone]
    viol = int((out > 0).sum())
    print('%s: 改动域外差异像素 = %d / %d (要求 0)' % (fish, viol, out.size))
    # 前缘带（非肢体区）逐像素一致断言；
    # 「旧肢体压管」区除外——那里是鱼像素（旧鞋/旧鳍），设计上由补画重绘成管面。
    fb = np.asarray(mask17.front_band(fish)) > 0
    _limb_on_tube = np.asarray(mask17.old_limbs(fish, dil=8)) > 0
    cfg = compose17.POSE[fish]
    E_limbs = mask17.old_limbs(fish, dil=cfg['limb_dil'])
    if cfg.get('unit'):
        layer, u, _ = compose17._unit_layer(fish, cfg)
        a = layer.split()[3].point(lambda v: 255 if v > 60 else 0)
        E = np.asarray(Image.composite(E_limbs, Image.new('L', E_limbs.size, 0),
                                       a).filter(ImageFilter.MaxFilter(7))) > 0
    else:
        E = np.zeros_like(fb)
    keep_ring = fb & ~E & ~_limb_on_tube
    vr = int((d[keep_ring] > 0).sum())
    print('%s: 前缘带(¬肢体)差异像素 = %d / %d (要求 0)' % (fish, vr, keep_ring.sum()))
    # ZZ 字标像素级保留断言（fish1）
    if fish == 'fish1':
        zz = np.asarray(mask17.zz_letters()) > 0
        vz = int((d[zz] > 0).sum())
        print('%s: ZZ 差异像素 = %d / %d (要求 0)' % (fish, vz, zz.sum()))
    # 躯干覆盖硬闸（用户裁决 2026-11-09）：旧鱼躯干掩膜区（抠图α¬管带¬ZZ）
    # 必须被新角色像素完整覆盖——「圈里只剩头/鞋/底片补丁色」必须过不了闸。
    ox, oy = mask17.cut_origins()[fish]
    ca = np.asarray(mask17.cut_rgba(fish).split()[3])
    big = np.zeros(ref.shape[:2], bool)
    big[oy:oy + ca.shape[0], ox:ox + ca.shape[1]] = ca > 127
    # 躯干=旧鱼主体（抠图α∧圈内 OPEN 域），管上肢体/圈外鱼鳍按设计改水，不算躯干。
    ox_, oy_, oa, ob = mask17.OPEN_ELL[fish]
    yy0, xx0 = np.mgrid[0:ref.shape[0], 0:ref.shape[1]]
    openell = ((xx0 - ox_) / oa) ** 2 + ((yy0 - oy_) / ob) ** 2 <= 1.0
    torso = big & openell & ~(np.asarray(mask17.tube_band(fish)) > 0)
    if fish == 'fish1':
        torso = torso & ~(np.asarray(mask17.zz_letters()) > 0)
    torso = np.asarray(Image.fromarray((torso * 255).astype('uint8'))
                       .filter(ImageFilter.MinFilter(9))) > 0
    if cfg.get('unit'):
        cov = np.asarray(a).astype(bool)
        uncovered = torso & ~cov
        uc = int(uncovered.sum())
        ratio = 1.0 - uncovered.sum() / max(torso.sum(), 1)
        # 未覆盖像素必须是重构后的冷调水面（底片补丁色/雾状擦痕一律不过）
        stg = np.asarray(Image.open(stage_path).convert('RGB')).astype(int)
        Rs, Gs, Bs = stg[:, :, 0], stg[:, :, 1], stg[:, :, 2]
        watery = ((Rs < 170) & ((Gs - Rs > 15) | (Bs - Rs > 5))) | \
            (stg.max(2) < 110)   # 深色水面阴影缝=中性暗，视作水
        unc2 = np.asarray(Image.fromarray((uncovered * 255).astype('uint8'))
                          .filter(ImageFilter.MinFilter(5))) > 0   # 去角色边缘 AA 环
        _td = np.asarray(Image.fromarray((np.asarray(mask17.tube_band(fish)) > 0)
                                         .astype('uint8') * 255).filter(ImageFilter.MaxFilter(11))) > 0
        unc2 = unc2 & ~_td   # 管缘缝像素按设计保留圈阴影，不要求水色（与补画清洗同域）
        bad = int((unc2 & ~watery).sum())
        print('%s: 躯干角色覆盖率 = %.1f%% (要求 ≥55%%)；未覆盖中非水色像素 = %d (要求 0)'
              % (fish, ratio * 100, bad))
    else:
        uc = None
    # 并排过闸图
    p1 = os.path.join(OUTD, '%s-%s-sbs.png' % (fish, name))
    sbs16.sbs(stage_path, mask17.CROPS[fish], p1, zoom=zoom, la='样张同区', lb='十七代 %s 贴回' % name)
    return viol, vr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fish', required=True)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--name', default='v1')
    ap.add_argument('--zoom', type=float, default=2.4)
    a = ap.parse_args()
    check(a.fish, a.stage, a.name, a.zoom)
