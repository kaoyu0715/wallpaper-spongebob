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
    keep_ring = fb & ~E & ~_limb_on_tube & ~compose17.cut_alpha_mask(fish)   # 压圈鱼发丝=鱼像素非圈面
    vr = int((d[keep_ring] > 0).sum())
    print('%s: 前缘带(¬肢体)差异像素 = %d / %d (要求 0)' % (fish, vr, keep_ring.sum()))
    # ZZ 字标像素级保留断言（fish1）
    if fish == 'fish1':
        zz = np.asarray(mask17.zz_letters()) > 0
        vz = int((d[zz] > 0).sum())
        print('%s: ZZ 差异像素 = %d / %d (要求 0)' % (fish, vz, zz.sum()))
    # 补底清晰度硬闸（用户裁决 2026-11-09 二轮）：补底区高频能量低于周边样张水面
    # 即不过——涂抹糊块/灰绿补丁/白抹痕一律拦下。上版 unit17 为固化反例（须 FAIL）。
    def _hp(arr):
        a = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)
        bl = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype('uint8'))
                        .filter(ImageFilter.GaussianBlur(2))).astype(np.float32)
        return a - bl
    def _rms(hp, m):
        v = hp[m]
        return float(np.sqrt((v ** 2).mean()))
    fillm = compose17.inpaint_base(fish) & ~(np.asarray(mask17.tube_band(fish)) > 0)
    _uc = np.asarray(Image.fromarray(np.asarray(a).astype('uint8')).filter(ImageFilter.MaxFilter(9))) > 0
    fillm = fillm & ~_uc          # 只测水面：角色本体高对比不计入
    sur = np.asarray(Image.fromarray((fillm * 255).astype('uint8'))
                     .filter(ImageFilter.MaxFilter(65))) > 0
    sur = sur & ~fillm & ~(np.asarray(mask17.tube_band(fish)) > 0) \
        & ~(np.asarray(mask17.footprint(fish, margin=14)) > 127)
    if fish == 'fish1':
        sur = sur & ~(np.asarray(mask17.zz_letters()) > 0)
    hp_ref = _hp(ref.astype(np.float32))
    hp_st = _hp(st.astype(np.float32))
    hf_ref = _rms(hp_ref, sur)
    r = _rms(hp_st, fillm) / max(hf_ref, 1e-6)
    print('%s: 补底区清晰度比(全局) = %.2f (要求 ≥0.85)' % (fish, r))
    # 局部糊斑检测：逐 20px 瓦片，高频能量低于其“周边样张水面环带”50% 即计糊斑块
    # （基线=局部而非全局：平缓水区的干净克隆不应误伤）。
    tsize = 20
    Hf, Wf = fillm.shape
    surr = np.asarray(Image.fromarray((fillm * 255).astype('uint8'))
                      .filter(ImageFilter.MaxFilter(81))) > 0
    surr = surr & ~fillm & ~(np.asarray(mask17.tube_band(fish)) > 0) & ~_uc
    # 只测**被改动**的像素：按设计未动的样张原像素（圈缘影/保留水）不算糊斑候选，
    # 改后瓦片覆盖 <30% 时跳过（避免半原样瓦片被均值拖低误判）。
    changed = np.abs(st.astype(int) - ref.astype(int)).sum(2) > 24
    deficit = ntot = 0
    for by in range(0, Hf, tsize):
        for bx in range(0, Wf, tsize):
            t = fillm[by:by + tsize, bx:bx + tsize]
            if t.sum() < 0.6 * t.size:
                continue
            tc = t & changed[by:by + tsize, bx:bx + tsize]
            if tc.sum() < 0.3 * t.sum():
                continue
            med = np.median(ref[by:by + tsize, bx:bx + tsize].reshape(-1, 3), 0)
            if (med[0] > 140 and med[0] - med[1] > 80 and med[2] < 100) or \
               (med[0] > 190 and med[1] > 170 and med[2] > 170):
                continue   # REF 中位=圈条纹：糊斑闸只管水面瓦片（条纹道另核）
            _tR, _tG, _tB = st[by:by + tsize, bx:bx + tsize, 0].astype(int), st[by:by + tsize, bx:bx + tsize, 1].astype(int), st[by:by + tsize, bx:bx + tsize, 2].astype(int)
            _st_st = ((_tR - _tG > 50) & (_tR - _tB > 60) & (_tR > 90)) | \
                     ((_tR > 190) & (_tR - _tG > -18) & (_tR - _tG < 40) & (_tB > 140))
            _nr = np.asarray(Image.fromarray((_st_st.astype('uint8') * 255))
                             .filter(ImageFilter.MaxFilter(31))) > 0   # 管邻域
            tc = tc & ~(_st_st & _nr)          # 圈条纹克隆像素不入水面糊斑统计
            if tc.sum() < 0.3 * t.sum():
                continue
            ring_ = surr[by:by + tsize, bx:bx + tsize]
            if ring_.sum() < 20:
                continue
            rv = hp_ref[by:by + tsize, bx:bx + tsize][ring_]
            base = float(np.sqrt((rv ** 2).mean()))
            v = hp_st[by:by + tsize, bx:bx + tsize][tc]
            rr = float(np.sqrt((v ** 2).mean()))
            ntot += 1
            if rr < 0.5 * base:
                deficit += 1
                if os.environ.get('DBG1'):
                    print('  tile y%d x%d rr=%.2f base=%.2f' % (by, bx, rr, base))
    pct = 100.0 * deficit / max(ntot, 1)
    print('%s: 糊斑瓦片 = %d / %d (%.1f%%，要求 ≤2%%；涂抹糊块/灰绿补丁/白抹痕不过)'
          % (fish, deficit, ntot, pct))
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
            (stg.max(2) < 110) | \
            ((Bs - Rs > 12) & (Gs - Rs > 8) & (Rs < 225))   # 亮青涟漪水(B≫R)也是水(实测)
        # 深色水面阴影缝=中性暗，视作水
        refg0 = np.asarray(Image.open(mask17.REF).convert('RGB')).astype(int)
        watery = watery | (np.abs(stg - refg0).sum(2) <= 8)   # 样张原像素未动=合法(圈白/星芒)
        _R, _G, _B = stg[:, :, 0], stg[:, :, 1], stg[:, :, 2]
        stripe_st = ((_R - _G > 50) & (_R - _B > 60) & (_R > 90)) | \
                    ((_R > 190) & (_R - _G > -18) & (_R - _G < 40) & (_B > 140))
        # 与 compose 条纹道同判据（深红/粉影红同属圈条纹,鱼鳍压圈处克隆合法）
        watery = watery | stripe_st
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
