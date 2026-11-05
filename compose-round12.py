# 十二代合成（issue #25：角色间相对高度校准 + 海绵宝宝腿身比硬闸）。
# 相对十一代的三项变化（对症 issue #25 任务1/任务2，实测口径见 delivery/recipe.md §4.2）：
#   T1 SB 腿身比：单元改用十二代短腿单元（g11cut/sb-right12_00001，prompt 显式短腿约束，
#      生成后 腿/身=0.752 仍略超带宽，由 sb_proportion.shorten_legs 确定性缝合至 0.68=
#      官方宽块立绘实测值），并在贴入前强制 硬闸C 腿身比 ∈ [0.55, 0.75]（官方经典立绘
#      实测 0.54–0.70，样张地面线反推 ≈0.62——带内即官方比例）。
#   T2 相对高度表按样张实测重建（SB=1.0）：派大星 1.22→1.45（样张 1.48±0.06、官方并排图
#      1.43 交叉一致）、章鱼哥 1.41→1.50（样张重建 1.7±0.3，受 C5 50% 线顶置 1.50）、
#      蟹老板 1.29→1.10（样张 1.10±0.05：官方并排图无蟹老板参照，样张为唯一依据）、
#      珊迪 1.25→1.32（样张 1.29±0.08、官方并排图 1.33 交叉一致）、小蜗 0.40 不变。
#      贴入高相应重排（同 SB 档 600–630 下按比例换算），后排章鱼哥 gy 收至 2555（站台上沿）。
#   T3 沙岛/场泡/盔圈粉花/C4 皮肤必检/硬闸A/B/TOP_FLOOR 与十一代完全同实现（零回归）。
# 用法: python3 compose-round12.py --bg bg-flat11.png --scene all --outdir output/round12
#       [--no-leg-fix]
import argparse, json, os, sys
import importlib.util

spec = importlib.util.spec_from_file_location(
    'compose_round11', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compose-round11.py'))
c11 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['compose-round11.py']
spec.loader.exec_module(c11)
sys.argv = _argv

m9 = c11.m9
import sb_proportion as SP

LEG_FIX_TARGET = 0.68

# T2: §4.2 新档（SB=1.0 基线档 600–630 不变；其余=新相对比例 × 615±15）
NEW_TIER = {'sb': (600, 630), 'patrick': (850, 920), 'squidward': (880, 940),
            'krabs': (640, 700), 'sandy': (785, 840), 'gary': (230, 255)}

# T2: 场景表（相对十一代只动 贴入高 与 章鱼哥后排 gy；位置/倾角/泡泡带不动）
_H = {'pat_l': 885, 'patrick_r': 885, 'sq_front': 915, 'sq_annoyed': 915,
      'krabs': 670, 'sandy_l': 810, 'sandy_c': 810}
SCENES12 = {}
for sid, sc in c11.SCENES11.items():
    chars = []
    for key, h, cx, gy, tilt in sc['chars']:
        if key in _H:
            h = _H[key]
        if key.startswith('sq') and gy in (2520, 2530):
            gy = 2555   # 后排名单 2470–2560 内收沿，保 TOP≥1600（C5 零回归）
        chars.append((key, h, cx, gy, tilt))
    SCENES12[sid] = dict(sc, chars=chars)

UNITS12 = dict(c11.UNITS11)
G11 = '/mnt/g/workspace/comfyui/output/g11cut'
UNITS12['sb_r'] = os.path.join(G11, 'sb-right12_00001.png')
UNITS12['new_sandy_c'] = os.path.join(G11, 'sandy-cheer11_00001.png')
UNITS12['new_sandy_l'] = os.path.join(G11, 'sandy-left11_00001.png')

_leg_stat = {}


def _sb_palette12(fig):
    """硬闸C 注入点：SB 单元在调色前做腿身比手术修正（确定性）并校验带宽。"""
    if not _leg_stat.get('enabled', True):
        return _REAL_PALETTE(fig)
    r0 = SP.leg_ratio(fig)[0]
    if r0 is None:
        raise SystemExit('硬闸C: 无法测量 SB 单元腿身比 —— 拒绝合成')
    fig = SP.shorten_legs(fig, target=LEG_FIX_TARGET)
    r1 = SP.leg_ratio(fig)[0]
    if not (SP.LEG_BAND[0] <= r1 <= SP.LEG_BAND[1]):
        raise SystemExit('硬闸C SB腿身比 %.3f ∉ [%.2f,%.2f] —— 拒绝合成' % (r1, *SP.LEG_BAND))
    print('硬闸C SB腿身比 %.3f →%.3f（目标%.2f）PASS' % (r0, r1, LEG_FIX_TARGET))
    out = _REAL_PALETTE(fig)
    _leg_stat['before'], _leg_stat['after'] = round(r0, 3), round(r1, 3)
    return out


import sb_palette
_REAL_PALETTE = sb_palette.sb_palette
sb_palette.sb_palette = _sb_palette12   # c11.compose_scene 调用点即走此处


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', required=True)
    ap.add_argument('--scene', default='all')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--no-leg-fix', action='store_true')
    args = ap.parse_args()
    _leg_stat['enabled'] = not args.no_leg_fix
    m9.TIER.update(NEW_TIER)
    c11.SCENES11 = SCENES12
    c11.UNITS11 = UNITS12
    os.makedirs(args.outdir, exist_ok=True)
    audit_dir = os.path.join(os.path.dirname(os.path.abspath(args.outdir.rstrip('/'))), 'audit12')
    ids = sorted(SCENES12) if args.scene == 'all' else [args.scene]
    rep = []
    for sid in ids:
        rep.extend(c11.compose_scene(sid, args.bg, os.path.join(args.outdir, 'r12-%s.png' % sid),
                                     audit_dir, sb_palette_on=True, flower_on=True, sandy_new=True))
        if rep:
            for r in rep:
                if r.get('key') == 'sb_r' and 'sb_leg_ratio' not in r:
                    r['sb_leg_ratio'] = _leg_stat.get('after')
                    r['sb_leg_ratio_raw'] = _leg_stat.get('before')
    rp = os.path.join(os.path.dirname(os.path.abspath(args.outdir.rstrip('/'))),
                      'paste-report-generation-12.json')
    json.dump(rep, open(rp, 'w'), ensure_ascii=False, indent=1)
    print('paste report →', rp)
