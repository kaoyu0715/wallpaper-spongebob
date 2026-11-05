# 十一代海绵宝宝单元调色（issue #23 差距 G3：十代海绵宝宝"发灰发绿、不够官方黄"）。
# 对症四点，全部为单元级后处理（不改 LoRA、不改种子，固定参数可复现）：
#   ① 身体淡黄绿 (≈240,240,110) → 样张官黄 (≈240,224,64)：掩膜内 G -= 0.12*(G-B)、B *= 0.55；
#   ② 亮绿气孔 (≈110,160,30) → 样张橄榄绿 (≈172,156,44)：向样张色 0.72 混合；
#   ③ 外缘灰白 AA 晕（RMBG 边缘残灰，贴入深绿底后呈"浅绿描边"观感）→ 距轮廓 3px 内、
#      低饱和亮像素向描边近黑 (20,18,16) 回色（兼得样张的加粗描边观感，G9 同受益）；
#   ④ 灰白眼球/衬衫 (S<0.12, 200≤V≤246) → 提纯白。
# 颜色锚点全部采样自 review/reference/target-ref.jpg 海绵宝宝区域。
import numpy as np
from PIL import Image


def _dilate(mask, k=1):
    out = mask.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def sb_palette(fig, verbose=True):
    arr = np.asarray(fig.convert('RGBA'))
    a = arr.astype(np.float64)
    R, G, B, A = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    vis = A > 10
    m = np.stack([R, G, B], -1)
    V = m.max(-1)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    S = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)

    # ① 身体黄 → 官黄
    body = vis & (R > 200) & (G > 190) & (B > 60) & (B < 170) & ((R - B) > 70) & ((G - B) > 50)
    dG = (G - B) * 0.12
    dBB = B * 0.45
    G2 = G - dG
    B2 = B - dBB

    # ② 气孔亮绿 → 橄榄
    pore = vis & (G > R + 10) & (G > B + 40) & (G > 110) & (R < 190)
    tgt = np.array([172.0, 156.0, 44.0])
    mix = 0.72
    R3 = R.copy(); G3 = G2.copy(); B3 = B2.copy()
    R3 = np.where(pore, R * (1 - mix) + tgt[0] * mix, R)
    G3 = np.where(pore, G2 * (1 - mix) + tgt[1] * mix, G2)
    B3 = np.where(pore, B2 * (1 - mix) + tgt[2] * mix, B2)

    # ③ 外缘灰白晕 → 描边近黑
    edge_in = _dilate(~vis, 3) & vis
    halo = edge_in & (S < 0.32) & (V > 140)
    w = np.clip((V - 120.0) / 120.0, 0.0, 0.9)
    R4 = np.where(halo, R3 * (1 - w) + 20 * w, R3)
    G4 = np.where(halo, G3 * (1 - w) + 18 * w, G3)
    B4 = np.where(halo, B3 * (1 - w) + 16 * w, B3)

    # ④ 灰白 → 纯白
    whites = vis & (S < 0.12) & (V >= 200) & (V <= 246)
    R5 = np.where(whites, 250, R4)
    G5 = np.where(whites, 250, G4)
    B5 = np.where(whites, 250, B4)

    out = a.copy()
    out[..., 0] = np.clip(R5, 0, 255)
    out[..., 1] = np.clip(G5, 0, 255)
    out[..., 2] = np.clip(B5, 0, 255)
    if verbose:
        print('sb_palette: body=%d pore=%d halo=%d whites=%d px'
              % (body.sum(), pore.sum(), halo.sum(), whites.sum()))
    return Image.fromarray(out.astype(np.uint8), 'RGBA')
