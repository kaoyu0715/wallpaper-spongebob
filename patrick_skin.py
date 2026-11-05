# 十代新增（issue #21）：派大星皮肤红点清理 + 成品审计探测器。
#
# 根因：Patrick_il LoRA 红色素泄漏/数据伪影——粉色皮肤上散落的暗红小点
# （与已修的描边偏红 fix_patrick_outline、短裤花斑 clean_patrick_blotches 同源）。
#
# clean_patrick_skin（配方 §4.3 新增第 3 步，三件套→四件套）：
#   在粉色皮肤区域内，把红度明显高于肤色（相对本图肤色中位数动态阈值）、
#   面积小（≤max_area）、且被粉色皮肤围合的异常红像素，回填为周围肤色
#   （逐孤岛取周围环带肤色中位数 + 高斯扩散，保 AA 边缘）。
#   不误伤：眼睛/嘴/描边/口腔红——大色块由面积上限排除；靠近轮廓/半透明的
#   描边带由 touches-alpha 排除；眼周、短裤上缘等被黑线或绿色围合的碎片
#   由围合占比（≥0.75 圈环为皮肤）排除。
#   顺序：blue → blotches → skin → outline。skin 放在 outline 修复之前，
#   避免轮廓带内的红点先被染成近黑残留；outline 核心描边线面积大不受影响。
#
# detect_patrick_specks（§5 成品审计 C4 皮肤必检的自动化部分）：
#   在成品图上按同一套判据数残留在皮肤内的暗红孤岛（阈值随成品比例放宽
#   下限、上限按缩放系数放大），返回孤岛列表；0 个 = 皮肤无红点。
#   贴黑描线的红碎屑不计（归 C4 既有的描边偏色检查口径）。
import numpy as np
from collections import deque
from PIL import Image, ImageFilter

NBR = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dilate(m, k=1):
    out = m.copy()
    for _ in range(k):
        p = np.pad(out, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def _components(mask):
    lab = -np.ones(mask.shape, np.int32)
    H_, W_ = mask.shape
    comps = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0] >= 0:
            continue
        q = deque([(y0, x0)])
        lab[y0, x0] = 0
        comp = []
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for dy, dx in NBR:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H_ and 0 <= nx < W_ and mask[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = 0
                    q.append((ny, nx))
        comps.append(comp)
    return comps


def _masks(arr):
    """返回 (visible, skin, cand, skin_med, skin_rg_med)。
    cand = 红度明显高于肤色的异常红像素（相对本图中位数动态阈值）。"""
    R, G, B, A = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    visible = A > 140
    # 肤色粉：沿用 §4.3 既有 skin 定义
    skin = visible & (R > 190) & (G > 110) & (G < 210) & (B > 100) & (B < 200) & (R > G + 30)
    if int(skin.sum()) < 20000:
        return visible, skin, np.zeros_like(visible), None, None
    skin_med = np.median(arr[skin][:, :3], axis=0)
    rg_med = float(np.median((R - G)[skin]))
    cand = (visible & (R > 110)
            & ((R - G) > rg_med + 12)
            & (G < skin_med[1] - 18)
            & (B > R - 170))
    return visible, skin, cand, skin_med, rg_med


def _enclosed(comp_coords, cand, skin, visible, ring_r=4, min_frac=0.75):
    """孤岛围合判定：不触碰半透明边（描边带），且外围环带皮肤占比 ≥ min_frac。"""
    ys = [c[0] for c in comp_coords]
    xs = [c[1] for c in comp_coords]
    y0, y1 = max(min(ys) - ring_r, 0), min(max(ys) + ring_r + 1, cand.shape[0])
    x0, x1 = max(min(xs) - ring_r, 0), min(max(xs) + ring_r + 1, cand.shape[1])
    cm = np.zeros(cand.shape, bool)
    cm[ys, xs] = True
    d2 = _dilate(cm, 2)
    if (d2 & ~visible).any():  # 触碰半透明边：属描边带，不属皮肤散点
        return False, None
    ring = _dilate(cm, ring_r) & ~cm
    ring_px = ring[y0:y1, x0:x1]
    tot = int(ring_px.sum())
    if tot == 0:
        return False, None
    sk = int((ring & skin)[y0:y1, x0:x1].sum())
    ok = sk / tot >= min_frac
    ring_coords = np.argwhere(ring & skin)
    return ok, ring_coords


def _find_specks(arr, max_area=400, min_area=6, ring_r=4, min_frac=0.75):
    visible, skin, cand, skin_med, rg_med = _masks(arr)
    if cand is None or skin_med is None or int(cand.sum()) == 0:
        return [], skin, cand, skin_med
    out = []
    for comp in _components(cand):
        if not (min_area <= len(comp) <= max_area):
            continue
        ok, ring_skin = _enclosed(comp, cand, skin, visible, ring_r, min_frac)
        if ok:
            out.append((comp, ring_skin))
    return out, skin, cand, skin_med


def clean_patrick_skin(fig, max_area=400, min_area=6, verbose=True):
    im = np.asarray(fig).astype(np.int32)
    specks, skin, cand, skin_med = _find_specks(im, max_area=max_area, min_area=min_area)
    if not specks:
        if verbose:
            print('skin: 未发现皮肤红点孤岛')
        return fig
    out = np.asarray(fig).copy()
    total = 0
    for comp, ring_skin in specks:
        cm = np.zeros(cand.shape, bool)
        ys = [c[0] for c in comp]
        xs = [c[1] for c in comp]
        cm[ys, xs] = True
        fillc = np.median(im[ring_skin[:, 0], ring_skin[:, 1], :3], axis=0).astype(np.uint8)
        patch = _dilate(cm, 1)  # 含 AA 圈 1px
        y0, y1 = max(min(ys) - 6, 0), min(max(ys) + 7, cm.shape[0])
        x0, x1 = max(min(xs) - 6, 0), min(max(xs) + 7, cm.shape[1])
        flat = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        flat[..., :3] = fillc
        flat[..., 3] = 255
        fill = Image.fromarray(flat, 'RGBA').filter(ImageFilter.GaussianBlur(2))
        mmask = Image.fromarray((patch[y0:y1, x0:x1] * 255).astype('uint8'), 'L') \
            .filter(ImageFilter.MaxFilter(3))
        sub = Image.fromarray(out[y0:y1, x0:x1], 'RGBA')
        out[y0:y1, x0:x1] = np.asarray(Image.composite(fill, sub, mmask))
        total += len(comp)
    if verbose:
        print('skin: 回填红点孤岛 %d 个 / %d px → 周围肤色（皮肤中位 %s）'
              % (len(specks), total, tuple(int(v) for v in skin_med)))
    return Image.fromarray(out, 'RGBA')


def detect_patrick_specks(rgb_img, paste_ratio=1.0, min_area=4, verbose=True):
    """成品审计：rgb 成品（或其裁切块）内残留的派大星皮肤红点孤岛数。
    成品按贴入比等比缩小，面积上限按 ratio^2 放大补偿，红度下限略放宽
    （LANCZOS 缩放会让小点混色变淡）。
    「红点」定义与 §4.3 skin 步一致：被粉色皮肤围合、**不贴黑描边**——
    环带邻域（3px + 1px 线宽容差）内有黑描边像素的红色碎屑属 C4 既有的
    「描边偏色」检查对象，不计入皮肤散点。"""
    a = np.asarray(rgb_img.convert('RGB')).astype(np.int32)
    rgba = np.dstack([a, np.full(a.shape[:2], 255, np.int32)])
    visible, skin, cand, skin_med, rg_med = _masks(rgba)
    if int(skin.sum()) < 3000:
        return []
    max_area = int(max(24, 400 * paste_ratio * paste_ratio))
    cand = cand & _dilate(skin, 8)  # 只查皮肤围合带内的红岛
    dark = (a[..., 0] < 95) & (a[..., 1] < 95) & (a[..., 2] < 95)
    near_line = _dilate(dark, 1)
    found = []
    for comp in _components(cand):
        if not (min_area <= len(comp) <= max_area):
            continue
        ys = [c[0] for c in comp]
        xs = [c[1] for c in comp]
        cm = np.zeros(cand.shape, bool)
        cm[ys, xs] = True
        ring = _dilate(cm, 3) & ~cm
        tot = int(ring.sum())
        if not tot or int((ring & skin).sum()) / tot < 0.6:
            continue
        if (ring & near_line).any():
            continue  # 贴描线的红碎屑：归描边偏色检查，不算皮肤散点
        found.append((len(comp), int(np.mean(ys)), int(np.mean(xs))))
    if verbose:
        print('detect: 残留红点孤岛 %d 个: %s' % (len(found), found[:8]))
    return found
