"""边缘残缺（edge_clip）共享逻辑：三引擎（realscene / multi_rs / gpu_engine）
独立副本最容易改漏、漏掉是静默错误的逻辑链集中于此。仅标准库/纯整数浮点，
multiprocessing spawn 子进程与 torch 路径均可导入。

用途：允许目标部分越出画布（贴边残缺），粘贴只保留画布交集区域，
标注与可见部分一致（clamp 自动收缩，且防御全越界/碎屑负宽框）。
"""

import random

MIN_VIS_FRAC = 0.3      # 保底可见比例（相对目标自身宽/高）。max_frac≤0.7 时恒成立
MIN_LABEL_NORM = 1e-3   # 病态/碎屑标注下限（归一化；640x480 下 ≈ 0.6px）


def visible_rect(bw, bh, x, y, tw, th, min_vis_frac=MIN_VIS_FRAC):
    """目标 (x,y,tw,th) 与画布 (bw,bh) 的可见交集 (ix0, iy0, ix1, iy1)。
    无交（可见为 0），或可见宽/高 < min_vis_frac*tw/th → 返回 None。
    min_vis_frac=0 时仅判"有无交集"。"""
    ix0, iy0 = max(0, x), max(0, y)
    ix1, iy1 = min(bw, x + tw), min(bh, y + th)
    if ix0 >= ix1 or iy0 >= iy1:
        return None
    if min_vis_frac > 0:
        if (ix1 - ix0) < min_vis_frac * tw or (iy1 - iy0) < min_vis_frac * th:
            return None
    return ix0, iy0, ix1, iy1


def find_anchor(bw, bh, tw, th, placed_boxes, edge_prob, edge_frac, max_attempts):
    """放置采样（与三引擎现有循环逐位一致 + 边缘残缺放宽）。
    - edge 触发（random.random() < edge_prob）：x = randint(-ox, bw-tw+ox)，
      ox = int(edge_frac*tw)（y 同理 oy）——两轴、每轴左右两方向均可越界；
    - 常规：x = randint(0, bw-tw)（与现状完全相同，prob=0 时逐位回归）；
    - 每候选先 visible_rect 保底（无交/可见不足 → continue 换样），
      再做 overlap 拒绝（safe_margin = max(tw,th)*0.1，AABB 判据同现状）。
    返回 (x, y, edge_mode) 或 None（尝试耗尽）。edge_mode=True 表示本次越界采样。"""
    ox = int(edge_frac * tw) if edge_prob > 0 else 0
    oy = int(edge_frac * th) if edge_prob > 0 else 0
    # 与两轴越界区间是否可能有交：边缘模式下抽样范围=axis span 两端各扩 ox/oy，
    # 若 ox >= bw - tw（目标接近于全宽），可能出现跨轴交织，由 visible_rect 兜底。
    edge_mode = bool(edge_prob > 0 and (ox > 0 or oy > 0)) and random.random() < edge_prob
    for _ in range(max_attempts):
        if edge_mode:
            x = random.randint(-ox, bw - tw + ox)
            y = random.randint(-oy, bh - th + oy)
        else:
            x = random.randint(0, bw - tw)
            y = random.randint(0, bh - th)
        if visible_rect(bw, bh, x, y, tw, th) is None:
            continue
        overlap = False
        safe_margin = max(tw, th) * 0.1
        for b in placed_boxes:
            dx = max(0, abs((x + tw / 2) - (b['x'] + b['width'] / 2)) - (tw + b['width']) / 2)
            dy = max(0, abs((y + th / 2) - (b['y'] + b['height'] / 2)) - (th + b['height']) / 2)
            if dx < safe_margin and dy < safe_margin:
                overlap = True
                break
        if not overlap:
            return x, y, edge_mode
    return None


def clamp_norm_bbox(xc, yc, wn, hn, to_border, min_norm=MIN_LABEL_NORM):
    """现有标注 clamp（x_min=max(TB, xc-wn/2) 等四行原样搬入）+ 病态防御。
    返回 (xc2, yc2, wn2, hn2)（归一化、非负宽高）或 None——
    任一维 clamp 后 ≤0（全越界/负宽），或 clamp 后宽/高 < min_norm（碎屑框丢弃）。"""
    x_min = max(0.0 + to_border, xc - wn / 2)
    x_max = min(1.0 - to_border, xc + wn / 2)
    y_min = max(0.0 + to_border, yc - hn / 2)
    y_max = min(1.0 - to_border, yc + hn / 2)
    if x_max <= x_min or y_max <= y_min:
        return None
    if (x_max - x_min) < min_norm or (y_max - y_min) < min_norm:
        return None
    return (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min
