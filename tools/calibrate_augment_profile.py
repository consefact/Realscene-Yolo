#!/usr/bin/env python3
"""根据真实标注图自动标定增强参数（数据驱动的"增强往哪调"工具）。

思路（移植自参考实现）：从带 YOLO 标注的真实图中裁出目标 Patch，统计亮度/清晰度/
噪声/压缩痕迹/色偏等分布 → 推荐出该任务的增强参数（概率+范围）；compare 模式再
把合成域 profile 与真实域对齐（align_strength + blend_value），输出覆盖 YAML。

输出 YAML 可直接放入 config.yaml 的 synth.profile（config.py 加载时与 synth.aug 深合并）：
  aug:         本仓库合成引擎能直接消费的项（brightness/contrast/hsv/gaussian 等）
  strategy:    策略提示（困难样本比例等，作用于人工决策）
  calibration: 统计摘要 + 原始推荐值（含本仓库菜单尚未实现的项，如 gamma/shadow/jpeg）

用法：
  # 只用真实域标定（推荐，反正增强目标是逼近真实场景）
  python tools/calibrate_augment_profile.py --mode single \
      --images_dir seven/real/images --labels_dir seven/real/labels \
      --out_yaml profiles/seven_aug.yaml

  # 真实域 vs 合成域对齐
  python tools/calibrate_augment_profile.py --mode compare \
      --images_dir seven/real/images --labels_dir seven/real/labels \
      --synthetic_images_dir seven/train_output --synthetic_labels_dir seven/train_output \
      --align_strength 1.0 --out_yaml profiles/seven_aug.yaml

之后把 config.yaml 的 synth.profile 设为 ./profiles/seven_aug.yaml 即生效。
"""
import argparse
import copy
import json
import os
import random
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class PatchStats:
    mean_luma: float
    std_luma: float
    lap_var: float
    noise_std: float
    blockiness: float
    mean_h: float
    mean_s: float
    mean_v: float
    illum_std: float


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def blend_value(src, dst, strength):
    t = float(clamp(strength, 0.0, 1.0))
    if is_number(src) and is_number(dst):
        return float(src) + (float(dst) - float(src)) * t
    if isinstance(src, list) and isinstance(dst, list) and len(src) == len(dst):
        if all(is_number(a) and is_number(b) for a, b in zip(src, dst)):
            return [blend_value(a, b, t) for a, b in zip(src, dst)]
    return copy.deepcopy(dst if t >= 0.5 else src)


# ============================================================================
# 统计
# ============================================================================

def resolve_dirs(images_dir, labels_dir):
    """图片与标注目录：支持 images/labels 并列结构，或同一目录（flat 图+txt）。"""
    images_dir = os.path.abspath(images_dir)
    if not labels_dir:
        labels_dir = images_dir   # flat：图片与 txt 同目录
    else:
        labels_dir = os.path.abspath(labels_dir)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images 目录不存在: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels 目录不存在: {labels_dir}")
    return images_dir, labels_dir


def list_label_files(labels_dir):
    out = []
    for base, _, files in os.walk(labels_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(base, fn))
    return out


def find_image_for_label(label_path, labels_dir, images_dir):
    rel = os.path.relpath(label_path, labels_dir)
    stem = os.path.splitext(rel)[0]
    for ext in IMAGE_EXTS:
        cand = os.path.join(images_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def _is_float_token(s):
    try:
        float(s)
        return True
    except Exception:
        return False


def load_yolo_boxes(label_path):
    """只解析框坐标，不依赖类别字段（class xc yc w h / xc yc w h 均兼容）。"""
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            xc = yc = bw = bh = None
            if len(parts) >= 4 and all(_is_float_token(x) for x in parts[:4]):
                xc, yc, bw, bh = map(float, parts[:4])
            elif len(parts) >= 5 and all(_is_float_token(x) for x in parts[1:5]):
                xc, yc, bw, bh = map(float, parts[1:5])
            else:
                nums = [float(x) for x in parts if _is_float_token(x)]
                if len(nums) >= 4:
                    xc, yc, bw, bh = nums[:4]
            if xc is None:
                continue
            if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(bw) and np.isfinite(bh)):
                continue
            boxes.append((xc, yc, bw, bh))
    return boxes


def extract_patch(img, box, expand=0.05, min_box=24):
    h, w = img.shape[:2]
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    if (x2 - x1) < min_box or (y2 - y1) < min_box:
        return None
    ex, ey = (x2 - x1) * expand, (y2 - y1) * expand
    x1 = int(clamp(np.floor(x1 - ex), 0, w - 1))
    y1 = int(clamp(np.floor(y1 - ey), 0, h - 1))
    x2 = int(clamp(np.ceil(x2 + ex), 1, w))
    y2 = int(clamp(np.ceil(y2 + ey), 1, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def calc_blockiness(gray):
    g = gray.astype(np.float32)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 1.0
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    all_x = float(dx.mean() + 1e-6)
    all_y = float(dy.mean() + 1e-6)
    bx = float(dx[:, 7::8].mean()) if dx.shape[1] > 8 else all_x
    by = float(dy[7::8, :].mean()) if dy.shape[0] > 8 else all_y
    return 0.5 * (bx / all_x + by / all_y)


def calc_patch_stats(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    mean_luma = float(gray_f.mean() / 255.0)
    std_luma = float(gray_f.std() / 255.0)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F)
    lap_var = float(lap.var())
    smooth = cv2.GaussianBlur(gray_f, (3, 3), 0)
    residual = gray_f - smooth
    noise_std = float(residual.std() / 255.0)
    blockiness = calc_blockiness(gray)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_h = float(hsv[:, :, 0].mean() / 180.0)
    mean_s = float(hsv[:, :, 1].mean() / 255.0)
    mean_v = float(hsv[:, :, 2].mean() / 255.0)
    illum_std = std_luma
    return PatchStats(mean_luma, std_luma, lap_var, noise_std, blockiness,
                      mean_h, mean_s, mean_v, illum_std)


def pctl(arr, q):
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(np.asarray(arr), q))


def summarize_stats(stats_list):
    names = ("mean_luma", "std_luma", "lap_var", "noise_std", "blockiness",
             "mean_h", "mean_s", "mean_v", "illum_std")
    out = {}
    for n in names:
        vals = [getattr(s, n) for s in stats_list]
        out[n] = {
            "p5": round(pctl(vals, 5), 4),
            "p50": round(pctl(vals, 50), 4),
            "p95": round(pctl(vals, 95), 4),
        }
    return out


def collect_stats(images_dir, labels_dir, max_samples, expand, min_box, seed):
    random.seed(seed)
    label_files = list_label_files(labels_dir)
    if not label_files:
        raise RuntimeError(f"未找到标注文件: {labels_dir}")
    random.shuffle(label_files)
    boxes = []
    for lp in label_files:
        ip = find_image_for_label(lp, labels_dir, images_dir)
        if ip is None:
            continue
        try:
            img = cv2.imread(ip)
        except Exception:
            continue
        if img is None:
            continue
        for b in load_yolo_boxes(lp):
            patch = extract_patch(img, b, expand=expand, min_box=min_box)
            if patch is not None:
                boxes.append(patch)
        if len(boxes) >= max_samples:
            break
    if not boxes:
        raise RuntimeError(f"未统计到任何足够大的标注框: {labels_dir}（min_box={min_box}）")
    print(f"  已统计 {len(boxes)} 个 patch（{images_dir}）")
    return [calc_patch_stats(p) for p in boxes], summary_note(boxes)


def summary_note(boxes):
    sizes = [p.shape[0] * p.shape[1] for p in boxes]
    return {"n_patch": len(boxes), "avg_px": int(np.mean(sizes))}


# ============================================================================
# 推荐 profile（输出适配本仓库 synth.aug 菜单 schema）
# ============================================================================

def recommend_menu(stats_list):
    """把统计分布映射为本仓库菜单能直接消费的项（概率+范围）。"""
    mean_l = [s.mean_luma for s in stats_list]
    noise_v = [s.noise_std for s in stats_list]
    lap_v = [s.lap_var for s in stats_list]
    h_v = [s.mean_h for s in stats_list]
    s_v = [s.mean_s for s in stats_list]
    v_v = [s.mean_v for s in stats_list]

    m5, m50, m95 = pctl(mean_l, 5), pctl(mean_l, 50), pctl(mean_l, 95)
    n50, n95 = pctl(noise_v, 50), pctl(noise_v, 95)
    l25, l50 = pctl(lap_v, 25), pctl(lap_v, 50)
    sh90 = pctl([s.illum_std for s in stats_list], 90)
    b90 = pctl([s.blockiness for s in stats_list], 90)

    # 亮度差（±）→ 乘性因子
    low_b = round(clamp((m5 - m50) * 255.0 * 1.6, -60, -6), 1)
    high_b = round(clamp((m95 - m50) * 255.0 * 1.6, 6, 60), 1)
    bright_lo = round(clamp(1.0 + low_b / 100.0, 0.6, 1.35), 2)
    bright_hi = round(clamp(1.0 + high_b / 100.0, 0.6, 1.35), 2)

    dyn = float(clamp(pctl(mean_l, 90) - pctl(mean_l, 10), 0.05, 0.45))
    c_span = float(clamp(0.12 + dyn * 0.7, 0.12, 0.42))
    contrast_range = [round(1.0 - c_span, 2), round(1.0 + c_span, 2)]

    hsv_delta = round(clamp(max(h_v) * 0.28, 0.03, 0.2), 3)
    hsv_s = round(clamp(np.std(s_v) * 2.8, 0.04, 0.30), 3)
    hsv_v = round(clamp(np.std(v_v) * 2.8, 0.03, 0.25), 3)
    hsv_delta = round(clamp(max(hsv_delta, hsv_s, hsv_v), 0.03, 0.2), 3)

    noise_lo = round(clamp(n50 * 0.75, 0.003, 0.12), 3)
    noise_hi = round(clamp(max(noise_lo + 0.008, n95 * 1.35), 0.01, 0.18), 3)

    hard_flags = sum(1 for s in stats_list
                     if s.mean_luma < 0.25 or s.noise_std > 0.06
                     or s.lap_var < 35 or s.blockiness > 1.3)
    hard_ratio = hard_flags / max(1, len(stats_list))
    aug_num_max = 6 if hard_ratio >= 0.18 else 5
    aug_extra_prob = round(clamp(0.20 + hard_ratio * 0.65, 0.2, 0.65), 2)
    final_noise_prob = round(clamp(0.25 + n50 * 8.0, 0.2, 0.85), 2)

    menu = {
        "base": {
            "num": [2, aug_num_max],
            "extra_prob": aug_extra_prob,
            "menu": [
                {"type": "brightness", "prob": 0.5, "range": [bright_lo, bright_hi]},
                {"type": "contrast", "prob": 0.5, "range": contrast_range},
                {"type": "color", "prob": 0.5, "range": [0.5, 1.5]},
                {"type": "hsv", "prob": 0.5, "delta": round(max(hsv_delta, hsv_s, hsv_v), 3)},
                {"type": "cutout", "prob": 0.5, "mask_size": [50, 100], "num_masks": [1, 5]},
            ],
        },
        "final": {
            "menu": [
                {"type": "gaussian", "prob": final_noise_prob, "var": [
                    round(noise_lo * 255.0 * 255.0, 1), round(noise_hi * 255.0 * 255.0, 1)]},
                {"type": "salt_pepper", "prob": 0.3, "amount": [0.001, 0.005]},
                {"type": "poisson", "prob": 0.3, "intensity": 0.1},
            ],
        },
    }
    raw = {
        "brightness_range_pc": [low_b, high_b],
        "contrast_range": contrast_range,
        "hsv_delta_suggested": [round(hsv_delta, 3), round(hsv_s, 3), round(hsv_v, 3)],
        "noise_factor_range": [noise_lo, noise_hi],
        "final_gaussian_noise_prob": final_noise_prob,
        "final_sharpen_prob": round(clamp(0.75 - max(0.0, (70.0 - l25) / 180.0), 0.25, 0.8), 2),
        "blur_kernel_suggested": [3, 5, 7] if l50 < 120 else [3, 5],
        "jpeg_artifact_quality_suggested": [45, 90] if b90 > 1.3 else [55, 95],
        "shadow_shade_suggested": [round(clamp(0.9 - sh90 * 0.7, 0.55, 0.9), 2), 0.98],
        "gamma_suggested": [round(1.0 - clamp(0.1 + dyn * 0.5, 0.1, 0.35), 2),
                            round(1.0 + clamp(0.1 + dyn * 0.5, 0.1, 0.35), 2)],
        "aug_num_max": aug_num_max,
        "aug_extra_prob": aug_extra_prob,
    }
    return menu, raw, hard_ratio


def build_profile(stats_list, align_strength=1.0, synth_menu=None, synth_raw=None,
                 synth_hard=None, source="domain"):
    """生成 profile。synth_* 提供时走 compare 对齐（合成域向真实域混合）。"""
    menu, raw, hard_ratio = recommend_menu(stats_list)

    if synth_menu is not None and align_strength > 0:
        # 逐项浅混合：合成域为基线，按 align_strength 向真实域靠拢
        def align_nums(sv, rv):
            return blend_value(sv, rv, align_strength)

        for layer in ("base", "final"):
            sm = synth_menu.get(layer, {}).get("menu", [])
            rm = menu.get(layer, {}).get("menu", [])
            by_type = {i["type"]: i for i in rm}
            merged = []
            for sv in sm:
                rv = by_type.get(sv["type"])
                if rv is None or sv.get("prob") is None:
                    merged.append(sv)
                    continue
                item = dict(sv)
                for key, vals in sv.items():
                    if key in ("type", "prob") or key not in rv:
                        continue
                    svv, rvv = vals, rv[key]
                    if (isinstance(svv, (list, tuple)) and isinstance(rvv, (list, tuple))
                            and len(svv) == 2 and len(rvv) == 2
                            and all(is_number(x) for x in svv) and all(is_number(x) for x in rvv)):
                        item[key] = [align_nums(svv[j], rvv[j]) for j in range(2)]
                    elif is_number(svv) and is_number(rvv):
                        item[key] = align_nums(svv, rvv)
                merged.append(item)
            # 补上合成域没有但真实域推荐的类型
            have = {i["type"] for i in merged}
            for rv in rm:
                if rv["type"] not in have:
                    merged.append(copy.deepcopy(rv))
            menu[layer] = {k: v for k, v in synth_menu.get(layer, {}).items() if k != "menu"}
            menu[layer]["menu"] = merged
            # 只在合成域层本身带 num（抽 N 机制）时保留；final 层保持独立概率语义
            if "num" in synth_menu.get(layer, {}):
                menu[layer]["num"] = synth_menu[layer]["num"]
        raw = None  # compare 模式 raw 仅供参考，不混
        hard_ratio = float(blend_value(synth_hard or 0.0, hard_ratio, align_strength))

    profile = {
        "aug": menu,
        "strategy": {
            "hard_case_ratio": round(hard_ratio, 4),
            "recommend_hard_case_sampling": hard_ratio > 0.15,
            "notes": [
                "基础增强用于覆盖常规光照/清晰度分布",
                "困难样本（低照度/高噪声/强压缩）比例高时，建议保留 10-20% 真实现场图进验证",
            ],
        },
        "calibration": {
            "source": source,
            "align_strength": round(float(clamp(align_strength, 0.0, 1.0)), 3),
            "raw_recommended": raw if raw is not None else {},
        },
    }
    return profile


def main():
    ap = argparse.ArgumentParser(description="根据现场标注图自动标定增强参数")
    ap.add_argument("--mode", choices=["single", "compare"], required=True)
    ap.add_argument("--images_dir", required=True, help="带标注的现场图目录（images/ 或 flat）")
    ap.add_argument("--labels_dir", default="", help="标注 txt 目录（flat 时省略）")
    ap.add_argument("--synthetic_images_dir", default="", help="合成域图目录（compare 必填）")
    ap.add_argument("--synthetic_labels_dir", default="", help="合成域标注目录（flat 时省略）")
    ap.add_argument("--max_samples", type=int, default=3000)
    ap.add_argument("--min_box", type=int, default=24)
    ap.add_argument("--expand", type=float, default=0.05)
    ap.add_argument("--align_strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_yaml", default="profiles/aug_profile.auto.yaml")
    args = ap.parse_args()

    images_dir, labels_dir = resolve_dirs(args.images_dir, args.labels_dir)

    if args.mode == "compare":
        s_img, s_lab = resolve_dirs(args.synthetic_images_dir, args.synthetic_labels_dir)
        real_stats, _ = collect_stats(images_dir, labels_dir, args.max_samples,
                                      args.expand, args.min_box, args.seed)
        synth_stats, _ = collect_stats(s_img, s_lab, args.max_samples,
                                       args.expand, args.min_box, args.seed)
        sm, sraw, shard = recommend_menu(synth_stats)
        profile = build_profile(real_stats, align_strength=args.align_strength,
                                synth_menu=sm, synth_raw=sraw, synth_hard=shard,
                                source="compare(real<-synth)")
    else:
        stats, _ = collect_stats(images_dir, labels_dir, args.max_samples,
                                 args.expand, args.min_box, args.seed)
        profile = build_profile(stats, source="single(real)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_yaml)), exist_ok=True)
    with open(args.out_yaml, "w", encoding="utf-8") as f:
        f.write("# 由 tools/calibrate_augment_profile.py 自动生成（数据驱动增强标定）\n"
                f"# 用法：config.yaml 的 synth.profile 设为 {args.out_yaml}（相对项目根）即覆盖 aug 段\n")
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)
    print(f"profile → {args.out_yaml}")

    rep = {
        "profile_path": os.path.abspath(args.out_yaml),
        "mode": args.mode,
        "align_strength": args.align_strength,
        "calibration": profile["calibration"],
        "strategy": profile["strategy"],
    }
    rep_path = os.path.splitext(args.out_yaml)[0] + "_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"report → {rep_path}")


if __name__ == "__main__":
    main()
