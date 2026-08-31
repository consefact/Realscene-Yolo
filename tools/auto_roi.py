"""自动测量"底板+中心识别区"目标图的 roi 参数（工具脚本，功能归 main 分支）。

场景：目标图 = 外围环/底板（不变） + 中心识别区（换图，真正检测目标）。
期望：贴图用完整图（环保留作上下文），标注只框中心识别区 —— config 的
target_types.<类型>.roi 即此语义（相对紧框的归一化比例）。

素材配套（本仓库 delivery/target 的约定）：
  clean/<类>.png      完整目标图（环+中心，alpha=圆盘），作为目标图使用
  inner/<类>.png      仅中心识别区（alpha 只覆盖内容），与 clean **同画布尺寸**
                      且坐标系一致 → 其 alpha bbox 就是中心识别区位置
  inner/<类>_overlay.jpg   clean+红圈的人工验证图

脚本做法：
  inner.<类>.png 的 alpha>0 内容包围盒 ÷ clean 画布尺寸 = roi [rx, ry, rw, rh]
  （对圆形底板，紧框=alpha bbox≈整图，故 ROI 相对整图 ≡ 相对紧框）
  无 inner 的类（如 H/cross，无外围靶标）→ roi: null，整图即目标。

用法：
  python tools/auto_roi.py --classes pillbox tent tank car bridge cross H \
      --no-roi H cross --clean-dir <clean> --inner-dir <inner> \
      --out_yaml /tmp/roi_params.yaml

输出：打印每类 roi 向量 + 可直接粘入 config.yaml 的 target_types 片段（每类一个
类型，因为引擎的一个类型只带一份 roi；同 roi 的类可合并成目录形式）。
"""
import argparse
import json
import os

import cv2
import numpy as np


def alpha_bbox(alpha):
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def analyze_roi(full_path, inner_path, scale=1.0):
    """返回 roi=[rx,ry,rw,rh]（相对整图画布）。full 用于校验尺寸一致性。"""
    full = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    if full is None:
        raise ValueError(f"无法读取完整目标图: {full_path}")
    inner = cv2.imread(inner_path, cv2.IMREAD_UNCHANGED)
    if inner is None:
        raise ValueError(f"无法读取中心识别区图: {inner_path}")

    h, w = inner.shape[:2]
    fh, fw = full.shape[:2]
    if (h, w) != (fh, fw):
        print(f"  ⚠️ 画布不一致：inner {w}x{h} vs full {fw}x{fh}，以 full 为准（inner 视为等画布平移）")
    a = inner[:, :, 3] if inner.ndim == 3 and inner.shape[2] == 4 else None
    if a is None:
        # 无 alpha：退化为整图即目标
        print("  ⚠️ inner 无 alpha 通道，返回整图 roi")
        return [0.0, 0.0, 1.0, 1.0], (0, 0, w, h)
    bb = alpha_bbox(a)
    if bb is None:
        print("  ⚠️ inner 全透明")
        return None, None
    x0, y0, x1, y1 = bb
    hh, ww = full.shape[:2]
    roi = [x0 / ww, y0 / hh, (x1 - x0) / ww, (y1 - y0) / hh]
    return [round(v, 4) for v in roi], bb


def main():
    ap = argparse.ArgumentParser(description="推算底板+中心识别区目标图的 roi 参数")
    ap.add_argument("--classes", nargs="+", required=True, help="类别列表（需与 config.classes 一致）")
    ap.add_argument("--no-roi", nargs="*", default=[], help="不作 roi 的类（整图即目标）")
    ap.add_argument("--clean-dir", required=True, help="完整目标图目录（含 <类>.png）")
    ap.add_argument("--inner-dir", required=True, help="中心识别区目录（含 <类>.png，与 clean 同画布）")
    ap.add_argument("--out-json", default="", help="输出 roi 明细 json")
    ap.add_argument("--out-yaml", default="", help="输出 config target_types 片段 yaml")
    args = ap.parse_args()

    results = {}
    for c in args.classes:
        full = os.path.join(args.clean_dir, c + ".png")
        inner = os.path.join(args.inner_dir, c + ".png")
        if not os.path.exists(full):
            print(f"{c:10s} 无 clean 图 → 跳过")
            results[c] = None
            continue
        if not os.path.exists(inner):
            print(f"{c:10s} 无 inner 图 → 整图即目标 (roi: null)")
            results[c] = None
            continue
        full_img = cv2.imread(full, cv2.IMREAD_UNCHANGED)
        roi, bb = analyze_roi(full, inner)
        results[c] = None if (c in args.no_roi) else roi
        if c in args.no_roi:
            print(f"{c:10s} 整图即目标 (roi: null)")
        else:
            print(f"{c:10s} 中心bbox(px)={bb} 画布={full_img.shape[1]}x{full_img.shape[0]} "
                  f"roi={roi}")

    # target_types 片段：每类一个 type（roi 不同则必须分开）
    if args.out_yaml:
        classdir = os.path.join(args.clean_dir, "{cls}")
        # 实际建议：把 clean 图按类分子目录放或直接沿用现在目录
        # 我们只输出 yaml 结构，目录由用户在 config 里定
        lines = ["# 自动生成的 target_types 片段（粘贴进 config.yaml synth.target_types）"]
        lines.append("# 背景：贴图=clean 全图（环保留），标注=<roi> 只框中心识别区")
        lines.append("# H/cross 无环：roi: null 整图即目标，无需为它们建单独类型。")
        lines.append("# 引擎仅按类拆分：每个 roi 值一个 target_types 条目。")
        lines.append("target_types:")
        for c in args.classes:
            if results.get(c) is None:
                continue
            rx, ry, rw, rh = [float(v) for v in results[c]]
            lines.append(f"  roi_{c}:")
            lines.append(f"    dir: ./delivery-3/target/{c}")
            lines.append(f"    roi: [{rx}, {ry}, {rw}, {rh}]")
            lines.append("    scale: [0.3, 1.2]")
            lines.append("    rotate: [-180, 180]")
            lines.append("    perspective: [0.5, 0.20]")
            lines.append("    feather: 4")
            lines.append("    crop_transparent: true")
        yaml_text = "\n".join(lines)
        with open(args.out_yaml, "w", encoding="utf-8") as f:
            f.write(yaml_text + "\n")
        print(f"\n已写出片段: {args.out_yaml}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
