"""warp 环节最小环切诊断 —— 必须跑到"引擎里实际存在的那行代码",而不是等价重写。

此前修复链条的教训: 用本地 2.13 + 2.4.1-CPU 推断 2.4.1-GPU 行为,且反复"验证重写版"
而非引擎代码,等于没验证。本脚本对同一目标、同一矩阵,直接调用引擎函数,在目标
环境(4090 + torch 2.4.1)上一口气测全部候选路径:

  REF     cv2.warpPerspective 直调        (语义基准)
  GRID_g  grid_sample @ cuda, 无 clamp    ("grid_sample 本身"在 2.4.1 GPU 是否残)
  GRID_c  grid_sample @ cpu,   无 clamp    (对照)
  ENG_CV  引擎 _warp_rgba_cv               (gpu_warp_cv: true 的路径)
  ENG_GV  引擎 _warp_rgba_tensor @ cuda    (gpu_warp_cv: false + grid sample + clamp 的路径)

判据(相对 REF): 内容占比 <0.85 或 bbox 宽高 <0.9×REF 或 缺失/多出像素 ≠0 → ⚠
差值图: 红 = REF 有而本路无(真正的残缺), 绿 = 本路多出(污染/拖尾)。

用法:
  python realscene/diag_warp.py --gpu        # 服务器 4090 上跑这个
  python realscene/diag_warp.py --target xxx.png --out /tmp/diag
"""
import argparse
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "realscene"))
import importlib  # noqa: E402
gpu_engine = importlib.import_module("gpu_engine")
_warp_rgba_tensor = gpu_engine._warp_rgba_tensor
_warp_rgba_cv = gpu_engine._warp_rgba_cv


def make_hm(w, h, angle=35.0, persp=0.10):
    """确定性旋转+透视矩阵(模拟引擎 perspective 增强)。"""
    r = np.radians(angle)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s], [s, c]], np.float32)
    corn = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    src = (((corn - [w / 2, h / 2]) @ R.T + [w / 2, h / 2])).astype(np.float32)
    dst = src.copy()
    dst[1] += [w * 0.5 * persp, -h * 0.3 * persp]
    dst[2] += [0.0, h * 0.2 * persp]
    dst[3] += [-w * 0.3 * persp, h * 0.3 * persp]
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def grid_warp_generic(rgb, alpha, hm, out_w, out_h, device):
    """独立内嵌 grid_sample 实现(无 clamp),与引擎 _warp_rgba_tensor 的坐标换算一致。"""
    hm_inv = np.linalg.inv(hm)
    m = torch.tensor(hm_inv, dtype=torch.float32, device=device)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(device) / 255.0
    a = torch.from_numpy(alpha).float()[None, None].to(device) / 255.0
    h_in, w_in = rgb.shape[:2]
    ys, xs = torch.meshgrid(
        torch.arange(out_h, device=device, dtype=torch.float32),
        torch.arange(out_w, device=device, dtype=torch.float32), indexing="ij")
    ones = torch.ones_like(xs)
    dst = torch.stack([xs, ys, ones], -1).reshape(-1, 3, 1)
    src = (m @ dst).squeeze(-1)
    sx = (src[:, 0] / src[:, 2]) * 2.0 / (w_in - 1) - 1.0
    sy = (src[:, 1] / src[:, 2]) * 2.0 / (h_in - 1) - 1.0
    grid = torch.stack([sx, sy], 1).reshape(1, out_h, out_w, 2)
    rgb_o = F.grid_sample(t, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    a_o = F.grid_sample(a, grid, mode="nearest",
                        padding_mode="zeros", align_corners=False)
    return rgb_o, a_o


def to_np(rgb_o, a_o):
    rgb = (rgb_o[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy())
    a = (a_o[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy())
    return np.ascontiguousarray(rgb), np.ascontiguousarray(a)


def tight_bbox(a):
    ys, xs = np.nonzero(a > 127)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def find_first_target():
    """优先找带透明通道的真实目标(圆形板),边界缺陷只有圆弧上才显形。"""
    cands = []
    for tdir in (os.path.join(_ROOT, "delivery", "target", "clean"),
                 os.path.join(_ROOT, "delivery", "targets")):
        if os.path.isdir(tdir):
            for root, _, files in os.walk(tdir):
                for f in sorted(files):
                    if f.lower().endswith(".png"):
                        cands.append((0, os.path.join(root, f)))
                    elif f.lower().endswith(".jpg"):
                        cands.append((1, os.path.join(root, f)))
    if not cands:
        raise SystemExit("未找到目标图; 用 --target 指定一张")
    return min(cands, key=lambda t: t[0])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None)
    ap.add_argument("--gpu", action="store_true", help="在服务器 4090 上请务必加此参数")
    ap.add_argument("--out", default="diag_warp_out")
    ap.add_argument("--angle", type=float, default=35.0)
    ap.add_argument("--persp", type=float, default=0.10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"torch {torch.__version__} | cv2 {cv2.__version__} | "
          f"cuda_available={torch.cuda.is_available()}")

    tgt = args.target or find_first_target()
    img = cv2.imread(tgt, cv2.IMREAD_UNCHANGED)
    rgb = np.ascontiguousarray(img[..., :3])
    if img.ndim == 2 or img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, np.uint8)
    else:
        alpha = img[..., 3]

    bx, by, bw, bh = tight_bbox(alpha)
    rgb, alpha = rgb[by:by + bh, bx:bx + bw], alpha[by:by + bh, bx:bx + bw]
    h, w = alpha.shape
    hm = make_hm(w, h, args.angle, args.persp)
    out_w, out_h = int(w * 1.6), int(h * 1.6)
    T = np.array([[1, 0, (out_w - w) // 2], [0, 1, (out_h - h) // 2], [0, 0, 1]], np.float32)
    hm_full = T.astype(np.float32) @ hm
    print(f"target: {tgt}  ({w}x{h}), 矩阵: rot={args.angle} persp={args.persp} 输出 {out_w}x{out_h}")

    # ---- REF: cv2 直调 ----
    ref_rgb = cv2.warpPerspective(rgb, hm_full, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    ref_a = cv2.warpPerspective(alpha, hm_full, (out_w, out_h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    ref_mask = ref_a > 127

    runs = [("REF", ref_rgb, ref_a)]

    with torch.inference_mode():
        # GRID_c / GRID_g: grid_sample 本体
        runs.append(("GRID_c", *to_np(*grid_warp_generic(rgb, alpha, hm_full, out_w, out_h, "cpu"))))
        if args.gpu and torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda:0")
            except Exception as e:
                print(f"cuda 冒烟失败, GPU 分支跳过: {e}")
                args.gpu = False
        if args.gpu:
            runs.append(("GRID_g", *to_np(*grid_warp_generic(rgb, alpha, hm_full, out_w, out_h, "cuda:0"))))

        # 引擎真实路径: ENG_CV(引擎封装过的 cv2), ENG_GV(引擎 grid+clamp)
        t_rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        t_a = torch.from_numpy(alpha).float().unsqueeze(0).unsqueeze(0)
        rc, ac = _warp_rgba_cv(t_rgb, t_a, hm_full, out_h, out_w)
        runs.append(("ENG_CV", *to_np(rc / 255.0, ac / 255.0)))
        if args.gpu:
            rg, ag = _warp_rgba_tensor(t_rgb.to("cuda:0"), t_a.to("cuda:0"),
                                       hm_full, out_h, out_w)
            runs.append(("ENG_GV", *to_np(rg / 255.0, ag / 255.0)))

    rows, warns = [], []
    rw_, rh_ = tight_bbox(ref_a)[2:]
    print(f"{'名称':<7} 内容占比  bbox            缺失    多出   ")
    for name, r, a in runs:
        m = a > 127
        ratio = m.sum() / max(ref_mask.sum(), 1)
        bw_, bh_ = tight_bbox(a)[2:]
        missing = int((ref_mask & ~m).sum())
        extra = int((~ref_mask & m).sum())
        # 与 REF 的 RGB 差(双路都有的区域; 反映插值舍入级差异 vs 结构性差异)
        both = m & ref_mask
        rgb_diff = float(np.abs(r.astype(np.int32) - ref_rgb.astype(np.int32))[both].mean()) \
            if both.any() else float("nan")
        # 亚像素级(圆边缘插值取舍 <内容0.5%、bbox 差<1px)视为通过;
        # 真正的残缺/污染(missing 或 extra 大块像素)才标 ⚠
        ok = (ratio >= 0.995 and bw_ >= rw_ - 1 and bh_ >= rh_ - 1
              and missing < 0.005 * ref_mask.sum() and extra < 0.005 * ref_mask.sum())
        flag = "✅" if ok else "⚠ 残缺"
        print(f"{name:<7} {ratio:6.3f}  {bw_}x{bh_}(REF {rw_}x{rh_})  {missing:6d}  "
              f"{extra:6d}  RGB差{rgb_diff:5.1f}  {flag}")
        if not ok:
            warns.append(name)
        diff = np.zeros((out_h, out_w, 3), np.uint8)
        diff[ref_mask & ~m] = (0, 0, 255)
        diff[~ref_mask & m] = (0, 255, 0)
        rows.append((name, np.hstack([r, cv2.cvtColor(np.stack([a] * 3, -1), cv2.COLOR_RGB2BGR),
                                      diff])))

    maxw = max(r.shape[1] for _, r in rows)
    board = []
    for name, r in rows:
        pad = np.zeros((80, r.shape[1], 3), np.uint8)
        cv2.putText(pad, f"{name}  hstack=[rgb|alpha|diff]", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        board.append(np.vstack([pad, r]))
    mont = np.vstack(board)
    cv2.imwrite(os.path.join(args.out, "warp_compare.jpg"), mont,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n蒙太奇: {os.path.join(args.out, 'warp_compare.jpg')}")
    print("解读: 红=REF有而本路无(残缺), 绿=本路多出(污染)。")
    if warns:
        print(f"⚠ 残缺的路: {', '.join(warns)}")
        if "ENG_GV" in warns:
            print("  → 引擎 grid+clamp 在 2.4.1 GPU 上确认异常; 若 ENG_CV ✅ 说明 gpu_warp_cv:true 是对的, 残留问题在别处。")
        if "ENG_CV" in warns:
            print("  → 引擎封装的 cv2 也不对 —— 引擎 _warp_rgba_cv 有封装层 bug(与直调 cv2 不符)。")
        if "GRID_g" in warns and "ENG_GV" in warns:
            print("  → 2.4.1 GPU grid_sample 本体有缺陷(clamp 只是雪上加霜)。")
    else:
        print("✅ 本环境所有候选路径与 cv2 一致 —— 残缺不在 warp, 必须转查 bbox/label/paste/保存环节。")


if __name__ == "__main__":
    main()
