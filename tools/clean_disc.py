"""圆盘目标抠图清理：以外圈"蓝色环"拟合出一个圆，圆外 alpha 归零。

背景：delivery/target 这类圆盘（蓝白同心圆+黑十字）抠图时，四角透明了，
但圆盘上还粘着贴边残留（连通域分不开）。本工具用蓝色环颜色定位圆心/半径，
把圆外（残料、颗粒）一律清透明。

用法：
  python tools/clean_disc.py <输入目录> --out <输出目录>
  python tools/clean_disc.py <目录> --margin 6 --min-blue-hue 85 --max-blue-hue 135
参数（默认）：
  --margin 6          圆外保留余量（px）：蓝环外再保留几像素抗锯齿
  --hsv 蓝阀值：OpenCV H∈[85,135] S>100 V>80（蓝环很蓝时无需调）
  无蓝色检出时回退：重心 + 中位行跨度半径
输出：同名 PNG；并打印每张的拟合圆与清除的 α 像素数。
"""
import os
import sys
import glob
import argparse

import numpy as np
import cv2
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg")


def find_blue_ring(a, rgb):
    """定位蓝色环：(cx, cy, R) or None。取蓝色像素最大连通域的 minEnclosingCircle。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 135)
            & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 80))
    blue = blue.astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(blue, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == idx).astype(np.uint8)
    pts = cv2.findNonZero(m)
    if pts is None or len(pts) < 10:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return int(cx), int(cy), int(r)


def fallback_center_radius(a):
    """无蓝环回退：α>0 重心 + 各方向剖面半径的中位数（稳健于凸块）。"""
    ys, xs = np.where(a > 0)
    if not len(xs):
        return None
    cx, cy = int(xs.mean()), int(ys.mean())
    h, w = a.shape
    rads = []
    for ang in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        r = 0
        while True:
            x, y = int(cx + dx * (r + 1)), int(cy + dy * (r + 1))
            if not (0 <= x < w and 0 <= y < h):
                break
            if a[y, x] == 0:
                break
            r += 1
            if r > 2000:
                break
        rads.append(r)
    return cx, cy, int(np.median(rads))


def clean_one(src, out_dir, margin):
    im = Image.open(src)
    rgba = im.convert("RGBA")
    a = np.array(rgba)[:, :, 3]
    rgb = np.array(rgba)[:, :, :3]

    fit = find_blue_ring(a, rgb)
    if fit is None:
        fit = fallback_center_radius(a)
        method = "fallback(无蓝环)"
    else:
        method = "blue-ring"
    if fit is None:
        print(f"  ⚠️ {os.path.basename(src)}: 无内容，跳过"); return False
    cx, cy, r = fit

    h, w = a.shape
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    keep = dist <= r + margin
    removed = int(((a > 0) & ~keep).sum())
    a_new = np.where(keep, a, 0).astype(np.uint8)

    out = np.dstack([rgb, a_new])
    Image.fromarray(out.astype(np.uint8)).save(
        os.path.join(out_dir, os.path.basename(src)))
    print(f"  {os.path.basename(src)}: {method} 圆(cx={cx},cy={cy},R={r}) margin={margin} "
          f"| 清除 α 像素 {removed} 个")
    return True


def main():
    ap = argparse.ArgumentParser(description="圆盘目标：圆外 alpha 归零（去贴边残留）")
    ap.add_argument("indir", help="输入目录（png）")
    ap.add_argument("--out", default="", help="输出目录（默认 <indir>/clean/）")
    ap.add_argument("--margin", type=int, default=6, help="圆外保留余量 px（默认 6）")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out) if args.out else os.path.join(
        os.path.abspath(args.indir), "clean")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in sorted(glob.glob(os.path.join(args.indir, "*")))
             if f.lower().endswith(IMG_EXT)]
    if not files:
        print(f"❌ {args.indir} 下没有图片"); sys.exit(1)
    print(f"清理 {len(files)} 张 → {out_dir}")
    for f in files:
        clean_one(f, out_dir, args.margin)


if __name__ == "__main__":
    main()
