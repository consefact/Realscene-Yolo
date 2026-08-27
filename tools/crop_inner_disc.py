"""去掉外圈标靶（蓝白环+十字），只保留中间黑边内的实际图片。

原理：中心内容（深色圆盘/黑线环/椭圆环）外侧总有一圈粗白带（标靶最内圈），
检测"暗环→外侧粗白带"过渡，收集边界点 → 鲁棒 fitEllipse（圆是椭圆特例，
圆/椭圆通吃），椭圆内 alpha 保留、外全透明。

用法：
  python tools/crop_inner_disc.py <目标目录> --out <输出目录>
  python tools/crop_inner_disc.py delivery/target --out delivery/target/inner
  # 全图或指定文件均可：--files bridge,car 或 --skip house,cross

注意：输出不透明区铺白（透明=0 保留alpha），与 target_types.crop_transparent 兼容。
"""
import os, glob, argparse
import numpy as np, cv2
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg")


def load_rgba(p):
    arr = np.array(Image.open(p).convert("RGBA"))
    return arr[:, :, :3], arr[:, :, 3]


def black_ring_boundary(rgb, alpha, cx, cy):
    """每条射线：找第一处「暗环 run → 其后 ≤8px 内的 ≥12px 粗白带」→ 返回黑边外缘半径对应点。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    h, w = alpha.shape
    maxr = int(min(cx, cy, w - cx, h - cy)) - 2
    r0 = int(maxr * 0.10)
    pts = []
    for ang in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        rr = np.arange(r0, maxr)
        xs = np.round(cx + dx * rr).astype(int)
        ys = np.round(cy + dy * rr).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        rr, xs, ys = rr[ok], xs[ok], ys[ok]
        if len(rr) == 0:
            continue
        av = alpha[ys, xs]; g = gray[ys, xs]; vv = V[ys, xs]; ss = S[ys, xs]
        stop = np.where(av == 0)[0]
        end = stop[0] if len(stop) else len(rr)
        g, vv, ss, rr = g[:end], vv[:end], ss[:end], rr[:end]
        n = len(rr)
        dark = g < 95
        whiteband = (vv > 200) & (ss < 40)
        k = 0; boundary = None
        while k < n:
            if dark[k]:
                j = k
                while j < n and dark[j]:
                    j += 1
                t = j; ws = None
                while t < n and t - j <= 8:
                    if whiteband[t]:
                        ws = t; break
                    t += 1
                if ws is not None:
                    u = ws; wl = 0
                    while u < n and whiteband[u]:
                        wl += 1; u += 1
                    if wl >= 12:
                        boundary = rr[j - 1]
                        break
                k = j
            else:
                k += 1
        if boundary:
            pts.append((cx + dx * boundary, cy + dy * boundary))
    return np.array(pts, dtype=np.float32)


def robust_fit(pts, cx, cy):
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    med = np.median(r)
    inl = pts[(r > 0.8 * med) & (r < 1.2 * med)]
    if len(inl) < 10:
        inl = pts
    return cv2.fitEllipse(inl.astype(np.float32)), len(inl), len(pts)


def crop_one(p, out_dir):
    rgb, alpha = load_rgba(p)
    h, w = alpha.shape
    ys, xs = np.where(alpha > 0)
    if not len(xs):
        print(f"  ⚠️ {os.path.basename(p)}: 全透明，跳过"); return
    cx, cy = float(xs.mean()), float(ys.mean())
    for _ in range(2):  # 用拟合中心再拟合一次
        pts = black_ring_boundary(rgb, alpha, cx, cy)
        if len(pts) < 10:
            print(f"  ⚠️ {os.path.basename(p)}: 黑边边界点不足，跳过"); return
        ell, ni, nt = robust_fit(pts, cx, cy)
        (cx, cy), (MA, ma), angle = ell
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)), angle, 0, 360, 255, -1)
    result = np.dstack([rgb, np.where(mask > 0, alpha, 0).astype(np.uint8)])
    out = os.path.join(out_dir, os.path.basename(p))
    Image.fromarray(result).save(out)
    print(f"  {os.path.basename(p)}: 椭圆中心=({cx:.0f},{cy:.0f}) "
          f"轴=({MA:.0f}x{ma:.0f}) 角={angle:.1f} 内点{ni}/{nt} → {out}")


def main():
    ap = argparse.ArgumentParser(description="去掉标靶环，只留中间黑边内图片（圆/椭圆）")
    ap.add_argument("indir", help="输入目录")
    ap.add_argument("--out", default="", help="输出目录（默认 <indir>/inner/）")
    ap.add_argument("--files", default="", help="仅处理: bridge,car (逗号分隔，默认全部) ")
    ap.add_argument("--skip", default="", help="跳过: house,cross")
    ap.add_argument("--viz", action="store_true", help="额外输出 _overlay.jpg 拟合可视化")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out) if args.out else os.path.join(
        os.path.abspath(args.indir), "inner")
    os.makedirs(out_dir, exist_ok=True)

    all_files = [f for f in sorted(glob.glob(os.path.join(args.indir, "*")))
                 if f.lower().endswith(IMG_EXT)]
    if args.files:
        want = {s.strip() for s in args.files.split(",") if s.strip()}
        all_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in want]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        all_files = [f for f in all_files
                     if os.path.splitext(os.path.basename(f))[0] not in skip]
    if not all_files:
        print("❌ 没有待处理图片（检查 --files/--skip）"); return

    print(f"处理 {len(all_files)} 张 → {out_dir}")
    for f in all_files:
        crop_one(f, out_dir)
        if args.viz:
            rgb, alpha = load_rgba(f)
            im = Image.open(f).convert("RGBA")
            arr = np.array(im)
            a = arr[:, :, 3]
            ys, xs = np.where(a > 0)
            cx, cy = float(xs.mean()), float(ys.mean())
            for _ in range(2):
                pts = black_ring_boundary(rgb, alpha, cx, cy)
                ell, _, _ = robust_fit(pts, cx, cy)
                (cx, cy), (MA, ma), angle = ell
            ov = rgb.copy()
            cv2.ellipse(ov, (int(cx), int(cy)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (255, 0, 0), 5)
            Image.fromarray(ov).save(os.path.join(
                out_dir, os.path.splitext(os.path.basename(f))[0] + "_overlay.jpg"))


if __name__ == "__main__":
    main()
