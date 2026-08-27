"""给带透明通道的 PNG 铺上纯色背景（转成不透明 JPG/PNG）。

用途：把抠好的透明图平铺成"纯色背景 + 物体"的素材图，
方便作为 target（整图即目标）、或导出给不支持透明的工具/场景。
透明区域填纯色；物体区域保持原像素。

用法：
  python tools/flatten_png.py <path>                     # path 可为单张图或目录
  python tools/flatten_png.py <dir> --bg 40,110,180      # 指定背景色 (R,G,B)
  python tools/flatten_png.py <dir> --random             # 每张随机柔和彩色底（--bg 互斥）
  python tools/flatten_png.py <dir> --out <输出目录>       # 指定输出位置
输出：同名 .jpg（不透明），写入 <out>（默认 <path 同目录>/flat/）。
"""
import os
import sys
import random
import time
import argparse

# --- 载入统一配置（相对路径项目根）---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from PIL import Image
import colorsys

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def random_bg_rgb():
    """随机柔和彩色背影：H 任意、S/V 适中（分散、不刺眼、不极端暗）。"""
    h, s, v = random.random(), random.uniform(0.3, 0.9), random.uniform(0.4, 0.95)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def random_gray_rgb():
    """随机黑白灰背影：R=G=B，明度均匀（含极暗/极亮，模拟不同明度板面）。"""
    g = int(round(random.uniform(0.05, 0.95) * 255))
    return (g, g, g)


def flatten_one(src, bg_rgb, out_dir, stem=""):
    """单张：透明区域铺纯色 → 存 JPG。无 alpha 的图直接复制。"""
    with Image.open(src) as im:
        base = im.convert("RGBA")
        bg = Image.new("RGB", base.size, bg_rgb)
        bg.paste(base, mask=base.getchannel("A"))   # alpha=0 → 纯色；alpha 半透 → 色+图过度
        name = stem if stem else os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(out_dir, name + ".jpg")
        bg.save(out, quality=95)
        return out


def main():
    ap = argparse.ArgumentParser(description="给带透明通道的图片铺纯色背景（转不透明）")
    ap.add_argument("path", help="单张图片 / 目录（递归）")
    ap.add_argument("--bg", default="200,200,200", help="背景色 R,G,B（默认 200,200,200 浅灰）")
    ap.add_argument("--random", dest="random_bg", action="store_true",
                    help="每张图随机柔和彩色底（与 --bg 互斥；可与 --gray 联合）")
    ap.add_argument("--gray", dest="random_gray", action="store_true",
                    help="随机底改为黑白灰（R=G=B，需与 --random 联合使用）")
    ap.add_argument("--count", type=int, default=1,
                    help="每张原图生成 N 个不同随机底版本（仅 --random 下有意义，默认 1）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（--random 时可用，便于复现）")
    ap.add_argument("--out", default="", help="输出目录（默认 <path同目录>/flat/）")
    args = ap.parse_args()

    if args.random_bg and args.bg != "200,200,200":
        print("❌ --random 与 --bg 互斥，只能二选一。")
        sys.exit(1)
    if args.random_gray and not args.random_bg:
        print("❌ --gray 需与 --random 联合使用（要固定灰则用 --bg 80,80,80）。")
        sys.exit(1)
    if args.count > 1 and not args.random_bg:
        print(f"❌ --count {args.count} 只在 --random 下有意义（固定色重复产出相同内容）。")
        sys.exit(1)

    bg_rgb = None
    if not args.random_bg:
        try:
            bg_rgb = tuple(int(x) for x in args.bg.split(","))
            assert len(bg_rgb) == 3
        except Exception:
            print(f"❌ --bg 格式错误，应为 R,G,B（如 40,110,180），收到：{args.bg}")
            sys.exit(1)
    else:
        random.seed(args.seed)
        print("模式：每张随机背景色（HSV 柔和采样）" if not args.random_gray
              else "模式：每张随机背景色（黑白灰）")

    src = os.path.abspath(args.path)
    if os.path.isfile(src):
        files = [src]
    elif os.path.isdir(src):
        files = [os.path.join(r, f) for r, _, fs in os.walk(src)
                 for f in fs if f.lower().endswith(IMG_EXT)]
    else:
        print(f"❌ 输入不存在：{src}")
        sys.exit(1)

    out_dir = os.path.abspath(args.out) if args.out else os.path.join(
        os.path.dirname(src.rstrip(os.sep)), "flat")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    n_ok = n_skip = 0
    for f in files:
        try:
            with Image.open(f) as im:
                if im.mode not in ("RGBA", "LA") and "transparency" not in im.info:
                    print(f"  ⏭ 跳过（无透明通道）：{os.path.basename(f)}")
                    n_skip += 1
                    continue
            stem = os.path.splitext(os.path.basename(f))[0]
            # --count>1：这 N 个随机版本放进以原图命名的子文件夹
            dest = out_dir
            if args.count > 1:
                dest = os.path.join(out_dir, stem)
                os.makedirs(dest, exist_ok=True)
            for k in range(args.count):
                if args.random_gray:
                    color = random_gray_rgb()
                elif args.random_bg:
                    color = random_bg_rgb()
                else:
                    color = bg_rgb
                name = f"{stem}_{k}" if args.count > 1 else ""
                flatten_one(f, color, dest, stem=name)
                n_ok += 1
        except Exception as e:
            print(f"  ❌ {os.path.basename(f)}: {e}")

    dt = time.perf_counter() - t0
    rate = n_ok / dt if dt > 0 else 0.0
    print(f"完成：铺底 {n_ok} 张 → {out_dir}（用时 {dt:.1f}s，{rate:.1f} 张/s）"
          + (f"，跳过无透明 {n_skip}" if n_skip else ""))


if __name__ == "__main__":
    main()
