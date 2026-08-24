"""把 X-AnyLabeling / LabelMe 的 JSON 标注转成 YOLO txt，并把 图片+txt 放到目标目录。

- 类名映射用 config.yaml 的 classes（label 要与之一致，大小写不敏感兜底）。
- 矩形/多边形都按所有点的 min/max 取外接框 → 归一化 YOLO 框。
- 每张有 json 的图都会输出（无框的 json → 空标签，即背景负样本）。

用法:
  python tools/json2yolo.py <图片目录>              # → 输出到 config.finetune.real_dir
  python tools/json2yolo.py <图片目录> --out <目标目录>   # 可覆盖输出位置
"""
import os
import sys
import json
import glob
import shutil
import argparse
from collections import defaultdict

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config

CFG = load_config()
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def img_size(path, j):
    """优先用 json 里的宽高，缺失则打开图片读取。"""
    w, h = j.get("imageWidth"), j.get("imageHeight")
    if w and h:
        return int(w), int(h)
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def main():
    ap = argparse.ArgumentParser(description="LabelMe/X-AnyLabeling JSON → YOLO txt")
    ap.add_argument("source", help="含 图片 + .json 的目录")
    ap.add_argument("--out", default=CFG.finetune.real_dir, help="输出目录（默认 config.finetune.real_dir）")
    ap.add_argument("--clean", action="store_true", help="先清空输出的 images/labels（避免残留过时标注）")
    ap.add_argument("--bg-unlabeled", dest="bg_unlabeled", action="store_true",
                    help="把没有 json 的图当空背景负样本（空标签）一并纳入")
    args = ap.parse_args()

    classes = list(CFG.classes)
    idx = {c: i for i, c in enumerate(classes)}
    idx_lower = {c.lower(): i for i, c in enumerate(classes)}

    img_out = os.path.join(args.out, "images")
    lab_out = os.path.join(args.out, "labels")
    if args.clean:
        for dd in (img_out, lab_out):
            if os.path.isdir(dd):
                shutil.rmtree(dd)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(args.source, "*.json")))
    if not jsons:
        print(f"❌ {args.source} 下没有 .json"); sys.exit(1)

    per_class = defaultdict(int)
    unknown = defaultdict(int)
    n_img = n_box = n_empty = n_noimg = 0

    for jf in jsons:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
        stem = os.path.splitext(os.path.basename(jf))[0]
        # 定位图片：json 的 imagePath，或同名找扩展名
        img_path = None
        cand = os.path.join(args.source, os.path.basename(d.get("imagePath", "")))
        if d.get("imagePath") and os.path.exists(cand):
            img_path = cand
        else:
            for e in IMG_EXTS:
                p = os.path.join(args.source, stem + e)
                if os.path.exists(p):
                    img_path = p; break
        if not img_path:
            print(f"⚠️ 找不到 {stem} 对应图片，跳过"); n_noimg += 1; continue

        W, H = img_size(img_path, d)
        lines = []
        for s in d.get("shapes", []):
            label = s.get("label", "")
            cid = idx.get(label, idx_lower.get(label.lower()))
            if cid is None:
                unknown[label] += 1; continue
            pts = s.get("points", [])
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            xmin, xmax = max(0, min(xs)), min(W, max(xs))
            ymin, ymax = max(0, min(ys)), min(H, max(ys))
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            cx = (xmin + xmax) / 2 / W
            cy = (ymin + ymax) / 2 / H
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
            per_class[cid] += 1; n_box += 1

        shutil.copy(img_path, os.path.join(img_out, os.path.basename(img_path)))
        with open(os.path.join(lab_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        if not lines:
            n_empty += 1

    # 把没有 json 的图作为空背景负样本纳入
    n_bg = 0
    if args.bg_unlabeled:
        for f in sorted(os.listdir(args.source)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(args.source, stem + ".json")):
                continue  # 有标注的已处理
            shutil.copy(os.path.join(args.source, f), os.path.join(img_out, f))
            open(os.path.join(lab_out, stem + ".txt"), "w").close()  # 空标签=背景
            n_bg += 1

    print(f"转换完成 → {args.out}")
    print(f"  有标注图 {n_img} 张（其中空标签 {n_empty}）| 无json背景 {n_bg} 张 | 框 {n_box} 个 | 缺图跳过 {n_noimg}")
    print("  每类框数:")
    for c in classes:
        print(f"    {c:10s} {per_class[idx[c]]}")
    if unknown:
        print("  ⚠️ 未识别的 label（不在 classes 里，已跳过）:", dict(unknown))
        print("     → 请核对 X-AnyLabeling 里的类名与 config.yaml 的 classes 一致")


if __name__ == "__main__":
    main()
