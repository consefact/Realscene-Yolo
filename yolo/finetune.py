"""从 best.pt 出发，混入实拍标注数据继续训练（混合微调，避免灾难性遗忘）。

前置：实拍图需已标注。可先用
    python check/yolo/quick_test.py <实拍图目录> --save-labels
预标注，人工纠正后，把 图片 + 同名 .txt 放到 config 的 finetune.real_dir
（支持 real/images + real/labels，或 real/ 下 flat 的 图+txt）。

用法：
  python yolo/finetune.py --dry-run     # 只组装数据集 + 校验，不训练
  python yolo/finetune.py               # 组装 + 从 best.pt 混合微调
默认参数取 config.yaml 的 finetune 段，可用命令行覆盖。

混合策略：train = 全部合成训练图 + 实拍训练图×oversample；val = 实拍验证图（真实指标）。
合成数据在原地引用（不复制），只组装实拍侧，输出到 config.paths.dataset 同级目录的 real_ft/。
"""
import os
import sys
import glob
import shutil
import random
import argparse

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config

import yaml
from ultralytics import YOLO

CFG = load_config()
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def pick_device(requested):
    if requested is not None:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            (torch.zeros(8, device="cuda") + 1).sum().item()
            return 0
    except Exception:
        pass
    return "cpu"


def pair_real(real_dir):
    """返回 [(image_path, label_path), ...]。支持 images/+labels/ 或 flat。"""
    idir, ldir = os.path.join(real_dir, "images"), os.path.join(real_dir, "labels")
    if os.path.isdir(idir) and os.path.isdir(ldir):
        img_root, lab_root = idir, ldir
    else:
        img_root = lab_root = real_dir
    pairs = []
    if not os.path.isdir(img_root):
        return pairs
    for f in sorted(os.listdir(img_root)):
        if f.lower().endswith(IMG_EXTS):
            txt = os.path.join(lab_root, os.path.splitext(f)[0] + ".txt")
            if os.path.exists(txt):                       # 必须有标注
                pairs.append((os.path.join(img_root, f), txt))
    return pairs


def build_ft_dataset(pairs, out_dir, val_ratio, oversample):
    """把实拍对切分并复制到 out_dir/{train,val}/{images,labels}，train 按 oversample 复制。"""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out_dir, sp, sub), exist_ok=True)

    random.seed(0)
    pairs = pairs[:]
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    def put(pairs_, sp, dup):
        for img, txt in pairs_:
            stem = os.path.splitext(os.path.basename(img))[0]
            ext = os.path.splitext(img)[1]
            for k in range(dup):
                tag = f"_{k}" if dup > 1 else ""
                shutil.copy(img, os.path.join(out_dir, sp, "images", f"{stem}{tag}{ext}"))
                shutil.copy(txt, os.path.join(out_dir, sp, "labels", f"{stem}{tag}.txt"))

    put(train_pairs, "train", max(1, oversample))
    put(val_pairs, "val", 1)
    return len(train_pairs), len(val_pairs)


def main():
    ft = CFG.finetune
    src_weights = os.path.join(CFG.paths.yolo_runs, CFG.train.run_name, "weights", "best.pt")

    ap = argparse.ArgumentParser(description="从 best.pt 混入实拍数据混合微调")
    ap.add_argument("--weights", default=src_weights, help=f"起点权重（默认 {src_weights}）")
    ap.add_argument("--real-dir", dest="real_dir", default=ft.real_dir, help="实拍标注数据目录")
    ap.add_argument("--oversample", type=int, default=ft.oversample, help="实拍图训练集重复倍数")
    ap.add_argument("--epochs", type=int, default=ft.epochs)
    ap.add_argument("--lr0", type=float, default=ft.lr0, help="学习率（微调宜小）")
    ap.add_argument("--freeze", type=int, default=ft.freeze, help="冻结前 N 层")
    ap.add_argument("--name", default=ft.run_name, help="输出子目录")
    ap.add_argument("--imgsz", type=int, default=CFG.train.imgsz)
    ap.add_argument("--batch", type=int, default=CFG.train.batch)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=50, help="早停耐心：val 连续多少 epoch 无改善就停")
    ap.add_argument("--cache", action="store_true", help="把图像缓存到内存加速（数据集不大时推荐）")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="只组装+校验，不训练")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ 找不到起点权重：{args.weights}"); sys.exit(1)

    pairs = pair_real(args.real_dir)
    if not pairs:
        print(f"❌ {args.real_dir} 下没有 图片+同名.txt 对。\n"
              f"   先用 quick_test.py --save-labels 预标注、人工纠正后放进来。")
        sys.exit(1)

    # 微调混合集输出到 config.paths.dataset 同级目录（换任务自动跟随，不写死任务名）
    ft_dir = os.path.join(os.path.dirname(CFG.paths.dataset), "real_ft")
    n_tr, n_va = build_ft_dataset(pairs, ft_dir, CFG.dataset.val_ratio, args.oversample)
    print(f"实拍对 {len(pairs)}  → 训练 {n_tr}（×{max(1,args.oversample)} 重复） / 验证 {n_va}")

    synth_train = os.path.join(CFG.paths.dataset, "train", "images")
    real_train = os.path.join(ft_dir, "train", "images")
    real_val = os.path.join(ft_dir, "val", "images")

    # val 优先用实拍（真实指标）；实拍验证太少则回退合成 val 并提示
    val_sources = [real_val] if n_va >= 1 else []
    if n_va < 10:
        synth_val = os.path.join(CFG.paths.dataset, "val", "images")
        if os.path.isdir(synth_val):
            val_sources.append(synth_val)
            print(f"⚠️ 实拍验证图仅 {n_va} 张，已并入合成 val 兜底（真实指标仍以实拍为准）")

    data = {
        "path": _ROOT,
        "train": [synth_train, real_train],
        "val": val_sources,
        "nc": len(CFG.classes),
        "names": {i: n for i, n in enumerate(CFG.classes)},
    }
    data_yaml = os.path.join(_ROOT, "yolo", "craic_ft.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("# finetune.py 自动生成的混合数据集描述\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"混合数据集描述 → {data_yaml}")
    print(f"  train: 合成 {synth_train}  +  实拍 {real_train}")
    print(f"  val  : {val_sources}")

    # 校验数据集（不训练也能确认图/标签配对正确）
    try:
        from ultralytics.data.utils import check_det_dataset
        info = check_det_dataset(data_yaml)
        print("数据集校验通过 ✓")
    except Exception as e:
        print(f"⚠️ 数据集校验提示：{e}")

    if args.dry_run:
        print("\n[dry-run] 已组装并校验，未训练。去掉 --dry-run 即开始微调。")
        return

    device = pick_device(args.device)
    print(f"\n从 {args.weights} 微调  (lr0={args.lr0} freeze={args.freeze} device={device})")
    model = YOLO(args.weights)
    model.train(
        data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        lr0=args.lr0, freeze=args.freeze, optimizer=CFG.train.optimizer,
        patience=args.patience, cache=args.cache,
        project=CFG.paths.yolo_runs, name=args.name, device=device, pretrained=True,
    )


if __name__ == "__main__":
    main()
