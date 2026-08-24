"""用一组照片快速验证训练好的 YOLO 模型。

功能：
  - 批量推理，打印基本信息：每图检测、每类数量/平均置信度、无检测图、推理速度
  - 可选 --save         保存画框可视化图
  - 可选 --save-labels  导出 YOLO 格式 .txt 标注（可用于伪标注扩充数据）

默认权重取 config.yaml 的 yolo_runs/<run_name>/weights/best.pt，可用 --weights 覆盖。

示例：
  python check/yolo/quick_test.py pics/                     # 只看信息
  python check/yolo/quick_test.py pics/ --save              # 同时存画框图
  python check/yolo/quick_test.py pics/ --save --save-labels
  python check/yolo/quick_test.py a.jpg --weights runs/x/best.pt --conf 0.3
"""
import os
import sys
import glob
import argparse
from collections import defaultdict

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config

import cv2
from ultralytics import YOLO

CFG = load_config()
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def pick_device(requested):
    """未指定时自动选设备：CUDA 真正可用则用 GPU，否则回退 CPU。"""
    if requested is not None:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            (torch.zeros(8, device="cuda") + 1).sum().item()  # 跑一次真实 kernel 验证兼容性
            return 0
    except Exception:
        pass
    return "cpu"


def collect_images(source):
    """source 可以是目录 / 单张图 / 通配符。"""
    if os.path.isdir(source):
        files = [os.path.join(source, f) for f in sorted(os.listdir(source))]
    elif os.path.isfile(source):
        files = [source]
    else:
        files = sorted(glob.glob(source))
    return [f for f in files if f.lower().endswith(IMG_EXTS)]


def main():
    default_weights = os.path.join(CFG.paths.yolo_runs, CFG.train.run_name, "weights", "best.pt")
    ap = argparse.ArgumentParser(description="用一组照片快速验证 YOLO 模型")
    ap.add_argument("source", help="图片目录 / 单张图片 / 通配符")
    ap.add_argument("--weights", default=default_weights, help=f"模型权重（默认 {default_weights}）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值（默认 0.45）")
    ap.add_argument("--imgsz", type=int, default=CFG.train.imgsz, help="推理尺寸")
    ap.add_argument("--out", default=os.path.join(_ROOT, "predict_out"), help="输出目录")
    ap.add_argument("--save", action="store_true", help="保存画框可视化图")
    ap.add_argument("--save-labels", dest="save_labels", action="store_true",
                    help="导出 YOLO 格式 .txt 标注")
    ap.add_argument("--max", type=int, default=0, help="最多处理张数（0=全部）")
    ap.add_argument("--device", default=None, help="设备，如 cpu 或 0")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ 找不到权重：{args.weights}\n   用 --weights 指定，或先训练。")
        sys.exit(1)

    images = collect_images(args.source)
    if args.max > 0:
        images = images[:args.max]
    if not images:
        print(f"❌ {args.source} 下没有图片")
        sys.exit(1)

    device = pick_device(args.device)
    model = YOLO(args.weights)
    names = model.names  # 训练时嵌入的类名 {id: name}
    print(f"模型   : {args.weights}")
    print(f"类别({len(names)}): {list(names.values())}")
    print(f"设备   : {device}")
    print(f"图片   : {len(images)} 张   conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    print("-" * 64)

    vis_dir = lab_dir = None
    if args.save:
        vis_dir = os.path.join(args.out, "annotated"); os.makedirs(vis_dir, exist_ok=True)
    if args.save_labels:
        lab_dir = os.path.join(args.out, "labels"); os.makedirs(lab_dir, exist_ok=True)

    per_class = defaultdict(int)
    conf_sum = defaultdict(float)
    total_dets = 0
    zero_imgs = []
    speeds = []

    results = model.predict(source=images, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            device=device, stream=True, verbose=False)
    n = 0
    for r in results:
        n += 1
        name = os.path.basename(r.path)
        base = os.path.splitext(name)[0]
        b = r.boxes
        ndet = 0 if b is None else len(b)
        total_dets += ndet
        if r.speed:
            speeds.append(r.speed.get("inference", 0.0))

        if ndet == 0:
            zero_imgs.append(name)
            print(f"[{n}/{len(images)}] {name}: 无检测")
        else:
            parts = []
            for i in range(ndet):
                cid = int(b.cls[i]); cf = float(b.conf[i])
                per_class[cid] += 1; conf_sum[cid] += cf
                parts.append(f"{names[cid]}:{cf:.2f}")
            print(f"[{n}/{len(images)}] {name}: {ndet} 个  " + "  ".join(parts))

        if vis_dir:
            cv2.imwrite(os.path.join(vis_dir, name), r.plot())
        if lab_dir is not None:
            with open(os.path.join(lab_dir, base + ".txt"), "w") as f:
                for i in range(ndet):
                    cid = int(b.cls[i]); x, y, w, h = b.xywhn[i].tolist()
                    f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # ---- 汇总 ----
    print("-" * 64)
    print(f"处理 {n} 张 | 检测框 {total_dets} 个 | 无检测 {len(zero_imgs)} 张")
    if speeds:
        avg = sum(speeds) / len(speeds)
        print(f"平均推理 {avg:.1f} ms/张 (~{1000 / avg:.1f} FPS)")
    print("每类检测数 / 平均置信度:")
    for cid in sorted(names):
        c = per_class.get(cid, 0)
        mc = (conf_sum[cid] / c) if c else 0.0
        print(f"  {names[cid]:10s} {c:5d}   conf均值 {mc:.2f}")
    if zero_imgs:
        show = ", ".join(zero_imgs[:10]) + (" …" if len(zero_imgs) > 10 else "")
        print(f"无检测图片({len(zero_imgs)}): {show}")
    if vis_dir:
        print(f"画框图  → {vis_dir}")
    if lab_dir is not None:
        print(f"标注txt → {lab_dir}")


if __name__ == "__main__":
    main()
