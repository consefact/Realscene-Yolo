"""对同一个视频跑多个 YOLO 模型，各自保存打框视频并输出检测统计（串行执行）。

串行理由：逐个加载模型、各自推理，写起来最简单，也不抢 GPU 显存；
每个模型结果独立成片，便于对比不同 epoch/版本权重的效果。

用法：
  python yolo/compare_videos.py xxx.mp4                      # 默认跑 yolo_run_/best-*.pt
  python yolo/compare_videos.py xxx.mp4 --models a.pt b.pt --imgsz 640x480
  python yolo/compare_videos.py xxx.mp4 --conf 0.3 --out predict_video
"""
import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_imgsz(s):
    """'640x480' → (480, 640)，纯数字 → 正方形边长。"""
    if "x" in str(s):
        w, h = str(s).lower().split("x")
        return (int(h), int(w))
    return int(s)


def main():
    ap = argparse.ArgumentParser(description="一个视频跑多个 YOLO 模型，分别保存打框视频")
    ap.add_argument("source", help="视频文件路径")
    ap.add_argument("--models", nargs="+", default=None,
                    help="模型权重或目录（目录取其下所有 .pt），可多个；默认取 yolo_run_/best-*.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值（默认 0.45）")
    ap.add_argument("--imgsz", default="640x480", help="推理尺寸 WxH（默认 640x480）")
    ap.add_argument("--out", default=os.path.join(_ROOT, "..", "predict_video"),
                    help="输出目录（默认 ../predict_video）")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        print(f"❌ 找不到视频：{args.source}")
        sys.exit(1)
    weights = args.models
    if weights is None:
        weights = sorted(glob.glob(os.path.join(_ROOT, "..", "yolo_run_", "best-*.pt")))
    else:
        # 支持 文件 / 目录 / 混用：目录取其下所有 .pt
        expanded = []
        for w in weights:
            if os.path.isdir(w):
                found = sorted(glob.glob(os.path.join(w, "*.pt")))
                if not found:
                    print(f"⚠ 目录下没有 .pt：{w}")
                expanded.extend(found)
            elif os.path.isfile(w):
                expanded.append(w)
            else:
                print(f"⚠ 找不到模型，跳过：{w}")
        weights = expanded
    if not weights:
        print("❌ 没有模型：--models 指定，或在 yolo_run_/ 下放 best-*.pt")
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    # 源视频 fps（仅本地文件有效），用于统一输出帧率
    cap = cv2.VideoCapture(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    vid_stem = os.path.splitext(os.path.basename(args.source))[0]
    imgsz = parse_imgsz(args.imgsz)

    summary = []  # (权重名, 总帧数, 均值fps, 每类{名称: (个数, 置信度和)})
    for w in weights:
        name = os.path.splitext(os.path.basename(w))[0]
        model = YOLO(w)
        print(f"\n=== {w} | 类别({len(model.names)}): {list(model.names.values())}"
              f"  conf={args.conf} imgsz={args.imgsz}")

        out_path = os.path.join(args.out, f"{vid_stem}_{name}.mp4")
        writer = None
        per_cls = defaultdict(lambda: [0, 0.0])
        t0 = time.time()
        n = 0

        for r in model.predict(source=args.source, conf=args.conf, iou=args.iou,
                               imgsz=imgsz, stream=True, verbose=False):
            frame = r.plot()
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    print(f"⚠ 无法写出：{out_path}")
                    break
            writer.write(frame)
            n += 1
            if r.boxes is not None:
                for cid, cf in zip(r.boxes.cls, r.boxes.conf):
                    cname = model.names[int(cid)]
                    per_cls[cname][0] += 1
                    per_cls[cname][1] += float(cf)

        if writer:
            writer.release()
        fps_mean = n / (time.time() - t0) if n else 0.0
        print(f"  {n} 帧 | 平均 {fps_mean:.1f} fps | 保存 → {out_path}")
        if n:
            summary.append((name, n, fps_mean, dict(per_cls)))

    # ---- 汇总对比 ----
    print("\n" + "=" * 64)
    print(f"模型对比 | 视频 {args.source} | {len(summary)} 个模型")
    for name, n, fps_mean, per_cls in summary:
        print(f"\n[{name}] 帧数={n}  fps={fps_mean:.1f}")
        for cname, (cnt, conf_sum) in sorted(per_cls.items(), key=lambda kv: -kv[1][0]):
            print(f"  {cname:10s} {cnt:5d} 帧率平均 {conf_sum / cnt:.2f}")


if __name__ == "__main__":
    main()
