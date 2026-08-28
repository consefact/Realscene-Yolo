"""读取本地视频流（视频文件 / 摄像头 / RTSP），实时跑 YOLO 并弹窗显示打框。

示例：
  python yolo/preview_video.py xxx.mp4                 # 视频文件
  python yolo/preview_video.py 0                       # USB 摄像头 / 设备索引
  python yolo/preview_video.py rtsp://192.168.1.1:8554/s  # RTSP 流
  python yolo/preview_video.py xxx.mp4 --weights yolo/yolo11n.pt --conf 0.3 --max-h 1080
  python yolo/preview_video.py 0 --record                 # 自动命名保存录像
  python yolo/preview_video.py xxx.mp4 --record out.mp4    # 保存到指定文件
"""
import argparse
import os
import time

import cv2
from ultralytics import YOLO

_DEFAULT_WEIGHTS = (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolo_run_", "best-v3-2.pt"),
                    "yolo_run_/yolo11n.pt")


def main():
    ap = argparse.ArgumentParser(description="本地视频流跑 YOLO 实时打框预览")
    ap.add_argument("source", help="视频文件 / 摄像头索引(数字) / RTSP 或 http 流地址")
    ap.add_argument("--weights", default=None,
                    help=f"权重（默认微调模型，无则回退 {_DEFAULT_WEIGHTS[1]}）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值（默认 0.45）")
    ap.add_argument("--imgsz", default="640x480", help="推理尺寸 WxH（默认 640x480）")
    ap.add_argument("--device", default=None, help="设备，如 cpu 或 0")
    ap.add_argument("--max-h", type=int, default=900, help="显示前把帧缩放到的最大高度，0=不缩放（默认 900）")
    ap.add_argument("--record", nargs="?", const="auto", default=None,
                    help="把显示画面从头保存为视频；给路径则保存到该文件，单独 --record 自动命名")
    args = ap.parse_args()

    weights = args.weights
    if weights is None:
        weights = _DEFAULT_WEIGHTS[0] if os.path.exists(_DEFAULT_WEIGHTS[0]) else _DEFAULT_WEIGHTS[1]
    src = int(args.source) if args.source.isdigit() else args.source
    if "x" in str(args.imgsz):
        w, h = str(args.imgsz).lower().split("x")
        imgsz = (int(h), int(w))  # ultralytics 的 imgsz 元组按 (h, w) 传
    else:
        imgsz = int(args.imgsz)

    model = YOLO(weights)
    print(f"模型   : {weights}")
    print(f"类别({len(model.names)}): {list(model.names.values())}")
    print(f"来源   : {src}  conf={args.conf} iou={args.iou} imgsz={args.imgsz}")

    # stream=True 维持同一个 VideoCapture，避免每次迭代重开摄像头
    stream = model.predict(source=src, conf=args.conf, iou=args.iou, imgsz=imgsz,
                           device=args.device, stream=True, verbose=False)

    win = "yolo preview"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    fps, t0, n = 0.0, time.time(), 0
    writer, rec_path = None, None
    try:
        for r in stream:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            frame = r.plot()
            if args.max_h and frame.shape[0] > args.max_h:
                s = args.max_h / frame.shape[0]
                frame = cv2.resize(frame, (int(frame.shape[1] * s), args.max_h))
            n += 1
            if n % 5 == 0:
                now = time.time()
                fps = n / (now - t0)
                t0, n = now, 0
            cv2.putText(frame, f"{fps:.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)
            cv2.imshow(win, frame)

            if args.record:
                if writer is None:
                    rec_path = time.strftime("record_%Y%m%d_%H%M%S.mp4") if args.record == "auto" \
                        else args.record
                    writer = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps if fps > 0 else 30.0,
                                             (frame.shape[1], frame.shape[0]))
                    if writer.isOpened():
                        print(f"录像 → {rec_path}")
                    else:
                        writer = None
                        print(f"⚠ 无法写出视频文件：{rec_path}")
                if writer:
                    writer.write(frame)  # 所见即所得：保存显示的缩放画框帧
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if writer:
            writer.release()
            print(f"已保存 → {rec_path}")
        print("已退出")


if __name__ == "__main__":
    main()
