"""批量 GPU 合成入口。

用法：
  python realscene/gpu_synth.py --backend gpu --epochs 200
  python realscene/gpu_synth.py --backend auto --batch 32 --jpeg-workers 8

与 realscene.py 的关系：语义复刻（目标池/repeat/空镜/紧框/ROI/羽化/透视/菜单），
像素批量张量化、单进程（无多进程 pool，规避公用服务器卡死）。CPU 路径保持不动。
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import load_config
from gpu_engine import extract_synth_cfg, pick_device, BatchSynthesizer  # 同目录模块


def main():
    ap = argparse.ArgumentParser(description="批量 GPU 合成引擎")
    ap.add_argument("--backend", choices=["gpu", "auto", "cpu"], default=None,
                    help="覆盖 config 的 synth.backend")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="每批合成图数（gpu_batch）")
    ap.add_argument("--jpeg-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    sc = extract_synth_cfg(cfg)
    backend = args.backend or sc["BACKEND"]
    device = pick_device(backend)

    eng = BatchSynthesizer(sc, device=device,
                           batch_size=args.batch, jpeg_workers=args.jpeg_workers,
                           seed=args.seed)
    info = eng.summary()
    print("── GPU 合成 ──")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if args.epochs is None:
        return  # P0：仅加载校验
    eng.run(args.epochs)


if __name__ == "__main__":
    main()
