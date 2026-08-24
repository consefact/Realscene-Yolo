"""基于 config.yaml 训练 YOLO：先由配置自动生成数据集描述文件，再调用 ultralytics 训练。

换任务时无需改本文件——类别、路径、超参全部来自 config.yaml。
"""
import os
import sys

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config

import yaml
from ultralytics import YOLO

CFG = load_config()


def write_dataset_yaml(path):
    """由 config.yaml 生成 ultralytics 数据集描述文件（dataset.yaml）。"""
    data = {
        "path": CFG.paths.dataset,       # 数据集根（organize.py 输出）
        "train": "train/images",
        "val": "val/images",
        "nc": len(CFG.classes),
        "names": {i: c for i, c in enumerate(CFG.classes)},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 本文件由 y8train.py 依据 config.yaml 自动生成，请勿手改\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


if __name__ == "__main__":
    t = CFG.train
    data_yaml = write_dataset_yaml(os.path.join(_ROOT, "yolo", "dataset.yaml"))
    print(f"数据集配置已生成：{data_yaml}  (nc={len(CFG.classes)}, names={CFG.classes})")

    # 起始权重：存在则用绝对路径，否则退化为文件名交给 ultralytics 自动下载
    weights = CFG.paths.base_weights
    if not os.path.exists(weights):
        weights = os.path.basename(weights)

    model = YOLO(weights)  # 从预训练权重开始
    results = model.train(
        data=data_yaml,
        epochs=t.epochs,
        imgsz=t.imgsz,
        dropout=t.dropout,
        patience=t.patience,
        workers=t.workers,
        batch=t.batch,
        project=CFG.paths.yolo_runs,
        name=t.run_name,
        optimizer=t.optimizer,
        lr0=t.lr0,
        pretrained=True,
    )
