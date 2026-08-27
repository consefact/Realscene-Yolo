import os
import sys
import shutil
import random
import time

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()


def list_valid_pairs(src_dir):
    """列出所有有对应标签的图片"""
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    pairs = []
    for f in os.listdir(src_dir):
        if f.lower().endswith(supported_exts):
            label = os.path.splitext(f)[0] + '.txt'
            if os.path.exists(os.path.join(src_dir, label)):
                pairs.append(f)
    return pairs


def copy_pairs(file_list, src_dir, img_dst, label_dst):
    """复制图片和标签到目标目录"""
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)
    for f in file_list:
        base = os.path.splitext(f)[0]
        shutil.copy(os.path.join(src_dir, f), os.path.join(img_dst, f))
        shutil.copy(os.path.join(src_dir, base + '.txt'),
                    os.path.join(label_dst, base + '.txt'))
    print(f"  已复制 {len(file_list)} 对 → {img_dst}")


def main():
    t0 = time.perf_counter()
    src_dir = CFG.paths.synth_output
    target_dir = CFG.paths.dataset
    val_ratio = CFG.dataset.val_ratio  # 验证集比例

    pairs = list_valid_pairs(src_dir)
    if not pairs:
        print(f"错误：{src_dir} 下没有找到图片+标签对")
        return

    random.shuffle(pairs)
    split = int(len(pairs) * val_ratio)
    val_files = pairs[:split]
    train_files = pairs[split:]

    print(f"共 {len(pairs)} 对，训练集 {len(train_files)} / 验证集 {len(val_files)}")

    print("训练集...")
    copy_pairs(train_files, src_dir,
               os.path.join(target_dir, 'train', 'images'),
               os.path.join(target_dir, 'train', 'labels'))

    print("验证集...")
    copy_pairs(val_files, src_dir,
               os.path.join(target_dir, 'val', 'images'),
               os.path.join(target_dir, 'val', 'labels'))

    print(f"完成 → {target_dir}/（用时 {time.perf_counter() - t0:.1f}s）")


if __name__ == '__main__':
    main()