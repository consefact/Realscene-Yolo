import os
import time
from pathlib import Path
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description='Recursively crop images in a directory based on specified coordinates.')
    parser.add_argument('--source', required=True, help='源图片目录')
    parser.add_argument('--target', required=True, help='输出目录')
    parser.add_argument('--left', type=int, required=True, help='裁剪左边界(px)')
    parser.add_argument('--upper', type=int, required=True, help='裁剪上边界(px)')
    parser.add_argument('--right', type=int, required=True, help='裁剪右边界(px)')
    parser.add_argument('--lower', type=int, required=True, help='裁剪下边界(px)')
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    crop_box = (args.left, args.upper, args.right, args.lower)

    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    t0 = time.perf_counter()
    n_cropped = 0
    for file_path in source_dir.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            # 计算相对于源目录的相对路径
            relative_path = file_path.relative_to(source_dir)
            target_file = target_dir / relative_path

            # 创建目标目录
            target_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(file_path) as img:
                    cropped_img = img.crop(crop_box)
                    cropped_img.save(target_file)
                    n_cropped += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    dt = time.perf_counter() - t0
    rate = n_cropped / dt if dt > 0 else 0.0
    print(f"完成：裁剪 {n_cropped} 张 → {target_dir}（用时 {dt:.1f}s，{rate:.1f} 张/s）")

if __name__ == '__main__':
    main()
