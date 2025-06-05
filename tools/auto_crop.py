import os
from pathlib import Path
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description='Recursively crop images in a directory based on specified coordinates.')
    parser.add_argument('--source', required=True, help='Source directory path',default='/path/to/source')
    parser.add_argument('--target', required=True, help='Target directory path',default='/path/to/target')
    parser.add_argument('--left', type=int, required=True,default=0)
    parser.add_argument('--upper', type=int, required=True,default=0)
    parser.add_argument('--right', type=int, required=True,default=1000)
    parser.add_argument('--lower', type=int, required=True,default=1000)
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    crop_box = (args.left, args.upper, args.right, args.lower)

    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

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
                    print(f"Cropped {file_path} to {target_file}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    main()