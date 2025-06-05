import os
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将指定目录下的所有图片文件重命名为从1开始的索引+原扩展名形式')
    parser.add_argument('dir_path', type=str, help='图片所在的目录路径')
    args = parser.parse_args()

    dir_path = args.dir_path

    # 支持的图片扩展名（小写形式，用于判断是否是图片）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico'}

    # 获取目录下所有文件
    files = os.listdir(dir_path)

    # 筛选出图片文件
    image_files = []
    for f in files:
        full_path = os.path.join(dir_path, f)
        if os.path.isfile(full_path):  # 确保是文件
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_files.append(f)

    # 对图片文件排序，确保重命名顺序一致
    image_files.sort()

    # 从1开始重命名
    idx = 1
    for filename in image_files:
        original_ext = os.path.splitext(filename)[1]  # 保留原始扩展名（大小写不变）
        new_name = f"{idx}{original_ext}"
        old_path = os.path.join(dir_path, filename)
        new_path = os.path.join(dir_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
        idx += 1

if __name__ == '__main__':
    main()