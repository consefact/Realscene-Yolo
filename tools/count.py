"""递归统计目录下指定后缀的文件数量（数据体检小工具）。

用法：python tools/count.py <目录> [--ext .png]
"""
import os
import argparse


def count_files_recursive(directory, file_format=".txt"):
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(file_format.lower()):
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="递归统计目录下指定后缀的文件数量")
    parser.add_argument("directory", help="要统计的目录")
    parser.add_argument("--ext", default=".txt", help="文件后缀（默认 .txt）")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"无效目录：{args.directory}")
    else:
        n = count_files_recursive(args.directory, args.ext)
        print(f"'{args.directory}' 下 {args.ext} 文件数量：{n}")
