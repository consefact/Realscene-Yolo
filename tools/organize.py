import os
import shutil
import argparse

def copy_images_and_labels(src_dir, img_dst, label_dst):
    """
    从源目录中复制图片和对应的标签文件到目标目录
    """
    supported_image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(supported_image_exts):
            image_path = os.path.join(src_dir, filename)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(src_dir, label_filename)

            if os.path.exists(label_path):
                # 复制图片和标签
                shutil.copy(image_path, os.path.join(img_dst, filename))
                shutil.copy(label_path, os.path.join(label_dst, label_filename))
            else:
                print(f"⚠️ 警告：未找到与图片 {filename} 对应的标签文件 {label_filename}")

def main():
    parser = argparse.ArgumentParser(description='将训练集和验证集整理为 YOLO 数据集结构')
    parser.add_argument('train_source', help='训练集源目录（包含图片和标签）',default='/home/airhust/zyt/images/FORTRAIN',)
    parser.add_argument('val_source', help='验证集源目录（包含图片和标签）',default='/home/airhust/zyt/images/FORTEST')
    parser.add_argument('target_dir', help='输出 YOLO 数据集的根目录',default='/home/airhust/zyt/images/yolo_dataset')
    args = parser.parse_args()

    # 构建目标目录结构
    train_images = os.path.join(args.target_dir, 'train', 'images')
    train_labels = os.path.join(args.target_dir, 'train', 'labels')
    val_images = os.path.join(args.target_dir, 'val', 'images')
    val_labels = os.path.join(args.target_dir, 'val', 'labels')

    # 创建目标目录（如果不存在）
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    # 处理训练集
    print("🔄 正在处理训练集...")
    copy_images_and_labels(args.train_source, train_images, train_labels)

    # 处理验证集
    print("🔄 正在处理验证集...")
    copy_images_and_labels(args.val_source, val_images, val_labels)

    print("✅ 数据集整理完成！")

if __name__ == '__main__':
    main()