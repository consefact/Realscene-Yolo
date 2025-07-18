import os
import sys

def find_extra_images(folder_path):
    """
    查找没有对应标注文件的额外图片
    :param folder_path: 文件夹路径
    """
    # 获取所有文件列表
    all_files = os.listdir(folder_path)
    
    # 创建文件名集合（不含扩展名）
    base_names = set()
    for f in all_files:
        name, ext = os.path.splitext(f)
        if ext.lower() in ['.jpg', '.jpeg', '.txt']:
            base_names.add(name)
    
    # 查找没有对应标注文件的图片
    extra_images = []
    for f in all_files:
        if f.lower().endswith(('.jpg', '.jpeg')):
            base_name = os.path.splitext(f)[0]
            if f"{base_name}.txt" not in all_files:
                # 检查是否有大小写不同的情况
                txt_exists = any(
                    txt_file.lower() == f"{base_name}.txt".lower()
                    for txt_file in all_files
                    if txt_file.lower().endswith('.txt')
                )
                if not txt_exists:
                    extra_images.append(f)
    
    return extra_images

if __name__ == "__main__":
    # 获取文件夹路径
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("请输入文件夹路径: ").strip()
    
    if not os.path.isdir(folder_path):
        print(f"错误: 路径 {folder_path} 不是一个有效的文件夹")
        sys.exit(1)
    
    # 查找额外图片
    extra_images = find_extra_images(folder_path)
    
    # 输出结果
    if not extra_images:
        print("没有找到额外的图片")
    else:
        print("找到以下额外图片 (没有对应的标注文件):")
        for img in extra_images:
            print(f"- {img}")