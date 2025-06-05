import os
import random
from PIL import Image, ImageDraw, ImageFilter

def process_all_images(original_root, new_root, real_targets_dir=None, probability_use_real=0.5):
    """批量处理所有图片，为每个真实靶标生成一个图像"""
    # 获取所有真实靶标文件列表
    real_target_files = []
    if real_targets_dir is not None and os.path.exists(real_targets_dir):
        real_target_files = [f for f in os.listdir(real_targets_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for root, dirs, files in os.walk(original_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                original_path = os.path.join(root, file)
                relative_path = os.path.relpath(original_path, original_root)
                dir_name = os.path.dirname(relative_path)
                base_name = os.path.splitext(os.path.basename(file))[0]

                # 加载原始图像并调整尺寸为32x32
                original_img = Image.open(original_path).convert("RGB")
                original_img = original_img.resize((32, 32))

                # 判断是否使用真实靶标
                use_real = False
                if real_target_files:
                    use_real = random.random() < probability_use_real

                # 使用真实靶标生成多个图像
                if use_real and real_target_files:
                    for idx, target_file in enumerate(real_target_files):
                        # 构造输出文件名，包含靶标索引
                        new_file = f"{base_name}_real_{idx:03d}.png"
                        new_relative_path = os.path.join(dir_name, new_file)
                        output_path = os.path.join(new_root, new_relative_path)

                        # 加载真实靶标背景
                        target_path = os.path.join(real_targets_dir, target_file)
                        background = Image.open(target_path).convert("RGB")
                        background = background.resize((100, 100))  # 统一尺寸

                        # 创建遮罩并应用高斯模糊
                        mask = Image.new('L', original_img.size, 0)
                        draw_mask = ImageDraw.Draw(mask)
                        draw_mask.rectangle([0, 0, original_img.width, original_img.height], fill=255)
                        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))  # 调整radius可控制羽化程度

                        # 粘贴原始图像，使用mask实现边缘平滑
                        background.paste(original_img, (34, 34), mask)

                        # 保存图像
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        background.save(output_path)

                # 否则生成合成背景图像
                else:
                    new_file = f"{base_name}_synthetic.png"
                    new_relative_path = os.path.join(dir_name, new_file)
                    output_path = os.path.join(new_root, new_relative_path)

                    # 创建合成背景
                    background = Image.new('RGB', (100, 100), (0, 0, 0))
                    draw = ImageDraw.Draw(background)
                    draw.ellipse([(12, 12), (87, 87)], fill=(128, 128, 128))  # 灰色环
                    draw.ellipse([(25, 25), (75, 75)], fill=(255, 255, 255))  # 白色圆

                    # 粘贴原始图像
                    background.paste(original_img, (34, 34))

                    # 保存图像
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    background.save(output_path)

                # 释放资源
                original_img.close()

if __name__ == "__main__":
    original_dataset_dir = "/home/airhust/zyt/images/organized_pics"  # 原始数据集根目录
    new_dataset_dir = "/home/airhust/zyt/images/new_targets"  # 新生成数据集根目录
    real_targets_dir = "/home/airhust/zyt/images/pictures/background"          # 真实靶标图片目录
    probability_use_real = 1                 # 每次处理时使用真实靶标的概率（此处设为1）

    process_all_images(original_dataset_dir, new_dataset_dir, real_targets_dir, probability_use_real)
    print(f"处理完成，新图片保存至：{new_dataset_dir}")