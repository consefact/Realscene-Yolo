import os
import random
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程锁用于保护进度条更新
progress_lock = threading.Lock()

def count_files(root_dir):
    """统计目录中符合条件的文件数量"""
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
    return count

def process_single_image(args):
    """处理单个图片的任务"""
    (original_path, original_root, new_root, real_target_files, real_targets_dir, probability_use_real, pbar) = args
    
    relative_path = os.path.relpath(original_path, original_root)
    dir_name = os.path.dirname(relative_path)
    base_name = os.path.splitext(os.path.basename(os.path.basename(original_path)))[0]

    # 加载原始图像并调整尺寸为64x64
    original_img = Image.open(original_path).convert("RGB")
    original_img = original_img.resize((64, 64))

    # 判断是否使用真实靶标
    use_real = False
    if real_target_files:
        use_real = random.random() < probability_use_real

    processed_count = 0

    try:
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
                background = background.resize((194, 194))  # 修改为194x194

                # 创建遮罩并应用高斯模糊
                mask = Image.new('L', original_img.size, 0)
                draw_mask = ImageDraw.Draw(mask)
                draw_mask.rectangle([0, 0, original_img.width, original_img.height], fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(radius=10))  # radius 适当放大

                # 粘贴原始图像，使用mask实现边缘平滑
                background.paste(original_img, (65, 65), mask)  # 居中粘贴

                # 保存图像
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                background.save(output_path)
                
                processed_count += 1

        # 否则生成合成背景图像
        else:
            new_file = f"{base_name}_synthetic.png"
            new_relative_path = os.path.join(dir_name, new_file)
            output_path = os.path.join(new_root, new_relative_path)

            # 创建合成背景
            background = Image.new('RGB', (194, 194), (0, 0, 0))
            draw = ImageDraw.Draw(background)

            # 按比例缩放环和圆
            scale = 194 / 100
            gray_ring = [(12 * scale, 12 * scale), (87 * scale, 87 * scale)]
            white_circle = [(25 * scale, 25 * scale), (75 * scale, 75 * scale)]

            draw.ellipse(gray_ring, fill=(128, 128, 128))   # 灰色环
            draw.ellipse(white_circle, fill=(255, 255, 255)) # 白色圆

            # 粘贴原始图像
            background.paste(original_img, (65, 65))  # 居中粘贴

            # 保存图像
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            background.save(output_path)
            
            processed_count += 1

    finally:
        # 释放资源
        original_img.close()
        
        # 安全地更新进度条
        with progress_lock:
            pbar.update(1)
    
    return processed_count

def process_all_images(original_root, new_root, real_targets_dir=None, probability_use_real=0.5, max_workers=8):
    """批量处理所有图片，为每个真实靶标生成一个图像（多线程版本）"""
    # 获取所有真实靶标文件列表
    real_target_files = []
    if real_targets_dir is not None and os.path.exists(real_targets_dir):
        real_target_files = [f for f in os.listdir(real_targets_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 收集所有需要处理的文件
    tasks = []
    for root, dirs, files in os.walk(original_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                original_path = os.path.join(root, file)
                tasks.append((original_path, original_root, new_root, real_target_files, real_targets_dir, probability_use_real, None))

    # 统计总文件数用于进度条
    total_files = len(tasks)
    
    # 创建进度条
    pbar = tqdm(total=total_files, desc="处理图片", unit="张")
    
    # 为每个任务添加进度条引用
    tasks_with_pbar = [(task[0], task[1], task[2], task[3], task[4], task[5], pbar) for task in tasks]
    
    # 使用线程池执行任务
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_image, task): task for task in tasks_with_pbar}
        
        # 等待所有任务完成
        for future in as_completed(future_to_task):
            try:
                count = future.result()
                processed_count += count
            except Exception as exc:
                print(f'任务执行出错: {exc}')
    
    pbar.close()
    print(f"总共处理了 {processed_count} 个图像文件")

if __name__ == "__main__":
    original_dataset_dir = "/home/ling/zyt/cifar100_images"  # 原始数据集根目录
    new_dataset_dir = "/home/ling/zyt195/images/bigtargets"        # 新生成数据集根目录
    real_targets_dir = "/home/ling/zyt195/images/cut_class"    # 真实靶标图片目录
    probability_use_real = 1                                    # 每次处理时使用真实靶标的概率（此处设为1）
    max_workers = 32                                             # 线程数，可以根据CPU核心数调整

    process_all_images(original_dataset_dir, new_dataset_dir, real_targets_dir, probability_use_real, max_workers)
    print(f"\n处理完成，新图片保存至：{new_dataset_dir}")