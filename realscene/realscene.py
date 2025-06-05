import os
import random
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from config import CLASSES
from tqdm import tqdm

# 全局参数配置
REAL_IMAGES_DIR = "real_images"
SYNTHETIC_DIR = "new_cifar100_images"  # 合成靶标目录
REAL_SYNTHETIC_DIR = "real_synthetic_targets"  # 真实合成靶标目录
ORIGINAL_TARGET_DIR = "original_targets"  # 原始识别区域目录
OUTPUT_DIR = "NTDATA"

BASE_SIZE = (1280, 1280)  # YOLO训练尺寸
NUM_TARGETS_PER_IMAGE = 3  # 每张图像目标数量
MIN_CROP_RATIO = 0.3  # 最小裁剪比例
MIN_TARGET_RATIO = 0.2  # 目标最小占比
MAX_CROP_RATIO = 0.9  # 最大裁剪比例
MAX_TARGET_RATIO = 0.9  # 目标最大占比
MAX_TARGET_FAILURE = 3  # 最大失败次数
MAX_OVERLAP_ATTEMPTS = 20  # 最大重叠检测次数
TO_BORDER = 1e-6  # 边界安全距离
NUM_ROUNDS = 6000  # 总生成轮数
class_names = CLASSES  # 从config.py导入类别名称
APPLY_GEOMETRIC_AUG = False
# 靶标类型定义
TARGET_TYPES = {
    "synthetic": 100,      # 合成靶标
    "real_synthetic": 100, # 真实合成靶标
    "original": 32         # 原始识别区域
}

def apply_geometric_augmentation(base):
    """几何变换增强"""
    if not APPLY_GEOMETRIC_AUG:
        return base
    if random.random() < 0.3:  # 提高应用概率
        w, h = base.size
        distortion = 0.1  # 增加形变幅度
        
        # 生成四边形顶点
        dx1 = random.randint(0, int(w*distortion))
        dy1 = random.randint(0, int(h*distortion))
        dx2 = random.randint(0, int(w*distortion))
        dy2 = random.randint(0, int(h*distortion))
        dx3 = random.randint(0, int(w*distortion))
        dy3 = random.randint(0, int(h*distortion))
        dx4 = random.randint(0, int(w*distortion))
        dy4 = random.randint(0, int(h*distortion))
        
        points = [
            (dx1, dy1),                   # 左上角
            (w - dx2, dy2),               # 右上角
            (w - dx3, h - dy3),           # 右下角
            (dx4, h - dy4)                # 左下角
        ]
        
        base = base.transform(
            base.size,
            Image.QUAD,
            [points[0][0], points[0][1],
             points[1][0], points[1][1],
             points[2][0], points[2][1],
             points[3][0], points[3][1]],
            resample=Image.BICUBIC
        )
    return base

def apply_base_augmentation(base_image):
    """基础增强（亮度/对比度/几何变换）"""
    base = base_image.convert("RGB")
    
    # 亮度调整
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(base)
        base = enhancer.enhance(random.uniform(0.6, 1.35))
    
    # 对比度调整
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(base)
        base = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 饱和度调整
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(base)
        base = enhancer.enhance(random.uniform(0.5, 1.5))
    
    # 几何变换
    base = apply_geometric_augmentation(base)
    
    # HSV颜色扰动
    if random.random() > 0.5:
        img_np = np.array(base).astype("float32") / 255.0
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv_factor = np.random.uniform(-0.1, 0.1, 3)
        img_hsv = np.clip(img_hsv + hsv_factor, 0, 1)
        base = Image.fromarray((cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB) * 255).astype(np.uint8))
    
    return base

def random_crop(image_path):
    """随机裁剪并增强"""
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    crop_width = random.randint(
        int(original_width * MIN_CROP_RATIO),
        int(original_width * MAX_CROP_RATIO)
    )
    crop_height = int(crop_width * 3 / 4)  # 保持4:3比例
    
    x = random.randint(0, original_width - crop_width)
    y = random.randint(0, original_height - crop_height)
    cropped = img.crop((x, y, x+crop_width, y+crop_height))
    resized_img = cropped.resize(BASE_SIZE, Image.LANCZOS)
    
    return apply_base_augmentation(resized_img)

def load_target_images():
    """加载所有类型靶标"""
    target_images = []
    dir_type_map = [
        (SYNTHETIC_DIR, "synthetic"),
        (REAL_SYNTHETIC_DIR, "real_synthetic"),
        (ORIGINAL_TARGET_DIR, "original")
    ]
    
    for target_dir, target_type in dir_type_map:
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    try:
                        img = Image.open(path).convert("RGBA")
                        category = os.path.basename(os.path.dirname(path))
                        class_id = class_names.index(category)
                        target_images.append(
                            (img, class_id, len(target_images), target_type)
                        )
                    except Exception as e:
                        print(f"加载失败：{path}，原因：{e}")
    return target_images

def apply_augmentation(target_tuple):
    """靶标增强"""
    target, class_id, original_idx, target_type = target_tuple
    scale = random.uniform(0.5, 1.0)
    
    # 根据类型调整缩放范围
    if target_type == "original":
        scale = random.uniform(0.8, 1.2)
    
    new_size = (int(target.width * scale), int(target.height * scale))
    target = target.resize(new_size, Image.LANCZOS)
    
    # 旋转与翻转
    angle = random.randint(-45, 45)
    target = target.rotate(angle, resample=Image.BICUBIC, expand=True).convert("RGBA")
    if random.random() > 0.5:
        target = target.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 色彩增强
    enhancer = ImageEnhance.Brightness(target)
    target = enhancer.enhance(random.uniform(0.7, 1.35))
    enhancer = ImageEnhance.Contrast(target)
    target = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.25:
        enhancer = ImageEnhance.Sharpness(target)
        target = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.35:
        enhancer = ImageEnhance.Color(target)
        target = enhancer.enhance(random.uniform(0.5, 1.5))
    
    return (target, class_id, original_idx, target_type)

def place_target(base, target_tuple, placed_bboxes):
    """放置靶标并确保不重叠"""
    target, class_id, original_idx, target_type = target_tuple
    base = base.convert("RGBA")
    target = target.convert("RGBA")
    base_width, base_height = base.size
    target_width, target_height = target.size
    
    # 根据类型确定有效区域
    effective_w, effective_h = {
        "original": (target_width, target_height),
        "synthetic": (32, 32),
        "real_synthetic": (32, 32)
    }[target_type]
    
    for _ in range(MAX_OVERLAP_ATTEMPTS):
        x = random.randint(0, base_width - target_width)
        y = random.randint(0, base_height - target_height)
        new_bbox = {"x": x, "y": y, "width": effective_w, "height": effective_h}
        
        # 检查重叠
        overlap = False
        for bbox in placed_bboxes:
            bx1, by1 = bbox['x'], bbox['y']
            bx2, by2 = bx1 + bbox['width'], by1 + bbox['height']
            tx1, ty1 = new_bbox['x'], new_bbox['y']
            tx2, ty2 = tx1 + new_bbox['width'], ty1 + new_bbox['height']
            
            if not (tx2 < bx1 or tx1 > bx2 or ty2 < by1 or ty1 > by2):
                overlap = True
                break
                
        if not overlap:
            break
    else:
        return base, None
        
    base.paste(target, (x, y), target.getchannel('A'))
    return (
        base.convert("RGB"),
        {
            "original_idx": original_idx,
            "x": x,
            "y": y,
            "width": effective_w,
            "height": effective_h
        }
    )

def process_round(round_num, target_images, used_targets):
    """处理单轮生成"""
    # 提前检查是否所有目标都已使用
    if all(used_targets):
        return
    
    # 随机选择1-50中的基底图片
    real_image_idx = random.choice(range(1, 51))
    
    real_path = os.path.join(REAL_IMAGES_DIR, f"{real_image_idx}.jpg")
    output_image_path = os.path.join(
        OUTPUT_DIR, 
        f"E{epoch}_R{round_num}_img{real_image_idx}.jpg"
    )
    label_path = output_image_path.replace(".jpg", ".txt")
    
    base = random_crop(real_path)
    bboxes = []
    placed_bboxes = []
    failed_attempts = 0
    
    # 保持原有目标放置逻辑
    for _ in range(NUM_TARGETS_PER_IMAGE):
        available_targets = [
            t for t in target_images if not used_targets[t[2]]
        ]
        if not available_targets:
            break
            
        target_tuple = random.choice(available_targets)
        augmented = apply_augmentation(target_tuple)
        target, _, _, target_type = augmented
        
        # 动态计算缩放比例（保持原有逻辑）
        current_width, current_height = target.size
        min_dim = min(current_width, current_height)
        max_allowed_width = BASE_SIZE[0]
        max_allowed_height = BASE_SIZE[1]
        
        width_scale = max_allowed_width / current_width
        height_scale = max_allowed_height / current_height
        max_safe_scale = min(width_scale, height_scale)
        
        # 缩放逻辑（保持不变）
        if target_type == "original":
            min_scale = 1.0
            max_scale = min(3.0, max_safe_scale)
        else:
            min_scale = (MIN_TARGET_RATIO * min(BASE_SIZE)) / min_dim
            max_scale = min(
                (MAX_TARGET_RATIO * min(BASE_SIZE)) / min_dim,
                max_safe_scale
        )
        
        scale = random.uniform(min_scale, max_scale)
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        target = target.resize((new_width, new_height), Image.LANCZOS)
        
        # 尝试放置目标（保持原有逻辑）
        placed_base, pixel_bbox = place_target(
            base.copy(),
            (target, *augmented[1:]),
            placed_bboxes
        )
        
        if pixel_bbox:
            placed_bboxes.append(pixel_bbox)
            base = placed_base

            # 新增：根据 target_type 调整标注框位置和尺寸（保持不变）
            target_x = pixel_bbox['x']
            target_y = pixel_bbox['y']
            target_width = pixel_bbox['width']
            target_height = pixel_bbox['height']

            if target_type in ["synthetic", "real_synthetic"]:
                roi_x = 34
                roi_y = 34
                roi_w = 32
                roi_h = 32

                abs_x = target_x + roi_x * (target_width / 100)
                abs_y = target_y + roi_y * (target_height / 100)
                abs_w = roi_w * (target_width / 100)
                abs_h = roi_h * (target_height / 100)

                x_center = (abs_x + abs_w / 2) / BASE_SIZE[0]
                y_center = (abs_y + abs_h / 2) / BASE_SIZE[1]
                width_norm = abs_w / BASE_SIZE[0]
                height_norm = abs_h / BASE_SIZE[1]

            elif target_type == "original":
                x_center = (pixel_bbox['x'] + pixel_bbox['width'] / 2) / BASE_SIZE[0]
                y_center = (pixel_bbox['y'] + pixel_bbox['height'] / 2) / BASE_SIZE[1]
                width_norm = pixel_bbox['width'] / BASE_SIZE[0]
                height_norm = pixel_bbox['height'] / BASE_SIZE[1]

            # 边界修正
            x_min = max(0.0 + TO_BORDER, x_center - width_norm / 2)
            x_max = min(1.0 - TO_BORDER, x_center + width_norm / 2)
            y_min = max(0.0 + TO_BORDER, y_center - height_norm / 2)
            y_max = min(1.0 - TO_BORDER, y_center + height_norm / 2)

            bboxes.append({
                "class_id": augmented[1],
                "x_center": (x_min + x_max) / 2,
                "y_center": (y_min + y_max) / 2,
                "width": x_max - x_min,
                "height": y_max - y_min
            })
            used_targets[pixel_bbox["original_idx"]] = True
            failed_attempts = 0
        else:
            failed_attempts += 1
            if failed_attempts >= MAX_TARGET_FAILURE:
                break
    
    # 最终增强处理（保持原有逻辑）
    img_np = np.array(base)
    if random.random() < 0.5:
        img_np = apply_final_noise(img_np)
        
    base = Image.fromarray(img_np).convert("RGB")
    
    # 保存结果（保持原有逻辑）
    try:
        base.save(output_image_path, quality=95)
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                line = (
                    f"{bbox['class_id']} "
                    f"{bbox['x_center']:.6f} "
                    f"{bbox['y_center']:.6f} "
                    f"{bbox['width']:.6f} "
                    f"{bbox['height']:.6f}\n"
                )
                f.write(line)
    except Exception as e:
        print(f"处理失败：{real_path}，原因：{e}")

def apply_final_noise(img_np):
    """最终噪声增强"""
    # 高斯噪声
    if random.random() < 0.5:
        mean = 0
        var = random.uniform(1, 10)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    
    # 椒盐噪声
    if random.random() < 0.3:
        s_vs_p = 0.5
        amount = random.uniform(0.001, 0.005)
        out = np.copy(img_np)
        
        # 椒噪声
        num_salt = np.ceil(amount * img_np.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img_np.shape]
        out[tuple(coords)] = 0
        
        # 盐噪声
        num_pepper = np.ceil(amount * img_np.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img_np.shape]
        out[tuple(coords)] = 255
        
        img_np = out
        
    return img_np

def apply_cutout(img_np):
    """Cutout增强"""
    if random.random() < 0.3:
        mask_size = random.randint(20, 60)
        num_masks = random.randint(1, 3)
        h, w = img_np.shape[:2]
        
        for _ in range(num_masks):
            x = random.randint(0, w - mask_size)
            y = random.randint(0, h - mask_size)
            img_np[y:y+mask_size, x:x+mask_size] = np.random.randint(0, 256, (mask_size, mask_size, 3))
            
    return img_np

def one_epoch():
    """单轮生成"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_images = load_target_images()
    num_targets = len(target_images)
    used_targets = [False] * num_targets
    start_round = 0
    
    for round_num in tqdm(
        range(NUM_ROUNDS),
        desc=f"Epoch :{epoch + 1}",
        total=NUM_ROUNDS,
        dynamic_ncols=True,
        miniters=1
    ):
        if all(used_targets):
            print(f"Round {round_num}: 所有目标已用完，停止该轮处理")
            break
            
        process_round(round_num, target_images, used_targets)

def count_txt_files_recursive(directory):
    """统计标签文件数量"""
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.txt'):
                count += 1
    return count

def main(epochs=10):
    """主函数"""
    global epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        one_epoch()
        print(f"Epoch {epoch + 1} 完成！")
    print("所有轮次完成！")

if __name__ == "__main__":
    main(6)
    print(f"生成的标签文件数量：{count_txt_files_recursive(OUTPUT_DIR)}")