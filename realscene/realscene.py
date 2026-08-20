import os
import sys
import random
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from tqdm import tqdm

# --- 载入统一配置（config.yaml 是唯一配置源）---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()

# 路径
REAL_IMAGES_DIR = CFG.paths.backgrounds     # 背景图目录
OUTPUT_DIR = CFG.paths.synth_output         # 合成结果

# 合成参数
_S = CFG.synth
EPOCHS = _S.epochs                          # 总轮数
BASE_SIZE = tuple(_S.base_size)             # YOLO训练尺寸
NUM_TARGETS_PER_IMAGE = _S.num_targets_per_image
MIN_CROP_RATIO = _S.min_crop_ratio
MIN_TARGET_RATIO = _S.min_target_ratio
MAX_CROP_RATIO = _S.max_crop_ratio
MAX_TARGET_RATIO = _S.max_target_ratio
MAX_TARGET_FAILURE = _S.max_target_failure
MAX_OVERLAP_ATTEMPTS = _S.max_overlap_attempts
TO_BORDER = float(_S.to_border)             # 边界安全距离
NUM_ROUNDS = _S.num_rounds
APPLY_GEOMETRIC_AUG = _S.apply_geometric_aug
APPLY_INK_REFLECTION = _S.apply_ink_reflection

# 目标类型：{类型名: {dir, roi, scale}}
#   roi=None            → 整张目标图就是检测框
#   roi=[rx,ry,rw,rh]   → 目标是底板，只框内部 ROI（相对目标框的比例）
TARGET_TYPES = dict(_S.target_types)
TARGET_ROI = {t: (list(spec["roi"]) if spec.get("roi") else None)
              for t, spec in TARGET_TYPES.items()}
TARGET_SCALE = {t: tuple(spec.get("scale", [0.5, 1.0])) for t, spec in TARGET_TYPES.items()}

CLASSES = list(CFG.classes)                 # 唯一类别源；下标即 class_id
class_names = CLASSES
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
        img_np = np.array(base).astype(np.float32) / 255.0
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv_factor = np.random.uniform(-0.1, 0.1, 3)
        img_hsv = np.clip(img_hsv + hsv_factor, 0, 1).astype(np.float32)
        base = Image.fromarray((cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB) * 255).astype(np.uint8))
    
    # Cutout增强
    if random.random() > 0.5:
        img_np = np.array(base)
        img_np = apply_cutout(img_np)
        base = Image.fromarray(img_np)
    return base

def list_background_images():
    """列出背景图目录下所有图片文件"""
    if not os.path.isdir(REAL_IMAGES_DIR):
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    for f in os.listdir(REAL_IMAGES_DIR):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(REAL_IMAGES_DIR, f))
    return files


def random_crop(image_path):
    """随机裁剪并增强"""
    img = Image.open(image_path)
    original_width, original_height = img.size

    if original_width < 2 or original_height < 2:
        return Image.new("RGB", BASE_SIZE, (128, 128, 128))

    crop_width = random.randint(
        max(1, int(original_width * MIN_CROP_RATIO)),
        max(1, int(original_width * MAX_CROP_RATIO))
    )
    crop_height = int(crop_width * 3 / 4)  # 保持4:3比例
    crop_height = min(crop_height, original_height)

    x = random.randint(0, max(1, original_width - crop_width))
    y = random.randint(0, max(1, original_height - crop_height))
    cropped = img.crop((x, y, x+crop_width, y+crop_height))
    resized_img = cropped.resize(BASE_SIZE, Image.LANCZOS)

    return apply_base_augmentation(resized_img)

def load_target_images():
    """加载所有类型目标（依据 config.yaml 的 synth.target_types）"""
    target_images = []
    # 遍历配置里声明、且实际存在的目标目录
    dir_type_map = []
    for target_type, spec in TARGET_TYPES.items():
        target_dir = spec.get("dir")
        if target_dir and os.path.isdir(target_dir):
            dir_type_map.append((target_dir, target_type))
        else:
            print(f"跳过 target_type '{target_type}'：目录不存在 {target_dir}")

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
    # 每类目标的初始缩放范围来自配置
    lo, hi = TARGET_SCALE.get(target_type, (0.5, 1.0))
    scale = random.uniform(lo, hi)

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
    base = base.convert("RGBA")
    target, class_id, original_idx, target_type = target_tuple
    base_width, base_height = base.size
    target_width, target_height = target.size

    # 如果 target 比底图还大，跳过放置
    if target_width > base_width or target_height > base_height:
        return base.convert("RGB"), None

    placed = False
    for _ in range(MAX_OVERLAP_ATTEMPTS * 2):
        # 使用随机生成策略确保坐标在图像内
        x = random.randint(0, base_width - target_width)
        y = random.randint(0, base_height - target_height)
        new_bbox = {"x": x, "y": y, "width": target_width, "height": target_height}

        # 检查重叠
        overlap = False
        safe_margin = max(target_width, target_height) * 0.1
        for bbox in placed_bboxes:
            dx = max(0, abs((x + target_width / 2) - (bbox['x'] + bbox['width'] / 2)) - (target_width + bbox['width']) / 2)
            dy = max(0, abs((y + target_height / 2) - (bbox['y'] + bbox['height'] / 2)) - (target_height + bbox['height']) / 2)
            if dx < safe_margin and dy < safe_margin:
                overlap = True
                break
        if not overlap:
            placed = True
            break

    if not placed:
        return base, None

    base.paste(target, (x, y), target.getchannel('A'))
    return (
        base.convert("RGB"),
        {
            "original_idx": original_idx,
            "x": x,
            "y": y,
            "width": target_width,
            "height": target_height,
            "target_type": target_type
        }
    )

def process_round(round_num, target_images, used_targets, bg_paths):
    """处理单轮生成"""
    # 提前检查是否所有目标都已使用
    if all(used_targets):
        return

    # 随机选择一张背景图片
    real_path = random.choice(bg_paths)
    fname = os.path.splitext(os.path.basename(real_path))[0]
    output_image_path = os.path.join(
        OUTPUT_DIR,
        f"E{epoch}_R{round_num}_{fname}.jpg"
    )
    label_path = output_image_path.replace(".jpg", ".txt")
    
    base = random_crop(real_path)
    bboxes = []
    placed_bboxes = []
    failed_attempts = 0
    
    # 创建空间分区
    grid_size = max(2, min(5, int(NUM_TARGETS_PER_IMAGE**0.5)))
    available_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(available_cells)
    
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
        
        placed_base, pixel_bbox = place_target(
            base.copy(),
            (target, *augmented[1:]),
            placed_bboxes
        )
        
        if pixel_bbox:
            # 添加实际靶标尺寸到已放置列表
            placed_bboxes.append({
                "x": pixel_bbox['x'],
                "y": pixel_bbox['y'],
                "width": pixel_bbox['width'],
                "height": pixel_bbox['height']
            })
            
            base = placed_base
            target_type = pixel_bbox['target_type']
            target_width = pixel_bbox['width']
            target_height = pixel_bbox['height']

            # ROI：roi=None → 整图即目标；roi=[rx,ry,rw,rh] → 只框底板内部区域
            roi = TARGET_ROI.get(target_type)
            if roi:
                rx, ry, rw, rh = roi
                abs_x = pixel_bbox['x'] + rx * target_width
                abs_y = pixel_bbox['y'] + ry * target_height
                abs_w = rw * target_width
                abs_h = rh * target_height
            else:
                abs_x = pixel_bbox['x']
                abs_y = pixel_bbox['y']
                abs_w = target_width
                abs_h = target_height

            # 计算归一化坐标
            x_center = (abs_x + abs_w / 2) / BASE_SIZE[0]
            y_center = (abs_y + abs_h / 2) / BASE_SIZE[1]
            width_norm = abs_w / BASE_SIZE[0]
            height_norm = abs_h / BASE_SIZE[1]

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

    # 泊松噪声
    if random.random() < 0.3:  # 30% 的概率添加泊松噪声
        intensity_scale = 0.1  # 控制噪声强度，建议 0.1~0.3
        # 将图像转换为浮点型以支持泊松分布
        img_float = img_np.astype(np.float32)
        # 生成泊松噪声，lambda = 像素值 * 强度系数
        noise = np.random.poisson(lam=img_float * intensity_scale)
        # 将噪声叠加到原始图像
        img_np = np.clip(img_float + noise, 0, 255).astype(np.uint8)  

    # 油墨反光
    if random.random() < 0.3:
        img_np = apply_ink_reflection(img_np)  
    return img_np

def apply_ink_reflection(img_np):
    """模拟油墨反光效果"""
    if not APPLY_INK_REFLECTION:
        return img_np
    h, w = img_np.shape[:2]
    # 创建一个空白的遮罩图层
    reflection_mask = np.zeros((h, w), dtype=np.float32)
        
    # 随机生成反光区域（椭圆形）
    center_x = random.randint(int(w * 0.2), int(w * 0.8))
    center_y = random.randint(int(h * 0.2), int(h * 0.8))
    radius_x = random.randint(10, 40)
    radius_y = random.randint(10, 40)
    angle = random.randint(0, 360)

    # 使用椭圆绘制反光区域
    cv2.ellipse(reflection_mask, (center_x, center_y), (radius_x, radius_y), angle, 0, 360, 1.0, -1)

    # 对遮罩进行高斯模糊，模拟扩散效果
    reflection_mask = cv2.GaussianBlur(reflection_mask, (21, 21), 0)

    # 增强亮度（模拟反光）
    brightness_factor = random.uniform(0.5, 1.5)
    reflection_mask = np.clip(reflection_mask * brightness_factor, 0, 1)

    # 将反光叠加到图像上（使用“亮光”混合模式）
    img_float = img_np.astype(np.float32) / 255.0
    img_float = np.clip(img_float + reflection_mask[..., None], 0, 1)
    img_np = (img_float * 255).astype(np.uint8)

    return img_np



def apply_cutout(img_np):
    """Cutout增强"""
    mask_size = random.randint(50, 100)
    num_masks = random.randint(1, 5)
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
    bg_paths = list_background_images()

    if not bg_paths:
        print("错误：backgrounds 目录下没有图片，请先用 run.py 拍摄背景图")
        return
    if not target_images:
        print("错误：没有加载到任何靶标图片，请先运行 generate_letters.py")
        return

    print(f"背景图: {len(bg_paths)} 张, 靶标: {len(target_images)} 个")
    num_targets = len(target_images)
    used_targets = [False] * num_targets

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

        process_round(round_num, target_images, used_targets, bg_paths)

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
    main(EPOCHS)
    print(f"生成的标签文件数量：{count_txt_files_recursive(OUTPUT_DIR)}")