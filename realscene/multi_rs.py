import multiprocessing
import random
import os
import numpy as np
from tqdm import tqdm
import time
from PIL import Image, ImageEnhance
import cv2
import ctypes

# 全局参数配置
REAL_IMAGES_DIR = "/home/ling/zyt195/images/floor"
SYNTHETIC_DIR = "none"  # 合成靶标目录
REAL_SYNTHETIC_DIR = "/home/ling/zyt195/images/targets"  # 真实合成靶标目录
ORIGINAL_TARGET_DIR = "no"  # 原始识别区域目录
OUTPUT_DIR = "/home/ling/zyt195/images/ANOTRAIN"
EPOCHS = 1  # 总轮数
BASE_SIZE = (1280, 1280)  # YOLO训练尺寸
NUM_TARGETS_PER_IMAGE = 8  # 每张图像目标数量
MIN_CROP_RATIO = 0.4  # 最小裁剪比例
MIN_TARGET_RATIO = 0.2  # 目标最小占比
MAX_CROP_RATIO = 0.9  # 最大裁剪比例
MAX_TARGET_RATIO = 0.6  # 目标最大占比
MAX_TARGET_FAILURE = 8  # 最大失败次数
MAX_OVERLAP_ATTEMPTS = 20  # 最大重叠检测次数
TO_BORDER = 1e-6  # 边界安全距离
NUM_ROUNDS = 100  # 总生成轮数

APPLY_GEOMETRIC_AUG = False
APPLY_INK_REFLECTION = False  # 是否应用油墨反光效果
# 靶标类型定义
TARGET_TYPES = {
    "synthetic": 194,      # 合成靶标
    "real_synthetic": 194, # 真实合成靶标
    "original": 64         # 原始识别区域
}

CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
class_names = CLASSES  # 从config.py导入类别名称

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

def count_txt_files_recursive(directory):
    """统计标签文件数量"""
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.txt'):
                count += 1
    return count
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


def place_target(base, target_tuple, placed_bboxes):
    base = base.convert("RGBA")
    target, class_id, original_idx, target_type = target_tuple
    base_width, base_height = base.size
    target_width, target_height = target.size

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

# 共享内存管理器（Linux优化版）
class SharedTargetManager:
    def __init__(self, target_images):
        self.target_images = target_images
        self.num_targets = len(target_images)
        # 创建共享内存数组（用于标记已使用靶标）
        self.used_targets = multiprocessing.Array(ctypes.c_bool, [False] * self.num_targets)
        self.lock = multiprocessing.Lock()
    
    def get_available_target(self):
        """获取一个未使用的靶标（进程安全）"""
        with self.lock:
            available_indices = [i for i in range(self.num_targets) if not self.used_targets[i]]
            if not available_indices:
                return None, -1  # 所有靶标都已使用
            
            idx = random.choice(available_indices)
            self.used_targets[idx] = True  # 标记为已使用
            return self.target_images[idx], idx
    
    def all_targets_used(self):
        """检查是否所有靶标都已使用"""
        with self.lock:
            return all(self.used_targets)
    
    def release_target(self, idx):
        """释放靶标（标记为未使用）"""
        if 0 <= idx < self.num_targets:
            with self.lock:
                self.used_targets[idx] = False

# 全局共享管理器（每个epoch初始化）
shared_manager = None

def process_round(args):
    """处理单轮生成（使用全局共享管理器）"""
    round_num, epoch = args
    global shared_manager
    
    # 设置进程独立随机种子
    seed = (os.getpid() + int(time.time() * 1000) + round_num) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    
    # 检查是否所有目标都已使用
    if shared_manager.all_targets_used():
        return
    
    # 随机选择基底图片
    real_image_idx = random.choice(range(1, 104))
    real_path = os.path.join(REAL_IMAGES_DIR, f"{real_image_idx}.jpg")
    
    # 文件名加入进程ID防止冲突
    output_image_path = os.path.join(
        OUTPUT_DIR, 
        f"E{epoch}_P{os.getpid()}_R{round_num}_img{real_image_idx}.jpg"
    )
    label_path = output_image_path.replace(".jpg", ".txt")
    
    base = random_crop(real_path)
    bboxes = []
    placed_bboxes = []
    failed_attempts = 0
    
    # 记录本轮使用的靶标索引，用于错误处理
    used_in_round = []
    
    for _ in range(NUM_TARGETS_PER_IMAGE):
        # 获取可用靶标（进程安全）
        target_data = shared_manager.get_available_target()
        if not target_data:
            break  # 没有可用靶标
            
        target_tuple, target_idx = target_data
        used_in_round.append(target_idx)
        
        augmented = apply_augmentation(target_tuple)
        target, class_id, _, target_type = augmented
        
        # 动态计算缩放比例
        current_width, current_height = target.size
        min_dim = min(current_width, current_height)
        max_allowed_width = BASE_SIZE[0]
        max_allowed_height = BASE_SIZE[1]
        
        width_scale = max_allowed_width / current_width
        height_scale = max_allowed_height / current_height
        max_safe_scale = min(width_scale, height_scale)
        
        # 缩放逻辑
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
        
        # 创建包含原始索引的新元组
        new_target_tuple = (target, class_id, target_idx, target_type)
        placed_base, pixel_bbox = place_target(
            base.copy(),
            new_target_tuple,
            placed_bboxes
        )
        
        if pixel_bbox:
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

            # 修正标注框计算
            if target_type in ["synthetic", "real_synthetic"]:
                roi_x = 34 * target_width / 92.0
                roi_y = 34 * target_height / 92.0
                roi_w = 32 * target_width / 92.0
                roi_h = 32 * target_height / 92.0

                abs_x = pixel_bbox['x'] + roi_x
                abs_y = pixel_bbox['y'] + roi_y
                abs_w = roi_w
                abs_h = roi_h

            elif target_type == "original":
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
                "class_id": class_id,
                "x_center": (x_min + x_max) / 2,
                "y_center": (y_min + y_max) / 2,
                "width": x_max - x_min,
                "height": y_max - y_min
            })
            failed_attempts = 0
        else:
            # 放置失败时释放靶标
            shared_manager.release_target(target_idx)
            used_in_round.remove(target_idx)
            failed_attempts += 1
            if failed_attempts >= MAX_TARGET_FAILURE:
                break
    
    # 最终增强处理
    img_np = np.array(base)
    if random.random() < 0.5:
        img_np = apply_final_noise(img_np)
        
    base = Image.fromarray(img_np).convert("RGB")
    
    # 保存结果
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
        # 保存失败时释放所有使用的靶标
        for idx in used_in_round:
            shared_manager.release_target(idx)

def one_epoch():
    """并行化单轮生成（使用全局共享管理器）"""
    global shared_manager  # 声明全局变量
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_images = load_target_images()
    shared_manager = SharedTargetManager(target_images)  # 初始化全局管理器
    
    # 创建进程池
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(processes=num_cores)
    
    # 准备任务参数（不再传递manager）
    tasks = [(round_num, epoch) for round_num in range(NUM_ROUNDS)]

    # 使用tqdm进度条
    completed = 0
    early_termination = False
    with tqdm(total=NUM_ROUNDS, desc=f"Epoch {epoch+1}") as pbar:
        # 处理结果
        try:
            for result in pool.imap_unordered(process_round, tasks):
                pbar.update()
                completed += 1
                
                # 提前终止检查
                if shared_manager.all_targets_used() and not early_termination:
                    print(f"所有靶标已用完，提前终止轮次生成 (已完成 {completed}/{NUM_ROUNDS} 轮)")
                    early_termination = True
                    pool.terminate()  # 终止剩余任务
                    break
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            pool.terminate()
        finally:
            pool.close()
            pool.join()

def main(epochs=10):
    """主函数"""
    global epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        one_epoch()
        print(f"Epoch {epoch + 1} 完成！")
    print("所有轮次完成！")

if __name__ == "__main__":
    # 确保在Linux上正确使用fork
    if multiprocessing.get_start_method() != "fork":
        multiprocessing.set_start_method("fork", force=True)
    
    main(EPOCHS)
    print(f"生成的标签文件数量：{count_txt_files_recursive(OUTPUT_DIR)}")
    
    