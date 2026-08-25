import os
import sys
import random
from PIL import Image
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
BACKGROUND_RATIO = _S.get("background_ratio", 0.0)  # 空镜负样本比例
MIN_CROP_RATIO = _S.min_crop_ratio
MIN_TARGET_RATIO = _S.min_target_ratio
MAX_CROP_RATIO = _S.max_crop_ratio
MAX_TARGET_RATIO = _S.max_target_ratio
MAX_TARGET_FAILURE = _S.max_target_failure
MAX_OVERLAP_ATTEMPTS = _S.max_overlap_attempts
TO_BORDER = float(_S.to_border)             # 边界安全距离
NUM_ROUNDS = _S.num_rounds
JPEG_QUALITY = int(_S.get("jpeg_quality", 95))

# 增强菜单（from config.yaml synth.aug；校准 profile 已由 config.py 合并）
AUG_MENUS = dict(_S.get("aug", {}))

# 目标类型：{类型名: {dir, roi, scale, rotate, feather, crop_transparent}}
#   roi=None            → 整张目标图就是检测框
#   roi=[rx,ry,rw,rh]   → 目标是底板，只框内部 ROI（相对目标框的比例）
TARGET_TYPES = dict(_S.target_types)
TARGET_ROI = {t: (list(spec["roi"]) if spec.get("roi") else None)
              for t, spec in TARGET_TYPES.items()}
TARGET_SCALE = {t: tuple(spec.get("scale", [0.5, 1.0])) for t, spec in TARGET_TYPES.items()}
TARGET_ROTATE = {t: tuple(spec.get("rotate", [-45, 45])) for t, spec in TARGET_TYPES.items()}
TARGET_FEATHER = {t: int(spec.get("feather", 0)) for t, spec in TARGET_TYPES.items()}
TARGET_CROP_TRANSPARENT = {t: bool(spec.get("crop_transparent", True)) for t, spec in TARGET_TYPES.items()}

CLASSES = list(CFG.classes)                 # 唯一类别源；下标即 class_id
class_names = CLASSES


# ============================================================================
# 增强算子库（numpy/cv2；输入输出 np.uint8 RGB，需要保 alpha 时传入 RGBA）
# ============================================================================

def _f(img):
    return img.astype(np.float32)


def aug_brightness(img, range_):
    factor = random.uniform(*range_)
    return np.clip(_f(img) * factor, 0, 255).astype(np.uint8)


def aug_contrast(img, range_):
    factor = random.uniform(*range_)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((_f(img) - mean) * factor + mean, 0, 255).astype(np.uint8)


def aug_color(img, range_):
    factor = random.uniform(*range_)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray3 = np.stack([gray] * 3, axis=2).astype(np.float32)
    return np.clip(_f(img) * factor + gray3 * (1.0 - factor), 0, 255).astype(np.uint8)


def aug_sharpness(img, range_):
    amount = random.uniform(*range_)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return np.clip(cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0), 0, 255).astype(np.uint8)


def aug_hsv(img, delta=0.1):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    hsv = np.clip(hsv + np.random.uniform(-delta, delta, 3), 0, 1)
    return cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def aug_cutout(img, mask_size=(50, 100), num_masks=(1, 5)):
    out = img.copy()
    h, w = out.shape[:2]
    ms = max(4, min(w, h) // 4)
    for _ in range(random.randint(*num_masks)):
        size = random.randint(mask_size[0], min(mask_size[1], ms))
        x = random.randint(0, max(0, w - size))
        y = random.randint(0, max(0, h - size))
        out[y:y + size, x:x + size] = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return out


def aug_geometric(img, distortion=0.1):
    """透视扰动（默认关闭；开启后目标是贴入前的背景层，不影响标注框）。"""
    h, w = img.shape[:2]
    dx = lambda: random.randint(0, int(w * distortion))
    dy = lambda: random.randint(0, int(h * distortion))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx(), dy()], [w - dx(), dy()], [w - dx(), h - dy()], [dx(), h - dy()]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def aug_flip(img, *a, **kw):
    return img[:, ::-1].copy()


def aug_gaussian(img, var=(1, 10)):
    sigma = random.uniform(var[0], var[1]) ** 0.5
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(_f(img) + noise, 0, 255).astype(np.uint8)


def aug_salt_pepper(img, amount=(0.001, 0.005)):
    out = img.copy()
    a = random.uniform(*amount)
    n = int(np.ceil(a * out.size * 0.5))
    for pix in (0, 255):
        coords = [np.random.randint(0, max(1, i - 1), n) for i in out.shape]
        out[tuple(coords)] = pix
    return out


def aug_poisson(img, intensity=0.1):
    f = _f(img)
    noise = np.random.poisson(lam=f * intensity)
    return np.clip(f + noise, 0, 255).astype(np.uint8)


def aug_ink_reflection(img, *a, **kw):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    rx, ry = random.randint(10, 40), random.randint(10, 40)
    cv2.ellipse(mask, (cx, cy), (rx, ry), random.randint(0, 360), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = np.clip(mask * random.uniform(0.5, 1.5), 0, 1)
    return np.clip(_f(img) + mask[..., None] * 255, 0, 255).astype(np.uint8)


APPLY_MAP = {
    "brightness": aug_brightness, "contrast": aug_contrast, "color": aug_color,
    "sharpness": aug_sharpness, "hsv": aug_hsv, "cutout": aug_cutout,
    "geometric": aug_geometric, "flip": aug_flip,
    "gaussian": aug_gaussian, "salt_pepper": aug_salt_pepper,
    "poisson": aug_poisson, "ink_reflection": aug_ink_reflection,
}


def apply_aug_menu(img, menu_def):
    """按菜单应用增强。img 为 np.uint8 RGB（RGBA 时自动分离 alpha 处理 RGB，再拼回）。
    menu_def 带 num → 每图抽 num 种 + extra_prob 补抽；不带 → 每项独立 prob。"""
    items = menu_def.get("menu", [])
    chosen = list(items)
    if "num" in menu_def:
        n = random.randint(*menu_def["num"])
        chosen = random.sample(items, min(n, len(items)))
        if random.random() < menu_def.get("extra_prob", 0.0) and len(chosen) < len(items):
            chosen.append(random.choice([i for i in items if i not in chosen]))
    for item in chosen:
        if random.random() < item.get("prob", 1.0):
            # config 键用 range，算子参数用 range_（避免与内建 range 冲突）
            kwargs = {("range_" if k == "range" else k): v
                      for k, v in item.items() if k not in ("type", "prob")}
            func = APPLY_MAP.get(item["type"])
            if func is None:
                print(f"⚠️ 未知增强类型：{item['type']}，跳过")
                continue
            is_rgba = img.ndim == 3 and img.shape[2] == 4
            if is_rgba:
                rgb, alpha = img[:, :, :3].copy(), img[:, :, 3]
                rgb = func(rgb, **kwargs)
                if item["type"] == "flip":
                    alpha = alpha[:, ::-1].copy()
                img = np.dstack([rgb, alpha])
            else:
                img = func(img, **kwargs)
    return img


def _build_edge_feather_alpha(w, h, radius):
    """边缘羽化 alpha：离边缘 radius 内线性渐变（抠图硬边 → 柔和）。"""
    if radius <= 0:
        return np.full((h, w), 255, dtype=np.uint8)
    yy, xx = np.indices((h, w), dtype=np.float32)
    dist = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy])
    alpha = np.clip((dist + 1.0) / float(radius), 0.0, 1.0)
    return (alpha * 255.0).astype(np.uint8)


def _crop_transparent(img, room=0):
    """裁掉 alpha=0 的透明边距（保留 room px 余量）。返回裁剪后图；无透明边距则原样。"""
    if img.shape[2] != 4:
        return img
    a = img[:, :, 3]
    if a.all() or not a.any():
        return img
    ys, xs = np.where(a > 0)
    y0, y1 = max(0, ys.min() - room), min(a.shape[0], ys.max() + 1 + room)
    x0, x1 = max(0, xs.min() - room), min(a.shape[1], xs.max() + 1 + room)
    return img[y0:y1, x0:x1]


def _rotate_rgba(img, angle):
    """旋转 RGBA（RGB 双线性、alpha 邻近），返回 (旋转后图, expand 画布内紧致 bbox 或 None)。
    bbox 为 (x1,y1,x2,y2)，相对 expand 画布。"""
    if abs(angle) < 0.01:
        return img, None
    h, w = img.shape[:2]
    cos_a, sin_a = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
    # 消除 cos(90°)≈6e-17 类浮点误差：接近整值(0/1)时钳位
    if cos_a < 1e-10: cos_a = 0.0
    if sin_a < 1e-10: sin_a = 0.0
    if abs(cos_a - 1.0) < 1e-10: cos_a = 1.0
    if abs(sin_a - 1.0) < 1e-10: sin_a = 1.0
    max_w = int(np.ceil(w * cos_a + h * sin_a))
    max_h = int(np.ceil(w * sin_a + h * cos_a))
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m[0, 2] += (max_w - w) / 2.0
    m[1, 2] += (max_h - h) / 2.0
    rgb = cv2.warpAffine(img[:, :, :3], m, (max_w, max_h), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = cv2.warpAffine(img[:, :, 3], m, (max_w, max_h), flags=cv2.INTER_NEAREST, borderValue=0)
    # 紧致 bbox：原图四角经过旋转+平移后取轴对齐（定义域为 expand 画布）
    pts = np.float32([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).reshape(-1, 3)
    rot = pts @ m.T
    x1, y1 = int(np.floor(rot[:, 0].min())), int(np.floor(rot[:, 1].min()))
    x2, y2 = int(np.ceil(rot[:, 0].max())), int(np.ceil(rot[:, 1].max()))
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(max_w, x2); y2 = min(max_h, y2)
    return np.dstack([rgb, alpha]), (x1, y1, x2, y2)


# ============================================================================
# 流水线
# ============================================================================

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
    resized = cv2.resize(np.array(cropped), BASE_SIZE, interpolation=cv2.INTER_LANCZOS4)

    return apply_aug_menu(resized, AUG_MENUS.get("base", {"menu": []}))


def load_target_images():
    """加载所有类型目标（依据 config.yaml 的 synth.target_types）。

    目标目录顶层支持两种布局：
      <dir>/<类名>/xxx.png   文件夹类，类名 = 子目录名（文件可任意嵌套）
      <dir>/<类名>.png       单图类，类名 = 文件名去掉扩展名（无需建文件夹）
    同名冲突（同名的文件夹和图片文件同时存在）→ 报错退出。
    """
    target_images = []
    dir_type_map = []
    for target_type, spec in TARGET_TYPES.items():
        target_dir = spec.get("dir")
        if target_dir and os.path.isdir(target_dir):
            dir_type_map.append((target_dir, target_type))
        else:
            print(f"跳过 target_type '{target_type}'：目录不存在 {target_dir}")

    IMG_EXT = ('.png', '.jpg', '.jpeg')

    for target_dir, target_type in dir_type_map:
        dir_classes, file_classes = [], []
        for entry in sorted(os.listdir(target_dir)):
            full = os.path.join(target_dir, entry)
            if os.path.isdir(full):
                dir_classes.append(entry)
            elif entry.lower().endswith(IMG_EXT):
                file_classes.append(os.path.splitext(entry)[0])

        # 同名冲突：文件夹类名 与 单图文件名（任意扩展名）同时存在 → 报错退出
        conflicts = [
            dc for dc in dir_classes
            if any(os.path.exists(os.path.join(target_dir, dc + e)) for e in IMG_EXT)
        ]
        if conflicts:
            print(f"❌ '{target_dir}' 下类名冲突（既是文件夹又是单图）：{conflicts}")
            print("   请只保留一种布局：要么删除文件夹（把图平铺到该目录），要么删除同名图片文件。")
            sys.exit(1)

        def add_image(path, category):
            nonlocal target_images
            try:
                img = np.array(Image.open(path).convert("RGBA"))
                if TARGET_CROP_TRANSPARENT.get(target_type, True):
                    img = _crop_transparent(img)
                class_id = class_names.index(category)
                target_images.append((img, class_id, len(target_images), target_type))
            except Exception as e:
                print(f"加载失败：{path}，原因：{e}")

        for dc in dir_classes:
            class_dir = os.path.join(target_dir, dc)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(IMG_EXT):
                        add_image(os.path.join(root, file), dc)   # 类名 = 顶层目录名
        for fc in file_classes:
            for e in IMG_EXT:
                path = os.path.join(target_dir, fc + e)
                if os.path.exists(path):
                    add_image(path, fc)                          # 类名 = 文件 stem
    return target_images


def apply_augmentation(target_tuple):
    """目标增强：透明裁剪 → 缩放 → RGB 菜单增强+翻转 → 旋转（带紧框 bbox）。"""
    img, class_id, original_idx, target_type = target_tuple
    scale_lo, scale_hi = TARGET_SCALE.get(target_type, (0.5, 1.0))
    scale = random.uniform(scale_lo, scale_hi)
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    img = apply_aug_menu(img, AUG_MENUS.get("target", {"menu": []}))

    rot_lo, rot_hi = TARGET_ROTATE.get(target_type, (-45, 45))
    angle = round(random.uniform(rot_lo, rot_hi), 1)
    img, tight = _rotate_rgba(img, angle)

    return (img, class_id, original_idx, target_type, tight)


def place_target(base, target_tuple, placed_bboxes):
    """把增强后的目标贴到 base（numpy alpha 融合）。返回 (base, bbox dict 或 None)。
    bbox 为粘贴后的检测框（紧框目标 = 物体真实旋转后范围；否则 = 粘贴画布整张）。"""
    base_rgba = base  # np RGBA
    target, class_id, original_idx, target_type, tight = target_tuple
    bh, bw = base_rgba.shape[:2]
    th, tw = target.shape[:2]

    # 超出画布直接放弃
    if tw > bw or th > bh:
        return base, None

    placed = False
    for _ in range(MAX_OVERLAP_ATTEMPTS * 2):
        x = random.randint(0, bw - tw)
        y = random.randint(0, bh - th)

        overlap = False
        safe_margin = max(tw, th) * 0.1
        for bbox in placed_bboxes:
            dx = max(0, abs((x + tw / 2) - (bbox['x'] + bbox['width'] / 2)) - (tw + bbox['width']) / 2)
            dy = max(0, abs((y + th / 2) - (bbox['y'] + bbox['height'] / 2)) - (th + bbox['height']) / 2)
            if dx < safe_margin and dy < safe_margin:
                overlap = True
                break
        if not overlap:
            placed = True
            break

    if not placed:
        return base, None

    # 相邻区域 alpha 融合（含边缘羽化）
    alpha = target[:, :, 3].astype(np.float32)
    feather = TARGET_FEATHER.get(target_type, 0)
    if feather > 0:
        alpha = alpha * _build_edge_feather_alpha(tw, th, feather).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0, 255).astype(np.float32) / 255.0
    region = base_rgba[y:y + th, x:x + tw].astype(np.float32)
    src = target[:, :, :3].astype(np.float32)
    blended = alpha[..., None] * src + (1.0 - alpha[..., None]) * region
    base_rgba[y:y + th, x:x + tw] = np.clip(blended, 0, 255).astype(np.uint8)

    # 检测框：紧框目标用物体真实范围（相对粘贴画布），否则整张
    if tight:
        bx1, by1, bx2, by2 = tight
        pixel_bbox = {
            "original_idx": original_idx,
            "x": x + bx1, "y": y + by1,
            "width": bx2 - bx1, "height": by2 - by1,
            "target_type": target_type,
        }
    else:
        pixel_bbox = {
            "original_idx": original_idx,
            "x": x, "y": y, "width": tw, "height": th,
            "target_type": target_type,
        }
    return base_rgba.copy(), pixel_bbox


def process_round(round_num, target_images, used_targets, bg_paths):
    """处理单轮生成"""
    if all(used_targets):
        return

    real_path = random.choice(bg_paths)
    fname = os.path.splitext(os.path.basename(real_path))[0]
    output_image_path = os.path.join(OUTPUT_DIR, f"E{epoch}_R{round_num}_{fname}.jpg")
    label_path = output_image_path.replace(".jpg", ".txt")

    base = random_crop(real_path)
    bboxes = []
    placed_bboxes = []
    failed_attempts = 0

    target_count = 0 if random.random() < BACKGROUND_RATIO else NUM_TARGETS_PER_IMAGE

    for _ in range(target_count):
        available_targets = [t for t in target_images if not used_targets[t[2]]]
        if not available_targets:
            break

        target_tuple = random.choice(available_targets)
        augmented = apply_augmentation(target_tuple)
        target, class_id, _, target_type, tight = augmented
        tw, th = target.shape[1], target.shape[0]

        # 目标缩放到占画布 MIN/MAX 比例
        min_dim = min(tw, th)
        max_safe_scale = min(BASE_SIZE[0] / tw, BASE_SIZE[1] / th)
        min_scale = (MIN_TARGET_RATIO * min(BASE_SIZE)) / min_dim
        max_scale = min((MAX_TARGET_RATIO * min(BASE_SIZE)) / min_dim, max_safe_scale)
        if max_scale <= 0 or min_scale > max_scale:
            failed_attempts += 1
            if failed_attempts >= MAX_TARGET_FAILURE:
                break
            continue
        scale = random.uniform(min_scale, max_scale)
        new_w, new_h = max(1, int(tw * scale)), max(1, int(th * scale))
        target = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # 紧框随缩放等比放大（子图尺寸）
        if tight:
            bx1, by1, bx2, by2 = tight
            tight = (int(round(bx1 * scale)), int(round(by1 * scale)),
                     int(round(bx2 * scale)), int(round(by2 * scale)))

        augmented = (target, class_id, _, target_type, tight)
        placed_base, pixel_bbox = place_target(base.copy(), augmented, placed_bboxes)

        if pixel_bbox:
            placed_bboxes.append({
                "x": pixel_bbox['x'], "y": pixel_bbox['y'],
                "width": pixel_bbox['width'], "height": pixel_bbox['height'],
            })
            base = placed_base

            target_type = pixel_bbox['target_type']
            target_width = pixel_bbox['width']
            target_height = pixel_bbox['height']

            # ROI：roi=None → 紧框/整图即目标；roi=[rx,ry,rw,rh] → 只框底板内部区域
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

            x_center = (abs_x + abs_w / 2) / BASE_SIZE[0]
            y_center = (abs_y + abs_h / 2) / BASE_SIZE[1]
            width_norm = abs_w / BASE_SIZE[0]
            height_norm = abs_h / BASE_SIZE[1]

            x_min = max(0.0 + TO_BORDER, x_center - width_norm / 2)
            x_max = min(1.0 - TO_BORDER, x_center + width_norm / 2)
            y_min = max(0.0 + TO_BORDER, y_center - height_norm / 2)
            y_max = min(1.0 - TO_BORDER, y_center + height_norm / 2)

            bboxes.append({
                "class_id": class_id,
                "x_center": (x_min + x_max) / 2,
                "y_center": (y_min + y_max) / 2,
                "width": x_max - x_min,
                "height": y_max - y_min,
            })
            used_targets[pixel_bbox["original_idx"]] = True
            failed_attempts = 0
        else:
            failed_attempts += 1
            if failed_attempts >= MAX_TARGET_FAILURE:
                break

    # 整图噪声
    img_np = base[:, :, :3]
    img_np = apply_aug_menu(img_np, AUG_MENUS.get("final", {"menu": []}))
    result = Image.fromarray(img_np)

    # 保存
    try:
        result.save(output_image_path, quality=JPEG_QUALITY)
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


def one_epoch():
    """单轮生成"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_images = load_target_images()
    bg_paths = list_background_images()

    if not bg_paths:
        print("错误：backgrounds 目录下没有图片，请先用 run.py 拍摄背景图")
        return
    if not target_images:
        print("错误：没有加载到任何靶标图片，请先生成目标图")
        return

    print(f"背景图: {len(bg_paths)} 张, 靶标: {len(target_images)} 个")
    used_targets = [False] * len(target_images)

    for round_num in tqdm(range(NUM_ROUNDS), desc=f"Epoch :{epoch + 1}",
                          total=NUM_ROUNDS, dynamic_ncols=True, miniters=1):
        if all(used_targets):
            print(f"Round {round_num}: 所有目标已用完，停止该轮处理")
            break
        process_round(round_num, target_images, used_targets, bg_paths)


def count_txt_files_recursive(directory):
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
