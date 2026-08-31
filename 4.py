import os
import cv2
import random
import numpy as np
import shutil
import yaml
from tqdm import tqdm
import multiprocessing
import argparse


def load_images_and_labels(root_dir):
    backgrounds = []
    image_paths = []
    label_paths = []
    class_names = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")

    required_subdirs = ["background", "label", "image"]
    for subdir in required_subdirs:
        if not os.path.exists(os.path.join(root_dir, subdir)):
            raise FileNotFoundError(f"必需的子目录 '{subdir}' 不存在于 {root_dir}")

    classes_txt_candidates = [
        os.path.join(root_dir, "classes.txt"),
        os.path.join(root_dir, "label", "classes.txt"),
    ]
    classes_txt_path = next((p for p in classes_txt_candidates if os.path.exists(p)), None)
    if classes_txt_path is None:
        raise FileNotFoundError(f"未找到类别文件: {classes_txt_candidates[0]}")
    with open(classes_txt_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    background_dir = os.path.join(root_dir, "background")
    for bg_file in os.listdir(background_dir):
        if bg_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            bg_img = cv2.imread(os.path.join(background_dir, bg_file))
            if bg_img is not None:
                backgrounds.append(bg_img)
    if not backgrounds:
        print("警告: 没有找到背景图片")

    image_dir = os.path.join(root_dir, "image")
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(root_dir, "label", os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(label_path):
                image_paths.append(img_path)
                label_paths.append(label_path)
            else:
                print(f"警告: 图片 {img_file} 没有对应的标签文件")

    return backgrounds, image_paths, label_paths, class_names


def apply_motion_blur_cuda(image, blur_length, blur_angle):
    """
    CUDA加速的动态模糊
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        kernel = np.zeros((blur_length, blur_length), dtype=np.float32)
        center = blur_length // 2
        angle_rad = np.deg2rad(blur_angle)
        dx = int(np.cos(angle_rad) * center)
        dy = int(np.sin(angle_rad) * center)
        x1, y1 = center - dx, center - dy
        x2, y2 = center + dx, center + dy
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)
        kernel = kernel / np.sum(kernel)

        gpu_kernel = cv2.cuda_GpuMat()
        gpu_kernel.upload(kernel)

        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        gpu_result = cv2.cuda.filter2D(gpu_image, -1, gpu_kernel)
        return gpu_result.download()
    else:
        return apply_motion_blur_cpu(image, blur_length, blur_angle)


def apply_motion_blur_cpu(image, blur_length, blur_angle):
    """
    CPU版本的动态模糊
    """
    angle_rad = np.deg2rad(blur_angle)
    kernel = np.zeros((blur_length, blur_length), dtype=np.float32)
    center = blur_length // 2
    dx = int(np.cos(angle_rad) * center)
    dy = int(np.sin(angle_rad) * center)
    x1, y1 = center - dx, center - dy
    x2, y2 = center + dx, center + dy
    cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)
    kernel = kernel / np.sum(kernel)

    return cv2.filter2D(image, -1, kernel)


def rotate_and_scale_image_cuda(image, angle, scale):
    """
    CUDA加速的旋转缩放（保持透明区域）
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        gpu_M = cv2.cuda_GpuMat(M)
        gpu_rotated = cv2.cuda.warpAffine(gpu_image, gpu_M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        rotated = gpu_rotated.download()
        return rotated, M, (new_w, new_h)
    else:
        return rotate_and_scale_image_cpu(image, angle, scale)


def rotate_and_scale_image_cpu(image, angle, scale):
    """
    CPU版本的旋转缩放（保持透明区域）
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated, M, (new_w, new_h)


def transform_bbox(bbox, img_size, transform_matrix, new_img_size):
    """将单个YOLO bbox按照仿射变换映射到新图像坐标系"""
    if len(bbox) != 4:
        print(f"警告: bbox格式不正确: {bbox}")
        return [0.5, 0.5, 0.1, 0.1]

    x_center, y_center, width, height = bbox
    orig_h, orig_w = img_size
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        print(f"警告: 无效的边界框坐标: {bbox}")
        return [0.5, 0.5, 0.1, 0.1]

    abs_x = x_center * orig_w
    abs_y = y_center * orig_h
    abs_w = width * orig_w
    abs_h = height * orig_h
    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2

    points = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1],
    ])
    transformed_points = np.dot(points, transform_matrix.T)

    new_x1 = np.min(transformed_points[:, 0])
    new_y1 = np.min(transformed_points[:, 1])
    new_x2 = np.max(transformed_points[:, 0])
    new_y2 = np.max(transformed_points[:, 1])
    new_w, new_h = new_img_size

    new_x_center = ((new_x1 + new_x2) / 2) / new_w
    new_y_center = ((new_y1 + new_y2) / 2) / new_h
    new_width = (new_x2 - new_x1) / new_w
    new_height = (new_y2 - new_y1) / new_h
    new_width = max(0.02, min(1.0, new_width))
    new_height = max(0.02, min(1.0, new_height))
    return [new_x_center, new_y_center, new_width, new_height]


def blend_with_alpha_cuda(background, foreground, x, y):
    """
    CUDA加速的透明混合
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        bg_gpu = cv2.cuda_GpuMat()
        bg_gpu.upload(background)

        fg_gpu = cv2.cuda_GpuMat()
        fg_gpu.upload(foreground)

        result = cv2.cuda.addWeighted(bg_gpu, 1.0, fg_gpu, 1.0, 0.0)
        return result.download()
    else:
        return blend_with_alpha_cpu(background, foreground, x, y)


def blend_with_alpha_cpu(background, foreground, x, y):
    """
    CPU版本的透明混合
    """
    if foreground.shape[2] == 3:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)

    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)
    if x1 >= x2 or y1 >= y2:
        return background

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)
    fg_roi = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
    fg_alpha = fg_roi[:, :, 3] / 255.0
    fg_alpha = np.expand_dims(fg_alpha, axis=-1)
    bg_roi = background[y1:y2, x1:x2]
    blended = fg_roi[:, :, :3] * fg_alpha + bg_roi * (1 - fg_alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def draw_yolo_labels(image, labels, class_names):
    """
    在图像上绘制YOLO格式的标签
    """
    h, w = image.shape[:2]
    for label in labels:
        if len(label) == 5:
            class_id, x_center, y_center, width, height = label
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"

            x = int(x_center * w)
            y = int(y_center * h)
            box_w = int(width * w)
            box_h = int(height * h)

            x1 = x - box_w // 2
            y1 = y - box_h // 2
            x2 = x + box_w // 2
            y2 = y + box_h // 2

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x1
            text_y = y1 - baseline - 5
            if text_y - text_height < 0:
                text_y = y2 + text_height + baseline + 5

            text_x = max(0, text_x)
            text_y = min(h - 1, text_y)
            cv2.rectangle(image,
                          (text_x, text_y - text_height - baseline),
                          (text_x + text_width, text_y + baseline),
                          color, -1)
            cv2.putText(image, class_name, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def process_one_image(args):
    i, backgrounds, images, labels, class_names, output_dir, draw_labels, debug, angle_min, angle_max = args
    output_files = []

    bg = random.choice(backgrounds).copy()
    bg_h, bg_w = bg.shape[:2]

    num_blocks = random.randint(8, 20)
    for _ in range(num_blocks):
        block_w = random.randint(bg_w // 30, bg_w // 8)
        block_h = random.randint(bg_h // 30, bg_h // 8)
        x1 = random.randint(0, bg_w - block_w)
        y1 = random.randint(0, bg_h - block_h)
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(bg, (x1, y1), (x1 + block_w, y1 + block_h), color, thickness=-1)

    if random.random() < 0.5:
        noise = np.random.randint(0, 50, (bg_h, bg_w, 3), dtype=np.uint8)
        bg = cv2.add(bg, noise)

    num_objects = random.randint(1, min(5, len(images)))
    placed_objects = []
    output_labels = []

    for _ in range(num_objects):
        idx = random.randint(0, len(images) - 1)
        img_path = images[idx]
        label_path = labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        apply_blur = random.random() < 0.3
        original_img = img.copy()
        if apply_blur:
            blur_length = random.randint(5, 15)
            blur_angle = random.randint(0, 179)
            rgb_channel = img[:, :, :3]
            alpha_channel = img[:, :, 3] if img.shape[2] == 4 else None
            blurred_rgb = apply_motion_blur_cuda(rgb_channel, blur_length, blur_angle)
            if alpha_channel is not None:
                img = np.dstack([blurred_rgb, alpha_channel])
            else:
                img = blurred_rgb

        img_h, img_w = img.shape[:2]
        scale = random.uniform(0.5, 1.0) 
        rotation_angle = random.uniform(angle_min, angle_max)
        rotated_img, M, (new_w, new_h) = rotate_and_scale_image_cuda(img, rotation_angle, scale)

        try:
            with open(label_path, 'r') as f:
                bbox_lines = f.readlines()
        except Exception as e:
            print(f"读取标签文件错误: {label_path}, {e}")
            continue

        transformed_bboxes = []
        orig_img_h, orig_img_w = original_img.shape[:2]
        for line in bbox_lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    bbox = list(map(float, parts[1:]))
                    transformed_bbox = transform_bbox(bbox, (orig_img_h, orig_img_w), M, (new_w, new_h))
                    transformed_bboxes.append((int(parts[0]), transformed_bbox))
                except Exception as e:
                    print(f"标签变换错误: {e}, bbox: {bbox}")
                    continue
            else:
                print(f"标签格式错误: {line.strip()}")
                continue

        if not transformed_bboxes:
            continue

        max_attempts = 50
        placed = False
        for _ in range(max_attempts):
            x = random.randint(0, max(1, bg_w - new_w))
            y = random.randint(0, max(1, bg_h - new_h))
            overlap = False
            new_rect = (x, y, x + new_w, y + new_h)
            for rect in placed_objects:
                if not (new_rect[2] < rect[0] or new_rect[0] > rect[2] or new_rect[3] < rect[1] or new_rect[1] > rect[3]):
                    overlap = True
                    break
            if not overlap:
                bg = blend_with_alpha_cuda(bg, rotated_img, x, y)
                placed_objects.append(new_rect)

                for class_id, bbox in transformed_bboxes:
                    new_x_center = (bbox[0] * new_w + x) / bg_w
                    new_y_center = (bbox[1] * new_h + y) / bg_h
                    new_width = bbox[2] * new_w / bg_w
                    new_height = bbox[3] * new_h / bg_h
                    new_x_center = max(0.0, min(1.0, new_x_center))
                    new_y_center = max(0.0, min(1.0, new_y_center))
                    new_width = max(0.02, min(1.0, new_width))
                    new_height = max(0.02, min(1.0, new_height))
                    output_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
                placed = True
                break
        if not placed:
            continue

    output_filename = f"aug_{i:04d}"
    is_train = random.random() < 0.8
    split_dir = "train" if is_train else "val"

    output_img_path = os.path.join(output_dir, "images", split_dir, output_filename + ".jpg")
    if draw_labels or debug:
        label_data = []
        for line in output_labels:
            parts = line.strip().split()
            if len(parts) == 5:
                label_data.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        annotated_bg = draw_yolo_labels(bg.copy(), label_data, class_names)
        if draw_labels:
            cv2.imwrite(output_img_path, annotated_bg)
        else:
            cv2.imwrite(output_img_path, bg)
        if debug:
            debug_dir = os.path.join(output_dir, "带上标记框的生成后的图片", split_dir)
            os.makedirs(debug_dir, exist_ok=True)
            debug_img_path = os.path.join(debug_dir, output_filename + ".jpg")
            cv2.imwrite(debug_img_path, annotated_bg)
    else:
        cv2.imwrite(output_img_path, bg)

    output_label_path = os.path.join(output_dir, "labels", split_dir, output_filename + ".txt")
    with open(output_label_path, 'w') as f:
        f.writelines(output_labels)
    output_files.append(output_img_path)
    return output_files


def place_images_on_background(backgrounds, images, labels, class_names, output_dir, num_output_images=10, draw_labels=False, debug=False, angle_min=0.0, angle_max=360.0):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    if debug:
        os.makedirs(os.path.join(output_dir, "带上标记框的生成后的图片", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "带上标记框的生成后的图片", "val"), exist_ok=True)

    all_output_files = []
    args_list = [
        (i, backgrounds, images, labels, class_names, output_dir, draw_labels, debug, angle_min, angle_max)
        for i in range(num_output_images)
    ]
    cpu_count = min(multiprocessing.cpu_count(), 24)
    with multiprocessing.Pool(cpu_count) as pool:
        for result in tqdm(pool.imap_unordered(process_one_image, args_list), total=num_output_images, desc="多进程增强数据"):
            if result:
                all_output_files.extend(result)
    return all_output_files


def create_yaml_file(output_dir, class_names):
    yaml_content = {
        'train': os.path.join('.', 'images', 'train'),
        'val': os.path.join('.', 'images', 'val'),
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)


def create_class_file(output_dir, class_names):
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    class_file_path = os.path.join(label_dir, "classes.txt")
    with open(class_file_path, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}\n")
    print(f"已创建类别文件: {class_file_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO数据增强工具')
    parser.add_argument('--draw_labels', action='store_true', help='在生成的图片上直接绘制YOLO标签')
    parser.add_argument('--debug', action='store_true', help='额外保存带标注框的生成图片到 output_dir/带上标记框的生成后的图片')
    parser.add_argument('--angle_min', type=float, default=-5.0, help='贴上对象的随机旋转最小角度（度）')
    parser.add_argument('--angle_max', type=float, default=5.0, help='贴上对象的随机旋转最大角度（度）')
    parser.add_argument('--angle_range', nargs=2, type=float, metavar=('MIN', 'MAX'), help='旋转角度范围，格式：--angle_range 0 180')
    parser.add_argument('--num_output_images', type=int, default=36000, help='输出图片数量')
    args = parser.parse_args()

    if args.angle_range is not None:
        args.angle_min, args.angle_max = args.angle_range

    if args.angle_min > args.angle_max:
        args.angle_min, args.angle_max = args.angle_max, args.angle_min

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input_data")
    output_dir = os.path.join(current_dir, "yolo_dataset")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"绘制标签: {args.draw_labels}")
    print(f"调试模式: {args.debug}")
    print(f"旋转角度范围: [{args.angle_min}, {args.angle_max}] 度")

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"检测到CUDA设备，可用GPU数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
        print(f"GPU名称: {cv2.cuda.getDeviceName(0)}")
    else:
        print("未检测到CUDA设备，将使用CPU优化版本")

    try:
        backgrounds, image_paths, label_paths, class_names = load_images_and_labels(input_dir)
        if not backgrounds:
            print("错误：没有找到背景图片")
            return
        if not image_paths:
            print("错误：没有找到类别图片")
            return
        if not class_names:
            print("错误：没有找到类别文件夹")
            return
        print(f"找到 {len(backgrounds)} 张背景图片")
        print(f"找到 {len(image_paths)} 张类别图片")
        print(f"找到 {len(class_names)} 个类别: {', '.join(class_names)}")
        print("开始生成增强数据集...")

        all_output_files = place_images_on_background(
            backgrounds,
            image_paths,
            label_paths,
            class_names,
            output_dir,
            num_output_images=args.num_output_images,
            draw_labels=args.draw_labels,
            debug=args.debug,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
        )
        create_yaml_file(output_dir, class_names)
        create_class_file(output_dir, class_names)
        print(f"\n数据集生成完成，共生成 {len(all_output_files)} 张图片")
        print(f"数据集已保存到: {output_dir}")
        if args.debug:
            print(f"调试标注图已保存到: {os.path.join(output_dir, '带上标记框的生成后的图片')}")
        print(f"YAML配置文件已生成: {os.path.join(output_dir, 'dataset.yaml')}")
        print(f"类别文件已生成: {os.path.join(output_dir, 'labels', 'classes.txt')}")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("请检查:")
        print(f"1. 输入目录 {input_dir} 是否存在")
        print("2. 输入目录结构是否正确:")
        print("   input_data/")
        print("   ├── background/       # 背景图目录")
        print("   ├── image/            # 待贴图目录，放所有对象图片")
        print("   ├── label/            # 与 image 同名的 YOLO 标签目录")
        print("   ├── classes.txt       # 类别名列表，按顺序定义 class_id")
        print("   └── ...")
        print("3. 确保所有图片和标签文件都是有效的")
        print("4. 确保标签文件格式正确 (class_id x_center y_center width height)")


if __name__ == "__main__":
    main()
