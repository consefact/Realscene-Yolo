import os
import random
import argparse
import cv2
import colorsys
from ultralytics import YOLO
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
def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Image Prediction")
    parser.add_argument("input_dir", type=str, help="Path to input directory with images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to process")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有图片文件
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in os.listdir(args.input_dir) 
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    if not image_files:
        print("No valid images found in the input directory")
        return
    
    # 生成100种不同颜色（基于HSV色轮）
    num_classes = 100
    colors = []

    for i in range(num_classes):
        # 计算HSV色调值（0-360度）
        hue = i * (360 / num_classes)
        # 转换为RGB颜色（饱和度1.0，亮度1.0）
        r, g, b = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
        # 转换为RGB整数格式（0-255）
        color = (int(r * 255), int(g * 255), int(b * 255))
        colors.append(color)

    # 随机选择图片
    selected_images = random.sample(
        image_files, 
        k=min(len(image_files), args.num_images)
    )

    # 加载YOLOv8n模型
    model = YOLO("/home/ling/zyt195/yolo_run/secupgrade/weights/best.pt")

    for img_name in selected_images:
        img_path = os.path.join(args.input_dir, img_name)
        output_path = os.path.join(args.output_dir, img_name)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue


        results = model.predict(source=img, save=False)
        result = results[0]

        # 检查是否有检测结果
        if result.boxes.xyxy.shape[0] == 0:
            print(f"No detections found in {img_name}")
            continue

        # 获取所有检测框、类别和置信度
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # 遍历每个检测结果
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cls_int = int(cls)

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls_int], 2)
        
            # 添加类别标签
            label = f"Class {CLASSES[cls_int]} ({conf:.2f})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1
            text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + 5
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls_int], 2)

        # 保存结果
        cv2.imwrite(output_path, img)
        print(f"Processed: {img_name} -> {output_path}")

if __name__ == "__main__":
    main()