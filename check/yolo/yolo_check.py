import os
import sys
import random
import argparse
import cv2
import colorsys
from ultralytics import YOLO

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()
CLASSES = list(CFG.classes)
def main():
    parser = argparse.ArgumentParser(description="YOLOv8n Image Prediction")
    parser.add_argument("input_dir", type=str, help="Path to input directory with images")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to process")
    default_weights = os.path.join(CFG.paths.yolo_runs, CFG.train.run_name, "weights", "best.pt")
    parser.add_argument("--weights", type=str, default=default_weights,
                        help="训练得到的 .pt 权重路径（默认取 config 里 run 的 best.pt）")
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
    
    # 按类别数生成颜色（基于HSV色轮）
    num_classes = len(CLASSES)
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
    model = YOLO(args.weights)

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