import os
import sys
from PIL import Image, ImageDraw, ImageFont
import colorsys

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()
CLASSES = list(CFG.classes)
def draw_yolo_boxes(input_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按实际类别数生成颜色
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

    # 尝试加载较大字体
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # 尝试加载Arial字体
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)  # Linux系统常见字体
        except:
            font = ImageFont.load_default()  # 使用默认字体（可能较小）
            print("Warning: Large font not found. Using default font.")
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        img_path = os.path.join(input_dir, filename)

        if not os.path.exists(txt_path):
            print(f"Warning: No annotation for {filename}")
            continue

        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # 坐标转换
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    half_width = box_width * img_width / 2
                    half_height = box_height * img_height / 2

                    # 矩形坐标
                    x_min = x_center_px - half_width
                    y_min = y_center_px - half_height
                    x_max = x_center_px + half_width
                    y_max = y_center_px + half_height

                    # 获取对应颜色
                    color = colors[class_id]

                    # 绘制矩形框
                    draw.rectangle(
                        [(x_min, y_min), (x_max, y_max)],
                        outline=color,
                        width=3
                    )

                    # 添加类别标签
                    text = f"{CLASSES[class_id]}"
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_x = x_min
                    text_y = y_min - text_height

                    # 确保文本在图像范围内
                    if text_y < 0:
                        text_y = y_min + 5  # 如果顶部放不下，放在框下
                    draw.text(
                        (text_x, text_y),
                        text,
                        fill=color,  # 白色文字更易读
                        font=font,
                    )

            # 保存处理后的图像
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_directory = CFG.paths.synth_output   # 输入目录
    output_directory = CFG.paths.check_output  # 输出目录
    draw_yolo_boxes(input_directory, output_directory)