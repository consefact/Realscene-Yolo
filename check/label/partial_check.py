import os
import sys
import random
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
def get_valid_files(input_dir):
    """获取所有有效图片文件（有对应txt标注）"""
    valid_files = []
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            valid_files.append(filename)
    return valid_files

def draw_bounding_boxes(input_dir, output_dir, files_to_draw):
    """绘制标注框并保存"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 按实际类别数生成颜色
    num_classes = len(CLASSES)
    colors = []
    for i in range(num_classes):
        hue = i * (360 / num_classes)
        r, g, b = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
        colors.append((int(r*255), int(g*255), int(b*255)))
    
    # 尝试加载大字体
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
            print("Warning: Large font not found. Using default font.")

    # 处理选中的文件
    for filename in files_to_draw:
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        img_path = os.path.join(input_dir, filename)
        
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

                    # 绘制矩形框
                    color = colors[class_id]
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
                    #text_width, text_height = font.getsize(text)
                    text_x = x_min
                    text_y = y_min - text_height

                    # 确保文本在图像范围内
                    if text_y < 0:
                        text_y = y_min + 5  # 如果顶部放不下，放在框下
                    draw.text(
                        (text_x, text_y),
                        text,
                        fill='black',
                        font=font
                    )

            # 保存处理后的图像
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_directory = CFG.paths.synth_output
    output_directory = CFG.paths.check_output
    num_samples = CFG.check.num_samples  # 抽样数量（改 config.yaml 的 check.num_samples）

    # 获取所有有效文件
    valid_files = get_valid_files(input_directory)
    if not valid_files:
        print("No valid files found in the input directory!")
        exit()

    # 随机选取样本（不足则全取）
    selected_files = random.sample(valid_files, k=min(num_samples, len(valid_files)))

    # 执行可视化
    draw_bounding_boxes(input_directory, output_directory, selected_files)