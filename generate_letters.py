"""「整图即目标」范例生成器：为每个类别绘制一张类名文字图（两种场景样式）。

场景1：蓝白同心圆 + 白色文字（白底）
场景2：白色正方形 + 黑色文字（白底）

参数来自 config.yaml：classes / generate_letters.output_dir / generate_letters.num_per_class。
本脚本仅作范例——真实项目通常直接把各类目标裁剪图放进 targets/<类名>/ 即可。
"""
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()

OUTPUT_DIR = CFG.generate_letters.output_dir
NUM_PER_CLASS = CFG.generate_letters.num_per_class
CLASSES = list(CFG.classes)     # 逐类绘制类名文字
FONT_SIZES = [28, 32, 36, 40, 44, 48, 52, 56]

# Linux 常用粗体字体路径，按优先级尝试
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def load_font(size):
    """尝试加载字体，失败则用默认字体"""
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    print("警告：未找到 TrueType 字体，使用 PIL 默认字体（效果较差）")
    return ImageFont.load_default()


def draw_style1(draw, w, h):
    """场景1：蓝白同心圆（外蓝→中白→内蓝，蓝底白字）
    外蓝环宽度 = R/5, 白环宽度 = R/10, 剩下为内圈蓝底
    返回 (r_outer, r_inner) 供字体大小计算"""
    cx, cy = w // 2, h // 2

    # 蓝色系随机变化
    blue_color = (
        random.randint(0, 30),
        random.randint(60, 120),
        random.randint(160, 220),
    )

    r_outer = random.randint(70, 88)        # 总半径
    r_middle = int(r_outer * 0.8)           # 外蓝环占 1/5
    r_inner  = int(r_outer * 0.7)           # 白环占 1/10，内圈蓝底占 7/10

    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        fill=blue_color,
    )
    draw.ellipse(
        [cx - r_middle, cy - r_middle, cx + r_middle, cy + r_middle],
        fill=(255, 255, 255),
    )
    draw.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        fill=blue_color,
    )

    return r_inner  # 内圈半径，用于约束字母大小


def draw_style2(draw, w, h):
    """场景2：白色正方形 + 浅灰边框，返回正方形半边长用于字体约束"""
    margin = random.randint(10, 25)
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    # 白色填充
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
    half_side = (w - 2 * margin) / 2
    return half_side


def pick_font_fit(letter, max_half_diag):
    """选尽可能大但半对角线不超过 max_half_diag 的字体"""
    for size in sorted(FONT_SIZES, reverse=True):
        font = load_font(size)
        bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), letter, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        half_diag = ((tw / 2) ** 2 + (th / 2) ** 2) ** 0.5
        if half_diag <= max_half_diag:
            return font, tw, th
    font = load_font(FONT_SIZES[0])
    return font, tw, th


def generate():
    for letter in CLASSES:
        out_dir = os.path.join(OUTPUT_DIR, letter)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(NUM_PER_CLASS):
            w, h = 194, 194
            style = random.choice(["concentric", "square"])

            if style == "concentric":
                img = Image.new("RGB", (w, h), (255, 255, 255))  # 白底
                draw = ImageDraw.Draw(img)
                r_inner = draw_style1(draw, w, h)
                text_color = (255, 255, 255)

                # 根据内圈半径自适应选字体
                font, tw, th = pick_font_fit(letter, r_inner * 0.85)
            else:
                img = Image.new("RGB", (w, h), (255, 255, 255))  # 白底，无黑边
                draw = ImageDraw.Draw(img)
                half_side = draw_style2(draw, w, h)
                text_color = (0, 0, 0)

                # 与同心圆同等约束的字体大小
                font, tw, th = pick_font_fit(letter, half_side * 0.85)

            # 居中画字母，加微小随机偏移
            x = (w - tw) // 2 + random.randint(-3, 3)
            y = (h - th) // 2 + random.randint(-3, 3)
            draw.text((x, y), letter, fill=text_color, font=font)

            img.save(os.path.join(out_dir, f"{style}_{i:04d}.png"))

    print(f"生成完成 → {OUTPUT_DIR}/  (每类 {NUM_PER_CLASS} 张，共 {len(CLASSES)} 类：{CLASSES})")


if __name__ == "__main__":
    generate()
