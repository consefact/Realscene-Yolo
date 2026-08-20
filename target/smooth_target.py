"""底板 + ROI 范例生成器：把识别区裁剪图贴到"底板"中心，产出底板目标图。

- 输入：objects_dir/<类名>/*.png  （识别区，按类分子目录）
- 输出：output_dir/<类名>/*.png    （识别区贴在底板中心的目标图）
- 与 realscene 的配合：把某个 target_type 的 dir 指向这里的 output_dir，
  并把它的 roi 设为 [offset/board, offset/board, obj/board, obj/board]
  （默认 100/32/34 → roi=[0.34, 0.34, 0.32, 0.32]），标注就只框内部识别区。

所有参数来自 config.yaml 的 smooth_target 段。
"""
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFilter

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()


def make_synthetic_board(board_size):
    """合成底板：黑底 + 灰环 + 白圆（几何按 100 基准缩放）。"""
    board = Image.new('RGB', (board_size, board_size), (0, 0, 0))
    draw = ImageDraw.Draw(board)
    s = board_size / 100.0
    draw.ellipse([(12 * s, 12 * s), (87 * s, 87 * s)], fill=(128, 128, 128))  # 灰色环
    draw.ellipse([(25 * s, 25 * s), (75 * s, 75 * s)], fill=(255, 255, 255))  # 白色圆
    return board


def process_all_images(objects_dir, output_dir, real_targets_dir, prob_use_real,
                       board_size, object_size, paste_offset):
    """为每张识别区图生成底板目标图（可选贴到真实底板照片上）。"""
    # 真实底板照片列表（可选）
    real_target_files = []
    if real_targets_dir and os.path.isdir(real_targets_dir):
        real_target_files = [f for f in os.listdir(real_targets_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for root, dirs, files in os.walk(objects_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            original_path = os.path.join(root, file)
            relative_path = os.path.relpath(original_path, objects_dir)
            dir_name = os.path.dirname(relative_path)   # 保留类别子目录结构
            base_name = os.path.splitext(os.path.basename(file))[0]

            # 加载识别区并缩放
            original_img = Image.open(original_path).convert("RGB")
            original_img = original_img.resize((object_size, object_size))

            use_real = bool(real_target_files) and (random.random() < prob_use_real)

            if use_real:
                # 每张识别区在每个真实底板上各生成一张
                for idx, target_file in enumerate(real_target_files):
                    new_file = f"{base_name}_real_{idx:03d}.png"
                    output_path = os.path.join(output_dir, dir_name, new_file)

                    background = Image.open(os.path.join(real_targets_dir, target_file)).convert("RGB")
                    background = background.resize((board_size, board_size))

                    # 羽化遮罩，实现边缘平滑
                    mask = Image.new('L', original_img.size, 0)
                    ImageDraw.Draw(mask).rectangle(
                        [0, 0, original_img.width, original_img.height], fill=255)
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

                    background.paste(original_img, (paste_offset, paste_offset), mask)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    background.save(output_path)
            else:
                # 使用合成底板
                new_file = f"{base_name}_synthetic.png"
                output_path = os.path.join(output_dir, dir_name, new_file)

                background = make_synthetic_board(board_size)
                background.paste(original_img, (paste_offset, paste_offset))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                background.save(output_path)

            original_img.close()


if __name__ == "__main__":
    st = CFG.smooth_target
    real_dir = st.real_targets_dir
    if isinstance(real_dir, str) and real_dir.lower() in ("none", "no"):
        real_dir = None

    process_all_images(
        objects_dir=st.objects_dir,
        output_dir=st.output_dir,
        real_targets_dir=real_dir,
        prob_use_real=st.prob_use_real,
        board_size=st.board_size,
        object_size=st.object_size,
        paste_offset=st.paste_offset,
    )
    print(f"处理完成，底板目标图保存至：{st.output_dir}")
