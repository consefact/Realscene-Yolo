import os
from PIL import Image, ImageDraw, ImageFilter          
def main(base_name):     
    new_file = f"{base_name}_synthetic.png"
    factor = 4
    # 创建合成背景
    background = Image.new('RGB', (factor*100, factor*100), (0, 0, 0))
    draw = ImageDraw.Draw(background)
    draw.ellipse([(factor*12, factor*12), (factor*87, factor*87)], fill=(128, 128, 128))  # 灰色环
    draw.ellipse([(factor*25, factor*25), (factor*75, factor*75)], fill=(255, 255, 255))  # 白色圆

    # 粘贴原始图像
    background.save(new_file)

if __name__ == "__main__":
    main("base_target")