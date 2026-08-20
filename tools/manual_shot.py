import cv2
import os
import sys
from datetime import datetime

# --- 载入统一配置 ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, "config.yaml")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from config import load_config
CFG = load_config()

# 全局变量定义
SAVEPATH = CFG.capture.save_dir
NOWCLASS = CFG.capture.class_name  # 保存图片的子目录名（分类拍摄用）

def main():
    # 创建主保存目录（若不存在）
    os.makedirs(SAVEPATH, exist_ok=True)

    # 构建子目录路径
    save_dir = os.path.join(SAVEPATH, NOWCLASS)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(CFG.capture.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.capture.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.capture.frame_height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            cv2.imshow('Frame', frame)

            key = cv2.waitKey(1) & 0xFF

            # 按 'C' 或 'c' 键拍照
            if key == ord('c') or key == ord('C'):
                # 使用毫秒级时间戳命名文件
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
                filename = f"frame_{timestamp}.jpg"
                framepath = os.path.join(save_dir, filename)

                success = cv2.imwrite(framepath, frame)
                if success:
                    print(f"Saved: {framepath}")
                else:
                    print(f"Failed to save: {framepath}")

            # 按 'q' 键退出
            if key == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()