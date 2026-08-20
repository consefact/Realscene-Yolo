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

SAVEPATH = CFG.capture.save_dir


def main():
    # 创建保存目录（若不存在）
    os.makedirs(SAVEPATH, exist_ok=True)
    cap = cv2.VideoCapture(CFG.capture.camera_index)
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.capture.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.capture.frame_height)

    every_n = CFG.capture.auto_every_n
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # 每隔 N 帧保存一次（降低保存频率）
            frame_count += 1
            if frame_count % every_n == 0:
                # 使用毫秒级时间戳避免冲突
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]  # 截取前3位毫秒
                framepath = os.path.join(SAVEPATH, f"frame_{timestamp}.jpg")
                success = cv2.imwrite(framepath, frame)
                if not success:
                    print(f"Failed to save {framepath}")

            # 显示画面（可选）
            cv2.imshow('Frame', frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()
