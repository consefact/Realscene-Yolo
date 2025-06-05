import cv2
import os
from datetime import datetime

# 全局变量定义
SAVEPATH = './save_frame'
NOWCLASS = 'nowclass'  # 指定保存图片的子目录名

def main():
    # 创建主保存目录（若不存在）
    os.makedirs(SAVEPATH, exist_ok=True)

    # 构建子目录路径
    save_dir = os.path.join(SAVEPATH, NOWCLASS)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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