import cv2
import os
from datetime import datetime

SAVEPATH = './save_frame'


def main():
    # 创建保存目录（若不存在）
    os.makedirs(SAVEPATH, exist_ok=True)
    cap = cv2.VideoCapture(0)
    # 设置摄像头参数（例如分辨率为640x480）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # 每隔5帧保存一次（降低保存频率）
            frame_count += 1
            if frame_count % 5 == 0:
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