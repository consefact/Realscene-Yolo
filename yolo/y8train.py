from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# model = YOLO("./runs/detect/train/weights/best.pt")
results = model.train(data="craic.yaml", epochs=100, imgsz=640,patience=30, workers=8, batch=128,save_period=3,cache='True', name='y8train')