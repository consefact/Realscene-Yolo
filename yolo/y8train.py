from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO("/home/airhust/zyt/yolo_run/sectrain/weights/best.pt")
# model = YOLO("./runs/detect/train/weights/best.pt")
results = model.train(data="craic.yaml", epochs=400, imgsz=640,dropout = 0.5, patience=40, workers=8, batch=128,save_period=3,cache='True',project = '/home/airhust/zyt/yolo_run',name='sectrain2',optimizer='AdamW',cls = 5.5, dfl = 1.0, box = 6.5)