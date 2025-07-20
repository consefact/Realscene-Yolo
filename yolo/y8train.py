from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
# model = YOLO("/home/airhust/zyt/yolo_run/sectrain/weights/best.pt")
model = YOLO("/home/ling/zyt195/yolo_run/firsttry/weights/best.pt")
# results = model.train(data="craic.yaml", epochs=125, imgsz=640,dropout = 0.5, patience=50, workers=8, batch=128,save_period=3,cache='True',project = '/home/ling/zyt195/yolo_run',name='firsttry',optimizer='AdamW',cls = 5.5, dfl = 1.0, box = 6.5)
results = model.train(data="craic.yaml", epochs=50, imgsz=640,dropout = 0.5, patience=50, workers=8, batch=128,save_period=3,cache='True',project = '/home/ling/zyt195/yolo_run',name='secupgrade',optimizer = "SGD", lr0=0.001,pretrained=True,cls = 5.5, dfl = 1.0, box = 6.5)