from ultralytics import YOLO
import scribal_char_spotting.config as cfg

epochs = 400
imgsz = cfg.TILE_SIZE

# Load a COCO-pretrained YOLOv8m model
model = YOLO("yolov8m.pt")

# Display model information (optional)
model.info()

# Train the model for specified n0. of epochs
results = model.train(data=cfg.YOLO_YAML_PATH, epochs=epochs, imgsz=imgsz)

# Save the trained model
model.save(cfg.YOLO_SAVE_PATH)
