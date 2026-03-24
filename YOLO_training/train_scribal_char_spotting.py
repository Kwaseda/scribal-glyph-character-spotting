from ultralytics import YOLO
import scribal_char_spotting.config as cfg

epochs = 200
imgsz = 512

# Load a COCO-pretrained YOLOv8m model
model = YOLO("yolov8m.pt")

# Display model information (optional)
model.info()

# Train the model for specified n0. of epochs
results = model.train(
    data="configs/scribal-glyph-charspotting.yaml", epochs=epochs, imgsz=imgsz
)

# Save the trained model
# model.save("/YOLO_training/saved_models/exp_train_7138h.pt") # Didn't work so used Google Colab. Also, training time was too long
