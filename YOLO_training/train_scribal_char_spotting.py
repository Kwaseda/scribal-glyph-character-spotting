"""Train YOLOv8m on the tiled dataset.

Training for this project actually ran on Google Colab; the notebook beside this
file is the record of those runs. This script is the local equivalent and needs
a GPU to finish in reasonable time.

The dataset root is read from configs/scribal-glyph-charspotting.yaml, which
points at the Colab mount by default. Edit its `path` key before running here.
"""

from ultralytics import YOLO

import scribal_char_spotting.config as cfg

EPOCHS = 200
IMGSZ = 512
PATIENCE = 50

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    model.info()

    model.train(
        data=cfg.YOLO_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        patience=PATIENCE,
        # Glyphs are chiral: a mirrored 'b' is a 'd', so horizontal flips would
        # teach the detector the wrong class.
        fliplr=0.0,
        project=cfg.YOLO_PATH,
        name="exp_train_local",
    )
