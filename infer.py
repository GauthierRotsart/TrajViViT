import os
from pathlib import Path
from ultralytics import YOLO

# Initialize YOLOv5 model
model = YOLO("runs/detect/train7/weights/best.pt",type="v8")  # or device='cuda:0'

# Create output directory
output_dir = Path("pred")
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through images in the validation set
val_images_dir = Path("/home/dani/data/DroneDataset/nexus/video1/detection/val/images")
for image_path in val_images_dir.glob("*.jpg"):
    # Perform inference
    results = model.predict(image_path)
    boxes = results[0].boxes.xywhn
    classes = results[0].boxes.cls
    rstr=""
    for cls, box in zip(classes, boxes):
        rstr += f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
    # Save predictions to a file
    output_file_path = output_dir / f"{image_path.stem}.txt"
    with open(output_file_path, "w") as f:
        f.write(rstr)
