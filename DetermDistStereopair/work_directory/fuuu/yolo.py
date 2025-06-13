from ultralytics import YOLO
import os
from pathlib import Path
import cv2

def detect_objects(image_path, crop_dir="results/crops"):
    model = YOLO("yolov8n.pt")

    os.makedirs(crop_dir, exist_ok=True)
    filename = Path(image_path).stem
    results = model(image_path)
    img = cv2.imread(image_path)

    objects = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        cropped_img = img[y1:y2, x1:x2]
        crop_filename = f"{crop_dir}/{filename}_{i}_{class_name}.jpg"
        cv2.imwrite(crop_filename, cropped_img)

        objects.append({
            "image": crop_filename,
            "class": class_name,
            "bbox": (x1, y1, x2, y2)
        })

    return objects

def pair_objects(objects1, objects2):
    pairs = []
    used_indices = set()

    for obj1 in objects1:
        matched = False
        for i, obj2 in enumerate(objects2):
            if i in used_indices:
                continue
            if obj1["class"] == obj2["class"]:
                pairs.append((obj1, obj2))
                used_indices.add(i)
                matched = True
                break
        if not matched:
            # pairs.append((obj1, None))
            pass

    # Add unpaired from second list
    # for i, obj2 in enumerate(objects2):
    #     if i not in used_indices:
    #         pairs.append((None, obj2))

    return pairs