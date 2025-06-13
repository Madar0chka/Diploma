import os
import time

from resnet import resnet
from yolo import detect_objects, pair_objects
from distance import compute_depth
from labels import annotate_img

def main():
    start_time = time.time()

    image_width_px = 4000
    horizontal_fov_deg = 67
    stereo_baseline_cm = 20

    os.makedirs("results", exist_ok=True)

    left_image_path = 'images/left_30cm.jpg'
    right_image_path = 'images/right_30cm.jpg'

    left_objects = detect_objects(left_image_path)
    right_objects = detect_objects(right_image_path)

    paired_objects = pair_objects(left_objects, right_objects)
    print(paired_objects)

    for idx, (left_obj, right_obj) in enumerate(paired_objects):
        print(f"\nPair {idx + 1}:")
        print("Left:", left_obj["image"] if left_obj else "None")
        print("Right:", right_obj["image"] if right_obj else "None")
        print(left_obj)

    enhanced_pairs = resnet(paired_objects)

    for idx, (left_obj, right_obj) in enumerate(enhanced_pairs):
        print(f"\nPair {idx + 1}:")
        print("Left:", left_obj["image"] if left_obj else "None")
        print("Right:", right_obj["image"] if right_obj else "None")
        print(left_obj)

    compute_depth(enhanced_pairs, image_width_px, horizontal_fov_deg, stereo_baseline_cm)

    annotate_img(left_image_path, enhanced_pairs)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Time: {duration:.4f} sec")

if __name__ == "__main__":
    main()
