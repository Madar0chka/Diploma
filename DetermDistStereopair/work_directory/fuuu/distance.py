import math

def compute_depth(object_pairs, img_width_px, fov_degrees, baseline_in_cm):
    # Horizontal field of view в радіанах
    fov_radians = math.radians(fov_degrees)
    focal_length = img_width_px / (2 * math.tan(fov_radians / 2))

    for left_obj, right_obj in object_pairs:
        box_l = left_obj["bbox"]
        box_r = right_obj["bbox"]

        center_l_x = (box_l[0] + box_l[2]) / 2
        center_r_x = (box_r[0] + box_r[2]) / 2

        disparity_px = center_l_x - center_r_x

        if disparity_px == 0:
            print(' Неможливо обчислити відстань (нульова диспаратність)')
            left_obj['distance'] = 0
            right_obj['distance'] = 0

            left_obj.pop('image', None)
            right_obj.pop('image', None)
            continue

        depth_cm = (focal_length * baseline_in_cm) / disparity_px

        print("Left:", left_obj.get("image"))
        print("Right:", right_obj.get("image"))
        print('Обчислена відстань:', depth_cm)

        left_obj['distance'] = depth_cm
        right_obj['distance'] = depth_cm

        left_obj.pop('image', None)
        right_obj.pop('image', None)