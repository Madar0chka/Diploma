from pathlib import Path
import cv2


def draw_text_with_bg(canvas, caption, anchor):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness = 5
    fg_color = (0, 0, 0)
    bg_color = (255, 255, 255)
    x, y = anchor

    (w, h), baseline = cv2.getTextSize(caption, font_face, scale, thickness)

    top_left_corner = (x, y - h - baseline)
    bottom_right_corner = (x + w, y)

    cv2.rectangle(canvas, top_left_corner, bottom_right_corner, bg_color, thickness=cv2.FILLED)
    cv2.putText(canvas, caption, (x, y - baseline), font_face, scale, fg_color, thickness)


def annotate_img(input_path, matched_objs):
    image = cv2.imread(input_path)
    img_name = Path(input_path).stem

    print(matched_objs)

    for det_l, det_r in matched_objs:
        x_min, y_min, x_max, y_max = det_l["bbox"]
        category = det_l["class"]
        dist_cm = det_l["distance"]

        if dist_cm <= 0:
            continue

        tag = f"{category}, {dist_cm:.1f} cm"

        draw_text_with_bg(image, tag, (x_min, y_min))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=10)

        # Альтернативний спосіб:
        # cv2.putText(
        #     image,
        #     tag,
        #     (x_min, y_min - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=3,
        #     color=(0, 0, 255),
        #     thickness=10
        # )

    output_path = f'results/annotated/{img_name}_annotated.jpg'
    cv2.imwrite(output_path, image)