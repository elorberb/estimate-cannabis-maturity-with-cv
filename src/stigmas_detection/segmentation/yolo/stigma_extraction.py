import os

import cv2
import numpy as np
from ultralytics import YOLO


class StigmaExtraction:
    @staticmethod
    def extract_segmented_objects(
        image_rgb: np.ndarray, result, image_name: str, save_dir: str | None = None
    ) -> list:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if len(masks) == 0:
            return []

        extracted_segments = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, mask in enumerate(masks):
            resized_mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
            x_min, y_min, x_max, y_max = map(int, boxes[i])
            cropped_object = segmented_object[y_min:y_max, x_min:x_max]
            cropped_mask = binary_mask[y_min:y_max, x_min:x_max]

            cropped_rgba = cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGRA)
            cropped_rgba[..., 3] = (cropped_mask * 255).astype(np.uint8)
            extracted_segments.append(cropped_rgba)

            if save_dir:
                save_path = os.path.join(save_dir, f"{image_name}_pistil_{i + 1}.png")
                cv2.imwrite(save_path, cropped_rgba)

        return extracted_segments

    @staticmethod
    def run_inference(image_path: str, model_checkpoint: str, save_dir: str) -> list:
        model = YOLO(model_checkpoint)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_num = os.path.basename(image_path).split(".")[0]

        results = model.predict(image)
        result = results[0]

        return StigmaExtraction.extract_segmented_objects(image_rgb, result, save_dir=save_dir, image_name=image_num)


if __name__ == "__main__":
    _image_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_1_2024_12_05/lab/6/IMG_0626.JPG"
    _image_name = os.path.basename(_image_path).split(".")[0]
    _save_dir = os.path.join(
        "/home/etaylor/code_projects/thesis/src/stigmas_detection/segmentation/yolo/extracted_stigmas",
        _image_name,
    )

    _segmented_objects = StigmaExtraction.run_inference(
        _image_path,
        "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8m-seg_fine_tuned.pt",
        _save_dir,
    )
    print(f"Saved {len(_segmented_objects)} stigmas objects to: {_save_dir}")
