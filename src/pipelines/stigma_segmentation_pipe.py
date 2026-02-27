import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


class StigmaSegmentationPipeline:
    @staticmethod
    def extract_segmented_objects(image_rgb: np.ndarray, result, save_dir: str | None = None) -> list:
        if result.masks is None:
            return []
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
            extracted_segments.append(cropped_object)
            if save_dir:
                save_path = os.path.join(save_dir, f"segment_{i+1}.png")
                cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
        return extracted_segments

    @staticmethod
    def classify_stigma_colors(segmented_object: np.ndarray) -> tuple[float, float, float]:
        hsv_image = cv2.cvtColor(segmented_object, cv2.COLOR_RGB2HSV)
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(
            cv2.inRange(hsv_image, np.array([35, 40, 40], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
            cv2.MORPH_CLOSE, kernel,
        )
        white_mask = cv2.morphologyEx(
            cv2.inRange(hsv_image, np.array([0, 0, 220], dtype=np.uint8), np.array([180, 30, 255], dtype=np.uint8)),
            cv2.MORPH_CLOSE, kernel,
        )
        orange_mask = cv2.morphologyEx(
            cv2.inRange(hsv_image, np.array([10, 80, 70], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
            cv2.MORPH_CLOSE, kernel,
        )

        not_orange = cv2.bitwise_not(orange_mask)
        white_exclusive = cv2.bitwise_and(white_mask, not_orange)
        green_exclusive = cv2.bitwise_and(green_mask, cv2.bitwise_not(cv2.bitwise_or(orange_mask, white_exclusive)))
        total_union = cv2.bitwise_or(orange_mask, cv2.bitwise_or(white_exclusive, green_exclusive))
        total_pixels = cv2.countNonZero(total_union)

        if total_pixels == 0:
            return 0.0, 0.0, 0.0
        return (
            cv2.countNonZero(green_exclusive) / float(total_pixels),
            cv2.countNonZero(white_exclusive) / float(total_pixels),
            cv2.countNonZero(orange_mask) / float(total_pixels),
        )

    @staticmethod
    def run_pipeline(image_path: str, model: YOLO, save_dir: str | None = None) -> tuple[list, list]:
        image = cv2.imread(image_path)
        if image is None:
            return [], []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = model.predict(image)[0]

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image_rgb)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            axes[1].imshow(annotated)
            axes[1].set_title("Annotated Image")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "annotated.png"), dpi=300, bbox_inches="tight")
            plt.close()

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        segmentation_save_dir = os.path.join(save_dir, f"{base_name}_segments") if save_dir else None
        segmented_objects = StigmaSegmentationPipeline.extract_segmented_objects(
            image_rgb, result, save_dir=segmentation_save_dir
        )

        ratios_list = []
        for idx, segment in enumerate(segmented_objects):
            green_ratio, white_ratio, orange_ratio = StigmaSegmentationPipeline.classify_stigma_colors(segment)
            ratios_list.append({
                "segment_index": idx + 1,
                "green_ratio": green_ratio,
                "white_ratio": white_ratio,
                "orange_ratio": orange_ratio,
            })

        if save_dir:
            json_path = os.path.join(save_dir, f"{base_name}_stigma_ratios.json")
            with open(json_path, "w") as fp:
                json.dump(ratios_list, fp, indent=4)

        return segmented_objects, ratios_list

    @staticmethod
    def process_all_folders(parent_folder_path: str, model: YOLO, output_dir: str) -> None:
        stigmas_output_dir = os.path.join(output_dir, "stigmas")
        os.makedirs(stigmas_output_dir, exist_ok=True)
        distinct_stigmas_output_dir = os.path.join(output_dir, "distinct_stigmas")
        os.makedirs(distinct_stigmas_output_dir, exist_ok=True)

        subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]
        for folder_path in subfolders:
            folder_name = os.path.basename(folder_path)
            folder_output_dir = os.path.join(stigmas_output_dir, folder_name)
            os.makedirs(folder_output_dir, exist_ok=True)

            image_files = [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                base_file_name = os.path.splitext(image_file)[0]
                image_output_dir = os.path.join(folder_output_dir, base_file_name)
                os.makedirs(image_output_dir, exist_ok=True)

                segmented_objects, _ = StigmaSegmentationPipeline.run_pipeline(
                    image_path, model, save_dir=image_output_dir
                )
                for idx, seg_img in enumerate(segmented_objects):
                    seg_filename = f"{base_file_name}_segment_{idx+1}.png"
                    seg_path = os.path.join(distinct_stigmas_output_dir, seg_filename)
                    cv2.imwrite(seg_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    _model = YOLO(
        "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8m-seg_fine_tuned.pt"
    )
    StigmaSegmentationPipeline.process_all_folders(
        "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/images/day_1_2024_05_30/greenhouse",
        _model,
        "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_1/results/faster_rcnn/day_1_2024_05_30/greenhouse",
    )
