import argparse
import json
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

GREEN_DATA: list[tuple[int, int, int]] = [
    (214, 217, 162),
    (202, 207, 149),
    (195, 201, 155),
    (216, 222, 194),
    (195, 199, 140),
    (233, 237, 210),
    (203, 207, 148),
    (192, 196, 120),
    (200, 204, 130),
    (198, 204, 134),
    (208, 213, 159),
    (221, 225, 175),
    (236, 240, 203),
    (212, 218, 180),
    (203, 209, 173),
    (232, 237, 197),
    (198, 202, 151),
    (241, 244, 215),
    (127, 135, 76),
    (130, 134, 73),
    (127, 126, 70),
    (211, 216, 162),
    (216, 219, 166),
    (210, 213, 156),
    (207, 213, 151),
    (220, 224, 173),
    (210, 218, 169),
    (211, 219, 172),
    (205, 214, 167),
    (198, 209, 151),
    (174, 189, 146),
    (206, 212, 174),
    (208, 215, 173),
    (208, 215, 181),
    (208, 214, 168),
    (127, 137, 76),
    (185, 190, 132),
    (154, 143, 87),
    (210, 205, 149),
    (217, 215, 154),
    (218, 221, 168),
    (240, 236, 201),
    (245, 250, 218),
    (218, 223, 159),
    (218, 221, 164),
    (212, 218, 146),
    (212, 219, 139),
    (216, 223, 146),
    (206, 209, 138),
    (195, 193, 118),
    (211, 209, 158),
    (223, 221, 173),
    (212, 211, 167),
]

ORANGE_DATA: list[tuple[int, int, int]] = [
    (136, 90, 77),
    (95, 47, 27),
    (135, 99, 67),
    (145, 99, 65),
    (128, 88, 62),
    (141, 100, 56),
    (172, 146, 89),
    (227, 206, 163),
    (225, 198, 145),
    (231, 207, 159),
    (240, 220, 193),
    (214, 193, 148),
    (156, 109, 53),
    (163, 121, 63),
    (150, 99, 46),
    (164, 132, 93),
    (179, 142, 90),
    (150, 105, 50),
    (161, 118, 67),
    (193, 169, 125),
    (206, 192, 165),
    (191, 165, 106),
    (189, 145, 98),
    (213, 169, 124),
    (209, 177, 136),
    (211, 176, 134),
    (209, 179, 143),
    (207, 181, 148),
    (200, 176, 142),
    (192, 163, 131),
    (160, 115, 58),
    (167, 133, 96),
    (167, 127, 68),
    (185, 146, 105),
    (193, 172, 141),
    (141, 99, 74),
    (183, 148, 116),
    (173, 136, 107),
    (185, 154, 123),
    (187, 165, 141),
]


class PistilsPipeline:
    @staticmethod
    def load_model(checkpoint_path: str) -> YOLO:
        return YOLO(checkpoint_path)

    @staticmethod
    def segment_pistils(image_bgr: np.ndarray, model: YOLO) -> object:
        results = model.predict(source=image_bgr, conf=0.3, iou=0.45)
        return results[0]

    @staticmethod
    def extract_segmented_objects(image_rgb: np.ndarray, result, save_dir: str | None = None) -> list:
        if result.masks is None or result.masks.data is None:
            return []
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
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
                save_path = os.path.join(save_dir, f"pistil_{i+1}.png")
                cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
        return extracted_segments

    @staticmethod
    def classify_colors(
        pistil_bgr: np.ndarray,
        green_data: list[tuple[int, int, int]],
        orange_data: list[tuple[int, int, int]],
        black_thresh: int = 10,
        white_thresh: int = 240,
        save_images: bool = False,
        output_dir: str | None = None,
        save_filename: str | None = None,
    ) -> tuple[float, float]:
        pistil_rgb = cv2.cvtColor(pistil_bgr, cv2.COLOR_BGR2RGB)
        h, w = pistil_rgb.shape[:2]
        bg_mask = np.all(pistil_rgb < black_thresh, axis=-1) | np.all(pistil_rgb > white_thresh, axis=-1)
        pistil_flat = pistil_rgb.reshape(-1, 3).astype(np.float32)
        valid_indices = np.where(~bg_mask.reshape(-1))[0]
        if len(valid_indices) == 0:
            return 0.0, 0.0

        valid_pixels = pistil_flat[valid_indices]
        green_arr = np.array(green_data, dtype=np.float32)
        orange_arr = np.array(orange_data, dtype=np.float32)
        pixel_expanded = valid_pixels[:, np.newaxis, :]
        dist2_green = np.sum((pixel_expanded - green_arr[np.newaxis, :, :]) ** 2, axis=2)
        dist2_orange = np.sum((pixel_expanded - orange_arr[np.newaxis, :, :]) ** 2, axis=2)
        is_green_valid = np.min(dist2_green, axis=1) < np.min(dist2_orange, axis=1)

        labels = np.zeros((h * w,), dtype=np.uint8)
        labels[valid_indices[is_green_valid]] = 1
        labels[valid_indices[~is_green_valid]] = 2
        labels_2d = labels.reshape(h, w)

        green_count = np.count_nonzero(labels_2d == 1)
        orange_count = np.count_nonzero(labels_2d == 2)
        total_valid = green_count + orange_count
        green_ratio = green_count / total_valid
        orange_ratio = orange_count / total_valid

        if save_images and output_dir:
            classified_img = np.zeros((h, w, 3), dtype=np.uint8)
            classified_img[labels_2d == 1] = (0, 255, 0)
            classified_img[labels_2d == 2] = (255, 165, 0)
            os.makedirs(output_dir, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(pistil_rgb)
            axes[0].set_title("Original Pistil (RGB)")
            axes[0].axis("off")
            axes[1].imshow(classified_img)
            axes[1].set_title("Classified (1-NN)")
            axes[1].axis("off")
            plt.tight_layout()
            vis_path = os.path.join(output_dir, save_filename or "classified_vis.png")
            plt.savefig(vis_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.close("all")

        return green_ratio, orange_ratio

    @staticmethod
    def process_image(
        image_path: str,
        model: YOLO,
        green_data: list[tuple[int, int, int]],
        orange_data: list[tuple[int, int, int]],
        output_dir: str,
        save_images: bool = False,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = PistilsPipeline.segment_pistils(image_bgr, model)

        analysis_dir = os.path.join(output_dir, "pistils_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        extracted_dir = os.path.join(analysis_dir, "extracted_pistils")
        colors_dir = os.path.join(analysis_dir, "pistils_colors_images")

        if save_images:
            os.makedirs(extracted_dir, exist_ok=True)
            os.makedirs(colors_dir, exist_ok=True)
            annotated_bgr = result.plot(labels=False, conf=False)
            cv2.imwrite(os.path.join(analysis_dir, "annotated_segmentation.png"), annotated_bgr)

        pistils = PistilsPipeline.extract_segmented_objects(
            image_rgb, result, save_dir=extracted_dir if save_images else None
        )

        classification_results = []
        for idx, pistil_crop in enumerate(pistils):
            pistil_crop_bgr = cv2.cvtColor(pistil_crop, cv2.COLOR_RGB2BGR)
            g_ratio, o_ratio = PistilsPipeline.classify_colors(
                pistil_crop_bgr,
                green_data,
                orange_data,
                save_images=save_images,
                output_dir=colors_dir if save_images else None,
                save_filename=f"pistil_{idx+1}.png" if save_images else None,
            )
            classification_results.append(
                {"pistil_index": idx + 1, "green_ratio": g_ratio, "orange_ratio": o_ratio}
            )

        with open(os.path.join(analysis_dir, "pistils_color_ratios.json"), "w") as f:
            json.dump(classification_results, f, indent=4)

        if classification_results:
            num_pistils = len(classification_results)
            aggregated = {
                "num_pistils": num_pistils,
                "average_green_ratio": sum(r["green_ratio"] for r in classification_results) / num_pistils,
                "average_orange_ratio": sum(r["orange_ratio"] for r in classification_results) / num_pistils,
            }
            with open(os.path.join(analysis_dir, "aggregated_pistils_color_ratios.json"), "w") as f:
                json.dump(aggregated, f, indent=4)

    @staticmethod
    def process_folder(
        folder_path: str,
        model: YOLO,
        green_data: list[tuple[int, int, int]],
        orange_data: list[tuple[int, int, int]],
        output_base_folder: str,
        save_images: bool = False,
    ) -> None:
        folder_name = os.path.basename(folder_path)
        folder_output_dir = os.path.join(output_base_folder, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)

        def extract_number(filename: str) -> int:
            match = re.search(r"\d+", filename)
            return int(match.group()) if match else -1

        image_files = sorted(
            [
                f for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
            key=extract_number,
        )
        for image_file in image_files[1:]:
            image_path = os.path.join(folder_path, image_file)
            base_name = os.path.splitext(image_file)[0]
            image_output_dir = os.path.join(folder_output_dir, base_name)
            os.makedirs(image_output_dir, exist_ok=True)
            PistilsPipeline.process_image(
                image_path, model, green_data, orange_data, image_output_dir, save_images
            )

    @staticmethod
    def process_all_folders(
        parent_folder: str,
        model: YOLO,
        green_data: list[tuple[int, int, int]],
        orange_data: list[tuple[int, int, int]],
        output_base_folder: str,
        save_images: bool = False,
    ) -> None:
        subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
        for folder in subfolders:
            PistilsPipeline.process_folder(folder, model, green_data, orange_data, output_base_folder, save_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_input_folder", type=str, required=True)
    parser.add_argument("--output_base_folder", type=str, required=True)
    _args = parser.parse_args()

    _model = PistilsPipeline.load_model(
        "/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8s_seg_fine_tuned.pt"
    )
    PistilsPipeline.process_all_folders(
        _args.parent_input_folder, _model, GREEN_DATA, ORANGE_DATA, _args.output_base_folder, save_images=True
    )
