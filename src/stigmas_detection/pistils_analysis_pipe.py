import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


class PistilsAnalysisPipeline:
    @staticmethod
    def load_model(checkpoint_path: str) -> YOLO:
        return YOLO(checkpoint_path)

    @staticmethod
    def segment_pistils(image_bgr: np.ndarray, model: YOLO) -> object:
        results = model.predict(image_bgr)
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
                save_path = os.path.join(save_dir, f"pistil_{i + 1}.png")
                cv2.imwrite(save_path, cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
        return extracted_segments

    @staticmethod
    def visualize_segmented_objects(segmented_objects: list) -> None:
        if not segmented_objects:
            return
        fig, axes = plt.subplots(1, len(segmented_objects), figsize=(16, 8))
        if len(segmented_objects) == 1:
            axes.imshow(segmented_objects[0])
            axes.set_title("Segment 1")
            axes.axis("off")
        else:
            for idx, obj in enumerate(segmented_objects):
                axes[idx].imshow(obj)
                axes[idx].set_title(f"Segment {idx + 1}")
                axes[idx].axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_color_segmentation(seg_obj_rgb: np.ndarray, model: YOLO) -> object:
        image_bgr = cv2.cvtColor(seg_obj_rgb, cv2.COLOR_RGB2BGR)
        results = model.predict(image_bgr)
        return results[0]

    @staticmethod
    def visualize_color_segmentation(original_rgb: np.ndarray, annotated_bgr: np.ndarray) -> None:
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Pistil (RGB)")
        axes[0].axis("off")
        axes[1].imshow(annotated_rgb)
        axes[1].set_title("Color Model Segmentation")
        axes[1].axis("off")
        plt.show()

    @staticmethod
    def extract_color_model_masks(result, image_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_shape[:2]
        empty = np.zeros((height, width), dtype=np.uint8)
        if result.masks is None or result.masks.data is None:
            return empty, empty

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        pred_green_mask = np.zeros((height, width), dtype=np.uint8)
        pred_orange_mask = np.zeros((height, width), dtype=np.uint8)

        for i, mask in enumerate(masks):
            resized_mask = cv2.resize(mask, (width, height))
            binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
            if classes[i] == 0:
                pred_green_mask = cv2.bitwise_or(pred_green_mask, binary_mask)
            elif classes[i] == 1:
                pred_orange_mask = cv2.bitwise_or(pred_orange_mask, binary_mask)

        return pred_green_mask, pred_orange_mask

    @staticmethod
    def hsv_fallback_orange_green(seg_obj_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(seg_obj_rgb, cv2.COLOR_RGB2HSV)
        orange_raw = cv2.inRange(hsv, np.array([0, 50, 50], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
        green_raw = cv2.inRange(hsv, np.array([40, 30, 30], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8))
        fallback_orange = orange_raw.copy()
        fallback_green = cv2.bitwise_and(green_raw, cv2.bitwise_not(fallback_orange))
        return fallback_orange, fallback_green

    @staticmethod
    def combine_model_and_fallback(
        seg_obj_rgb: np.ndarray,
        object_mask: np.ndarray,
        pred_orange_mask: np.ndarray,
        pred_green_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        model_union = cv2.bitwise_or(pred_orange_mask, pred_green_mask)
        model_union_in_obj = cv2.bitwise_and(model_union, object_mask)
        missing_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(model_union_in_obj))

        fb_orange_full, fb_green_full = PistilsAnalysisPipeline.hsv_fallback_orange_green(seg_obj_rgb)
        fb_orange = cv2.bitwise_and(fb_orange_full, missing_mask)
        fb_green = cv2.bitwise_and(fb_green_full, missing_mask)

        final_orange = cv2.bitwise_or(pred_orange_mask, fb_orange)
        final_green = cv2.bitwise_or(pred_green_mask, fb_green)

        overlap = cv2.bitwise_and(final_orange, final_green)
        if cv2.countNonZero(overlap) > 0:
            final_green = cv2.bitwise_and(final_green, cv2.bitwise_not(final_orange))

        union_now = cv2.bitwise_or(final_orange, final_green)
        leftover = cv2.bitwise_and(object_mask, cv2.bitwise_not(cv2.bitwise_and(union_now, object_mask)))
        final_green = cv2.bitwise_or(final_green, leftover)

        total_pixels = cv2.countNonZero(object_mask)
        if total_pixels == 0:
            return final_orange, final_green, 0.0, 0.0

        orange_ratio = cv2.countNonZero(final_orange) / float(total_pixels)
        green_ratio = cv2.countNonZero(final_green) / float(total_pixels)
        return final_orange, final_green, orange_ratio, green_ratio

    @staticmethod
    def run_pipeline(
        image_path: str,
        pistils_model_checkpoint: str,
        color_model_checkpoint: str,
        save_dir: str | None = None,
    ) -> None:
        pistils_model = PistilsAnalysisPipeline.load_model(pistils_model_checkpoint)
        color_model = PistilsAnalysisPipeline.load_model(color_model_checkpoint)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pistils_result = PistilsAnalysisPipeline.segment_pistils(image_bgr, pistils_model)
        annotated_rgb = cv2.cvtColor(pistils_result.plot(), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(annotated_rgb)
        axes[1].set_title("Pistils Segmentation")
        axes[1].axis("off")
        plt.show()

        pistils = PistilsAnalysisPipeline.extract_segmented_objects(image_rgb, pistils_result, save_dir=save_dir)
        PistilsAnalysisPipeline.visualize_segmented_objects(pistils)

        for i, seg_obj in enumerate(pistils):
            color_result = PistilsAnalysisPipeline.run_color_segmentation(seg_obj, color_model)
            annotated_color_bgr = color_result.plot()
            PistilsAnalysisPipeline.visualize_color_segmentation(seg_obj, annotated_color_bgr)

            height, width = seg_obj.shape[:2]
            pred_green_mask, pred_orange_mask = PistilsAnalysisPipeline.extract_color_model_masks(
                color_result, (height, width)
            )

            gray_obj = cv2.cvtColor(seg_obj, cv2.COLOR_RGB2GRAY)
            _, object_mask = cv2.threshold(gray_obj, 1, 255, cv2.THRESH_BINARY)

            final_orange, final_green, orange_ratio, green_ratio = PistilsAnalysisPipeline.combine_model_and_fallback(
                seg_obj_rgb=seg_obj,
                object_mask=object_mask,
                pred_orange_mask=pred_orange_mask,
                pred_green_mask=pred_green_mask,
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(seg_obj)
            axes[0].set_title("Segmented Pistil (RGB)")
            axes[0].axis("off")
            axes[1].imshow(final_green, cmap="gray")
            axes[1].set_title("Final Green Mask")
            axes[1].axis("off")
            axes[2].imshow(final_orange, cmap="gray")
            axes[2].set_title("Final Orange Mask")
            axes[2].axis("off")
            plt.show()

            total_ratio = green_ratio + orange_ratio
            print(
                f"Pistil {i+1} => Green: {green_ratio*100:.2f}%, "
                f"Orange: {orange_ratio*100:.2f}% (Sum={total_ratio*100:.2f}%)"
            )


if __name__ == "__main__":
    PistilsAnalysisPipeline.run_pipeline(
        image_path="/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/yolo_formatted/images/val/day_4_IMG_5942.jpg",
        pistils_model_checkpoint="/home/etaylor/code_projects/thesis/checkpoints/stigmas_segmentation/yolo/fine_tuned/yolov8s_seg_fine_tuned.pt",
        color_model_checkpoint="/home/etaylor/code_projects/thesis/checkpoints/stigmas_color_segmentation/yolo/fine_tuned/yolov8n-seg_fine_tuned.pt",
    )
