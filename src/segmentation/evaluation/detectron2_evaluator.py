import json
import os
import re

import cv2

from src.segmentation.evaluation.base_evaluator import BaseEvaluator


class Detectron2Evaluator(BaseEvaluator):
    def __init__(self, num_classes: int, coco_annotations_file_path: str) -> None:
        super().__init__(num_classes)
        self.coco_data = self.load_coco_annotations(coco_annotations_file_path)

    @staticmethod
    def load_coco_annotations(file_path: str) -> dict:
        with open(file_path, "r") as f:
            return json.load(f)

    def get_annotations_for_patch(self, file_name: str) -> tuple[dict, list]:
        image_entry = next((img for img in self.coco_data["images"] if img["file_name"] == file_name), None)
        if not image_entry:
            raise ValueError(f"No image with file name {file_name} found in annotations.")
        image_id = image_entry["id"]
        annotations = [ann for ann in self.coco_data["annotations"] if ann["image_id"] == image_id]
        return image_entry, annotations

    def parse_annotations(self, file_path: str) -> list:
        _, annotations = self.get_annotations_for_patch(file_path)
        ground_truths = []
        for annotation in annotations:
            bbox = annotation["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            ground_truths.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "class_id": annotation["category_id"] - 1,
            })
        return ground_truths

    def get_annotations_for_image_patches(self, image_number: str) -> dict:
        patches_gt_boxes = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.\w+$")
        for img in self.coco_data["images"]:
            if pattern.match(img["file_name"]):
                patches_gt_boxes[img["file_name"]] = self.parse_annotations(img["file_name"])
        return patches_gt_boxes

    def get_annotations_for_dataset(self, images_directory: str) -> dict:
        image_numbers = self.get_image_numbers(images_directory)
        return {
            image_number: self.get_annotations_for_image_patches(image_number)
            for image_number in image_numbers
        }

    def parse_model_outputs(self, outputs: dict) -> list:
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()

        return [
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "score": score,
                "class_id": class_id - 1,
            }
            for (x_min, y_min, x_max, y_max), score, class_id in zip(pred_boxes, scores, pred_classes)
        ]

    def predict_and_parse_image_patches(self, image_number: str, images_directory: str, predictor) -> dict:
        parsed_outputs_by_patch = {}
        for file_name in os.listdir(images_directory):
            if image_number in file_name and "label_ground-truth" not in file_name:
                image_path = os.path.join(images_directory, file_name)
                outputs = predictor(cv2.imread(image_path))
                parsed_outputs_by_patch[file_name] = self.parse_model_outputs(outputs)
        return parsed_outputs_by_patch

    def predict_and_parse_dataset(self, images_directory: str, predictor) -> dict:
        image_numbers = self.get_image_numbers(images_directory)
        return {
            image_number: self.predict_and_parse_image_patches(image_number, images_directory, predictor)
            for image_number in image_numbers
        }
