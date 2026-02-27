import os
import re

from src.segmentation.evaluation.base_evaluator import BaseEvaluator


class UltralyticsEvaluator(BaseEvaluator):
    def __init__(self, num_classes: int, image_size: int) -> None:
        super().__init__(num_classes)
        self.image_size = image_size

    def parse_annotations(self, file_path: str) -> list:
        ground_truths = []
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * self.image_size
                y_center = float(parts[2]) * self.image_size
                width = float(parts[3]) * self.image_size
                height = float(parts[4]) * self.image_size
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                ground_truths.append({"bbox": [x_min, y_min, x_max, y_max], "class_id": class_id})
        return ground_truths

    def get_annotations_for_image_patches(self, image_number: str, annotations_directory: str) -> dict:
        patches_gt_boxes = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.txt$")
        for file_name in os.listdir(annotations_directory):
            if pattern.match(file_name):
                file_path = os.path.join(annotations_directory, file_name)
                gt_boxes = self.parse_annotations(file_path)
                patches_gt_boxes[file_name.replace(".txt", ".png")] = gt_boxes
        return patches_gt_boxes

    def get_annotations_for_dataset(self, annotations_directory: str) -> dict:
        image_numbers = set()
        for file_name in os.listdir(annotations_directory):
            match = re.match(r"(IMG_\d+)_p\d+\.txt", file_name)
            if match:
                image_numbers.add(match.group(1))

        return {
            image_number: self.get_annotations_for_image_patches(image_number, annotations_directory)
            for image_number in image_numbers
        }

    def parse_model_outputs(self, outputs) -> list:
        parsed_outputs = []
        for detection in outputs[0].boxes.data:
            x_min, y_min, x_max, y_max, confidence, class_id = detection.cpu().numpy()
            parsed_outputs.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "score": confidence,
                "class_id": int(class_id),
            })
        return parsed_outputs

    def predict_and_parse_image_patches(self, image_number: str, images_directory: str, predictor) -> dict:
        parsed_outputs_by_patch = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.(jpg|png)$")
        for file_name in os.listdir(images_directory):
            if pattern.match(file_name):
                image_path = os.path.join(images_directory, file_name)
                outputs = predictor(image_path)
                parsed_outputs_by_patch[file_name] = self.parse_model_outputs(outputs)
        return parsed_outputs_by_patch

    def predict_and_parse_dataset(self, images_directory: str, predictor) -> dict:
        image_numbers = self.get_image_numbers(images_directory)
        return {
            image_number: self.predict_and_parse_image_patches(image_number, images_directory, predictor)
            for image_number in image_numbers
        }
