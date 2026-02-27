import os
import re

from src.segmentation.evaluation.ultralytics_evaluator import UltralyticsEvaluator


class YoloNasEvaluator(UltralyticsEvaluator):
    def __init__(self, num_classes: int, image_size: int) -> None:
        super().__init__(num_classes, image_size)

    def parse_model_outputs(self, outputs) -> list:
        bboxes = outputs.prediction.bboxes_xyxy
        confidences = outputs.prediction.confidence
        class_ids = outputs.prediction.labels

        return [
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "score": confidence,
                "class_id": int(class_id),
            }
            for (x_min, y_min, x_max, y_max), confidence, class_id in zip(bboxes, confidences, class_ids)
        ]

    def predict_and_parse_image_patches(self, image_number: str, images_directory: str, predictor) -> dict:
        parsed_outputs_by_patch = {}
        pattern = re.compile(rf"^{image_number}_p\d+\.(jpg|png)$")
        for file_name in os.listdir(images_directory):
            if pattern.match(file_name):
                image_path = os.path.join(images_directory, file_name)
                outputs = predictor.predict(image_path, fuse_model=False)
                parsed_outputs_by_patch[file_name] = self.parse_model_outputs(outputs)
        return parsed_outputs_by_patch

    def predict_and_parse_dataset(self, images_directory: str, predictor) -> dict:
        image_numbers = self.get_image_numbers(images_directory)
        return {
            image_number: self.predict_and_parse_image_patches(image_number, images_directory, predictor)
            for image_number in image_numbers
        }
