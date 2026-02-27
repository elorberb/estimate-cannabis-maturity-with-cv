import copy
import os
import re
from abc import ABC, abstractmethod

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

CLASS_NAMES = ("Clear", "Cloudy", "Amber")


class BaseEvaluator(ABC):
    def __init__(self, num_classes: int = 3) -> None:
        self.num_classes = num_classes

    @abstractmethod
    def parse_annotations(self, file_path: str) -> list:
        pass

    @abstractmethod
    def parse_model_outputs(self, outputs) -> list:
        pass

    @staticmethod
    def iou(box_a: list, box_b: list) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter_area / float(box_a_area + box_b_area - inter_area)

    @staticmethod
    def calculate_precision(true_positives: int, false_positives: int) -> float:
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def calculate_recall(true_positives: int, false_negatives: int) -> float:
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
        normalized_matrix = confusion_matrix.astype(np.float32)
        for i in range(len(normalized_matrix)):
            row_sum = normalized_matrix[i, :].sum()
            if row_sum > 0:
                normalized_matrix[i, :] /= row_sum
        return normalized_matrix

    @staticmethod
    def get_image_numbers(images_directory: str) -> set:
        image_numbers = set()
        pattern = re.compile(r"IMG_\d+")
        for file_name in os.listdir(images_directory):
            match = pattern.search(file_name)
            if match:
                image_numbers.add(match.group(0))
        return image_numbers

    def compute_confusion_matrix(
        self,
        matches: list,
        misclassifications: list,
        false_positives: list,
        false_negatives: list,
        gt_boxes: list,
        pred_boxes: list,
        normalize: bool = False,
        single_class: bool = False,
    ) -> np.ndarray:
        num_classes = 1 if single_class else self.num_classes
        cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

        for gt_idx, pred_idx, _ in matches:
            gt_class = 0 if single_class else gt_boxes[gt_idx]["class_id"]
            pred_class = 0 if single_class else pred_boxes[pred_idx]["class_id"]
            cm[gt_class, pred_class] += 1

        for gt_idx, pred_idx, _ in misclassifications:
            true_class = 0 if single_class else gt_boxes[gt_idx]["class_id"]
            pred_class = 0 if single_class else pred_boxes[pred_idx]["class_id"]
            cm[true_class, pred_class] += 1

        for pred in false_positives:
            pred_class = 0 if single_class else pred["class_id"]
            cm[num_classes, pred_class] += 1

        for fn in false_negatives:
            true_class = 0 if single_class else fn["class_id"]
            cm[true_class, num_classes] += 1

        if normalize:
            cm = cm.astype(np.float32)
            for i in range(len(cm)):
                if cm[i, :].sum() > 0:
                    cm[i, :] /= cm[i, :].sum()

        return cm

    def match_predictions(
        self, gt_boxes: list, pred_boxes: list, iou_thresh: float = 0.5
    ) -> tuple[list, list, list, list]:
        matches = []
        misclassifications = []
        false_positives = []
        detected = set()

        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_match = None

            for gt_idx, gt in enumerate(gt_boxes):
                current_iou = self.iou(pred["bbox"], gt["bbox"])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match = (gt_idx, pred_idx, current_iou)

            if best_match and best_iou >= iou_thresh:
                gt_idx, pred_idx, _ = best_match
                if gt_idx not in detected:
                    if gt_boxes[gt_idx]["class_id"] == pred["class_id"]:
                        matches.append(best_match)
                    else:
                        misclassifications.append(best_match)
                    detected.add(gt_idx)
            else:
                false_positives.append(pred)

        false_negatives = [
            {"bbox": gt["bbox"], "class_id": gt["class_id"]}
            for gt_idx, gt in enumerate(gt_boxes)
            if gt_idx not in detected
        ]

        return matches, misclassifications, false_positives, false_negatives

    def calculate_metrics(self, confusion_matrix: np.ndarray, single_class: bool = False) -> dict:
        if single_class:
            tp = confusion_matrix[0, 0]
            fp = confusion_matrix[0, 1]
            fn = confusion_matrix[1, 0]
            precision = self.calculate_precision(tp, fp)
            recall = self.calculate_recall(tp, fn)
            return {"precision": precision, "recall": recall, "f1": self.calculate_f1_score(precision, recall)}

        num_classes = confusion_matrix.shape[0] - 1
        precision_arr = np.zeros(num_classes)
        recall_arr = np.zeros(num_classes)
        f1_arr = np.zeros(num_classes)
        overall_tp = overall_fp = overall_fn = 0

        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            precision_arr[i] = self.calculate_precision(tp, fp)
            recall_arr[i] = self.calculate_recall(tp, fn)
            f1_arr[i] = self.calculate_f1_score(precision_arr[i], recall_arr[i])
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        overall_precision = self.calculate_precision(overall_tp, overall_fp)
        overall_recall = self.calculate_recall(overall_tp, overall_fn)
        return {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": self.calculate_f1_score(overall_precision, overall_recall),
            "class_wise_precision": precision_arr,
            "class_wise_recall": recall_arr,
            "class_wise_f1": f1_arr,
        }

    def evaluate_patch(
        self, gt_boxes: list, pred_boxes: list, iou_thresh: float = 0.5, single_class: bool = False
    ) -> dict:
        working_gt_boxes = copy.deepcopy(gt_boxes)
        working_pred_boxes = copy.deepcopy(pred_boxes)

        if single_class:
            for box in working_gt_boxes:
                box["class_id"] = 0
            for box in working_pred_boxes:
                box["class_id"] = 0

        matches, misclassifications, false_positives, false_negatives = self.match_predictions(
            working_gt_boxes, working_pred_boxes, iou_thresh
        )
        confusion_matrix = self.compute_confusion_matrix(
            matches, misclassifications, false_positives, false_negatives,
            working_gt_boxes, working_pred_boxes, single_class=single_class,
        )
        normalized_confusion_matrix = self.compute_confusion_matrix(
            matches, misclassifications, false_positives, false_negatives,
            working_gt_boxes, working_pred_boxes, normalize=True, single_class=single_class,
        )

        return {
            "matches": matches,
            "misclassifications": misclassifications,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "confusion_matrix": confusion_matrix,
            "normalized_confusion_matrix": normalized_confusion_matrix,
        }

    def evaluate_image(
        self,
        patches_gt_boxes_dict: dict,
        patches_pred_boxes_dict: dict,
        iou_thresh: float = 0.5,
        single_class: bool = False,
    ) -> dict:
        matrix_size = 2 if single_class else (self.num_classes + 1)
        global_conf_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        detailed_results = {}

        for patch_key in patches_gt_boxes_dict.keys():
            patch_result = self.evaluate_patch(
                patches_gt_boxes_dict[patch_key],
                patches_pred_boxes_dict[patch_key],
                iou_thresh,
                single_class,
            )
            global_conf_matrix += patch_result["confusion_matrix"]
            detailed_results[patch_key] = patch_result

        return {
            "patch_results": detailed_results,
            "confusion_matrix": global_conf_matrix,
            "normalized_confusion_matrix": self.normalize_confusion_matrix(global_conf_matrix),
        }

    def evaluate_dataset(
        self,
        dataset_gt_boxes_dict: dict,
        dataset_pred_boxes_dict: dict,
        iou_thresh: float = 0.5,
        single_class: bool = False,
    ) -> dict:
        matrix_size = 2 if single_class else (self.num_classes + 1)
        dataset_conf_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        detailed_dataset_results = {}

        for image_number in dataset_gt_boxes_dict.keys():
            image_results = self.evaluate_image(
                dataset_gt_boxes_dict[image_number],
                dataset_pred_boxes_dict[image_number],
                iou_thresh,
                single_class,
            )
            dataset_conf_matrix += image_results["confusion_matrix"]
            detailed_dataset_results[image_number] = image_results

        metrics = self.calculate_metrics(confusion_matrix=dataset_conf_matrix, single_class=single_class)
        return {
            "image_results": detailed_dataset_results,
            "metrics": metrics,
            "confusion_matrix": dataset_conf_matrix,
            "normalized_confusion_matrix": self.normalize_confusion_matrix(dataset_conf_matrix),
        }

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: tuple = CLASS_NAMES) -> None:
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="0.2f" if cm.dtype == np.float32 else "d",
            cmap="Blues",
            xticklabels=list(class_names) + ["Background"],
            yticklabels=list(class_names) + ["Background"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    @staticmethod
    def plot_boxes(image: "Image.Image | str", boxes: list, class_label: int | None = None, is_ground_truth: bool = True) -> None:
        if isinstance(image, str):
            image = Image.open(image)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        class_labels = {0: "clear", 1: "cloudy", 2: "amber"}
        colors = {0: "grey", 1: "white", 2: "orange"}
        linestyle = "dashed" if is_ground_truth else "solid"

        for box in boxes:
            if class_label is None or box["class_id"] == class_label:
                x_min, y_min, x_max, y_max = box["bbox"]
                cls_id = box["class_id"]
                rect = patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=1.5,
                    edgecolor=colors[cls_id],
                    facecolor="none",
                    linestyle=linestyle,
                    label=class_labels[cls_id],
                )
                ax.add_patch(rect)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()
