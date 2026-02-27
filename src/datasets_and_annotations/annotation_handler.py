import os
import random
import shutil
from pathlib import Path

import numpy as np
from segments.utils import export_dataset
from ultralytics.data.converter import convert_coco

from src.config.settings import SEGMENTS_FOLDER
from src.datasets_and_annotations.segmentsai_handler import SegmentsAIHandler

CLASS_NAMES = {0: "trichome", 1: "clear", 2: "cloudy", 3: "amber"}


class AnnotationHandler:
    def __init__(self) -> None:
        self._segments_handler = SegmentsAIHandler()

    def convert_coco_to_segments(
        self, image: np.ndarray, outputs
    ) -> tuple[np.ndarray, list]:
        segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
        annotations = []
        instances = outputs["instances"]

        for i in range(len(instances.pred_classes)):
            instance_id = i + 1
            category_id = int(instances.pred_classes[i])
            mask = instances.pred_masks[i].cpu()
            segmentation_bitmap[mask] = instance_id
            annotations.append({"id": instance_id, "category_id": category_id})

        return segmentation_bitmap, annotations

    def convert_segments_to_coco_format(
        self,
        dataset_name: str,
        release_version: str,
        export_format: str = "coco-instance",
        output_dir: str = ".",
    ) -> tuple:
        dataset = self._segments_handler.get_dataset_instance(dataset_name, version=release_version)
        export_json_path, saved_images_path = export_dataset(
            dataset, export_format=export_format, export_folder=output_dir
        )

        annotations_folder = os.path.join(os.path.dirname(saved_images_path), "annotations")
        os.makedirs(annotations_folder, exist_ok=True)
        new_export_json_path = os.path.join(annotations_folder, os.path.basename(export_json_path))
        shutil.move(export_json_path, new_export_json_path)

        return dataset, new_export_json_path, saved_images_path

    @staticmethod
    def _link_image(src_dir: str, dst_dir: str, img_name: str) -> None:
        os.symlink(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))

    @staticmethod
    def _copy_label(label_dir: str, dst_label_dir: str, img_name: str) -> None:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_path = os.path.join(label_dir, label_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst_label_dir, label_name))

    @staticmethod
    def _link_images_and_copy_labels(
        images: list[str],
        source_dir: str,
        target_img_dir: str,
        label_dir: str,
        target_label_dir: str,
    ) -> None:
        for img_name in images:
            AnnotationHandler._link_image(source_dir, target_img_dir, img_name)
            AnnotationHandler._copy_label(label_dir, target_label_dir, img_name)

    @staticmethod
    def _list_images(directory: str) -> list[str]:
        return [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith((".jpg", ".png"))
        ]

    @staticmethod
    def _split_list(items: list, ratio: float) -> tuple[list, list]:
        random.shuffle(items)
        split_idx = int(len(items) * ratio)
        return items[:split_idx], items[split_idx:]

    @staticmethod
    def prepare_train_val_splits(
        image_dir: str,
        label_dir: str,
        train_percentage: float,
        output_base_dir: str,
    ) -> None:
        train_img_dir = os.path.join(output_base_dir, "images/train")
        val_img_dir = os.path.join(output_base_dir, "images/val")
        train_label_dir = os.path.join(output_base_dir, "labels/train")
        val_label_dir = os.path.join(output_base_dir, "labels/val")

        for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            os.makedirs(d, exist_ok=True)

        all_images = AnnotationHandler._list_images(image_dir)
        train_images, val_images = AnnotationHandler._split_list(all_images, train_percentage)

        AnnotationHandler._link_images_and_copy_labels(train_images, image_dir, train_img_dir, label_dir, train_label_dir)
        AnnotationHandler._link_images_and_copy_labels(val_images, image_dir, val_img_dir, label_dir, val_label_dir)

    @staticmethod
    def setup_yolo_dataset_directory(
        image_dir: str, label_dir: str, output_base_dir: str
    ) -> None:
        img_output_dir = os.path.join(output_base_dir, "images")
        label_output_dir = os.path.join(output_base_dir, "labels")
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)
        all_images = AnnotationHandler._list_images(image_dir)
        AnnotationHandler._link_images_and_copy_labels(
            all_images, image_dir, img_output_dir, label_dir, label_output_dir
        )

    @staticmethod
    def create_yaml(dataset_path: str, yaml_path: str, train_dir: str = "images/train", val_dir: str = "images/val") -> None:
        yaml_content = f"""path: {dataset_path}
train: {train_dir}
val: {val_dir}

names:
  0: trichome
  1: clear
  2: cloudy
  3: amber
"""
        Path(yaml_path).write_text(yaml_content)

    @staticmethod
    def convert_coco_to_yolo_single(
        annotations_folder_name: str,
        dataset_version: str,
        saving_yaml_path: str,
        train_percentage: float = 0.8,
    ) -> str:
        annotations_dir = f"{SEGMENTS_FOLDER}/{annotations_folder_name}/annotations"
        output_dir = f"{annotations_dir}/yolo"
        image_dir = f"{SEGMENTS_FOLDER}/{annotations_folder_name}/{dataset_version}"
        label_dir = f"{output_dir}/labels/export_coco-instance_{annotations_folder_name}_{dataset_version}"
        organized_path = f"{output_dir}_split"

        convert_coco(labels_dir=annotations_dir, save_dir=output_dir, use_segments=True)
        AnnotationHandler.prepare_train_val_splits(image_dir, label_dir, train_percentage, organized_path)

        yaml_file_path = os.path.join(saving_yaml_path, f"{annotations_folder_name}_{dataset_version}_data.yaml")
        AnnotationHandler.create_yaml(organized_path, yaml_file_path)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        return yaml_file_path

    @staticmethod
    def convert_coco_to_yolo_train_test(
        train_folder: str,
        train_version: str,
        test_folder: str,
        test_version: str,
        saving_yaml_path: str,
    ) -> str:
        train_annotations_dir = f"{SEGMENTS_FOLDER}/{train_folder}/annotations"
        train_image_dir = f"{SEGMENTS_FOLDER}/{train_folder}/{train_version}"
        train_output_dir = f"{train_annotations_dir}/yolo"
        train_label_dir = f"{train_output_dir}/labels/export_coco-instance_{train_folder}_{train_version}"
        train_organized = f"{train_output_dir}_split"

        test_annotations_dir = f"{SEGMENTS_FOLDER}/{test_folder}/annotations"
        test_image_dir = f"{SEGMENTS_FOLDER}/{test_folder}/{test_version}"
        test_output_dir = f"{test_annotations_dir}/yolo"
        test_label_dir = f"{test_output_dir}/labels/export_coco-instance_{test_folder}_{test_version}"
        test_organized = f"{test_output_dir}_split"

        convert_coco(labels_dir=train_annotations_dir, save_dir=train_output_dir, use_segments=False)
        convert_coco(labels_dir=test_annotations_dir, save_dir=test_output_dir, use_segments=False)

        AnnotationHandler.setup_yolo_dataset_directory(train_image_dir, train_label_dir, train_organized)
        AnnotationHandler.setup_yolo_dataset_directory(test_image_dir, test_label_dir, test_organized)

        yaml_content = f"""train: {train_organized}/images
val: {test_organized}/images

names:
  0: trichome
  1: clear
  2: cloudy
  3: amber
"""
        yaml_file_path = os.path.join(saving_yaml_path, f"{train_folder}.yaml")
        Path(yaml_file_path).write_text(yaml_content)

        return yaml_file_path
