import os
import shutil
import random
import numpy as np
from pathlib import Path
from segments.utils import export_dataset
from ultralytics.data.converter import convert_coco

from src.datasets_and_annotations.segmentsai_handler import SegmentsAIHandler
import config

SEGMENTS_HANDLER = SegmentsAIHandler()

CLASS_NAMES = {0: "trichome", 1: "clear", 2: "cloudy", 3: "amber"}


def convert_coco_to_segments(image, outputs):
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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def convert_segments_to_coco_format(dataset_name, release_version, export_format="coco-instance", output_dir="."):
    dataset = SEGMENTS_HANDLER.get_dataset_instance(dataset_name, version=release_version)
    export_json_path, saved_images_path = export_dataset(dataset, export_format=export_format, export_folder=output_dir)

    annotations_folder = ensure_dir(os.path.join(os.path.dirname(saved_images_path), "annotations"))
    new_export_json_path = os.path.join(annotations_folder, os.path.basename(export_json_path))
    shutil.move(export_json_path, new_export_json_path)

    return dataset, new_export_json_path, saved_images_path


def link_image(src_dir, dst_dir, img_name):
    os.symlink(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))


def copy_label(label_dir, dst_label_dir, img_name):
    label_name = os.path.splitext(img_name)[0] + ".txt"
    src_path = os.path.join(label_dir, label_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(dst_label_dir, label_name))


def link_images_and_copy_labels(images, source_dir, target_img_dir, label_dir, target_label_dir):
    for img_name in images:
        link_image(source_dir, target_img_dir, img_name)
        copy_label(label_dir, target_label_dir, img_name)


def list_images(directory):
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith((".jpg", ".png"))]


def split_list(items, ratio):
    random.shuffle(items)
    split_idx = int(len(items) * ratio)
    return items[:split_idx], items[split_idx:]


def prepare_train_val_splits(image_dir, label_dir, train_percentage, output_base_dir):
    train_img_dir = ensure_dir(os.path.join(output_base_dir, "images/train"))
    val_img_dir = ensure_dir(os.path.join(output_base_dir, "images/val"))
    train_label_dir = ensure_dir(os.path.join(output_base_dir, "labels/train"))
    val_label_dir = ensure_dir(os.path.join(output_base_dir, "labels/val"))

    all_images = list_images(image_dir)
    train_images, val_images = split_list(all_images, train_percentage)

    link_images_and_copy_labels(train_images, image_dir, train_img_dir, label_dir, train_label_dir)
    link_images_and_copy_labels(val_images, image_dir, val_img_dir, label_dir, val_label_dir)


def setup_yolo_dataset_directory(image_dir, label_dir, output_base_dir):
    img_output_dir = ensure_dir(os.path.join(output_base_dir, "images"))
    label_output_dir = ensure_dir(os.path.join(output_base_dir, "labels"))
    all_images = list_images(image_dir)
    link_images_and_copy_labels(all_images, image_dir, img_output_dir, label_dir, label_output_dir)


def create_yaml(dataset_path, yaml_path, train_dir="images/train", val_dir="images/val"):
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


def remove_dir_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def convert_coco_to_yolo_single(annotations_folder_name, dataset_version, saving_yaml_path, train_percentage=0.8):
    annotations_dir = f"{config.SEGMENTS_FOLDER}/{annotations_folder_name}/annotations"
    output_dir = f"{annotations_dir}/yolo"
    image_dir = f"{config.SEGMENTS_FOLDER}/{annotations_folder_name}/{dataset_version}"
    label_dir = f"{output_dir}/labels/export_coco-instance_{annotations_folder_name}_{dataset_version}"
    organized_path = f"{output_dir}_split"

    convert_coco(labels_dir=annotations_dir, save_dir=output_dir, use_segments=True)
    prepare_train_val_splits(image_dir, label_dir, train_percentage, organized_path)

    yaml_file_path = os.path.join(saving_yaml_path, f"{annotations_folder_name}_{dataset_version}_data.yaml")
    create_yaml(organized_path, yaml_file_path)
    remove_dir_if_exists(output_dir)

    return yaml_file_path


def convert_coco_to_yolo_train_test(train_folder, train_version, test_folder, test_version, saving_yaml_path):
    train_annotations_dir = f"{config.SEGMENTS_FOLDER}/{train_folder}/annotations"
    train_image_dir = f"{config.SEGMENTS_FOLDER}/{train_folder}/{train_version}"
    train_output_dir = f"{train_annotations_dir}/yolo"
    train_label_dir = f"{train_output_dir}/labels/export_coco-instance_{train_folder}_{train_version}"
    train_organized = f"{train_output_dir}_split"

    test_annotations_dir = f"{config.SEGMENTS_FOLDER}/{test_folder}/annotations"
    test_image_dir = f"{config.SEGMENTS_FOLDER}/{test_folder}/{test_version}"
    test_output_dir = f"{test_annotations_dir}/yolo"
    test_label_dir = f"{test_output_dir}/labels/export_coco-instance_{test_folder}_{test_version}"
    test_organized = f"{test_output_dir}_split"

    convert_coco(labels_dir=train_annotations_dir, save_dir=train_output_dir, use_segments=False)
    convert_coco(labels_dir=test_annotations_dir, save_dir=test_output_dir, use_segments=False)

    setup_yolo_dataset_directory(train_image_dir, train_label_dir, train_organized)
    setup_yolo_dataset_directory(test_image_dir, test_label_dir, test_organized)

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
