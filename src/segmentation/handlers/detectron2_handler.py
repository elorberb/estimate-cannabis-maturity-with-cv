import csv
import os
import random

import cv2
import detectron2
import pandas as pd
import seaborn as sns
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from src.annotations.annotation_handler import AnnotationHandler

DETECTRON2_CHECKPOINT_BASE_PATH = "checkpoints/detectron2"

DETECTION_MODELS = [
    "COCO-Detection/faster_rcnn_R_101_C4_3x",
    "COCO-Detection/faster_rcnn_R_101_DC5_3x",
    "COCO-Detection/faster_rcnn_R_101_FPN_3x",
    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x",
    "COCO-Detection/retinanet_R_101_FPN_3x",
]

SEGMENTATION_MODELS = [
    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x",
]


class Detectron2Handler:
    def __init__(self) -> None:
        self._annotation_handler = AnnotationHandler()

    @staticmethod
    def print_version_info() -> None:
        torch_version = ".".join(torch.__version__.split(".")[:2])
        cuda_version = torch.version.cuda
        print("torch:", torch_version, "; cuda:", cuda_version)
        print("detectron2:", detectron2.__version__)

    @staticmethod
    def register_coco_dataset(name: str, json_path: str, images_path: str) -> tuple:
        register_coco_instances(name, {}, json_path, images_path)
        return MetadataCatalog.get(name), DatasetCatalog.get(name)

    def register_dataset(self, dataset_name: str, release_version: str) -> tuple:
        _, export_json_path, saved_images_path = self._annotation_handler.convert_segments_to_coco_format(
            dataset_name, release_version
        )
        return Detectron2Handler.register_coco_dataset(dataset_name, export_json_path, saved_images_path)

    @staticmethod
    def split_dataset(dataset_dicts: list, train_ratio: float = 0.8) -> tuple[list, list]:
        shuffled = dataset_dicts.copy()
        random.shuffle(shuffled)
        split_idx = int(train_ratio * len(shuffled))
        return shuffled[:split_idx], shuffled[split_idx:]

    @staticmethod
    def register_split(name: str, dataset_dicts: list, thing_classes: list):
        DatasetCatalog.register(name, lambda d=dataset_dicts: d)
        MetadataCatalog.get(name).set(thing_classes=thing_classes)
        return MetadataCatalog.get(name)

    def register_and_split_dataset(
        self, dataset_name: str, release_version: str, train_ratio: float = 0.8
    ) -> tuple:
        metadata, dataset_dicts = self.register_dataset(dataset_name, release_version)
        train_dicts, test_dicts = Detectron2Handler.split_dataset(dataset_dicts, train_ratio)

        train_metadata = Detectron2Handler.register_split(
            f"{dataset_name}_train", train_dicts, metadata.thing_classes
        )
        test_metadata = Detectron2Handler.register_split(
            f"{dataset_name}_test", test_dicts, metadata.thing_classes
        )

        return train_metadata, train_dicts, test_metadata, test_dicts

    def prepare_and_register_datasets(
        self,
        dataset_name_train: str,
        dataset_name_test: str,
        release_train: str,
        release_test: str,
    ) -> tuple:
        _, train_json, train_images = self._annotation_handler.convert_segments_to_coco_format(
            dataset_name_train, release_train
        )
        _, test_json, test_images = self._annotation_handler.convert_segments_to_coco_format(
            dataset_name_test, release_test
        )

        register_coco_instances(dataset_name_train, {}, train_json, train_images)
        register_coco_instances(dataset_name_test, {}, test_json, test_images)

        return (
            MetadataCatalog.get(dataset_name_train),
            DatasetCatalog.get(dataset_name_train),
            MetadataCatalog.get(dataset_name_test),
            DatasetCatalog.get(dataset_name_test),
        )

    @staticmethod
    def visualize_sample(d: dict, metadata, scale: float = 0.5) -> None:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
        out = visualizer.draw_dataset_dict(d)
        image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()

    @staticmethod
    def visualize_samples(dataset_dicts: list, metadata, indices: list | None = None, scale: float = 0.5) -> None:
        samples = dataset_dicts if indices is None else [dataset_dicts[i] for i in indices]
        for d in samples:
            Detectron2Handler.visualize_sample(d, metadata, scale)

    @staticmethod
    def visualize_prediction(d: dict, metadata, predictor, scale: float = 0.5) -> None:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=scale, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()

    @staticmethod
    def evaluate_model_on_dataset(cfg, predictor) -> dict:
        evaluator = COCOEvaluator("my_dataset_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_output"))
        val_loader = build_detection_test_loader(cfg, "my_dataset_val")
        return inference_on_dataset(predictor.model, val_loader, evaluator)

    @staticmethod
    def extract_object_info_to_csv(
        input_images_directory: str, output_csv_path: str, predictor, metadata
    ) -> str:
        with open(output_csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])

            for image_filename in os.listdir(input_images_directory):
                image_path = os.path.join(input_images_directory, image_filename)
                image = cv2.imread(image_path)
                outputs = predictor(image)

                mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
                class_labels = outputs["instances"].pred_classes.to("cpu").numpy()
                labeled_mask = label(mask)
                props = regionprops(labeled_mask)

                for i, prop in enumerate(props):
                    class_label = class_labels[i] if i < len(class_labels) else "Unknown"
                    class_name = (
                        "Unknown" if class_label == "Unknown" else metadata.thing_classes[class_label]
                    )
                    csvwriter.writerow([
                        os.path.basename(image_path), class_name, i + 1,
                        prop.area, prop.centroid, prop.bbox,
                    ])

        return f"Object-level information saved to {output_csv_path}"

    @staticmethod
    def plot_class_statistics(output_csv_path: str, metadata_train) -> None:
        df = pd.read_csv(output_csv_path)
        class_names = metadata_train.thing_classes

        avg_objects = df.groupby(["File Name", "Class Name"])["Object Number"].count().reset_index()
        avg_objects = avg_objects.groupby("Class Name")["Object Number"].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Class Name", y="Object Number", data=avg_objects, order=class_names)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        avg_area = df.groupby("Class Name")["Area"].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Class Name", y="Area", data=avg_area, order=class_names)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
