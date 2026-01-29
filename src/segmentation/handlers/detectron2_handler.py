import os
import random
import csv
import cv2
import torch
import detectron2
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from skimage.measure import regionprops, label

from src.datasets_and_annotations.segmentsai_handler import SegmentsAIHandler
from src.datasets_and_annotations import annotation_handler


SEGMENTS_HANDLER = SegmentsAIHandler()
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


def print_version_info():
    torch_version = ".".join(torch.__version__.split(".")[:2])
    cuda_version = torch.version.cuda
    print("torch:", torch_version, "; cuda:", cuda_version)
    print("detectron2:", detectron2.__version__)


def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


def convert_to_coco(dataset_name, release_version):
    return annotation_handler.convert_segments_to_coco_format(
        dataset_name=dataset_name,
        release_version=release_version,
    )


def register_coco_dataset(name, json_path, images_path):
    register_coco_instances(name, {}, json_path, images_path)
    return MetadataCatalog.get(name), DatasetCatalog.get(name)


def register_dataset(dataset_name, release_version):
    _, export_json_path, saved_images_path = convert_to_coco(dataset_name, release_version)
    return register_coco_dataset(dataset_name, export_json_path, saved_images_path)


def split_dataset(dataset_dicts, train_ratio=0.8):
    shuffled = dataset_dicts.copy()
    random.shuffle(shuffled)
    split_idx = int(train_ratio * len(shuffled))
    return shuffled[:split_idx], shuffled[split_idx:]


def register_split(name, dataset_dicts, thing_classes):
    DatasetCatalog.register(name, lambda d=dataset_dicts: d)
    MetadataCatalog.get(name).set(thing_classes=thing_classes)
    return MetadataCatalog.get(name)


def register_and_split_dataset(dataset_name, release_version, train_ratio=0.8):
    metadata, dataset_dicts = register_dataset(dataset_name, release_version)
    train_dicts, test_dicts = split_dataset(dataset_dicts, train_ratio)

    train_name = f"{dataset_name}_train"
    test_name = f"{dataset_name}_test"

    train_metadata = register_split(train_name, train_dicts, metadata.thing_classes)
    test_metadata = register_split(test_name, test_dicts, metadata.thing_classes)

    return train_metadata, train_dicts, test_metadata, test_dicts


def convert_and_register_datasets(dataset_name_train, dataset_name_test, release_train, release_test):
    _, train_json, train_images = convert_to_coco(dataset_name_train, release_train)
    _, test_json, test_images = convert_to_coco(dataset_name_test, release_test)

    register_coco_instances(dataset_name_train, {}, train_json, train_images)
    register_coco_instances(dataset_name_test, {}, test_json, test_images)

    metadata_train = MetadataCatalog.get(dataset_name_train)
    dataset_dicts_train = DatasetCatalog.get(dataset_name_train)
    metadata_test = MetadataCatalog.get(dataset_name_test)
    dataset_dicts_test = DatasetCatalog.get(dataset_name_test)

    return metadata_train, dataset_dicts_train, metadata_test, dataset_dicts_test


def visualize_image(img, metadata, visualizer_func, scale=0.5):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=scale)
    out = visualizer_func(visualizer)
    image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


def visualize_sample(d, metadata, scale=0.5):
    img = cv2.imread(d["file_name"])
    visualize_image(img, metadata, lambda v: v.draw_dataset_dict(d), scale)


def visualize_samples(dataset_dicts, metadata, indices=None, scale=0.5):
    samples = dataset_dicts if indices is None else [dataset_dicts[i] for i in indices]
    for d in samples:
        visualize_sample(d, metadata, scale)


def visualize_prediction(d, metadata, predictor, scale=0.5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=scale, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_rgb = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


def visualize_predictions(dataset_dicts, metadata, predictor, indices=None, scale=0.5):
    samples = dataset_dicts if indices is None else [dataset_dicts[i] for i in indices]
    for d in samples:
        visualize_prediction(d, metadata, predictor, scale)


def evaluate_model(cfg, predictor):
    evaluator = COCOEvaluator("my_dataset_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_output"))
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    return inference_on_dataset(predictor.model, val_loader, evaluator)


def get_masks_and_labels(outputs):
    mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)
    class_labels = outputs["instances"].pred_classes.to("cpu").numpy()
    return mask, class_labels


def extract_region_props(mask):
    labeled_mask = label(mask)
    return regionprops(labeled_mask)


def write_csv_row(csvwriter, filename, class_name, obj_num, area, centroid, bbox):
    csvwriter.writerow([filename, class_name, obj_num, area, centroid, bbox])


def get_class_name(class_label, metadata):
    if class_label == "Unknown":
        return "Unknown"
    return metadata.thing_classes[class_label]


def process_image_objects(image_path, predictor, metadata, csvwriter):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    mask, class_labels = get_masks_and_labels(outputs)
    props = extract_region_props(mask)

    for i, prop in enumerate(props):
        class_label = class_labels[i] if i < len(class_labels) else "Unknown"
        class_name = get_class_name(class_label, metadata)
        write_csv_row(csvwriter, os.path.basename(image_path), class_name, i + 1, prop.area, prop.centroid, prop.bbox)


def extract_object_info_to_csv(input_images_directory, output_csv_path, predictor, metadata):
    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])

        for image_filename in os.listdir(input_images_directory):
            image_path = os.path.join(input_images_directory, image_filename)
            process_image_objects(image_path, predictor, metadata, csvwriter)

    return f"Object-level information saved to CSV file at {output_csv_path}"


def compute_avg_objects_per_class(df):
    avg_objects = df.groupby(["File Name", "Class Name"])["Object Number"].count().reset_index()
    return avg_objects.groupby("Class Name")["Object Number"].mean().reset_index()


def plot_bar_chart(data, x_col, y_col, title, xlabel, ylabel, order=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=data, order=order)
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_avg_objects_per_class(df, class_names):
    avg_objects = compute_avg_objects_per_class(df)
    plot_bar_chart(avg_objects, "Class Name", "Object Number",
                   "Average Number of Objects per Image for Each Class",
                   "Class Name", "Average Number of Objects per Image", class_names)


def compute_avg_area_per_class(df):
    return df.groupby("Class Name")["Area"].mean().reset_index()


def plot_avg_area_per_class(df, class_names):
    avg_area = compute_avg_area_per_class(df)
    plot_bar_chart(avg_area, "Class Name", "Area",
                   "Average Area of Objects for Each Class",
                   "Class Name", "Average Area of Objects", class_names)


def plot_class_statistics(output_csv_path, metadata_train):
    df = pd.read_csv(output_csv_path)
    class_names = metadata_train.thing_classes
    plot_avg_objects_per_class(df, class_names)
    plot_avg_area_per_class(df, class_names)
