import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from segments import SegmentsClient, SegmentsDataset
from segments.utils import bitmap2file

import config

load_dotenv()


class SegmentsAIHandler:
    def __init__(self):
        api_key = os.getenv("SEGMENTS_API_KEY")
        self.client = SegmentsClient(api_key)

    @staticmethod
    def get_dataset_identifier(image_number, week, zoom_type="3x_regular"):
        return f"etaylor/cannabis_patches_{week}_{zoom_type}_{image_number}"

    def get_dataset_instance(self, dataset_name, version="v0.1"):
        release = self.client.get_release(dataset_name, version)
        return SegmentsDataset(release)

    def create_dataset(self, identifier, description, task_type, task_attributes):
        return self.client.add_dataset(identifier, description, task_type, task_attributes)

    def add_collaborator(self, dataset_id, user, role="annotator"):
        return self.client.add_dataset_collaborator(dataset_id, user, role)

    def upload_image(self, dataset_id, image_path):
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            asset = self.client.upload_asset(f, filename)

        sample_attrs = {"image": {"url": asset.url}}
        return self.client.add_sample(dataset_id, filename, sample_attrs)

    def upload_images(self, dataset_id, folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as f:
                asset = self.client.upload_asset(f, filename)

            sample_attrs = {"image": {"url": asset.url}}
            self.client.add_sample(dataset_id, filename, sample_attrs)

    def visualize_images(self, *images):
        for i, image in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(np.array(image))
        plt.show()

    def visualize_dataset(self, dataset_id, version="v0.1"):
        release = self.client.get_release(dataset_id, version)
        dataset = SegmentsDataset(release)

        for sample in dataset:
            try:
                self.visualize_images(sample["image"], sample["segmentation_bitmap"])
            except TypeError:
                pass

    def upload_annotation(self, sample_uuid, bitmap, annotation_data):
        seg_file = bitmap2file(bitmap)
        asset = self.client.upload_asset(seg_file, "label.png")

        label_attrs = {
            "format_version": "0.1",
            "annotations": annotation_data,
            "segmentation_bitmap": {"url": asset.url},
        }
        self.client.add_label(sample_uuid, "ground-truth", label_attrs, label_status="PRELABELED")

    def should_skip_sample(self, sample, only_patches):
        if not only_patches:
            return False
        return sample.name.endswith(".JPG") or "_p" not in sample.name

    def copy_sample_if_valid(self, sample, dest_dataset_id, label_status):
        label = self.client.get_label(sample.uuid)
        if not label or label.label_status != label_status:
            return False

        new_sample = self.client.add_sample(dest_dataset_id, sample.name, sample.attributes)
        self.client.add_label(new_sample.uuid, "ground-truth", label.attributes, label_status=label.label_status)
        return True

    def copy_dataset(self, source_id, dest_id, label_status="REVIEWED", only_patches=False):
        samples = self.client.get_samples(source_id)
        for sample in samples:
            if not self.should_skip_sample(sample, only_patches):
                self.copy_sample_if_valid(sample, dest_id, label_status)

    def decrement_category_ids(self, dataset_id, labelset="ground-truth"):
        samples = self.client.get_samples(dataset_id)

        for sample in samples:
            label = self.client.get_label(sample.uuid, labelset)
            if not label or not hasattr(label.attributes, "annotations"):
                continue

            attrs = label.attributes.dict() if not isinstance(label.attributes, dict) else label.attributes

            for annotation in attrs["annotations"]:
                if annotation.get("category_id", 0) > 0:
                    annotation["category_id"] -= 1

            self.client.update_label(sample.uuid, labelset, attrs)

    def get_trichome_distribution(self, image_number):
        if "etaylor" not in image_number:
            full_week, full_zoom = config.find_image_details(image_number)
            week = full_week.split("_")[0]
            zoom_type = full_zoom.split("_")[0] + "r"
            dataset_id = self.get_dataset_identifier(
                image_number,
                week=config.WEEKS_DIR[week],
                zoom_type=config.ZOOM_TYPES_DIR[zoom_type],
            )
        else:
            dataset_id = image_number

        samples = self.client.get_samples(dataset_id)
        distribution = {"clear": 0, "cloudy": 0, "amber": 0}

        for sample in samples:
            label = self.client.get_label(sample.uuid)
            for annotation in label.attributes.annotations:
                trichome_type = config.ANNOTATIONS_CLASS_MAPPINGS.get(annotation.category_id, 0)
                if trichome_type in distribution:
                    distribution[trichome_type] += 1

        return distribution
