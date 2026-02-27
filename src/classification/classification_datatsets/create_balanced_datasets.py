import os
import random
import shutil


class BalancedDatasetCreator:
    @staticmethod
    def print_image_distribution(dataset_path: str) -> None:
        if not os.path.exists(dataset_path):
            return
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                print(f"Class '{class_name}': {len(os.listdir(class_path))} images")

    @staticmethod
    def create_balanced_train_test_split(
        original_path: str, balanced_path: str, dataset_num: int, target_count: int = 200
    ) -> None:
        train_path = os.path.join(original_path, "train")
        balanced_train_path = os.path.join(balanced_path, f"balanced_{dataset_num}", "train")
        test_path = os.path.join(balanced_path, f"balanced_{dataset_num}", "val")
        os.makedirs(balanced_train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        def split_images_to_train_test(
            source_class_path: str, target_train_path: str, target_test_path: str, target_count: int = 200
        ) -> None:
            images = os.listdir(source_class_path)
            random.shuffle(images)
            train_images = images[:target_count]
            test_images = images[target_count:]
            for i, img_name in enumerate(train_images):
                shutil.copy(
                    os.path.join(source_class_path, img_name),
                    os.path.join(target_train_path, f"{i}_{img_name}"),
                )
            for img_name in test_images:
                shutil.copy(
                    os.path.join(source_class_path, img_name),
                    os.path.join(target_test_path, img_name),
                )

        for class_name in os.listdir(train_path):
            source_class_path = os.path.join(train_path, class_name)
            target_class_train_path = os.path.join(balanced_train_path, class_name)
            target_class_test_path = os.path.join(test_path, class_name)
            os.makedirs(target_class_train_path, exist_ok=True)
            os.makedirs(target_class_test_path, exist_ok=True)
            split_images_to_train_test(
                source_class_path, target_class_train_path, target_class_test_path, target_count
            )

    @staticmethod
    def create_balanced_split(
        original_path: str, balanced_path: str, dataset_num: int, target_count: int = 200
    ) -> None:
        train_path = os.path.join(original_path, "train")
        original_val_path = os.path.join(original_path, "val")
        balanced_train_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "train")
        new_val_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "val")
        test_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "test")
        os.makedirs(balanced_train_path, exist_ok=True)
        os.makedirs(new_val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        def balance_class_images(
            source_class_path: str, target_class_path: str, val_class_path: str, target_count: int = 200
        ) -> None:
            images = os.listdir(source_class_path)
            random.shuffle(images)
            train_images = images[:target_count]
            val_images = images[target_count:]
            for i, img_name in enumerate(train_images):
                shutil.copy(
                    os.path.join(source_class_path, img_name),
                    os.path.join(target_class_path, f"{i}_{img_name}"),
                )
            val_split = len(val_images) // 2
            for img_name in val_images[:val_split]:
                shutil.copy(
                    os.path.join(source_class_path, img_name),
                    os.path.join(val_class_path, img_name),
                )
            class_test_path = os.path.join(target_class_path, os.path.basename(source_class_path))
            os.makedirs(class_test_path, exist_ok=True)
            for img_name in val_images[val_split:]:
                shutil.copy(
                    os.path.join(source_class_path, img_name),
                    os.path.join(class_test_path, img_name),
                )

        for class_name in os.listdir(train_path):
            source_class_path = os.path.join(train_path, class_name)
            target_class_path = os.path.join(balanced_train_path, class_name)
            val_class_path = os.path.join(new_val_path, class_name)
            os.makedirs(target_class_path, exist_ok=True)
            os.makedirs(val_class_path, exist_ok=True)
            os.makedirs(os.path.join(test_path, class_name), exist_ok=True)
            balance_class_images(source_class_path, target_class_path, val_class_path)

        for class_name in os.listdir(original_val_path):
            original_class_path = os.path.join(original_val_path, class_name)
            new_val_class_path = os.path.join(new_val_path, class_name)
            os.makedirs(new_val_class_path, exist_ok=True)
            for img_name in os.listdir(original_class_path):
                shutil.copy(
                    os.path.join(original_class_path, img_name),
                    os.path.join(new_val_class_path, img_name),
                )

    @staticmethod
    def split_val_to_val_test(balanced_path: str, dataset_num: int, split_ratio: float = 0.5) -> None:
        val_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "val")
        test_path = os.path.join(balanced_path, f"train_set_{dataset_num}", "test")

        for class_name in os.listdir(val_path):
            class_val_path = os.path.join(val_path, class_name)
            class_test_path = os.path.join(test_path, class_name)
            os.makedirs(class_test_path, exist_ok=True)

            images = os.listdir(class_val_path)
            random.shuffle(images)
            split_point = int(len(images) * split_ratio)
            for img_name in images[split_point:]:
                shutil.move(
                    os.path.join(class_val_path, img_name),
                    os.path.join(class_test_path, img_name),
                )

    @staticmethod
    def create_balanced_test_set(source_dir: str, dest_dir: str, num_samples_per_class: int = 26) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        for class_name in os.listdir(source_dir):
            class_source_path = os.path.join(source_dir, class_name)
            class_dest_path = os.path.join(dest_dir, class_name)
            if not os.path.isdir(class_source_path):
                continue
            os.makedirs(class_dest_path, exist_ok=True)
            image_files = [
                f for f in os.listdir(class_source_path)
                if os.path.isfile(os.path.join(class_source_path, f))
            ]
            sampled_files = (
                image_files if len(image_files) < num_samples_per_class
                else random.sample(image_files, num_samples_per_class)
            )
            for file_name in sampled_files:
                shutil.copy2(
                    os.path.join(class_source_path, file_name),
                    os.path.join(class_dest_path, file_name),
                )

    @staticmethod
    def split_images_by_ratio(
        source_class_path: str,
        target_train_path: str,
        target_test_path: str,
        train_ratio: float = 0.75,
        random_split: bool = True,
    ) -> None:
        images = [
            f for f in os.listdir(source_class_path)
            if os.path.isfile(os.path.join(source_class_path, f))
        ]
        if random_split:
            random.shuffle(images)
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        os.makedirs(target_train_path, exist_ok=True)
        os.makedirs(target_test_path, exist_ok=True)

        for i, img_name in enumerate(train_images):
            shutil.copy2(
                os.path.join(source_class_path, img_name),
                os.path.join(target_train_path, f"{i}_{img_name}"),
            )
        for i, img_name in enumerate(test_images):
            shutil.copy2(
                os.path.join(source_class_path, img_name),
                os.path.join(target_test_path, f"{i}_{img_name}"),
            )

    @staticmethod
    def create_train_test_split_by_ratio(
        dataset_path: str, split_dataset_path: str, train_ratio: float = 0.8, random_split: bool = True
    ) -> None:
        os.makedirs(split_dataset_path, exist_ok=True)
        train_root = os.path.join(split_dataset_path, "train")
        test_root = os.path.join(split_dataset_path, "val")
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path) or class_name.lower() in ["train", "val"]:
                continue
            target_train_path = os.path.join(train_root, class_name)
            target_test_path = os.path.join(test_root, class_name)
            os.makedirs(target_train_path, exist_ok=True)
            os.makedirs(target_test_path, exist_ok=True)
            BalancedDatasetCreator.split_images_by_ratio(
                class_path, target_train_path, target_test_path, train_ratio, random_split
            )

    @staticmethod
    def print_dataset_stats(split_dataset_path: str) -> None:
        train_path = os.path.join(split_dataset_path, "train")
        test_path = os.path.join(split_dataset_path, "val")
        if not os.path.isdir(train_path) or not os.path.isdir(test_path):
            return

        for split_name, split_path in [("TRAIN", train_path), ("TEST", test_path)]:
            total = 0
            for class_name in os.listdir(split_path):
                class_dir = os.path.join(split_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                n_images = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                total += n_images
                print(f"  Class '{class_name}': {n_images} images")
            print(f"Total images in {split_name.lower()} folder: {total}")


if __name__ == "__main__":
    _dataset_path = "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/good_quality"
    _balanced_datasets_folder = "/home/etaylor/code_projects/thesis/classification_datasets/trichome_classification/balanced_datasets"

    for run in range(1, 6):
        BalancedDatasetCreator.create_train_test_split_by_ratio(
            dataset_path=_dataset_path,
            split_dataset_path=os.path.join(_balanced_datasets_folder, f"balanced_{run}"),
            train_ratio=0.75,
            random_split=True,
        )
