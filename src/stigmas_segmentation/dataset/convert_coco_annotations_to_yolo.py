from ultralytics.data.converter import convert_coco


class CocoToYoloConverter:
    @staticmethod
    def convert(labels_dir: str, save_dir: str, use_segments: bool = True) -> None:
        convert_coco(labels_dir=labels_dir, save_dir=save_dir, use_segments=use_segments)


if __name__ == "__main__":
    CocoToYoloConverter.convert(
        labels_dir="/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/annotations",
        save_dir="/home/etaylor/code_projects/thesis/segments/etaylor_stigmas_dataset/yolo",
    )
