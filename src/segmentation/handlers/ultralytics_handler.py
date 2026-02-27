import ultralytics
from IPython import display as ipydisplay

ULTRALYTICS_MODELS = [
    "rtdetr-x.pt",
    "yolov5xu.pt",
    "yolov8x.pt",
    "yolov9c.pt",
]


class UltralyticsHandler:
    @staticmethod
    def validate_version_and_gpu() -> None:
        ipydisplay.clear_output(wait=True)
        ultralytics.checks()
