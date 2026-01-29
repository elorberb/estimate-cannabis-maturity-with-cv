from IPython import display as ipydisplay
import ultralytics


ULTRALYTICS_MODELS = [
    "rtdetr-x.pt",
    "yolov5xu.pt",
    "yolov8x.pt",
    "yolov9c.pt",
]


def validate_version_and_gpu():
    ipydisplay.clear_output(wait=True)
    ultralytics.checks()
