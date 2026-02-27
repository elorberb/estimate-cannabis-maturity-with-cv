import argparse
import logging
import os

import torch
from ultralytics import RTDETR, YOLO, settings

from src.config.settings import (
    ULTRALYTICS_DATASETS_DIR,
    ULTRALYTICS_RUNS_DIR,
    ULTRALYTICS_WEIGHTS_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ULTRALYTICS_CHECKPOINT_PATHS = "/home/etaylor/code_projects/thesis/checkpoints/ultralytics"


class UltralyticsTrainer:
    @staticmethod
    def empty_cuda_cache() -> None:
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

    @staticmethod
    def setup_ultralytics_settings() -> None:
        settings.update({
            "runs_dir": ULTRALYTICS_RUNS_DIR,
            "weights_dir": ULTRALYTICS_WEIGHTS_DIR,
            "datasets_dir": ULTRALYTICS_DATASETS_DIR,
        })
        logger.info("Ultralytics settings updated:\n%s", settings)

    @staticmethod
    def train_model(config_yaml: str, model_checkpoint: str, epochs: int = 100, imgsz: int = 512):
        model_checkpoint_path = os.path.join(ULTRALYTICS_CHECKPOINT_PATHS, model_checkpoint)

        model = RTDETR(model_checkpoint_path) if model_checkpoint == "rtdetr-x.pt" else YOLO(model_checkpoint_path)
        results = model.train(data=config_yaml, epochs=epochs, imgsz=imgsz)
        return model, results

    @staticmethod
    def validate_model(model) -> None:
        valid_results = model.val()
        logger.info("Validation results:\n%s", valid_results)

    @staticmethod
    def run(args) -> None:
        UltralyticsTrainer.empty_cuda_cache()
        UltralyticsTrainer.setup_ultralytics_settings()

        logger.info(f"Starting training of Ultralytics model {args.checkpoint}")
        model, results = UltralyticsTrainer.train_model(args.config, args.checkpoint, args.epochs, args.imgsz)
        logger.info("Training results:\n%s", results)

        UltralyticsTrainer.validate_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)

    UltralyticsTrainer.run(parser.parse_args())
