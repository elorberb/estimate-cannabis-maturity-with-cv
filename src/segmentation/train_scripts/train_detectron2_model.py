import argparse
import json
import logging
import os
from datetime import datetime

import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from src.segmentation.handlers.detectron2_handler import Detectron2Handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

DETECTRON2_MODELS_PATH = "/home/etaylor/code_projects/thesis/checkpoints/detectron2"
DATASET_NAME_TRAIN = "etaylor/cannabis_patches_train_26-04-2024_15-44-44"
DATASET_NAME_TEST = "etaylor/cannabis_patches_test_26-04-2024_15-44-44"
RELEASE = "v0.1"


class Detectron2Trainer:
    @staticmethod
    def setup() -> None:
        torch.cuda.empty_cache()
        setup_logger()

    @staticmethod
    def train_and_evaluate(model: str) -> None:
        logger.info(f"Training and evaluating model {model}")
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        model_saving_path = os.path.join(DETECTRON2_MODELS_PATH, model, current_time)
        os.makedirs(model_saving_path, exist_ok=True)

        handler = Detectron2Handler()
        handler.prepare_and_register_datasets(
            DATASET_NAME_TRAIN, DATASET_NAME_TEST, RELEASE, RELEASE
        )

        cfg = get_cfg()
        cfg.OUTPUT_DIR = model_saving_path
        cfg.merge_from_file(model_zoo.get_config_file(f"{model}.yaml"))
        cfg.DATASETS.TRAIN = (DATASET_NAME_TRAIN,)
        cfg.DATASETS.TEST = ()
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{model}.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 18450
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        config_yaml_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
        with open(config_yaml_path, "w") as file:
            yaml.dump(cfg, file)

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)

        coco_evaluator = COCOEvaluator(
            DATASET_NAME_TEST, output_dir=os.path.join(cfg.OUTPUT_DIR, DATASET_NAME_TEST)
        )
        val_loader = build_detection_test_loader(cfg, DATASET_NAME_TEST)
        evaluation_results = inference_on_dataset(predictor.model, val_loader, coco_evaluator)

        output_dir = os.path.join(cfg.OUTPUT_DIR, os.path.basename(DATASET_NAME_TEST))
        os.makedirs(output_dir, exist_ok=True)

        results_saving_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_saving_path, "w") as file:
            json.dump(evaluation_results, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    Detectron2Trainer.setup()
    Detectron2Trainer.train_and_evaluate(args.model)
