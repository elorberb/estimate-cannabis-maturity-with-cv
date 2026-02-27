import argparse
import json

from fastai.vision.all import (
    ImageDataLoaders,
    Precision,
    Recall,
    Resize,
    error_rate,
    get_image_files,
    vision_learner,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.classification.utils import CLASSIFICATION_MODELS

TRAIN_DATASET_PATH = "/home/etaylor/code_projects/thesis/segments/etaylor_cannabis_patches_train_26-04-2024_15-44-44/trichome_dataset"
TEST_DATASET_PATH = "/home/etaylor/code_projects/thesis/segments/etaylor_cannabis_patches_test_26-04-2024_15-44-44/trichome_dataset"


class FastaiTrainer:
    @staticmethod
    def train_and_evaluate(
        model_name: str,
        model_func,
        train_path: str,
        test_path: str,
        epochs: int,
        output_path: str,
    ) -> dict:
        dls = ImageDataLoaders.from_folder(train_path, valid_pct=0.2, item_tfms=Resize(24))
        test_files = get_image_files(test_path)
        test_dl = dls.test_dl(test_files, with_labels=True)

        learn = vision_learner(
            dls, model_func, metrics=[error_rate, Precision(average="macro"), Recall(average="macro")]
        )
        learn.fine_tune(epochs)

        train_losses = [loss.item() for loss in learn.recorder.losses]  # noqa: F841
        preds, targs = learn.get_preds(dl=test_dl)

        if targs is not None:
            pred_classes = preds.argmax(dim=1).numpy()
            true_classes = targs.numpy()
            precision = precision_score(true_classes, pred_classes, average="macro")
            recall = recall_score(true_classes, pred_classes, average="macro")
            accuracy = accuracy_score(true_classes, pred_classes)
        else:
            precision = recall = accuracy = None

        results = {
            "model": model_name,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("epochs", type=int)
    _args = parser.parse_args()

    if _args.model_name not in CLASSIFICATION_MODELS:
        raise ValueError(f"Model name must be one of {list(CLASSIFICATION_MODELS.keys())}")

    _output_path = f"/home/etaylor/code_projects/thesis/src/classification/fastai/models_scores/{_args.model_name}_results.json"
    FastaiTrainer.train_and_evaluate(
        _args.model_name,
        CLASSIFICATION_MODELS[_args.model_name],
        TRAIN_DATASET_PATH,
        TEST_DATASET_PATH,
        _args.epochs,
        _output_path,
    )
