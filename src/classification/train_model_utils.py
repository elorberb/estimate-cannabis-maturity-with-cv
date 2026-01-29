from fastai.metrics import Precision, Recall
from fastai.vision.all import vision_learner, error_rate

from src.classification.utils import CLASSIFICATION_MODELS


def create_learner(dls, model_name):
    precision_macro = Precision(average="macro")
    recall_macro = Recall(average="macro")

    return vision_learner(
        dls=dls,
        arch=CLASSIFICATION_MODELS[model_name],
        metrics=[error_rate, precision_macro, recall_macro],
    )


def train_model(learner, epochs=25):
    learner.fine_tune(epochs=epochs)
    return learner.validate()


def train_across_dataloaders(dataloaders_dict, model_name, epochs=25):
    results = {}
    trained_models = {}

    for name, dls in dataloaders_dict.items():
        print(f"Training: {name}")
        learner = create_learner(dls, model_name)
        results[name] = train_model(learner, epochs)
        trained_models[name] = learner

    for config, result in results.items():
        print(f"{config}: {result}")

    return results, trained_models
