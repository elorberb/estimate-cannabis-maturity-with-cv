from fastai.metrics import Precision, Recall
from fastai.vision.all import error_rate, vision_learner

from src.classification.utils import CLASSIFICATION_MODELS


class ModelUtils:
    @staticmethod
    def create_learner(dls, model_name: str):
        return vision_learner(
            dls=dls,
            arch=CLASSIFICATION_MODELS[model_name],
            metrics=[error_rate, Precision(average="macro"), Recall(average="macro")],
        )

    @staticmethod
    def train_model(learner, epochs: int = 25) -> list:
        learner.fine_tune(epochs=epochs)
        return learner.validate()
