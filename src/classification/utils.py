from fastai.vision.all import *
from fastai.vision import *


CLASSIFICATION_MODELS = {
    "alexnet": models.alexnet,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
    "vgg16_bn": models.vgg16_bn,
    "vgg19_bn": models.vgg19_bn,
    "squeezenet1_1": models.squeezenet1_1,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
}

SMALL_MODELS = {
    "alexnet": models.alexnet,
    "resnet34": models.resnet34,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
}


def load_classification_model(model_path):
    return load_learner(model_path)


def resize_with_padding(size):
    return Resize(size, method="pad", pad_mode="zeros")


class RGB2HSV(Transform):
    def encodes(self, img: PILImage):
        return rgb2hsv(img)


def create_dataloader(dataset_path, item_tfms, batch_tfms=None, bs=16, valid_pct=0.25):
    return ImageDataLoaders.from_folder(
        path=dataset_path,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms or [],
        bs=bs,
        valid_pct=valid_pct,
    )


def get_dataloaders(dataset_path, size=128, bs=16, valid_pct=0.25):
    padding_tfm = resize_with_padding(size)
    resize_tfm = Resize(size)
    aug_tfms = aug_transforms(size=size)

    return {
        "raw": create_dataloader(dataset_path, resize_tfm, bs=bs, valid_pct=valid_pct),
        "resize_with_padding": create_dataloader(
            dataset_path, padding_tfm, aug_tfms, bs, valid_pct
        ),
        "convert_to_hsv": create_dataloader(
            dataset_path, resize_tfm, [RGB2HSV(), *aug_tfms], bs, valid_pct
        ),
        "normalize_pixels": create_dataloader(
            dataset_path, resize_tfm,
            [Normalize.from_stats(*imagenet_stats), *aug_tfms], bs, valid_pct
        ),
        "brightness_contrast": create_dataloader(
            dataset_path, resize_tfm,
            [Brightness(max_lighting=0.2, p=0.75), Contrast(max_lighting=0.2, p=0.75), *aug_tfms],
            bs, valid_pct
        ),
        "combined_all": create_dataloader(
            dataset_path, padding_tfm,
            [RGB2HSV(), *aug_transforms(size=size, flip_vert=True, max_rotate=10),
             Brightness(max_lighting=0.2, p=0.75), Contrast(max_lighting=0.2, p=0.75)],
            bs, valid_pct
        ),
    }
