import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SamHandler:
    @staticmethod
    def setup_sam(
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: str = "cuda",
    ) -> tuple:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return sam, mask_generator

    @staticmethod
    def create_instance_bitmap(mask_list: list) -> np.ndarray:
        instance_bitmap = np.zeros_like(mask_list[0]["segmentation"], dtype=bool)
        for seg in mask_list:
            instance_bitmap = np.logical_or(instance_bitmap, seg["segmentation"])
        return instance_bitmap

    @staticmethod
    def process_image_masks(image: np.ndarray, mask_generator: SamAutomaticMaskGenerator, name: str) -> dict:
        mask = mask_generator.generate(image)
        return {
            "mask": mask,
            "num_segments": len(mask),
            "instance_bitmap": SamHandler.create_instance_bitmap(mask),
        }

    @staticmethod
    def segment_images(images_names: list, mask_generator: SamAutomaticMaskGenerator) -> dict:
        return {
            name: SamHandler.process_image_masks(image, mask_generator, name)
            for image, name in images_names
        }
