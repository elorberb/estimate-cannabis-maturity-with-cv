import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def setup_sam():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam, mask_generator


def create_instance_bitmap(mask_list):
    instance_bitmap = np.zeros_like(mask_list[0]['segmentation'], dtype=bool)
    for seg in mask_list:
        instance_bitmap = np.logical_or(instance_bitmap, seg['segmentation'])
    return instance_bitmap


def process_image_masks(image, mask_generator, name):
    mask = mask_generator.generate(image)
    num_segments = len(mask)
    instance_bitmap = create_instance_bitmap(mask)

    return {
        'mask': mask,
        'num_segments': num_segments,
        'instance_bitmap': instance_bitmap,
    }


def segment_images(images_names, mask_generator):
    segmentation_dict = {}
    for image, name in images_names:
        segmentation_dict[name] = process_image_masks(image, mask_generator, name)
    return segmentation_dict
