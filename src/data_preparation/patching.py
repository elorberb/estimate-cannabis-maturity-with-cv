import os
import cv2
import pandas as pd


def calculate_padding(image_shape, patch_size):
    pad_height = (patch_size - image_shape[0] % patch_size) % patch_size
    pad_width = (patch_size - image_shape[1] % patch_size) % patch_size
    return pad_height, pad_width


def pad_image(image, pad_height, pad_width):
    return cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def extract_patches(padded_image, patch_size):
    patches = []
    for i in range(0, padded_image.shape[0], patch_size):
        for j in range(0, padded_image.shape[1], patch_size):
            patch = padded_image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, (i, j)))
    return patches


def cut_images(image, patch_size=512):
    pad_height, pad_width = calculate_padding(image.shape, patch_size)
    padded_image = pad_image(image, pad_height, pad_width)
    return extract_patches(padded_image, patch_size)


def create_patches_metadata(image_name, patches):
    rows = []
    for i, (_, coords) in enumerate(patches):
        rows.append({
            "patch_name": f"{image_name}_p{i}.png",
            "y": coords[0],
            "x": coords[1],
        })
    return pd.DataFrame(rows)


def save_patch(patch, path):
    cv2.imwrite(path, patch)


def save_patches(image_name, patches, saving_dir, csv_file_path):
    os.makedirs(saving_dir, exist_ok=True)

    new_metadata = create_patches_metadata(image_name, patches)

    for i, (patch, _) in enumerate(patches):
        patch_filename = f"{image_name}_p{i}.png"
        patch_path = os.path.join(saving_dir, patch_filename)
        save_patch(patch, patch_path)

    if os.path.exists(csv_file_path):
        existing = pd.read_csv(csv_file_path)
        metadata = pd.concat([existing, new_metadata], ignore_index=True)
    else:
        metadata = new_metadata

    metadata.to_csv(csv_file_path, index=False)


def apply_to_images(images_and_names, func):
    return [(func(image), name) for image, name in images_and_names]
