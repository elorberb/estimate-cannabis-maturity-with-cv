import os
import cv2


def extract_filename(file_path):
    basename = os.path.basename(file_path)
    name, _ = os.path.splitext(basename)
    return name


def get_image_paths(directory, extensions=(".png", ".jpg", ".jpeg")):
    extensions_lower = tuple(ext.lower() for ext in extensions)
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions_lower)
    ]


def read_image(path):
    return cv2.imread(path)


def read_images(input_path_or_list, transform_fn=None):
    if isinstance(input_path_or_list, str):
        paths = get_image_paths(input_path_or_list)
    elif isinstance(input_path_or_list, list):
        paths = input_path_or_list
    else:
        raise ValueError("Input must be a directory path or a list of image paths.")

    images = {}
    for path in paths:
        filename = os.path.basename(path)
        image = read_image(path)
        if transform_fn:
            image = transform_fn(image)
        images[filename] = image

    return images


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)
