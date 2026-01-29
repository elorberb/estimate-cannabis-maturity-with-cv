import cv2
import numpy as np


def to_grayscale(image):
    if len(image.shape) > 2:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def gradient_sharpness(image):
    gray = to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.average(magnitude)


def laplacian_sharpness(image):
    gray = to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)


def edge_sharpness(image):
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges) / float(edges.size)


def tenengrad_sharpness(image):
    gray = to_grayscale(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)


def fft_sharpness(image):
    gray = to_grayscale(image)
    f_transform = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shifted)
    return np.mean(magnitude)


def contrast_sharpness(image):
    gray = to_grayscale(image)
    return np.std(gray)


def calculate_sharpness(image):
    return edge_sharpness(image)
