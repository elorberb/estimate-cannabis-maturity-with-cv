import cv2
import numpy as np


class Sharpness:
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def gradient_sharpness(image: np.ndarray) -> float:
        gray = Sharpness.to_grayscale(image)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.average(magnitude))

    @staticmethod
    def laplacian_sharpness(image: np.ndarray) -> float:
        gray = Sharpness.to_grayscale(image)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    @staticmethod
    def edge_sharpness(image: np.ndarray) -> float:
        gray = Sharpness.to_grayscale(image)
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges) / float(edges.size))

    @staticmethod
    def tenengrad_sharpness(image: np.ndarray) -> float:
        gray = Sharpness.to_grayscale(image)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx**2 + gy**2))

    @staticmethod
    def fft_sharpness(image: np.ndarray) -> float:
        gray = Sharpness.to_grayscale(image)
        f_transform = np.fft.fft2(gray)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        return float(np.mean(magnitude))

    @staticmethod
    def contrast_sharpness(image: np.ndarray) -> float:
        gray = Sharpness.to_grayscale(image)
        return float(np.std(gray))

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        return Sharpness.edge_sharpness(image)
