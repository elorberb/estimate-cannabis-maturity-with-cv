"""Modal GPU inference functions for trichome and stigma analysis."""

import modal

app = modal.App("trichome-inference")

# Modal image with all ML dependencies
inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
    )
)

# TODO B1: mount model weights from Modal volume
# model_volume = modal.Volume.from_name("trichome-model-weights", create_if_missing=True)


# TODO B2: trichome inference function
# @app.function(gpu="T4", image=inference_image, volumes={"/models": model_volume})
# def detect_trichomes(image_bytes: bytes) -> dict:
#     ...


# TODO B3: stigma inference function
# @app.function(gpu="T4", image=inference_image, volumes={"/models": model_volume})
# def detect_stigmas(image_bytes: bytes) -> dict:
#     ...


# TODO B6: combined analysis (orchestrates B2-B5)
# @app.function(gpu="T4", image=inference_image, volumes={"/models": model_volume})
# def analyze_image(image_bytes: bytes) -> dict:
#     ...


@app.local_entrypoint()
def main() -> None:
    """Test entrypoint — run with: modal run app/modal/inference.py"""
    print("Modal inference app loaded successfully.")
