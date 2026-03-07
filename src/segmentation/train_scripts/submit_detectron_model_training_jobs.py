import os
import subprocess

from src.config.settings import PROJECT_ROOT
from src.segmentation.handlers.detectron2_handler import DETECTION_MODELS


class DetectronJobSubmitter:
    @staticmethod
    def create_and_submit_slurm_job(model_config: str) -> None:
        os.environ["JOB_NAME"] = f"train_{model_config.replace('/', '_')}"
        os.environ["OUTPUT_DIR"] = str(PROJECT_ROOT / "src/segmentation/notebooks/detectron2/train_logs")
        os.environ["MODEL_CONFIG"] = model_config

        template_path = str(PROJECT_ROOT / "cluster_instructions/launch_scripts/sbatch_gpu_train_detectron2_models.sh")
        command = f'envsubst < {template_path} > /tmp/{os.environ["JOB_NAME"]}.sh'
        subprocess.run(command, shell=True, check=True)
        subprocess.run(f'sbatch /tmp/{os.environ["JOB_NAME"]}.sh', shell=True, check=True)


if __name__ == "__main__":
    for model in DETECTION_MODELS:
        DetectronJobSubmitter.create_and_submit_slurm_job(model)
