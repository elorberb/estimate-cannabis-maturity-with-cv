import os
import subprocess

from src.segmentation.handlers.ultralytics_handler import ULTRALYTICS_MODELS


class UltraalyticsJobSubmitter:
    @staticmethod
    def create_and_submit_slurm_job(model_config: str, train_model: bool = True) -> None:
        os.environ["OUTPUT_DIR"] = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/train_logs"
        os.environ["MODEL_CONFIG"] = model_config

        if train_model:
            template_path = "/home/etaylor/cluster_instructions/launch_scripts/sbatch_gpu_train_ultralytics.sh"
            os.environ["JOB_NAME"] = f"train_{model_config.replace('.', '_')}"
        else:
            template_path = "/home/etaylor/cluster_instructions/launch_scripts/sbatch_gpu_tune_ultralytics.sh"
            os.environ["JOB_NAME"] = f"tune_{model_config.replace('.', '_')}"

        command = f'envsubst < {template_path} > /tmp/{os.environ["JOB_NAME"]}.sh'
        subprocess.run(command, shell=True, check=True)
        subprocess.run(f'sbatch /tmp/{os.environ["JOB_NAME"]}.sh', shell=True, check=True)


if __name__ == "__main__":
    for model in ULTRALYTICS_MODELS:
        UltraalyticsJobSubmitter.create_and_submit_slurm_job(model, train_model=False)
