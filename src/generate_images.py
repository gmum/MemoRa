import csv
import os
from pathlib import Path
import pandas as pd 
import math

import hydra
import tqdm

def run_experiment(config):
    model = hydra.utils.instantiate(config.model)
    model.load_pipeline()

    if config.exp.lora_weights is not None:
        lora_dir = str(Path(config.exp.lora_weights).parent)
        lora_file = str(Path(config.exp.lora_weights).name)
        model.load_lora(
            lora_dir=lora_dir,
            lora_scale=config.exp.lora_scale,
            lora_weights=lora_file,
        )

    
    gen_params = {
            k: v
            for k, v in config.exp.items()
            if k not in ["num_images", "lora_weights", "lora_scale", "seed"]
    }

    df = pd.read_csv(config.exp.csv_file)
    num_gpus = int(config.exp.gpus)
    gpu_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    per_gpu = math.ceil(len(df) / num_gpus)
    start_id = gpu_id * per_gpu
    end_id = min(start_id + per_gpu, len(df))
    df = df.iloc[start_id:end_id]

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        prompt = row.get("prompt")
        seed = int(row.get("evaluation_seed") or 2024)
        guidance_scale = float(row.get("evaluation_guidance") or 7.5)

        gen_params = {
            "prompt": prompt,
            "negative_prompt": config.exp.negative_prompt,
            "num_inference_steps": config.exp.num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": config.exp.height,
            "width": config.exp.width,
            "out_dir": config.exp.out_dir,
            "seed": int(seed),
            "idx": idx
        }

        img = model.generate_image(**gen_params)

@hydra.main(config_path="../configs", config_name="config_generate", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()
