import csv
import os
from pathlib import Path
import pandas as pd 
import math

import hydra
import tqdm

def run_experiment(config):
    model = hydra.utils.instantiate(config.evaluation)
    model.load_pipeline()

    #Load LoRA
    if config.generate_images.lora_weights is not None and config.generate_images.lora_scale is not None:
        lora_path = Path(config.lora.out_dir) / config.generate_images.lora_weights
        model.load_lora(
            lora_path=lora_path,
            lora_scale=config.generate_images.lora_scale,
        )

    #real index from csv
    df = pd.read_csv(config.generate_images.csv_path,index_col="index")

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        prompt = row.get("prompt")
        seed = int(row.get("evaluation_seed") or 2024)
        if "evaluation_guidance" in df.columns:
            guidance_scale = float(row.get("evaluation_guidance") or config.generate_images.guidance_scale)
        else:
            guidance_scale = float(config.generate_images.guidance_scale)

        generate_images_params = {
            "prompt": prompt,
            "num_inference_steps": config.generate_images.num_inference_steps,
            "guidance_scale": guidance_scale,
            "out_dir": config.generate_images.out_dir,
            "seed": int(seed),
            "idx": idx
        }

        img = model.generate_image(**generate_images_params)

@hydra.main(config_path="../configs", config_name="reunlearning_explicit_content", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()
