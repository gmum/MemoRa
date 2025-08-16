import csv
import os
from pathlib import Path

import hydra
import tqdm

from utils.nudity_eval import nudent_classify

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


    for i in tqdm.tqdm(range(config.exp.num_images)):
        model.generate_image(**gen_params, seed=config.exp.seed + i)

    stats = nudent_classify(config.exp.out_dir)
    print(stats)

@hydra.main(config_path="../configs", config_name="config_generate", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()
