from pathlib import Path
import pandas as pd

import hydra
import tqdm

def generate_images(config):
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

    start_id = config.generate_images.start_index
    end_id = start_id + config.generate_images.n_images
    if len(df) < end_id:
        end_id = len(df) 	
    df = df.iloc[start_id:end_id]

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        prompt = row.get("prompt")
        seed = int(row.get("evaluation_seed") or 2024)
        if "evaluation_guidance" in df.columns:
            guidance_scale = float(
                row.get("evaluation_guidance")
                if row.get("evaluation_guidance")
                else config.generate_images.guidance_scale
            )
        else:
            guidance_scale = float(config.generate_images.guidance_scale)

        generate_image_params = {
            "prompt": prompt,
            "num_inference_steps": config.generate_images.num_inference_steps,
            "guidance_scale": guidance_scale,
            "out_dir": config.generate_images.out_dir,
            "seed": int(seed),
            "idx": idx
        }

        img = model.generate_image(**generate_image_params, scheduler_name="lmsd")

@hydra.main(config_path="../configs", config_name="reunlearning_nudity", version_base=None)
def main(config):
    generate_images(config)

if __name__ == "__main__":
    main()
