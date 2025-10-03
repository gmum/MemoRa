import hydra
import copy
import torchvision
from diffusers import LMSDiscreteScheduler
from pathlib import Path
import pandas as pd
from utils.auto_memora_utils import sample_auto_memora_cfg

def auto_guidance(config):
    model = hydra.utils.instantiate(config.model)
    model.load_pipeline()

    model.scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
    )


    model_lora = copy.deepcopy(model)
    lora_path = Path(config.guidance.lora_path)
    model_lora.load_lora(lora_path=lora_path)

    df = pd.read_csv(config.guidance.csv_path, index_col="index")

    start_id = config.guidance.start_index
    end_id   = min(start_id + config.guidance.n_images, len(df))
    df = df.iloc[start_id:end_id]

    out_dir = Path(config.guidance.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        prompt = str(row["prompt"])
        seed = int(row.get("evaluation_seed", config.guidance.default_seed))

        imgs = sample_auto_memora_cfg(
            model.pipeline, model_lora.pipeline,
            prompt=prompt, negative_prompt="",
            steps=config.guidance.steps, height=512, width=512,
            guidance_scale=config.guidance.guidance_scale,
            w=config.guidance.w,
            seed=seed,
            main=config.guidance.main,
        )

        fname = out_dir / f"idx_{idx}_seed_{seed}.png"
        torchvision.utils.save_image(imgs, fname)


@hydra.main(config_path="../configs", config_name="auto_memora", version_base=None)
def main(config):
    auto_guidance(config)


if __name__ == "__main__":
    main()
