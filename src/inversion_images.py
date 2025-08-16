#import csv
import os
from pathlib import Path

import hydra
import tqdm

from utils.latent_iversion import slerp, encode_to_latent, sample, invert

def run_experiment(config):
    model_esd = hydra.utils.instantiate(config.model)
    model_esd.load_pipeline()

    model_orig = hydra.utils.instantiate(config.orig)
    model_orig.load_pipeline()

    gen_params = {k: v for k, v in config.exp.items()
                  if k not in ["seeds", "interp_steps", "out_dir"]}

    inv_params = {
        "prompt": config.exp.prompt,
        "guidance_scale": config.exp.guidance_scale,
        "num_inference_steps": config.exp.num_inference_steps,
        "negative_prompt": config.exp.negative_prompt,
    }

    out_dir = Path(config.exp.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for duet_idx, (seed_a, seed_b) in enumerate(config.exp.seeds):
        img_a = model_orig.generate_image(**gen_params, seed=int(seed_a))
        img_b = model_orig.generate_image(**gen_params, seed=int(seed_b))

        z_start_a = encode_to_latent(model_esd.pipeline, img_a)
        z_start_b = encode_to_latent(model_esd.pipeline, img_b)
         
        inv_a = invert(pipe= model_esd.pipeline, start_latents=z_start_a, **inv_params)
        inv_b = invert(pipe= model_esd.pipeline, start_latents=z_start_b, **inv_params)
         
        z_0_a = inv_a[-1].unsqueeze(0)
        z_0_b = inv_b[-1].unsqueeze(0)
        
        K = int(config.exp.interp_steps)
        for i in range(K + 1):
            t = i/K
            z_interp = slerp(z_0_a[0], z_0_b[0], t).unsqueeze(0)

            imgs = sample(model_esd.pipeline, start_latents=z_interp, **inv_params)
            img = imgs[0]

            name = f"{duet_idx}_slerp{i:02d}_t{t:.2f}.png"
            out_path = f"{config.exp.out_dir}/{name}" 
            img.save(out_path)
    model_esd.save_pipeline()


@hydra.main(config_path="../configs", config_name="config_inversion", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()
