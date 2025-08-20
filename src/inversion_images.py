#import csv
import os
from pathlib import Path

import hydra
import tqdm

from utils.latent_iversion import slerp, encode_to_latent, sample, invert

def run_experiment(config):
    model_unlearned = hydra.utils.instantiate(config.unlearned)
    model_unlearned.load_pipeline()

    model_original = hydra.utils.instantiate(config.original)
    model_original.load_pipeline()

    images_for_inversion_params = {
        "prompt": config.erasing_concept.prompt,
        "num_inference_steps": config.generate_images.num_inference_steps,
        "guidance_scale": config.generate_images.guidance_scale,
    }

    #creating a folder for photos after inversion and interpolation
    out_dir = Path(config.interpolation.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # seed_a, seed_b - seeds for two photos (img_a, img_b) during interpolation process
    for pair_index, (seed_a, seed_b) in enumerate(config.interpolation.seeds):
        image_a = model_original.generate_image(**images_for_inversion_params, seed=int(seed_a))
        image_b = model_original.generate_image(**images_for_inversion_params, seed=int(seed_b))

        #latents of photos after VAE
        z_start_a = encode_to_latent(model_unlearned.pipeline, image_a)
        z_start_b = encode_to_latent(model_unlearned.pipeline, image_b)

        #starting latents obtained by inversion
        inversion_a = invert(pipe= model_unlearned.pipeline, start_latents=z_start_a, **images_for_inversion_params)
        inversion_b = invert(pipe= model_unlearned.pipeline, start_latents=z_start_b, **images_for_inversion_params)
         
        z_T_a = inversion_a[-1].unsqueeze(0)
        z_T_b = inversion_b[-1].unsqueeze(0)
        
        #k = number of intervals between endpoints
        k = int(config.interpolation.interp_steps)
        for i in range(k + 1):
            interval = i/k
            interpolation_z = slerp(z_T_a[0], z_T_b[0], interval).unsqueeze(0)

            interpolation_img = sample(model_unlearned.pipeline, start_latents=interpolation_z, **images_for_inversion_params)
            interpolation_img = interpolation_img[0]

            file_name = f"{pair_index}_interval{interval:.2f}.png"
            out_path = f"{out_dir}/{file_name}" 
            interpolation_img.save(out_path)
    model_unlearned.save_pipeline(save_dir = config.interpolation.model_path)


@hydra.main(config_path="../configs", config_name="reunlearning_explicit_content", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()
