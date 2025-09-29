#import csv
from pathlib import Path

import hydra

from utils.latent_iversion import slerp, encode_to_latent, sample, invert

def inversion_process(config):
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
        z_0_a = encode_to_latent(model_unlearned.pipeline, image_a)
        z_0_b = encode_to_latent(model_unlearned.pipeline, image_b)

        #starting latents obtained by inversion
        intermediate_latents_a = invert(pipe= model_unlearned.pipeline, start_latents=z_0_a, **images_for_inversion_params)
        intermediate_latents_b = invert(pipe= model_unlearned.pipeline, start_latents=z_0_b, **images_for_inversion_params)
         
        #where T=50, 100 ...
        z_T_a = intermediate_latents_a[-(config.interpolation.start_step + 1)].unsqueeze(0)
        z_T_b = intermediate_latents_b[-(config.interpolation.start_step + 1)].unsqueeze(0)
        
        #k = number of intervals between endpoints
        k = int(config.interpolation.interp_steps)
        for i in range(k + 1):
            interval = i/k
            interpolation_z = slerp(z_T_a[0], z_T_b[0], interval).unsqueeze(0)

            interpolation_img = sample(model_unlearned.pipeline, start_step=config.interpolation.start_step, start_latents=interpolation_z, **images_for_inversion_params)
            interpolation_img = interpolation_img[0]

            file_name = f"{pair_index}_interval{interval:.2f}.png"
            out_path = f"{out_dir}/{file_name}" 
            interpolation_img.save(out_path)
    model_unlearned.save_pipeline(save_dir = config.interpolation.model_path)


@hydra.main(config_path="../configs", config_name="reunlearning_nudity", version_base=None)
def main(config):
    inversion_process(config)


if __name__ == "__main__":
    main()
