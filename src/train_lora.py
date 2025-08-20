import csv
import os

import hydra
import tqdm

from utils.lora_utils import create_json_metadata, train_lora


def run_experiment(config):
    #captions for images from dataset for training
    create_json_metadata(config.interpolation.out_dir, config.erasing_concept.prompt)

    #LoRA training
    train_lora(
        model_dir=config.interpolation.model_path,
        dataset_dir=config.interpolation.out_dir,
        output_dir=config.lora.out_dir,
        validation_prompt=config.erasing_concept.prompt,
        steps=config.lora.steps,
        rank=config.lora.rank,
        batch_size=config.lora.batch_size,
        learning_rate=config.lora.learning_rate,
        seed=config.lora.seed,
    )


@hydra.main(config_path="../configs", config_name="reunlearning_explicit_content", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()