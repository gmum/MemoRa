import csv
import os

import hydra
import tqdm
from nudenet import NudeDetector

valid_extensions = {".jpg", ".jpeg", ".png", ".gif"}

detector = NudeDetector()

detector_v2_default_classes = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]


def run_experiment(config):
    model = hydra.utils.instantiate(config.model)
    model.load_pipeline()
    model.generate_image(**config.exp)
    # TODO: Zrobic detekcje na podstawie wygenerowanych obrazow i zapisac wynik do CSV.


@hydra.main(config_path="../configs", config_name="config_generate", version_base=None)
def main(config):
    run_experiment(config)


if __name__ == "__main__":
    main()
