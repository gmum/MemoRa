from pathlib import Path

import hydra

from metrics.nudenet_detector import nudenet_classify


def calculate_nudity(config):
    folder = config.generate_images.out_dir
    result_path = Path(config.nudenet_classifier.out_dir) / config.nudenet_classifier.file_name
    stats = nudenet_classify(folder, result_path)
    print(stats)


@hydra.main(config_path="../configs", config_name="reunlearning_nudity", version_base=None)
def main(config):
    calculate_nudity(config)


if __name__ == "__main__":
    main()