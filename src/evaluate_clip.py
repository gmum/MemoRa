import hydra
from metrics.clip_metric import calculate_clip


def clip(config):
    result, _ = calculate_clip(
        config.generate_images.csv_path,
        config.generate_images.out_dir,
        config.clip.out_dir,
        config.clip.file_name,
    )
    print(result)


@hydra.main(
    config_path="../configs", config_name="reunlearning_nudity", version_base=None
)
def main(config):
    clip(config)


if __name__ == "__main__":
    main()
