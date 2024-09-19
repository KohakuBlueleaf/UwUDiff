import argparse

from omegaconf import OmegaConf
from hydra.utils import instantiate

from duwu.utils import get_images_recursively, instantiate_any
from duwu.metrics import compute_metrics, MetricConfig


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", type=str, nargs="+", default=["configs/demo_metrics.yaml"]
    )
    args = parser.parse_args()

    configs = []
    for config in args.configs:
        conf = OmegaConf.load(config)
        configs.append(conf)
    config = OmegaConf.merge(*configs)

    generated_image_dir = config.generated_image_dir
    generated_images = get_images_recursively(generated_image_dir)

    metrics_configs = [
        MetricConfig(**instantiate_any(metric_config)) for metric_config in config.metrics
    ]
    metrics = compute_metrics(metrics_configs, generated_images)
    for name, metric in metrics.items():
        print(f"{name}: {metric}")
