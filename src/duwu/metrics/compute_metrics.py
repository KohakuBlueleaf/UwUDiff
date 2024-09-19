from collections.abc import Callable, Sequence

from dataclasses import dataclass


@dataclass
class MetricConfig:

    name: str
    metric_func: Callable
    generated_dataset_func: Callable
    ref_dataset: Sequence | None = None


def compute_metrics(
    metric_configs: list[MetricConfig],
    generated_image_paths: list[str],
) -> dict[str, float]:
    metrics = {}
    for metric_config in metric_configs:
        generated_dataset = metric_config.generated_dataset_func(generated_image_paths)
        if metric_config.ref_dataset is None:
            metric = metric_config.metric_func(generated=generated_dataset)
        else:
            metric = metric_config.metric_func(
                generated=generated_dataset, reference=metric_config.ref_dataset
            )
        metrics[metric_config.name] = metric
    return metrics
