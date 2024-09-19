import argparse
import os

from omegaconf import OmegaConf

from duwu.loader import load_any
from duwu.utils import instantiate_any


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["configs/sampling/demo_sampling.yaml"],
    )
    args = parser.parse_args()

    configs = []
    for config in args.configs:
        conf = OmegaConf.load(config)
        configs.append(conf)
    config = OmegaConf.merge(*configs)

    unet = load_any(config.model_config.unet)
    te = load_any(config.model_config.te)
    vae = load_any(config.model_config.vae)

    sampling_func = instantiate_any(config.sampling_func)
    images = sampling_func(unet=unet, te=te, vae=vae)

    save = config.get("save_dir", None)
    if save is not None:
        os.makedirs(save, exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(save, f"{i}.png"))
